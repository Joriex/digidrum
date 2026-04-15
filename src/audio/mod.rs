//! Audio engine: cpal output stream + lock-free trigger queue.
//!
//! Tries hard to get a small, fixed buffer so latency is low and
//! deterministic. If the device won't honor a small buffer, we fall back to
//! its default and log what we got.

pub mod samples;
pub mod voices;

use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, Stream, StreamConfig, SupportedStreamConfigRange};
use crossbeam_queue::ArrayQueue;

pub use samples::{Sample, SampleBank, SampleId};

use self::voices::Mixer;

/// Fired from any thread (UI, detection); consumed by the audio callback.
#[derive(Clone, Copy, Debug)]
pub struct TriggerEvent {
    pub sample_id: SampleId,
    /// Linear velocity 0.0..1.0. Mapped to gain by the audio thread.
    pub velocity: f32,
}

/// Desired target. We try these in order until cpal accepts one.
const PREFERRED_SAMPLE_RATES: [u32; 3] = [48_000, 44_100, 96_000];
const PREFERRED_BUFFER_SIZES: [u32; 3] = [128, 256, 512];

pub struct AudioEngine {
    trigger_queue: Arc<ArrayQueue<TriggerEvent>>,
    /// Hot-swappable sample bank. UI thread clones, mutates, and stores; the
    /// audio callback reads the current snapshot with `load()` (wait-free).
    bank: Arc<ArcSwap<SampleBank>>,
    sample_rate: u32,
    channels: u16,
    buffer_frames: Option<u32>,
    #[allow(dead_code)] // kept alive for the lifetime of the engine
    stream: Stream,
}

impl AudioEngine {
    pub fn new(initial_bank: SampleBank) -> Result<Self> {
        let bank = Arc::new(ArcSwap::from_pointee(initial_bank));
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default output device")?;
        log::info!(
            "audio device: {}",
            device.name().unwrap_or_else(|_| "?".into())
        );

        let supported: Vec<SupportedStreamConfigRange> = device
            .supported_output_configs()
            .context("querying supported output configs")?
            .collect();

        let (config, format, buffer_frames) = pick_config(&supported)
            .ok_or_else(|| anyhow!("no usable output stream config"))?;

        log::info!(
            "audio stream: {} Hz, {} ch, format {:?}, buffer {:?} frames",
            config.sample_rate,
            config.channels,
            format,
            buffer_frames
        );

        let queue = Arc::new(ArrayQueue::<TriggerEvent>::new(256));
        let mut mixer = Mixer::new(config.sample_rate);
        let channels = config.channels as usize;

        let queue_cb = Arc::clone(&queue);
        let bank_cb = Arc::clone(&bank);

        let err_fn = |err| log::error!("audio stream error: {err}");

        // For now we only support f32 output. Most modern devices provide it.
        if format != SampleFormat::F32 {
            return Err(anyhow!(
                "output format {format:?} not supported (wants f32); patch Mixer to convert"
            ));
        }

        let stream = device
            .build_output_stream(
                &config,
                move |out: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    // Drain trigger queue into the mixer. `load()` is wait-free.
                    let bank_snap = bank_cb.load();
                    while let Some(ev) = queue_cb.pop() {
                        if let Some(s) = bank_snap.get(ev.sample_id) {
                            let gain = ev.velocity.clamp(0.0, 1.0);
                            mixer.trigger(Arc::clone(&s.data), s.source_sr, gain);
                        }
                    }
                    // Zero then additive mix.
                    for x in out.iter_mut() {
                        *x = 0.0;
                    }
                    mixer.render(out, channels);
                },
                err_fn,
                None,
            )
            .context("build_output_stream")?;

        stream.play().context("start audio stream")?;

        Ok(Self {
            trigger_queue: queue,
            bank,
            sample_rate: config.sample_rate,
            channels: config.channels,
            buffer_frames,
            stream,
        })
    }

    pub fn trigger(&self, ev: TriggerEvent) {
        // If the queue is full, drop oldest to keep the newest-most-relevant hit.
        if self.trigger_queue.push(ev).is_err() {
            let _ = self.trigger_queue.pop();
            let _ = self.trigger_queue.push(ev);
        }
    }

    pub fn trigger_queue(&self) -> Arc<ArrayQueue<TriggerEvent>> {
        Arc::clone(&self.trigger_queue)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn buffer_frames(&self) -> Option<u32> {
        self.buffer_frames
    }

    /// Current sample bank snapshot — cheap to clone (Arc).
    pub fn bank_snapshot(&self) -> Arc<SampleBank> {
        self.bank.load_full()
    }

    /// Add a sample to the bank. Clones the current bank, pushes the new
    /// sample, and publishes the updated bank atomically. Returns the new
    /// sample's id. The old `Arc<SampleBank>` is returned from `swap()` and
    /// dropped on *this* (UI) thread, never on the audio thread.
    pub fn add_sample(&self, sample: Sample) -> SampleId {
        let current = self.bank.load_full();
        let mut next = (*current).clone();
        let id = next.push(sample);
        self.bank.store(Arc::new(next));
        id
    }
}

/// Pick the lowest-latency config we can: prefer 48k, then a tiny fixed
/// buffer, f32. Falls back through progressively safer options.
fn pick_config(
    ranges: &[SupportedStreamConfigRange],
) -> Option<(StreamConfig, SampleFormat, Option<u32>)> {
    // Filter to f32 stereo/mono with 1+ channel.
    let f32_ranges: Vec<_> = ranges
        .iter()
        .filter(|r| r.sample_format() == SampleFormat::F32 && r.channels() >= 1)
        .collect();

    for &sr in &PREFERRED_SAMPLE_RATES {
        for range in &f32_ranges {
            let sr_min = range.min_sample_rate();
            let sr_max = range.max_sample_rate();
            if sr < sr_min || sr > sr_max {
                continue;
            }
            let channels = range.channels().min(2);
            for &bs in &PREFERRED_BUFFER_SIZES {
                let buffer_size = match range.buffer_size() {
                    cpal::SupportedBufferSize::Range { min, max } if bs >= *min && bs <= *max => {
                        BufferSize::Fixed(bs)
                    }
                    _ => continue,
                };
                let cfg = StreamConfig {
                    channels,
                    sample_rate: sr,
                    buffer_size,
                };
                return Some((cfg, SampleFormat::F32, Some(bs)));
            }
            // Device can't do a small fixed buffer — accept default.
            let cfg = StreamConfig {
                channels,
                sample_rate: sr,
                buffer_size: BufferSize::Default,
            };
            return Some((cfg, SampleFormat::F32, None));
        }
    }
    None
}
