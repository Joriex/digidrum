//! Polyphonic voice pool used inside the cpal render callback.
//!
//! Invariants (for realtime safety):
//!   * No allocation, no mutex, no syscalls.
//!   * Dropping a voice's `Arc<Vec<f32>>` must not deallocate — the sample
//!     bank keeps another `Arc` alive, so refcount only ever goes 2→1 here.

use std::sync::Arc;

pub const MAX_VOICES: usize = 32;

#[derive(Default)]
struct Voice {
    sample: Option<Arc<Vec<f32>>>,
    /// Fractional read position in the source sample, advanced by `rate`.
    pos: f64,
    /// Source-samples per output-sample (source_sr / engine_sr).
    rate: f64,
    gain: f32,
    active: bool,
}

pub struct Mixer {
    voices: [Voice; MAX_VOICES],
    /// Round-robin hint so we don't always start searching at voice 0.
    next: usize,
    engine_sr: u32,
}

impl Mixer {
    pub fn new(engine_sr: u32) -> Self {
        // Default::default() on arrays of non-Copy is a pain; use from_fn.
        let voices = std::array::from_fn(|_| Voice::default());
        Self {
            voices,
            next: 0,
            engine_sr,
        }
    }

    /// Claim a voice and start playing `sample`. If all voices are busy,
    /// steal the one that has been playing longest (highest pos/len ratio).
    pub fn trigger(&mut self, sample: Arc<Vec<f32>>, source_sr: u32, gain: f32) {
        let rate = source_sr as f64 / self.engine_sr as f64;

        // First try to find an inactive voice.
        for off in 0..MAX_VOICES {
            let i = (self.next + off) % MAX_VOICES;
            if !self.voices[i].active {
                self.assign(i, sample, rate, gain);
                self.next = (i + 1) % MAX_VOICES;
                return;
            }
        }
        // All busy — steal the one nearest its end.
        let (mut steal_idx, mut steal_prog) = (0usize, -1.0f64);
        for (i, v) in self.voices.iter().enumerate() {
            if let Some(s) = &v.sample {
                let prog = v.pos / s.len().max(1) as f64;
                if prog > steal_prog {
                    steal_prog = prog;
                    steal_idx = i;
                }
            }
        }
        self.assign(steal_idx, sample, rate, gain);
        self.next = (steal_idx + 1) % MAX_VOICES;
    }

    fn assign(&mut self, i: usize, sample: Arc<Vec<f32>>, rate: f64, gain: f32) {
        let v = &mut self.voices[i];
        v.sample = Some(sample);
        v.pos = 0.0;
        v.rate = rate;
        v.gain = gain;
        v.active = true;
    }

    /// Render `frames` frames into interleaved output of `channels` channels.
    /// Additive mix; caller zeroes the buffer first.
    pub fn render(&mut self, out: &mut [f32], channels: usize) {
        debug_assert!(out.len() % channels == 0);
        let frames = out.len() / channels;

        for v in self.voices.iter_mut() {
            if !v.active {
                continue;
            }
            let sample = match &v.sample {
                Some(s) => s,
                None => {
                    v.active = false;
                    continue;
                }
            };
            let data = sample.as_slice();
            let len = data.len();
            let gain = v.gain;
            let mut pos = v.pos;
            let rate = v.rate;

            for frame in 0..frames {
                let i = pos as usize;
                if i + 1 >= len {
                    // End of sample: deactivate and stop writing.
                    v.active = false;
                    // Drop the Arc here; bank still holds another so no dealloc.
                    // We can't mutate `v.sample = None` while `data` borrows it,
                    // so we break and zero the rest implicitly (already zero).
                    let _ = frame;
                    break;
                }
                let t = (pos - i as f64) as f32;
                let s = data[i] * (1.0 - t) + data[i + 1] * t;
                let s = s * gain;
                let base = frame * channels;
                // Mono sample → write same value to all channels.
                for ch in 0..channels {
                    out[base + ch] += s;
                }
                pos += rate;
            }
            v.pos = pos;
            if !v.active {
                v.sample = None;
            }
        }
    }
}
