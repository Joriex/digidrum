//! Sample bank: mono f32 buffers addressed by `SampleId`. Samples may be at
//! any source sample-rate; the mixer resamples on playback via linear
//! interpolation (good enough for drum one-shots; we can swap in rubato for
//! offline high-quality resampling once the sample editor lands).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

pub type SampleId = u32;

#[derive(Clone)]
pub struct Sample {
    /// Mono PCM in [-1.0, 1.0]. Arc so voices can hold a cheap reference.
    pub data: Arc<Vec<f32>>,
    pub source_sr: u32,
    pub name: String,
    /// Originating file path for disk-loaded samples; `None` for synthetic
    /// kit samples. Preset save uses this to make pad → sample mappings
    /// survive a restart.
    pub path: Option<PathBuf>,
}

#[derive(Clone, Default)]
pub struct SampleBank {
    samples: Vec<Sample>,
}

impl SampleBank {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    pub fn push(&mut self, sample: Sample) -> SampleId {
        let id = self.samples.len() as SampleId;
        self.samples.push(sample);
        id
    }

    pub fn get(&self, id: SampleId) -> Option<&Sample> {
        self.samples.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Load a mono/stereo WAV, downmixing stereo to mono by averaging.
pub fn load_wav(path: &Path) -> Result<Sample> {
    let reader =
        hound::WavReader::open(path).with_context(|| format!("opening wav {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sr = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("reading float wav samples")?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|r| r.map(|s| s as f32 * scale))
                .collect::<Result<Vec<_>, _>>()
                .context("reading int wav samples")?
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect()
    };

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("sample")
        .to_string();
    Ok(Sample {
        data: Arc::new(mono),
        source_sr: sr,
        name,
        path: Some(path.to_path_buf()),
    })
}

/// A synthetic drum-like click: short exponential-decay sine around 120 Hz
/// plus a tiny burst of noise, so M2 can make noise without any asset files.
pub fn synthetic_kick(sr: u32) -> Sample {
    let len = (sr as f32 * 0.30) as usize; // 300 ms
    let mut data = Vec::with_capacity(len);
    let decay = -1.0 / (sr as f32 * 0.08); // ~80 ms body
    let noise_decay = -1.0 / (sr as f32 * 0.005); // ~5 ms transient
    let mut phase = 0.0f32;
    for i in 0..len {
        let t = i as f32 / sr as f32;
        let env_body = (decay * i as f32).exp();
        let env_click = (noise_decay * i as f32).exp();
        // pitch sweep 200→55 Hz for a punchy kick feel
        let freq = 55.0 + 145.0 * (-8.0 * t).exp();
        phase += freq / sr as f32;
        let body = (phase * std::f32::consts::TAU).sin() * env_body;
        // cheap deterministic "noise" — hash-ish bit pattern
        let n = (((i as u32).wrapping_mul(2654435761)) as i32 as f32) / i32::MAX as f32;
        let click = n * env_click * 0.6;
        data.push((body + click) * 0.9);
    }
    Sample {
        data: Arc::new(data),
        source_sr: sr,
        name: "synth_kick".into(),
        path: None,
    }
}

pub fn synthetic_snare(sr: u32) -> Sample {
    let len = (sr as f32 * 0.25) as usize;
    let mut data = Vec::with_capacity(len);
    let noise_decay = -1.0 / (sr as f32 * 0.08);
    let tone_decay = -1.0 / (sr as f32 * 0.04);
    let mut phase = 0.0f32;
    for i in 0..len {
        let env_noise = (noise_decay * i as f32).exp();
        let env_tone = (tone_decay * i as f32).exp();
        phase += 220.0 / sr as f32;
        let tone = (phase * std::f32::consts::TAU).sin() * env_tone * 0.4;
        let n = (((i as u32).wrapping_mul(1597334677)) as i32 as f32) / i32::MAX as f32;
        let noise = n * env_noise;
        data.push((tone + noise) * 0.7);
    }
    Sample {
        data: Arc::new(data),
        source_sr: sr,
        name: "synth_snare".into(),
        path: None,
    }
}

pub fn synthetic_hat(sr: u32) -> Sample {
    let len = (sr as f32 * 0.10) as usize;
    let mut data = Vec::with_capacity(len);
    let decay = -1.0 / (sr as f32 * 0.03);
    for i in 0..len {
        let env = (decay * i as f32).exp();
        let n = (((i as u32).wrapping_mul(374761393)) as i32 as f32) / i32::MAX as f32;
        data.push(n * env * 0.5);
    }
    Sample {
        data: Arc::new(data),
        source_sr: sr,
        name: "synth_hat".into(),
        path: None,
    }
}
