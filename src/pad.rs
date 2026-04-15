//! Pad + Preset data model.
//!
//! Polygons are stored in normalized image coordinates (x and y in [0,1]).
//! When calibration lands in M7 these become pad-plane coordinates and the
//! detection thread applies the homography to transform them into image
//! coordinates per frame.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::audio::SampleId;

fn default_velocity_curve() -> f32 {
    1.0
}

pub type PadId = u32;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct MidiMapping {
    pub channel: u8,
    pub note: u8,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Pad {
    pub id: PadId,
    pub name: String,
    /// Polygon vertices, image-normalized; at least 3 for the pad to be active.
    pub polygon: Vec<[f32; 2]>,
    pub sample_id: SampleId,
    /// Path the sample was loaded from, if any. Populated on save so the
    /// preset survives a restart; `None` for synthetic / unsaved samples.
    #[serde(default)]
    pub sample_path: Option<PathBuf>,
    /// Mean |Δluma| per pixel above which a hit fires.
    pub threshold: f32,
    /// Metric value that maps to velocity = 1.0.
    pub metric_max: f32,
    pub refractory_ms: u32,
    /// Velocity response curve exponent. 1.0 = linear, <1 softer (expands
    /// quiet hits), >1 harder (compresses quiet hits). Applied to the
    /// normalized 0..1 velocity before audio gain and MIDI velocity.
    #[serde(default = "default_velocity_curve")]
    pub velocity_curve: f32,
    #[serde(default)]
    pub midi: Option<MidiMapping>,
    /// Display color, RGB 0..255.
    pub color: [u8; 3],
}

impl Pad {
    pub fn is_active(&self) -> bool {
        self.polygon.len() >= 3
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct Preset {
    pub pads: Vec<Pad>,
    /// Pad-plane → normalized-image homography. When present, `pad.polygon`
    /// vertices live in pad-plane coordinates and are transformed by this
    /// matrix to obtain image coordinates for display and detection.
    #[serde(default)]
    pub homography: Option<[[f32; 3]; 3]>,
    /// Last-selected MIDI output port name. Restored on load so the user
    /// doesn't have to re-pick their IAC / loopMIDI bus every session.
    #[serde(default)]
    pub midi_port: Option<String>,
    /// Human-readable name of the last-selected capture device, resolved
    /// back to an index on load by matching against the current device list.
    #[serde(default)]
    pub camera_name: Option<String>,
}

impl Preset {
    /// Next unused pad id: max(existing) + 1, or 0 for an empty preset.
    pub fn next_id(&self) -> PadId {
        self.pads.iter().map(|p| p.id).max().map(|m| m + 1).unwrap_or(0)
    }

    pub fn remove(&mut self, id: PadId) {
        self.pads.retain(|p| p.id != id);
    }
}

/// Small rotating palette so each new pad gets a distinct color.
pub const PALETTE: &[[u8; 3]] = &[
    [80, 200, 120],  // green
    [255, 140, 80],  // orange
    [100, 160, 255], // blue
    [230, 100, 200], // pink
    [240, 220, 90],  // yellow
    [160, 120, 240], // purple
];

pub fn pick_color(index: usize) -> [u8; 3] {
    PALETTE[index % PALETTE.len()]
}
