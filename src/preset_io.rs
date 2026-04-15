//! Preset serialization: JSON round-trip via `serde_json`.
//!
//! The on-disk format is literally the `Preset` struct serialized with
//! `serde_json::to_string_pretty`, so users can eyeball/diff/edit it.

use std::path::Path;

use anyhow::{Context, Result};

use crate::pad::Preset;

pub fn save_to_path(preset: &Preset, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(preset).context("serializing preset")?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn load_from_path(path: &Path) -> Result<Preset> {
    let json =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let preset = serde_json::from_str(&json)
        .with_context(|| format!("parsing preset JSON from {}", path.display()))?;
    Ok(preset)
}
