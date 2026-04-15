mod app;
mod audio;
mod calibration;
mod camera;
mod detect;
mod midi_out;
mod pad;
mod preset_io;

use std::sync::Arc;

use anyhow::Result;
use arc_swap::ArcSwap;

use crate::audio::{samples as samp, AudioEngine, SampleBank};
use crate::pad::{Pad, Preset};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    #[cfg(target_os = "macos")]
    {
        nokhwa::nokhwa_initialize(|granted| {
            log::info!("macOS camera permission granted: {granted}");
        });
    }

    let (frame_slot, stats_slot, camera_control, camera_errors) = camera::spawn(0);

    // Synthetic kit so the app makes noise without any WAVs on disk.
    let mut bank = SampleBank::new();
    let _kick = bank.push(samp::synthetic_kick(48_000));
    let _snare = bank.push(samp::synthetic_snare(48_000));
    let _hat = bank.push(samp::synthetic_hat(48_000));

    let audio = AudioEngine::new(bank)?;
    log::info!(
        "audio engine ready: {} Hz × {} ch, buffer {:?}",
        audio.sample_rate(),
        audio.channels(),
        audio.buffer_frames()
    );

    // Start with three rectangle-polygon pads across the lower half. The
    // user can drag their vertices, add new pads, delete them.
    let initial_preset = Preset {
        pads: vec![
            Pad {
                id: 0,
                name: "kick".into(),
                polygon: vec![[0.08, 0.40], [0.32, 0.40], [0.32, 0.80], [0.08, 0.80]],
                sample_id: 0,
                threshold: 6.0,
                metric_max: 25.0,
                refractory_ms: 60,
                velocity_curve: 1.0,
                sample_path: None,
                midi: None,
                color: pad::pick_color(0),
            },
            Pad {
                id: 1,
                name: "snare".into(),
                polygon: vec![[0.38, 0.40], [0.62, 0.40], [0.62, 0.80], [0.38, 0.80]],
                sample_id: 1,
                threshold: 6.0,
                metric_max: 25.0,
                refractory_ms: 60,
                velocity_curve: 1.0,
                sample_path: None,
                midi: None,
                color: pad::pick_color(1),
            },
            Pad {
                id: 2,
                name: "hat".into(),
                polygon: vec![[0.68, 0.40], [0.92, 0.40], [0.92, 0.80], [0.68, 0.80]],
                sample_id: 2,
                threshold: 6.0,
                metric_max: 25.0,
                refractory_ms: 60,
                velocity_curve: 1.0,
                sample_path: None,
                midi: None,
                color: pad::pick_color(2),
            },
        ],
        homography: None,
        midi_port: None,
        camera_name: None,
    };
    let state = Arc::new(ArcSwap::from_pointee(initial_preset.clone()));

    let midi_bus = midi_out::MidiBus::new()?;
    log::info!("midi: {} output ports available", midi_bus.ports().len());

    let detect_stats = detect::spawn(
        Arc::clone(&frame_slot),
        audio.trigger_queue(),
        Arc::clone(&state),
        midi_bus.sender(),
    );

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("drum-scan")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "drum-scan",
        native_options,
        Box::new(move |_cc| {
            Ok(Box::new(app::App::new(
                frame_slot,
                stats_slot,
                audio,
                detect_stats,
                state,
                initial_preset,
                midi_bus,
                camera_control,
                camera_errors,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe failed: {e}"))?;

    Ok(())
}
