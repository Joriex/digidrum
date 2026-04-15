//! Camera capture thread.
//!
//! Opens the requested camera at the highest framerate the device will give
//! us and publishes the latest decoded RGB frame into an `Arc<Mutex<_>>`
//! slot that the UI thread and the detect thread read. Supports live
//! switching between cameras via [`CameraControl::request`]: the capture
//! loop notices the atomic change, drops the current device, and reopens at
//! the new index without respawning the thread.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::query;
use nokhwa::utils::{ApiBackend, CameraIndex, CameraInfo, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;

/// One decoded RGB frame. Width x height in pixels, tightly packed RGB8.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// Monotonic timestamp at which the frame was received from the driver.
    pub captured_at: Instant,
    /// Monotonically increasing frame id, useful for detecting "is this new".
    pub seq: u64,
}

/// Shared latest-frame slot. `None` until the first frame arrives.
pub type FrameSlot = Arc<Mutex<Option<Arc<Frame>>>>;

/// UI-visible message about the last camera switch attempt. `None` means
/// clean state; `Some` is a one-shot that the UI reads and clears.
pub type ErrorSlot = Arc<Mutex<Option<String>>>;

/// Measured capture stats, updated once per second by the capture thread.
#[derive(Clone, Copy, Default, Debug)]
pub struct CaptureStats {
    pub fps: f32,
    pub width: u32,
    pub height: u32,
    /// Index of the currently open camera, or -1 while reopening.
    pub active_index: i32,
}

pub type StatsSlot = Arc<Mutex<CaptureStats>>;

/// A discovered capture device.
#[derive(Clone, Debug)]
pub struct CameraEntry {
    pub index: u32,
    pub name: String,
}

/// UI/app handle for asking the capture thread to switch devices. Cheap to
/// clone. Internally a single atomic — lock-free from either side.
#[derive(Clone)]
pub struct CameraControl {
    requested: Arc<AtomicI32>,
}

impl CameraControl {
    pub fn request(&self, index: u32) {
        self.requested.store(index as i32, Ordering::SeqCst);
    }

    pub fn requested_index(&self) -> u32 {
        self.requested.load(Ordering::SeqCst).max(0) as u32
    }
}

/// Enumerate available capture devices via nokhwa's auto backend.
pub fn list_cameras() -> Vec<CameraEntry> {
    let cams: Vec<CameraInfo> = match query(ApiBackend::Auto) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("camera enumeration failed: {e}");
            return Vec::new();
        }
    };
    let mut out = Vec::with_capacity(cams.len());
    for (fallback_idx, info) in cams.into_iter().enumerate() {
        let index = match info.index() {
            CameraIndex::Index(i) => *i,
            CameraIndex::String(_) => fallback_idx as u32,
        };
        out.push(CameraEntry {
            index,
            name: info.human_name(),
        });
    }
    out
}

/// Spawn the camera thread. Returns the shared slots plus a [`CameraControl`]
/// that lets the UI request a different device at runtime and an
/// [`ErrorSlot`] that surfaces switch failures to the UI for toast display.
pub fn spawn(initial_index: u32) -> (FrameSlot, StatsSlot, CameraControl, ErrorSlot) {
    install_camera_panic_filter();

    let frame_slot: FrameSlot = Arc::new(Mutex::new(None));
    let stats_slot: StatsSlot = Arc::new(Mutex::new(CaptureStats::default()));
    let error_slot: ErrorSlot = Arc::new(Mutex::new(None));
    let control = CameraControl {
        requested: Arc::new(AtomicI32::new(initial_index as i32)),
    };

    let frame_slot_t = Arc::clone(&frame_slot);
    let stats_slot_t = Arc::clone(&stats_slot);
    let error_slot_t = Arc::clone(&error_slot);
    let control_t = control.clone();

    thread::Builder::new()
        .name("camera".into())
        .spawn(move || {
            capture_supervisor(frame_slot_t, stats_slot_t, error_slot_t, control_t)
        })
        .expect("spawn camera thread");

    (frame_slot, stats_slot, control, error_slot)
}

/// Replace the default panic hook with one that stays silent for panics on
/// the camera thread (we catch and log them ourselves; the default hook's
/// stderr dump is just noise). Other threads keep the default behavior.
fn install_camera_panic_filter() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if thread::current().name() == Some("camera") {
                return;
            }
            default(info);
        }));
    });
}

/// Outer loop: open the currently-requested device, run the inner capture
/// loop until a switch is requested (or an error occurs), then retry.
///
/// Resilient against AVFoundation throwing NSExceptions (surfaced as Rust
/// panics) during format negotiation: each attempt is wrapped in
/// `catch_unwind`, and an index that panics is remembered so we don't
/// busy-panic while it's still requested. We also fall back from
/// `AbsoluteHighestFrameRate` to the device default on panic, since some
/// macOS cameras (Continuity, virtual cams) reject aggressive format
/// requests.
fn capture_supervisor(
    frame_slot: FrameSlot,
    stats_slot: StatsSlot,
    error_slot: ErrorSlot,
    control: CameraControl,
) {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let mut global_seq: u64 = 0;
    let mut last_failed: Option<u32> = None;
    let mut last_good: Option<u32> = None;

    loop {
        let idx = control.requested_index();

        // If this exact index just failed, wait for the user to pick a different
        // one before retrying (polling at UI-scale intervals, not hot).
        if last_failed == Some(idx) {
            thread::sleep(Duration::from_millis(250));
            continue;
        }

        if let Ok(mut s) = stats_slot.lock() {
            s.active_index = -1;
        }

        // Two attempts: high-fps first (gives us <15ms sensor latency on
        // compliant cams), then plain default so stubborn devices still work.
        // Both paths go through nokhwa's set_all on macOS, so a device that
        // rejects frame-duration changes will likely panic on both — that's
        // caught and the user is reverted to the last-working device below.
        let attempts = [
            RequestedFormatType::AbsoluteHighestFrameRate,
            RequestedFormatType::None,
        ];
        let mut opened = false;
        let mut last_reason: Option<String> = None;

        for fmt in attempts {
            let frame_slot = Arc::clone(&frame_slot);
            let stats_slot = Arc::clone(&stats_slot);
            let control = control.clone();
            let seq_out = AssertUnwindSafe(&mut global_seq);

            let result = catch_unwind(AssertUnwindSafe(move || {
                let AssertUnwindSafe(seq) = seq_out;
                let camera = match open_camera(idx, fmt) {
                    Ok(c) => c,
                    Err(e) => return Err(e),
                };
                if let Ok(mut s) = stats_slot.lock() {
                    s.active_index = idx as i32;
                }
                run_capture(camera, idx, &frame_slot, &stats_slot, &control, seq)
            }));

            match result {
                Ok(Ok(())) => {
                    last_failed = None;
                    opened = true;
                    break;
                }
                Ok(Err(e)) => {
                    let msg = format!("{e:#}");
                    log::error!("camera open failed (index {idx}, {fmt:?}): {msg}");
                    last_reason = Some(format!("{fmt:?}: {msg}"));
                }
                Err(panic) => {
                    let msg = panic_message(&panic);
                    log::error!("camera open panicked (index {idx}, {fmt:?}): {msg}");
                    last_reason = Some(format!("{fmt:?}: {msg}"));
                }
            }
            // Small cooldown so AVFoundation can release any session state
            // before the next attempt.
            thread::sleep(Duration::from_millis(150));
        }

        if opened {
            last_good = Some(idx);
            // Clear any stale error toast from a previous failure.
            if let Ok(mut e) = error_slot.lock() {
                *e = None;
            }
        } else {
            last_failed = Some(idx);
            let reason = last_reason.unwrap_or_else(|| "unknown failure".to_string());
            // If we have a known-good fallback AND it isn't the one we just
            // failed on, request it so the feed stays alive. The next outer
            // loop iteration will pick that up. Otherwise, surface a terminal
            // error toast and idle.
            match last_good {
                Some(fallback) if fallback != idx => {
                    log::warn!(
                        "camera {idx} unreachable ({reason}); reverting to {fallback}"
                    );
                    if let Ok(mut e) = error_slot.lock() {
                        *e = Some(format!(
                            "camera {idx} failed ({reason}) — reverted to camera {fallback}"
                        ));
                    }
                    control.request(fallback);
                }
                _ => {
                    log::warn!(
                        "camera {idx} unreachable ({reason}); waiting for a different selection"
                    );
                    if let Ok(mut e) = error_slot.lock() {
                        *e = Some(format!(
                            "camera {idx} failed ({reason}) — pick another device"
                        ));
                    }
                }
            }
        }
    }
}

fn panic_message(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = p.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else {
        "<non-string panic>".to_string()
    }
}

fn open_camera(index: u32, fmt: RequestedFormatType) -> Result<Camera> {
    let requested = RequestedFormat::new::<RgbFormat>(fmt);
    let mut camera = Camera::new(CameraIndex::Index(index), requested)
        .with_context(|| format!("opening camera {index}"))?;
    camera.open_stream().context("opening camera stream")?;
    Ok(camera)
}

/// Inner loop: pull frames until a switch is requested or the device errors.
fn run_capture(
    mut camera: Camera,
    opened_at: u32,
    frame_slot: &FrameSlot,
    stats_slot: &StatsSlot,
    control: &CameraControl,
    global_seq: &mut u64,
) -> Result<()> {
    let res = camera.resolution();
    log::info!(
        "camera {} open: {}x{}; format {:?}",
        opened_at,
        res.width_x,
        res.height_y,
        camera.frame_format()
    );

    let mut window_start = Instant::now();
    let mut window_frames: u32 = 0;

    loop {
        // Honor a pending switch request before grabbing the next frame.
        if control.requested_index() != opened_at {
            log::info!(
                "camera switch: {} → {}",
                opened_at,
                control.requested_index()
            );
            // Tear the stream down explicitly so AVFoundation releases the
            // device session before we try to open another one. Drop alone
            // races with the next open on some macOS devices.
            let _ = camera.stop_stream();
            return Ok(());
        }

        let buffer = camera
            .frame()
            .map_err(|e| anyhow!("camera frame error: {e}"))?;
        let captured_at = Instant::now();

        let decoded = buffer
            .decode_image::<RgbFormat>()
            .map_err(|e| anyhow!("camera decode error: {e}"))?;
        let (w, h) = decoded.dimensions();
        let rgb = decoded.into_raw();

        *global_seq = global_seq.wrapping_add(1);
        let frame = Arc::new(Frame {
            width: w,
            height: h,
            rgb,
            captured_at,
            seq: *global_seq,
        });
        if let Ok(mut slot) = frame_slot.lock() {
            *slot = Some(frame);
        }

        window_frames += 1;
        let elapsed = window_start.elapsed();
        if elapsed.as_secs_f32() >= 1.0 {
            let fps = window_frames as f32 / elapsed.as_secs_f32();
            if let Ok(mut s) = stats_slot.lock() {
                *s = CaptureStats {
                    fps,
                    width: w,
                    height: h,
                    active_index: opened_at as i32,
                };
            }
            log::info!("camera {} {}x{} @ {:.1} fps", opened_at, w, h, fps);
            window_start = Instant::now();
            window_frames = 0;
        }
    }
}
