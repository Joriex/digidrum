//! Motion-based hit detection over arbitrary polygon pads.
//!
//! The UI thread publishes a `Preset` snapshot via `ArcSwap`; the detect
//! thread reads the latest snapshot each frame, keeps per-pad runtime state
//! (rasterized mask + previous luma), and fires `TriggerEvent`s when the
//! mean |Δluma| over the masked pixels crosses a pad's threshold.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam_queue::ArrayQueue;

use crate::audio::TriggerEvent;
use crate::calibration;
use crate::camera::{Frame, FrameSlot};
use crate::midi_out::MidiSender;
use crate::pad::{Pad, PadId, Preset};

/// Per-pad live telemetry for the UI (meters, flash on hit).
#[derive(Clone, Debug, Default)]
pub struct PadStats {
    pub metric: f32,
    pub threshold: f32,
    pub last_hit: Option<Instant>,
}

pub type StatsSlot = Arc<Mutex<HashMap<PadId, PadStats>>>;

pub fn spawn(
    frame_slot: FrameSlot,
    trigger_queue: Arc<ArrayQueue<TriggerEvent>>,
    state: Arc<ArcSwap<Preset>>,
    midi: MidiSender,
) -> StatsSlot {
    let stats: StatsSlot = Arc::new(Mutex::new(HashMap::new()));
    let stats_thread = Arc::clone(&stats);

    thread::Builder::new()
        .name("detect".into())
        .spawn(move || detect_loop(frame_slot, trigger_queue, state, stats_thread, midi))
        .expect("spawn detect thread");

    stats
}

/// Everything the detect thread tracks per pad, keyed by `PadId`.
struct PadRuntime {
    /// Last polygon we rasterized for; if pad.polygon changes, rebuild.
    last_polygon: Vec<[f32; 2]>,
    /// Homography used at last build (so calibration changes trigger rebuild).
    last_homography: Option<[[f32; 3]; 3]>,
    /// Frame resolution at last build.
    last_res: (u32, u32),

    // AABB in pixels that the mask covers.
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    /// Dense w*h bitmap, 1 inside polygon else 0.
    mask: Vec<u8>,
    /// How many pixels the mask includes; used to normalize the metric.
    mask_count: u32,

    /// Previous-frame luma at masked pixels, same w*h layout as `mask`.
    prev_luma: Vec<u8>,
    /// Whether `prev_luma` has been primed with a real frame yet.
    primed: bool,

    last_hit: Option<Instant>,
    /// EMA-smoothed metric for the UI meter (not used to trigger).
    smoothed: f32,
}

impl PadRuntime {
    fn empty() -> Self {
        Self {
            last_polygon: Vec::new(),
            last_homography: None,
            last_res: (0, 0),
            x0: 0,
            y0: 0,
            w: 0,
            h: 0,
            mask: Vec::new(),
            mask_count: 0,
            prev_luma: Vec::new(),
            primed: false,
            last_hit: None,
            smoothed: 0.0,
        }
    }
}

fn detect_loop(
    frame_slot: FrameSlot,
    trigger_queue: Arc<ArrayQueue<TriggerEvent>>,
    state: Arc<ArcSwap<Preset>>,
    stats_slot: StatsSlot,
    midi: MidiSender,
) {
    let mut runtimes: HashMap<PadId, PadRuntime> = HashMap::new();
    let mut last_seq: u64 = 0;
    let alpha = 0.25f32;

    loop {
        // Block-ish wait for a fresh frame. Short sleep if none, to avoid busy-spin.
        let frame: Option<Arc<Frame>> = frame_slot.lock().ok().and_then(|g| g.as_ref().cloned());
        let Some(frame) = frame else {
            thread::sleep(Duration::from_millis(5));
            continue;
        };
        if frame.seq == last_seq {
            thread::sleep(Duration::from_micros(200));
            continue;
        }
        last_seq = frame.seq;

        let preset = state.load_full();

        // Garbage-collect runtimes for pads that no longer exist.
        let live: std::collections::HashSet<PadId> =
            preset.pads.iter().map(|p| p.id).collect();
        runtimes.retain(|id, _| live.contains(id));

        let mut stats_buf: HashMap<PadId, PadStats> = HashMap::with_capacity(preset.pads.len());

        for pad in &preset.pads {
            if !pad.is_active() {
                continue;
            }
            let rt = runtimes.entry(pad.id).or_insert_with(PadRuntime::empty);
            let res = (frame.width, frame.height);
            if rt.last_polygon != pad.polygon
                || rt.last_res != res
                || rt.last_homography != preset.homography
            {
                rebuild_mask(rt, pad, res, preset.homography.as_ref());
            }
            if rt.mask_count == 0 {
                continue;
            }

            let metric = compute_metric(rt, &frame);
            rt.smoothed = rt.smoothed * (1.0 - alpha) + metric * alpha;

            if rt.primed {
                let now = Instant::now();
                let in_refractory = rt
                    .last_hit
                    .map(|t| now.duration_since(t).as_millis() < pad.refractory_ms as u128)
                    .unwrap_or(false);

                if !in_refractory && metric >= pad.threshold {
                    let span = (pad.metric_max - pad.threshold).max(1.0);
                    let raw = ((metric - pad.threshold) / span).clamp(0.0, 1.0);
                    let curve = pad.velocity_curve.max(0.05);
                    let velocity = raw.powf(curve).clamp(0.0, 1.0);
                    let ev = TriggerEvent {
                        sample_id: pad.sample_id,
                        velocity,
                    };
                    if trigger_queue.push(ev).is_err() {
                        let _ = trigger_queue.pop();
                        let _ = trigger_queue.push(ev);
                    }
                    if let Some(m) = &pad.midi {
                        let midi_vel = ((velocity * 127.0).round() as i32).clamp(1, 127) as u8;
                        midi.fire(m.channel, m.note, midi_vel);
                    }
                    rt.last_hit = Some(now);
                    log::debug!(
                        "hit pad={} name='{}' metric={:.1} thr={:.1} vel={:.2}",
                        pad.id,
                        pad.name,
                        metric,
                        pad.threshold,
                        velocity
                    );
                }
            }

            stats_buf.insert(
                pad.id,
                PadStats {
                    metric: rt.smoothed,
                    threshold: pad.threshold,
                    last_hit: rt.last_hit,
                },
            );
        }

        if let Ok(mut s) = stats_slot.lock() {
            *s = stats_buf;
        }
    }
}

/// Sample luma into the pad's AABB and compute mean |cur - prev| over masked pixels.
fn compute_metric(rt: &mut PadRuntime, frame: &Frame) -> f32 {
    let stride = (frame.width * 3) as usize;
    let src = &frame.rgb;
    let w = rt.w as usize;

    let mut sum: u64 = 0;
    for row in 0..rt.h {
        let y = (rt.y0 + row) as usize;
        let row_start = y * stride + (rt.x0 as usize) * 3;
        let mask_row = (row as usize) * w;
        for col in 0..rt.w {
            let idx = mask_row + col as usize;
            if rt.mask[idx] == 0 {
                continue;
            }
            let off = row_start + (col as usize) * 3;
            let r = src[off] as u32;
            let g = src[off + 1] as u32;
            let b = src[off + 2] as u32;
            let luma = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            let prev = rt.prev_luma[idx];
            sum += (luma as i32 - prev as i32).unsigned_abs() as u64;
            rt.prev_luma[idx] = luma;
        }
    }

    if !rt.primed {
        rt.primed = true;
        return 0.0;
    }
    sum as f32 / rt.mask_count as f32
}

fn rebuild_mask(
    rt: &mut PadRuntime,
    pad: &Pad,
    res: (u32, u32),
    homography: Option<&[[f32; 3]; 3]>,
) {
    let (fw, fh) = res;
    // Transform pad-plane polygon into image-space once, then rasterize as before.
    let transformed: Vec<[f32; 2]> = match homography {
        Some(h) => pad.polygon.iter().map(|&v| calibration::apply(h, v)).collect(),
        None => pad.polygon.clone(),
    };
    let poly = &transformed;
    let mut min_x = 1.0f32;
    let mut max_x = 0.0f32;
    let mut min_y = 1.0f32;
    let mut max_y = 0.0f32;
    for v in poly {
        min_x = min_x.min(v[0]);
        max_x = max_x.max(v[0]);
        min_y = min_y.min(v[1]);
        max_y = max_y.max(v[1]);
    }
    let x0 = (min_x.clamp(0.0, 1.0) * fw as f32).floor() as u32;
    let y0 = (min_y.clamp(0.0, 1.0) * fh as f32).floor() as u32;
    let x1 = (max_x.clamp(0.0, 1.0) * fw as f32).ceil() as u32;
    let y1 = (max_y.clamp(0.0, 1.0) * fh as f32).ceil() as u32;
    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);
    let pixel_count = (w * h) as usize;

    let mut mask = vec![0u8; pixel_count];
    let mut count = 0u32;
    for row in 0..h {
        let py = (y0 + row) as f32 / fh as f32;
        for col in 0..w {
            let px = (x0 + col) as f32 / fw as f32;
            if point_in_polygon(px, py, poly) {
                mask[(row * w + col) as usize] = 1;
                count += 1;
            }
        }
    }

    rt.last_polygon = pad.polygon.clone();
    rt.last_homography = homography.copied();
    rt.last_res = res;
    rt.x0 = x0;
    rt.y0 = y0;
    rt.w = w;
    rt.h = h;
    rt.mask = mask;
    rt.mask_count = count;
    rt.prev_luma = vec![0u8; pixel_count];
    rt.primed = false;
}

/// Standard ray-casting point-in-polygon. `polygon` must be closed implicitly
/// (we wrap i=0 and j=n-1). Coordinates in normalized image space [0,1].
fn point_in_polygon(x: f32, y: f32, polygon: &[[f32; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);
        let crosses = (yi > y) != (yj > y)
            && x < (xj - xi) * (y - yi) / (yj - yi + f32::EPSILON) + xi;
        if crosses {
            inside = !inside;
        }
        j = i;
    }
    inside
}
