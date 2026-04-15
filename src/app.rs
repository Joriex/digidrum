//! Top-level eframe App: camera preview + polygon pad editor + live meters.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwap;
use eframe::egui;
use egui::{Color32, ColorImage, Pos2, Rect, Stroke, TextureHandle, TextureOptions};

use crate::audio::{AudioEngine, SampleId, TriggerEvent};
use crate::calibration;
use crate::camera::{
    self, CameraControl, CameraEntry, CaptureStats, ErrorSlot as CameraErrorSlot, Frame, FrameSlot,
    StatsSlot,
};
use crate::detect::{PadStats, StatsSlot as DetectStatsSlot};
use crate::midi_out::MidiBus;
use crate::pad::{self, MidiMapping, Pad, PadId, Preset};
use crate::preset_io;

/// Screen-space radius at which a click hits a vertex handle.
const VERTEX_HANDLE_RADIUS: f32 = 7.0;
/// Screen-space snap radius for closing an in-progress polygon.
const POLYGON_CLOSE_SNAP: f32 = 12.0;

enum EditMode {
    Idle,
    Drawing(Vec<[f32; 2]>),
    DraggingVertex { pad_id: PadId, vertex: usize },
    /// Collecting 4 image-space corner points in TL, TR, BR, BL order.
    Calibrating { points: Vec<[f32; 2]> },
}

pub struct App {
    frame_slot: FrameSlot,
    stats_slot: StatsSlot,
    detect_stats: DetectStatsSlot,
    audio: AudioEngine,

    state: Arc<ArcSwap<Preset>>,
    preset: Preset,
    midi: MidiBus,
    camera: CameraControl,
    /// One-shot error slot the capture thread writes into on failed switches.
    /// Polled each UI tick and surfaced as a status toast.
    camera_errors: CameraErrorSlot,
    /// Cached camera enumeration; refreshed via the "↻" button in the
    /// camera picker or when reapplying a preset's camera_name.
    cameras: Vec<CameraEntry>,
    /// Human-readable name of the device the user last picked. Echoed into
    /// the preset on save; used as the look-up key on load.
    current_camera_name: Option<String>,

    texture: Option<TextureHandle>,
    last_uploaded_seq: u64,
    last_stats: CaptureStats,
    pad_stats: std::collections::HashMap<PadId, PadStats>,

    mode: EditMode,
    selected: Option<PadId>,

    /// Last-used preset path; `Save` writes here without prompting.
    current_path: Option<PathBuf>,
    /// Transient status toast (file operations, errors); fades after `status_until`.
    status: Option<StatusToast>,
}

struct StatusToast {
    text: String,
    is_error: bool,
    until: Instant,
}

impl App {
    pub fn new(
        frame_slot: FrameSlot,
        stats_slot: StatsSlot,
        audio: AudioEngine,
        detect_stats: DetectStatsSlot,
        state: Arc<ArcSwap<Preset>>,
        preset: Preset,
        midi: MidiBus,
        camera: CameraControl,
        camera_errors: CameraErrorSlot,
    ) -> Self {
        let cameras = camera::list_cameras();
        let initial = camera.requested_index();
        let current_camera_name = cameras
            .iter()
            .find(|c| c.index == initial)
            .map(|c| c.name.clone());
        Self {
            frame_slot,
            stats_slot,
            detect_stats,
            audio,
            state,
            preset,
            midi,
            camera,
            camera_errors,
            cameras,
            current_camera_name,
            texture: None,
            last_uploaded_seq: 0,
            last_stats: CaptureStats::default(),
            pad_stats: Default::default(),
            mode: EditMode::Idle,
            selected: None,
            current_path: None,
            status: None,
        }
    }

    fn set_status(&mut self, text: impl Into<String>, is_error: bool) {
        self.status = Some(StatusToast {
            text: text.into(),
            is_error,
            until: Instant::now() + std::time::Duration::from_secs(if is_error { 6 } else { 3 }),
        });
    }

    fn new_preset(&mut self) {
        self.preset.pads.clear();
        self.selected = None;
        self.mode = EditMode::Idle;
        self.current_path = None;
        self.publish();
        self.set_status("new preset", false);
    }

    fn open_preset_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("drum-scan preset", &["json"])
            .set_title("Open preset")
            .pick_file()
        else {
            return;
        };
        match preset_io::load_from_path(&path) {
            Ok(p) => {
                self.preset = p;
                self.reload_preset_samples();
                self.reapply_runtime_state();
                self.selected = None;
                self.mode = EditMode::Idle;
                self.current_path = Some(path.clone());
                self.publish();
                self.set_status(format!("loaded {}", path.display()), false);
            }
            Err(e) => {
                log::error!("load failed: {e:#}");
                self.set_status(format!("load failed: {e}"), true);
            }
        }
    }

    /// For every pad that was saved with a `sample_path`, load the WAV back
    /// into the bank (deduping paths) and rewrite the pad's `sample_id` to
    /// point at the newly loaded slot. Pads without a path keep their id
    /// (still valid if it lives in the synthetic kit).
    fn reload_preset_samples(&mut self) {
        use std::collections::HashMap;
        let mut loaded: HashMap<PathBuf, SampleId> = HashMap::new();
        let mut failures = 0usize;
        for pad in &mut self.preset.pads {
            let Some(path) = pad.sample_path.clone() else {
                continue;
            };
            if let Some(&sid) = loaded.get(&path) {
                pad.sample_id = sid;
                continue;
            }
            match crate::audio::samples::load_wav(&path) {
                Ok(sample) => {
                    let sid = self.audio.add_sample(sample);
                    loaded.insert(path, sid);
                    pad.sample_id = sid;
                }
                Err(e) => {
                    log::error!("reload sample {} failed: {e:#}", path.display());
                    failures += 1;
                }
            }
        }
        if failures > 0 {
            self.set_status(format!("{failures} sample(s) failed to reload"), true);
        }
    }

    /// Apply ambient preset state (currently just the MIDI port) back onto
    /// the running systems after loading a preset from disk. Missing ports
    /// are surfaced as a transient error so the user knows why nothing is
    /// happening on the MIDI side.
    fn reapply_runtime_state(&mut self) {
        // --- MIDI port ---
        match self.preset.midi_port.clone() {
            Some(name) => {
                self.midi.refresh_ports();
                if self.midi.ports().iter().any(|p| p == &name) {
                    self.midi.set_port(Some(name));
                } else {
                    self.midi.set_port(None);
                    self.set_status(
                        format!("MIDI port '{name}' not available — select another"),
                        true,
                    );
                }
            }
            None => self.midi.set_port(None),
        }

        // --- Camera ---
        if let Some(name) = self.preset.camera_name.clone() {
            self.cameras = camera::list_cameras();
            match self.cameras.iter().find(|c| c.name == name) {
                Some(entry) => {
                    self.camera.request(entry.index);
                    self.current_camera_name = Some(entry.name.clone());
                }
                None => {
                    self.set_status(
                        format!("camera '{name}' not connected — keeping current device"),
                        true,
                    );
                }
            }
        }
    }

    /// Snapshot all ambient state (sample paths from the bank, current MIDI
    /// port from the bus) into the preset so that whatever `write_preset`
    /// serializes is a complete, round-trippable picture of the session.
    fn sync_runtime_state_into_preset(&mut self) {
        let bank = self.audio.bank_snapshot();
        for pad in &mut self.preset.pads {
            pad.sample_path = bank
                .get(pad.sample_id)
                .and_then(|s| s.path.clone());
        }
        self.preset.midi_port = self.midi.current().map(|s| s.to_string());
        self.preset.camera_name = self.current_camera_name.clone();
    }

    fn save_preset(&mut self) {
        if let Some(path) = self.current_path.clone() {
            self.write_preset(&path);
        } else {
            self.save_preset_as_dialog();
        }
    }

    fn save_preset_as_dialog(&mut self) {
        let dlg = rfd::FileDialog::new()
            .add_filter("drum-scan preset", &["json"])
            .set_title("Save preset as")
            .set_file_name("preset.json");
        let Some(path) = dlg.save_file() else {
            return;
        };
        self.write_preset(&path);
        self.current_path = Some(path);
    }

    fn write_preset(&mut self, path: &std::path::Path) {
        self.sync_runtime_state_into_preset();
        match preset_io::save_to_path(&self.preset, path) {
            Ok(()) => self.set_status(format!("saved {}", path.display()), false),
            Err(e) => {
                log::error!("save failed: {e:#}");
                self.set_status(format!("save failed: {e}"), true);
            }
        }
    }

    fn publish(&self) {
        self.state.store(Arc::new(self.preset.clone()));
    }

    /// Pad-plane → normalized image coords. Identity when no calibration set.
    fn pp_to_image(&self, p: [f32; 2]) -> [f32; 2] {
        match self.preset.homography.as_ref() {
            Some(h) => calibration::apply(h, p),
            None => p,
        }
    }

    /// Normalized image → pad-plane coords. Identity when no calibration set.
    fn image_to_pp(&self, p: [f32; 2]) -> [f32; 2] {
        match self.preset.homography.as_ref().and_then(calibration::invert) {
            Some(inv) => calibration::apply(&inv, p),
            None => p,
        }
    }

    fn begin_calibration(&mut self) {
        self.mode = EditMode::Calibrating { points: Vec::with_capacity(4) };
        self.set_status("click 4 corners: TL · TR · BR · BL (Esc to cancel)", false);
    }

    /// Finish 4-point calibration. `points` are in normalized image coords
    /// and interpreted as the TL/TR/BR/BL corners of the unit pad-plane.
    /// Existing pad polygons are transformed so they stay visually in place.
    fn commit_calibration(&mut self, points: [[f32; 2]; 4]) {
        let Some(new_h) = calibration::compute_homography(calibration::UNIT_SQUARE, points)
        else {
            self.set_status("calibration failed: degenerate points", true);
            self.mode = EditMode::Idle;
            return;
        };
        let Some(new_inv) = calibration::invert(&new_h) else {
            self.set_status("calibration failed: non-invertible", true);
            self.mode = EditMode::Idle;
            return;
        };
        let old_h = self.preset.homography;
        for pad in &mut self.preset.pads {
            for v in &mut pad.polygon {
                let img = match old_h.as_ref() {
                    Some(h) => calibration::apply(h, *v),
                    None => *v,
                };
                *v = calibration::apply(&new_inv, img);
            }
        }
        self.preset.homography = Some(new_h);
        self.mode = EditMode::Idle;
        self.publish();
        self.set_status("calibration applied", false);
    }

    fn clear_calibration(&mut self) {
        let Some(h) = self.preset.homography.take() else {
            return;
        };
        // Bake current pad-plane polygons back into image coords so pads stay
        // visually in place after removing the homography.
        for pad in &mut self.preset.pads {
            for v in &mut pad.polygon {
                *v = calibration::apply(&h, *v);
            }
        }
        self.publish();
        self.set_status("calibration cleared", false);
    }

    fn upload_frame_if_new(&mut self, ctx: &egui::Context) -> Option<egui::Vec2> {
        let frame: Option<Arc<Frame>> = self
            .frame_slot
            .lock()
            .ok()
            .and_then(|g| g.as_ref().cloned());
        let frame = frame?;
        if frame.seq == self.last_uploaded_seq {
            return self
                .texture
                .as_ref()
                .map(|t| egui::vec2(t.size()[0] as f32, t.size()[1] as f32));
        }
        let w = frame.width as usize;
        let h = frame.height as usize;
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in frame.rgb.chunks_exact(3) {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        let image = ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        match self.texture.as_mut() {
            Some(tex) if tex.size() == [w, h] => tex.set(image, TextureOptions::LINEAR),
            _ => {
                self.texture = Some(ctx.load_texture("camera", image, TextureOptions::LINEAR));
            }
        }
        self.last_uploaded_seq = frame.seq;
        Some(egui::vec2(w as f32, h as f32))
    }

    fn handle_trigger_keys(&self, ctx: &egui::Context) {
        // In Drawing mode we don't want the number keys to also fire samples;
        // let the user type freely. But this is a minor concern — fine for now.
        let keymap: [(egui::Key, SampleId); 3] = [
            (egui::Key::Num1, 0),
            (egui::Key::Num2, 1),
            (egui::Key::Num3, 2),
        ];
        ctx.input(|i| {
            for (key, id) in keymap {
                if i.key_pressed(key) {
                    self.audio.trigger(TriggerEvent {
                        sample_id: id,
                        velocity: 1.0,
                    });
                }
            }
            if i.key_pressed(egui::Key::Space) {
                self.audio.trigger(TriggerEvent {
                    sample_id: 0,
                    velocity: 1.0,
                });
            }
        });
    }

    fn paint_and_edit(&mut self, ui: &mut egui::Ui, tex: &TextureHandle, native: egui::Vec2) {
        let avail = ui.available_size();
        let scale = (avail.x / native.x).min(avail.y / native.y).max(0.01);
        let size = egui::vec2(native.x * scale, native.y * scale);

        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        painter.image(
            tex.id(),
            rect,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );

        // Convert between image-normalized (0..1) and screen coords within `rect`.
        let to_screen =
            |p: [f32; 2]| rect.min + egui::vec2(p[0] * rect.width(), p[1] * rect.height());
        let to_image = |p: Pos2| -> [f32; 2] {
            let rel = (p - rect.min) / rect.size();
            [rel.x.clamp(0.0, 1.0), rel.y.clamp(0.0, 1.0)]
        };

        // ----- Draw committed pads -----
        let now = Instant::now();
        let h_opt = self.preset.homography;
        let pp_to_img = |v: [f32; 2]| -> [f32; 2] {
            match h_opt.as_ref() {
                Some(h) => calibration::apply(h, v),
                None => v,
            }
        };
        for pad in &self.preset.pads {
            let selected = self.selected == Some(pad.id);
            let color = Color32::from_rgb(pad.color[0], pad.color[1], pad.color[2]);
            let last_hit = self.pad_stats.get(&pad.id).and_then(|s| s.last_hit);
            let flash = last_hit
                .map(|t| (1.0 - now.duration_since(t).as_secs_f32() / 0.120).max(0.0))
                .unwrap_or(0.0);
            let hit_color = Color32::from_rgb(255, 80, 60);
            let stroke_color = lerp_color(color, hit_color, flash);
            let stroke_width = if selected { 3.0 } else { 2.0 };

            if pad.polygon.len() >= 2 {
                // Transform pad-plane polygon into image space once.
                let poly_img: Vec<[f32; 2]> = pad.polygon.iter().copied().map(pp_to_img).collect();
                // Fill (polygon with alpha).
                let mut fill = stroke_color.to_array();
                let base_alpha = if selected { 70 } else { 35 };
                fill[3] = (base_alpha as f32 + 80.0 * flash).clamp(0.0, 200.0) as u8;
                let pts: Vec<Pos2> = poly_img.iter().copied().map(to_screen).collect();
                if pts.len() >= 3 {
                    painter.add(egui::Shape::convex_polygon(
                        pts.clone(),
                        Color32::from_rgba_unmultiplied(fill[0], fill[1], fill[2], fill[3]),
                        Stroke::new(stroke_width, stroke_color),
                    ));
                } else {
                    painter.line_segment([pts[0], pts[1]], Stroke::new(stroke_width, stroke_color));
                }

                // Vertex handles (larger on selected pad).
                let r = if selected {
                    VERTEX_HANDLE_RADIUS
                } else {
                    VERTEX_HANDLE_RADIUS - 2.0
                };
                for &v in &poly_img {
                    painter.circle_filled(to_screen(v), r, stroke_color);
                    painter.circle_stroke(to_screen(v), r, Stroke::new(1.0, Color32::WHITE));
                }

                // Name at centroid.
                if poly_img.len() >= 3 {
                    let centroid = polygon_centroid(&poly_img);
                    painter.text(
                        to_screen(centroid),
                        egui::Align2::CENTER_CENTER,
                        &pad.name,
                        egui::FontId::proportional(14.0),
                        Color32::WHITE,
                    );
                }

                // Meter along the AABB bottom: metric relative to threshold.
                if let Some(stats) = self.pad_stats.get(&pad.id) {
                    let aabb = polygon_aabb(&poly_img);
                    let aabb_screen = Rect::from_min_max(to_screen(aabb.0), to_screen(aabb.1));
                    let frac =
                        (stats.metric / stats.threshold.max(0.01)).clamp(0.0, 2.0) / 2.0;
                    let meter_h = 5.0;
                    let meter_rect = Rect::from_min_size(
                        egui::pos2(aabb_screen.left(), aabb_screen.bottom() + 2.0),
                        egui::vec2(aabb_screen.width() * frac, meter_h),
                    );
                    let meter_color = if stats.metric >= stats.threshold {
                        hit_color
                    } else {
                        stroke_color
                    };
                    painter.rect_filled(meter_rect, 0.0, meter_color);
                    let tick_x = aabb_screen.left() + aabb_screen.width() * 0.5;
                    painter.line_segment(
                        [
                            egui::pos2(tick_x, aabb_screen.bottom() + 2.0),
                            egui::pos2(tick_x, aabb_screen.bottom() + 2.0 + meter_h),
                        ],
                        Stroke::new(1.0, Color32::WHITE),
                    );
                }
            }
        }

        // ----- Draw in-progress polygon -----
        if let EditMode::Drawing(verts) = &self.mode {
            let pts: Vec<Pos2> = verts.iter().copied().map(pp_to_img).map(to_screen).collect();
            for w in pts.windows(2) {
                painter.line_segment([w[0], w[1]], Stroke::new(2.0, Color32::YELLOW));
            }
            // Rubber-band to cursor.
            if let Some(p) = response.hover_pos() {
                if let Some(first) = pts.first() {
                    let snap = pts.len() >= 3 && first.distance(p) <= POLYGON_CLOSE_SNAP;
                    let end = if snap { *first } else { p };
                    painter.line_segment(
                        [*pts.last().unwrap(), end],
                        Stroke::new(
                            2.0,
                            if snap {
                                Color32::from_rgb(120, 255, 180)
                            } else {
                                Color32::from_rgba_unmultiplied(255, 255, 100, 180)
                            },
                        ),
                    );
                    if snap {
                        painter.circle_stroke(
                            *first,
                            POLYGON_CLOSE_SNAP,
                            Stroke::new(1.5, Color32::from_rgb(120, 255, 180)),
                        );
                    }
                }
            }
            for p in &pts {
                painter.circle_filled(*p, 5.0, Color32::YELLOW);
                painter.circle_stroke(*p, 5.0, Stroke::new(1.0, Color32::WHITE));
            }
        }

        // ----- Draw calibration overlay -----
        if let EditMode::Calibrating { points } = &self.mode {
            let labels = ["TL", "TR", "BR", "BL"];
            let cal_color = Color32::from_rgb(100, 220, 255);
            let screen_pts: Vec<Pos2> = points.iter().copied().map(to_screen).collect();
            if screen_pts.len() >= 2 {
                for w in screen_pts.windows(2) {
                    painter.line_segment([w[0], w[1]], Stroke::new(2.0, cal_color));
                }
                if screen_pts.len() == 4 {
                    painter.line_segment(
                        [screen_pts[3], screen_pts[0]],
                        Stroke::new(2.0, cal_color),
                    );
                }
            }
            for (i, p) in screen_pts.iter().enumerate() {
                painter.circle_filled(*p, 6.0, cal_color);
                painter.circle_stroke(*p, 6.0, Stroke::new(1.5, Color32::WHITE));
                painter.text(
                    *p + egui::vec2(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    labels[i],
                    egui::FontId::proportional(14.0),
                    cal_color,
                );
            }
            // Hint label for the next point.
            if screen_pts.len() < 4 {
                if let Some(p) = response.hover_pos() {
                    painter.text(
                        p + egui::vec2(12.0, 12.0),
                        egui::Align2::LEFT_TOP,
                        format!("click {}", labels[screen_pts.len()]),
                        egui::FontId::proportional(13.0),
                        cal_color,
                    );
                }
            }
        }

        // ----- Handle interaction -----
        self.handle_editor_input(ui, &response, to_image, to_screen);
    }

    fn handle_editor_input(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        to_image: impl Fn(Pos2) -> [f32; 2],
        to_screen: impl Fn([f32; 2]) -> Pos2,
    ) {
        let ctx = ui.ctx();

        // Keyboard first (order matters: Enter / Esc must not also double-fire a click).
        let (enter, esc, backspace, delete) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::Backspace),
                i.key_pressed(egui::Key::Delete),
            )
        });

        match &mut self.mode {
            EditMode::Drawing(verts) => {
                if esc {
                    self.mode = EditMode::Idle;
                } else if enter && verts.len() >= 3 {
                    let polygon = std::mem::take(verts);
                    self.commit_polygon(polygon);
                } else if backspace {
                    verts.pop();
                    if verts.is_empty() {
                        self.mode = EditMode::Idle;
                    }
                }
            }
            EditMode::Idle => {
                if (delete || backspace) && self.selected.is_some() {
                    let id = self.selected.unwrap();
                    self.preset.remove(id);
                    self.selected = None;
                    self.publish();
                }
            }
            EditMode::DraggingVertex { .. } => {}
            EditMode::Calibrating { points } => {
                if esc {
                    self.mode = EditMode::Idle;
                    self.set_status("calibration cancelled", false);
                } else if enter && points.len() == 4 {
                    let pts = [points[0], points[1], points[2], points[3]];
                    self.commit_calibration(pts);
                } else if backspace {
                    points.pop();
                }
            }
        }

        // Pointer events scoped to this rect.
        let pointer = response.interact_pointer_pos();

        // Clicks (primary). On press we decide the action; on release (DraggingVertex)
        // we commit.
        if response.clicked() {
            if let Some(screen) = response.hover_pos() {
                let img = to_image(screen);
                let pp = self.image_to_pp(img);
                match &mut self.mode {
                    EditMode::Drawing(verts) => {
                        // Close if near the first vertex with enough vertices.
                        // `verts` are in pad-plane; project to image→screen.
                        if verts.len() >= 3 {
                            let first_img = match self.preset.homography.as_ref() {
                                Some(h) => calibration::apply(h, verts[0]),
                                None => verts[0],
                            };
                            let first_screen = to_screen(first_img);
                            if first_screen.distance(screen) <= POLYGON_CLOSE_SNAP {
                                let polygon = std::mem::take(verts);
                                self.commit_polygon(polygon);
                                return;
                            }
                        }
                        verts.push(pp);
                    }
                    EditMode::Idle => {
                        if let Some((pad_id, _)) = self.vertex_hit(screen, &to_screen) {
                            self.selected = Some(pad_id);
                        } else if let Some(pad_id) = self.pad_hit(pp) {
                            self.selected = Some(pad_id);
                        } else {
                            // Empty space → start a new polygon.
                            self.mode = EditMode::Drawing(vec![pp]);
                            self.selected = None;
                        }
                    }
                    EditMode::DraggingVertex { .. } => {}
                    EditMode::Calibrating { points } => {
                        if points.len() < 4 {
                            points.push(img);
                        }
                        if points.len() == 4 {
                            let pts = [points[0], points[1], points[2], points[3]];
                            self.commit_calibration(pts);
                        }
                    }
                }
            }
        }

        // Start drag: look at drag_started_by(PointerButton::Primary).
        if response.drag_started() {
            if let Some(screen) = pointer {
                if matches!(self.mode, EditMode::Idle) {
                    if let Some((pad_id, vertex)) = self.vertex_hit(screen, &to_screen) {
                        self.mode = EditMode::DraggingVertex { pad_id, vertex };
                        self.selected = Some(pad_id);
                    }
                }
            }
        }

        // Continue drag: update vertex position each frame we're dragging.
        if let EditMode::DraggingVertex { pad_id, vertex } = self.mode {
            if let Some(screen) = pointer {
                let img = to_image(screen);
                let pp = self.image_to_pp(img);
                if let Some(pad) = self.preset.pads.iter_mut().find(|p| p.id == pad_id) {
                    if vertex < pad.polygon.len() {
                        pad.polygon[vertex] = pp;
                    }
                }
                self.publish();
            }
        }
        if response.drag_stopped() {
            if matches!(self.mode, EditMode::DraggingVertex { .. }) {
                self.mode = EditMode::Idle;
            }
        }
    }

    fn vertex_hit(
        &self,
        screen: Pos2,
        to_screen: &impl Fn([f32; 2]) -> Pos2,
    ) -> Option<(PadId, usize)> {
        for pad in &self.preset.pads {
            for (i, v) in pad.polygon.iter().enumerate() {
                let img = self.pp_to_image(*v);
                if to_screen(img).distance(screen) <= VERTEX_HANDLE_RADIUS + 2.0 {
                    return Some((pad.id, i));
                }
            }
        }
        None
    }

    /// `pp` is a pad-plane coordinate — same space the polygons are stored in.
    fn pad_hit(&self, pp: [f32; 2]) -> Option<PadId> {
        for pad in self.preset.pads.iter().rev() {
            if pad.is_active() && point_in_polygon(pp, &pad.polygon) {
                return Some(pad.id);
            }
        }
        None
    }

    fn commit_polygon(&mut self, polygon: Vec<[f32; 2]>) {
        if polygon.len() < 3 {
            self.mode = EditMode::Idle;
            return;
        }
        let id = self.preset.next_id();
        let index = self.preset.pads.len();
        let sample_id = (index as SampleId) % 3;
        let pad = Pad {
            id,
            name: format!("pad {}", index + 1),
            polygon,
            sample_id,
            sample_path: None,
            threshold: 6.0,
            metric_max: 25.0,
            refractory_ms: 60,
            velocity_curve: 1.0,
            midi: None,
            color: pad::pick_color(index),
        };
        self.preset.pads.push(pad);
        self.selected = Some(id);
        self.mode = EditMode::Idle;
        self.publish();
    }

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("pads");
        ui.small("click empty space to draw · click vertex to drag · click pad to select");
        ui.small("Enter closes · Esc cancels · Del removes");
        ui.separator();

        // Pad list.
        let ids: Vec<PadId> = self.preset.pads.iter().map(|p| p.id).collect();
        let mut to_delete: Option<PadId> = None;
        for id in ids {
            let (name, color) = {
                let pad = self.preset.pads.iter().find(|p| p.id == id).unwrap();
                (pad.name.clone(), pad.color)
            };
            let selected = self.selected == Some(id);
            ui.horizontal(|ui| {
                let swatch = Color32::from_rgb(color[0], color[1], color[2]);
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, swatch);
                if ui.selectable_label(selected, &name).clicked() {
                    self.selected = Some(id);
                }
                if ui.small_button("✕").clicked() {
                    to_delete = Some(id);
                }
            });
        }
        if let Some(id) = to_delete {
            self.preset.remove(id);
            if self.selected == Some(id) {
                self.selected = None;
            }
            self.publish();
        }

        ui.horizontal(|ui| {
            if ui.button("clear all pads").clicked() {
                self.preset.pads.clear();
                self.selected = None;
                self.publish();
            }
        });

        ui.separator();
        ui.strong("calibration");
        ui.small("map the physical pad surface to a rectified plane");
        let calibrated = self.preset.homography.is_some();
        ui.label(if calibrated {
            "status: calibrated"
        } else {
            "status: none (pad-plane == image)"
        });
        ui.horizontal(|ui| {
            if ui
                .button(if calibrated { "re-calibrate…" } else { "calibrate…" })
                .clicked()
            {
                self.begin_calibration();
            }
            if calibrated && ui.button("clear").clicked() {
                self.clear_calibration();
            }
        });
        if let EditMode::Calibrating { points } = &self.mode {
            ui.label(format!("captured {} / 4 corners", points.len()));
            ui.small("Esc cancels · Backspace undoes last · Enter commits when 4");
        }

        ui.separator();
        ui.strong("camera");
        ui.horizontal(|ui| {
            let current = self
                .current_camera_name
                .clone()
                .unwrap_or_else(|| format!("index {}", self.camera.requested_index()));
            let mut pick: Option<CameraEntry> = None;
            egui::ComboBox::from_id_salt("camera_pick")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for cam in &self.cameras {
                        let sel = self.current_camera_name.as_deref() == Some(cam.name.as_str());
                        if ui
                            .selectable_label(sel, format!("{}. {}", cam.index, cam.name))
                            .clicked()
                        {
                            pick = Some(cam.clone());
                        }
                    }
                    if self.cameras.is_empty() {
                        ui.label("(no cameras found — try ↻)");
                    }
                });
            if ui.small_button("↻").on_hover_text("refresh camera list").clicked() {
                self.cameras = camera::list_cameras();
            }
            if let Some(cam) = pick {
                self.camera.request(cam.index);
                self.current_camera_name = Some(cam.name);
                // Drop the cached texture so we don't briefly render the old
                // frame at the new device's resolution.
                self.texture = None;
                self.last_uploaded_seq = 0;
            }
        });

        ui.separator();
        ui.strong("MIDI output");
        ui.horizontal(|ui| {
            let current = self.midi.current().unwrap_or("(none)").to_string();
            let mut pick: Option<Option<String>> = None;
            egui::ComboBox::from_id_salt("midi_port_pick")
                .selected_text(current)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(self.midi.current().is_none(), "(none)").clicked() {
                        pick = Some(None);
                    }
                    for name in self.midi.ports() {
                        let sel = self.midi.current() == Some(name.as_str());
                        if ui.selectable_label(sel, name).clicked() {
                            pick = Some(Some(name.clone()));
                        }
                    }
                });
            if ui.small_button("↻").on_hover_text("refresh port list").clicked() {
                self.midi.refresh_ports();
            }
            if let Some(choice) = pick {
                self.midi.set_port(choice);
            }
        });

        // Per-pad properties.
        ui.separator();
        if let Some(id) = self.selected {
            self.pad_editor(ui, id);
        } else {
            ui.label("select a pad to edit its properties");
        }
    }

    fn pad_editor(&mut self, ui: &mut egui::Ui, id: PadId) {
        // Collect sample catalog up-front so we don't borrow `self.audio`
        // while mutating `self.preset`.
        let bank_snap = self.audio.bank_snapshot();
        let sample_entries: Vec<(SampleId, String)> = bank_snap
            .samples()
            .iter()
            .enumerate()
            .map(|(i, s)| (i as SampleId, s.name.clone()))
            .collect();
        drop(bank_snap);

        let Some(pad) = self.preset.pads.iter_mut().find(|p| p.id == id) else {
            return;
        };
        let mut changed = false;

        ui.heading(format!("pad: {}", pad.name));
        ui.label(format!(
            "id {} · {} verts · sample s{}",
            pad.id,
            pad.polygon.len(),
            pad.sample_id
        ));
        ui.separator();

        // Name.
        ui.horizontal(|ui| {
            ui.label("name:");
            if ui.text_edit_singleline(&mut pad.name).changed() {
                changed = true;
            }
        });

        // Color.
        ui.horizontal(|ui| {
            ui.label("color:");
            let mut rgb = pad.color;
            if ui.color_edit_button_srgb(&mut rgb).changed() {
                pad.color = rgb;
                changed = true;
            }
        });

        ui.separator();
        ui.strong("sample");
        // Dropdown over loaded samples.
        let current_name = sample_entries
            .iter()
            .find(|(id, _)| *id == pad.sample_id)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("s{} (missing)", pad.sample_id));
        egui::ComboBox::from_id_salt(("sample_pick", pad.id))
            .selected_text(current_name)
            .show_ui(ui, |ui| {
                for (sid, name) in &sample_entries {
                    let selected = *sid == pad.sample_id;
                    if ui
                        .selectable_label(selected, format!("s{sid} · {name}"))
                        .clicked()
                    {
                        pad.sample_id = *sid;
                        changed = true;
                    }
                }
            });

        // Audition button — plays the currently-assigned sample.
        let audition = ui.button("▶ audition").clicked();
        let load_wav = ui.button("load WAV…").clicked();

        ui.separator();
        ui.strong("detection");
        ui.horizontal(|ui| {
            ui.label("threshold:");
            if ui
                .add(egui::Slider::new(&mut pad.threshold, 1.0..=40.0).logarithmic(true))
                .changed()
            {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("metric_max (→ vel 1.0):");
            if ui
                .add(egui::Slider::new(&mut pad.metric_max, 2.0..=80.0).logarithmic(true))
                .changed()
            {
                changed = true;
            }
        });
        // Keep metric_max strictly greater than threshold.
        if pad.metric_max <= pad.threshold {
            pad.metric_max = pad.threshold + 1.0;
            changed = true;
        }
        ui.horizontal(|ui| {
            ui.label("refractory (ms):");
            if ui
                .add(egui::Slider::new(&mut pad.refractory_ms, 10..=500))
                .changed()
            {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("velocity curve:")
                .on_hover_text("exponent applied to the 0..1 velocity. <1 softer, >1 harder");
            if ui
                .add(egui::Slider::new(&mut pad.velocity_curve, 0.3..=3.0))
                .changed()
            {
                changed = true;
            }
        });

        ui.separator();
        ui.strong("MIDI");
        let mut enabled = pad.midi.is_some();
        if ui.checkbox(&mut enabled, "send MIDI note on hit").changed() {
            pad.midi = if enabled {
                Some(MidiMapping { channel: 9, note: 36 + pad.id as u8 % 12 })
            } else {
                None
            };
            changed = true;
        }
        if let Some(m) = pad.midi.as_mut() {
            ui.horizontal(|ui| {
                ui.label("channel:");
                let mut ch1 = m.channel as u32 + 1; // show 1..=16
                if ui.add(egui::Slider::new(&mut ch1, 1..=16)).changed() {
                    m.channel = (ch1 - 1) as u8;
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("note:");
                let mut n = m.note as i32;
                if ui
                    .add(egui::Slider::new(&mut n, 0..=127).text(note_name(m.note)))
                    .changed()
                {
                    m.note = n.clamp(0, 127) as u8;
                    changed = true;
                }
            });
        }

        // Live stats readout.
        ui.separator();
        if let Some(st) = self.pad_stats.get(&pad.id) {
            ui.label(format!(
                "live metric: {:.1} / thr {:.1}  {}",
                st.metric,
                st.threshold,
                if st.metric >= st.threshold { "HIT" } else { "…" }
            ));
        }

        // Copy pad state we need for side-effects *before* releasing the mutable borrow.
        let pad_sample_id = pad.sample_id;

        // Publish if anything changed.
        if changed {
            self.publish();
        }

        // Side-effects that need non-pad borrows.
        if audition {
            self.audio.trigger(TriggerEvent {
                sample_id: pad_sample_id,
                velocity: 1.0,
            });
        }
        if load_wav {
            self.load_wav_for_selected(id);
        }
    }

    fn load_wav_for_selected(&mut self, pad_id: PadId) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("WAV audio", &["wav", "WAV"])
            .set_title("Load sample (WAV)")
            .pick_file()
        else {
            return;
        };
        match crate::audio::samples::load_wav(&path) {
            Ok(sample) => {
                let name = sample.name.clone();
                let sid = self.audio.add_sample(sample);
                if let Some(pad) = self.preset.pads.iter_mut().find(|p| p.id == pad_id) {
                    pad.sample_id = sid;
                }
                self.publish();
                self.set_status(format!("loaded sample: {name} → s{sid}"), false);
            }
            Err(e) => {
                log::error!("load wav failed: {e:#}");
                self.set_status(format!("load wav failed: {e}"), true);
            }
        }
    }
}

// ----- helpers -----

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let ar = a.to_array();
    let br = b.to_array();
    Color32::from_rgba_unmultiplied(
        lerp_u8(ar[0], br[0], t),
        lerp_u8(ar[1], br[1], t),
        lerp_u8(ar[2], br[2], t),
        lerp_u8(ar[3], br[3], t),
    )
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

/// Human-readable MIDI note label, e.g. 60 → "C4".
fn note_name(note: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let n = note as i32;
    let octave = n / 12 - 1;
    let name = NAMES[(n % 12) as usize];
    format!("{name}{octave}")
}

fn polygon_aabb(polygon: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let mut min = [1.0f32, 1.0];
    let mut max = [0.0f32, 0.0];
    for v in polygon {
        min[0] = min[0].min(v[0]);
        min[1] = min[1].min(v[1]);
        max[0] = max[0].max(v[0]);
        max[1] = max[1].max(v[1]);
    }
    (min, max)
}

fn polygon_centroid(polygon: &[[f32; 2]]) -> [f32; 2] {
    let n = polygon.len() as f32;
    let mut cx = 0.0f32;
    let mut cy = 0.0;
    for v in polygon {
        cx += v[0];
        cy += v[1];
    }
    [cx / n, cy / n]
}

fn point_in_polygon(p: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let (x, y) = (p[0], p[1]);
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i][0], polygon[i][1]);
        let (xj, yj) = (polygon[j][0], polygon[j][1]);
        let crosses =
            (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi + f32::EPSILON) + xi;
        if crosses {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        if let Ok(s) = self.stats_slot.lock() {
            self.last_stats = *s;
        }
        if let Ok(s) = self.detect_stats.lock() {
            self.pad_stats = s.clone();
        }

        // Surface camera-switch failures from the capture thread. If the
        // supervisor reverted to a known-good device, resync our picker label
        // to whatever the thread is now running.
        if let Some(msg) = self.camera_errors.lock().ok().and_then(|mut e| e.take()) {
            let active = self.last_stats.active_index;
            if active >= 0 {
                if let Some(entry) = self.cameras.iter().find(|c| c.index as i32 == active) {
                    self.current_camera_name = Some(entry.name.clone());
                }
            }
            self.set_status(msg, true);
        }

        self.handle_trigger_keys(&ctx);
        let native_size = self.upload_frame_if_new(&ctx);
        ctx.request_repaint();

        egui::TopBottomPanel::top("stats").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("drum-scan");
                ui.separator();
                // File actions.
                if ui.button("New").clicked() {
                    self.new_preset();
                }
                if ui.button("Open…").clicked() {
                    self.open_preset_dialog();
                }
                if ui.button("Save").clicked() {
                    self.save_preset();
                }
                if ui.button("Save as…").clicked() {
                    self.save_preset_as_dialog();
                }
                // Keyboard shortcuts: ⌘S / ⌘⇧S / ⌘O / ⌘N.
                let (save, save_as, open_, new_) = ctx.input(|i| {
                    (
                        i.modifiers.command
                            && !i.modifiers.shift
                            && i.key_pressed(egui::Key::S),
                        i.modifiers.command
                            && i.modifiers.shift
                            && i.key_pressed(egui::Key::S),
                        i.modifiers.command && i.key_pressed(egui::Key::O),
                        i.modifiers.command && i.key_pressed(egui::Key::N),
                    )
                });
                if save_as {
                    self.save_preset_as_dialog();
                } else if save {
                    self.save_preset();
                }
                if open_ {
                    self.open_preset_dialog();
                }
                if new_ {
                    self.new_preset();
                }
                ui.separator();
                if let Some(p) = &self.current_path {
                    ui.label(format!(
                        "preset: {}",
                        p.file_name().and_then(|s| s.to_str()).unwrap_or("?")
                    ));
                } else {
                    ui.label("preset: (unsaved)");
                }
                ui.separator();
                ui.label(format!(
                    "camera: {}x{} @ {:.1} fps",
                    self.last_stats.width, self.last_stats.height, self.last_stats.fps
                ));
                ui.separator();
                ui.label(format!(
                    "audio: {} Hz × {} ch, buf {}",
                    self.audio.sample_rate(),
                    self.audio.channels(),
                    self.audio
                        .buffer_frames()
                        .map(|b| b.to_string())
                        .unwrap_or_else(|| "default".into()),
                ));
                ui.separator();
                let mode_label = match self.mode {
                    EditMode::Idle => "mode: idle".to_string(),
                    EditMode::Drawing(ref v) => format!("mode: drawing ({} pts)", v.len()),
                    EditMode::DraggingVertex { .. } => "mode: dragging".to_string(),
                    EditMode::Calibrating { ref points } => {
                        format!("mode: calibrating ({}/4)", points.len())
                    }
                };
                ui.label(mode_label);
            });

            // Transient status/error line.
            if let Some(st) = &self.status {
                if Instant::now() < st.until {
                    let color = if st.is_error {
                        Color32::from_rgb(255, 120, 120)
                    } else {
                        Color32::from_rgb(140, 220, 160)
                    };
                    ui.colored_label(color, &st.text);
                } else {
                    self.status = None;
                }
            }
        });

        egui::SidePanel::right("pads")
            .resizable(true)
            .default_width(280.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.sidebar(ui);
                });
            });

        egui::TopBottomPanel::bottom("triggers").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("test triggers:");
                if ui.button("[1] kick").clicked() {
                    self.audio.trigger(TriggerEvent {
                        sample_id: 0,
                        velocity: 1.0,
                    });
                }
                if ui.button("[2] snare").clicked() {
                    self.audio.trigger(TriggerEvent {
                        sample_id: 1,
                        velocity: 1.0,
                    });
                }
                if ui.button("[3] hat").clicked() {
                    self.audio.trigger(TriggerEvent {
                        sample_id: 2,
                        velocity: 1.0,
                    });
                }
                ui.separator();
                ui.label("1/2/3 or space · click into frame to draw · drag vertices");
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| match (self.texture.clone(), native_size)
        {
            (Some(tex), Some(native)) => {
                ui.centered_and_justified(|ui| self.paint_and_edit(ui, &tex, native));
            }
            _ => {
                ui.centered_and_justified(|ui| {
                    ui.label("waiting for first camera frame…");
                });
            }
        });
    }
}
