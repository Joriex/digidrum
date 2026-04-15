# digidrum

> [!CAUTION]
> Be aware that this project is entirely vibe coded.

Turn any surface into a drum kit with a webcam.

Point a camera at a table (or pad, or cereal box), draw polygons over the
spots you want to hit, and play. Each polygon is a pad that triggers a
sample and/or a MIDI note when motion crosses its threshold. Angled
cameras are handled by a 4-point homography calibration so pads stay
locked to the physical surface.

Designed for low latency (<15 ms stick-to-sound target). Rust, `cpal`
audio, `nokhwa` capture, `egui` UI, `midir` MIDI out.

## Run

```
cargo run --release
```
