//! MIDI output via `midir`.
//!
//! A single worker thread owns the `MidiOutputConnection` and consumes
//! commands over an `mpsc` channel:
//!
//! - `SetPort(Some(name))` (re)connects to the named port.
//! - `SetPort(None)` disconnects.
//! - `Fire { channel, note, velocity }` sends note-on immediately and
//!   schedules a note-off 40 ms later so percussive plugins see proper
//!   on/off pairs.
//!
//! The UI thread owns the [`MidiBus`] and can also clone cheap
//! [`MidiSender`] handles for the detect thread. Sends are non-blocking
//! and lock-free from the caller's point of view.
//!
//! Latency: enumerating/opening ports is slow (hundreds of ms on macOS),
//! so that happens on the worker. Firing a note is a channel push +
//! one `MidiOutputConnection::send` call on the worker — well under a
//! millisecond.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use midir::{MidiOutput, MidiOutputConnection};

const CLIENT_NAME: &str = "drum-scan";
const PORT_NAME: &str = "drum-scan-out";
const NOTE_OFF_DELAY: Duration = Duration::from_millis(40);

enum Command {
    SetPort(Option<String>),
    Fire { channel: u8, note: u8, velocity: u8 },
}

pub struct MidiBus {
    tx: Sender<Command>,
    ports: Vec<String>,
    current: Option<String>,
}

#[derive(Clone)]
pub struct MidiSender {
    tx: Sender<Command>,
}

impl MidiSender {
    pub fn fire(&self, channel: u8, note: u8, velocity: u8) {
        let _ = self.tx.send(Command::Fire {
            channel,
            note,
            velocity,
        });
    }
}

impl MidiBus {
    pub fn new() -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Command>();
        thread::Builder::new()
            .name("midi".into())
            .spawn(move || midi_loop(rx))?;
        let ports = list_ports().unwrap_or_else(|e| {
            log::warn!("midi: enumerating ports failed: {e:#}");
            Vec::new()
        });
        Ok(Self {
            tx,
            ports,
            current: None,
        })
    }

    pub fn sender(&self) -> MidiSender {
        MidiSender { tx: self.tx.clone() }
    }

    pub fn ports(&self) -> &[String] {
        &self.ports
    }

    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    pub fn refresh_ports(&mut self) {
        match list_ports() {
            Ok(p) => self.ports = p,
            Err(e) => log::warn!("midi: port refresh failed: {e:#}"),
        }
    }

    pub fn set_port(&mut self, name: Option<String>) {
        self.current = name.clone();
        let _ = self.tx.send(Command::SetPort(name));
    }
}

fn list_ports() -> Result<Vec<String>> {
    let out = MidiOutput::new(CLIENT_NAME).map_err(|e| anyhow!("midi init: {e}"))?;
    let ports = out.ports();
    Ok(ports
        .iter()
        .filter_map(|p| out.port_name(p).ok())
        .collect())
}

fn connect(name: &str) -> Result<MidiOutputConnection> {
    let out = MidiOutput::new(CLIENT_NAME).map_err(|e| anyhow!("midi init: {e}"))?;
    let ports = out.ports();
    let port = ports
        .iter()
        .find(|p| out.port_name(p).ok().as_deref() == Some(name))
        .cloned()
        .ok_or_else(|| anyhow!("midi port '{name}' not found"))?;
    out.connect(&port, PORT_NAME)
        .map_err(|e| anyhow!("midi connect: {e}"))
}

fn midi_loop(rx: Receiver<Command>) {
    let mut conn: Option<MidiOutputConnection> = None;
    // Pending note-offs: (deadline, channel, note).
    let mut pending: Vec<(Instant, u8, u8)> = Vec::new();

    loop {
        let now = Instant::now();
        let next_off = pending
            .iter()
            .map(|(t, _, _)| *t)
            .min()
            .map(|t| t.saturating_duration_since(now))
            .unwrap_or(Duration::from_secs(60));

        match rx.recv_timeout(next_off) {
            Ok(Command::SetPort(name)) => {
                conn = None;
                if let Some(n) = name {
                    match connect(&n) {
                        Ok(c) => {
                            log::info!("midi: connected '{n}'");
                            conn = Some(c);
                        }
                        Err(e) => log::error!("midi connect failed: {e:#}"),
                    }
                } else {
                    log::info!("midi: disconnected");
                }
            }
            Ok(Command::Fire {
                channel,
                note,
                velocity,
            }) => {
                let ch = channel & 0x0F;
                let note = note & 0x7F;
                let vel = velocity.min(127);
                if let Some(c) = conn.as_mut() {
                    let _ = c.send(&[0x90 | ch, note, vel]);
                }
                pending.push((Instant::now() + NOTE_OFF_DELAY, ch, note));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        let now = Instant::now();
        pending.retain(|(t, channel, note)| {
            if *t <= now {
                if let Some(c) = conn.as_mut() {
                    let _ = c.send(&[0x80 | *channel, *note, 0]);
                }
                false
            } else {
                true
            }
        });
    }
}
