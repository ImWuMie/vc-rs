use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use rtrb::RingBuffer;
use thread_priority::{set_current_thread_priority, ThreadPriority};
use vc_core::dsp;
use vc_core::model_rvc::{
    set_process_gpu_priority, set_process_power_throttling, ChunkConverter, ChunkOutputConfig,
    ChunkStats, F0Config, GpuPriority, LiveParams, NoiseGateShaping, OutputDynamicsConfig,
    RvcPipeline, RvcPipelineConfig,
};
use vc_core::sola::SmoothingKind;
use vc_core::Provider;

use crate::audio::{self, AudioStream, RealtimeAudio};

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;
const COMMAND_CAPACITY: usize = 8;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AudioBackend {
    #[default]
    Cpal,
    Wasapi,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Smoother {
    #[default]
    Sola,
    Psola,
}

impl Smoother {
    fn kind(self) -> SmoothingKind {
        match self {
            Self::Sola => SmoothingKind::Sola,
            Self::Psola => SmoothingKind::Psola,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DenoiserMode {
    #[default]
    Off,
    NoiseGate,
    Rnnoise,
}

#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    pub model: Option<PathBuf>,
    pub embedder: Option<PathBuf>,
    pub embedder_output: Option<String>,
    pub f0_model: Option<PathBuf>,
    pub provider: Provider,
    pub gpu_priority: GpuPriority,
    pub gpu_device_id: u32,
    pub audio_backend: AudioBackend,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub wasapi_input_exclusive: bool,
    pub wasapi_output_exclusive: bool,
    pub wasapi_buffer_ms: u32,
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub smoother: Smoother,
    pub rvc_output_tail_discard_ms: u32,
    pub extra_convert_ms: u32,
    pub f0: F0Config,
    pub denoiser_mode: DenoiserMode,
    // Denoiser mode and gate attack/release/floor are static (set at load);
    // the gate threshold is live (see `LiveParams`).
    pub noise_gate_shaping: NoiseGateShaping,
    pub output_dynamics: OutputDynamicsConfig,
    pub passthrough: bool,
    pub debug_input_wav: Option<PathBuf>,
    pub debug_output_wav: Option<PathBuf>,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            model: None,
            embedder: None,
            embedder_output: None,
            f0_model: None,
            provider: Provider::Cpu,
            gpu_priority: GpuPriority::default(),
            gpu_device_id: 0,
            audio_backend: AudioBackend::Cpal,
            input_device: None,
            output_device: None,
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            smoother: Smoother::Sola,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            f0: F0Config::default(),
            denoiser_mode: DenoiserMode::Off,
            noise_gate_shaping: NoiseGateShaping::default(),
            output_dynamics: OutputDynamicsConfig::default(),
            passthrough: false,
            debug_input_wav: None,
            debug_output_wav: None,
        }
    }
}

impl RealtimeConfig {
    fn has_complete_model_set(&self) -> bool {
        self.model.is_some() && self.embedder.is_some() && self.f0_model.is_some()
    }

    pub fn validate(&self) -> Result<()> {
        if (self.wasapi_input_exclusive || self.wasapi_output_exclusive)
            && self.audio_backend != AudioBackend::Wasapi
        {
            bail!("WASAPI exclusive options require the WASAPI backend");
        }
        if self.chunk_ms == 0 {
            bail!("chunk size must be greater than zero");
        }
        let rms_mix_rate = self.output_dynamics.rms_mix_rate;
        if !(0.0..=1.0).contains(&rms_mix_rate) || !rms_mix_rate.is_finite() {
            bail!("RMS mix rate must be a finite value in 0.0..=1.0");
        }
        if !self.noise_gate_shaping.attack_ms.is_finite() || self.noise_gate_shaping.attack_ms < 0.0
        {
            bail!("noise gate attack must be a finite, non-negative value (ms)");
        }
        if !self.noise_gate_shaping.release_ms.is_finite()
            || self.noise_gate_shaping.release_ms < 0.0
        {
            bail!("noise gate release must be a finite, non-negative value (ms)");
        }
        if !(0.0..=1.0).contains(&self.noise_gate_shaping.floor) {
            bail!("noise gate floor must be in 0.0..=1.0");
        }
        if !self.passthrough
            && (self.model.is_none() || self.embedder.is_none() || self.f0_model.is_none())
        {
            bail!("model, embedder, and F0 model are required");
        }
        Ok(())
    }

    /// Builds the borrowed `RvcPipelineConfig` for the realtime worker, mapping
    /// the static engine config plus a live snapshot into the engine's load-time
    /// shape. Centralizing this here keeps the static→pipeline field copy in one
    /// place; `output_extra_ms` / `volume_excluded_ms` are derived from the
    /// crossfade/SOLA/tail knobs, not stored, so they live with the mapping.
    ///
    /// Only valid when all model paths are present. This includes switchable
    /// sessions whose initial route is passthrough; their RVC pipeline is still
    /// loaded for later live activation. The live snapshot seeds the load-time
    /// params; per-block updates still flow through `RvcPipeline::apply_live`.
    fn pipeline_config<'a>(
        &'a self,
        sample_rate: u32,
        chunk_samples: usize,
        live: &LiveParams,
    ) -> RvcPipelineConfig<'a> {
        let output_extra_ms = self
            .crossfade_ms
            .saturating_add(self.sola_search_ms)
            .saturating_add(self.rvc_output_tail_discard_ms);
        RvcPipelineConfig {
            model: self.model.as_ref().expect("validated"),
            embedder: self.embedder.as_ref().expect("validated"),
            embedder_output: self.embedder_output.as_deref(),
            f0_model: self.f0_model.as_ref().expect("validated"),
            provider: self.provider,
            gpu_priority: self.gpu_priority,
            gpu_device_id: self.gpu_device_id,
            sample_rate,
            chunk_samples,
            speaker_id: live.speaker_id,
            pitch_shift: live.pitch_shift,
            f0: self.f0.clone(),
            input_gain: live.input_gain,
            noise_gate_enabled: self.denoiser_mode == DenoiserMode::NoiseGate,
            noise_gate_threshold: live.noise_gate_threshold,
            noise_gate_shaping: self.noise_gate_shaping,
            output_extra_ms,
            volume_excluded_ms: self.crossfade_ms,
            extra_convert_ms: self.extra_convert_ms,
            output_gain: live.output_gain,
            output_dynamics: self.output_dynamics,
        }
    }
}

// `LiveParams` itself now lives in vc-core (re-exported via `lib.rs`) so the
// per-chunk live-update path is shared with the VST3 host callback; this is the
// lock-free worker-facing mirror that the audio side never touches.
#[derive(Default)]
struct AtomicLiveParams {
    pitch_shift: AtomicU32,
    speaker_id: AtomicI64,
    input_gain: AtomicU32,
    output_gain: AtomicU32,
    noise_gate_enabled: AtomicBool,
    noise_gate_threshold: AtomicU32,
}

impl AtomicLiveParams {
    fn new(value: LiveParams) -> Self {
        let this = Self::default();
        this.store(value);
        this
    }

    fn store(&self, value: LiveParams) {
        self.pitch_shift
            .store(value.pitch_shift.to_bits(), Ordering::Relaxed);
        self.speaker_id.store(value.speaker_id, Ordering::Relaxed);
        self.input_gain
            .store(value.input_gain.to_bits(), Ordering::Relaxed);
        self.output_gain
            .store(value.output_gain.to_bits(), Ordering::Relaxed);
        self.noise_gate_enabled
            .store(value.noise_gate_enabled, Ordering::Relaxed);
        self.noise_gate_threshold
            .store(value.noise_gate_threshold.to_bits(), Ordering::Relaxed);
    }

    fn load(&self) -> LiveParams {
        LiveParams {
            pitch_shift: f32::from_bits(self.pitch_shift.load(Ordering::Relaxed)),
            speaker_id: self.speaker_id.load(Ordering::Relaxed),
            input_gain: f32::from_bits(self.input_gain.load(Ordering::Relaxed)),
            output_gain: f32::from_bits(self.output_gain.load(Ordering::Relaxed)),
            noise_gate_enabled: self.noise_gate_enabled.load(Ordering::Relaxed),
            noise_gate_threshold: f32::from_bits(self.noise_gate_threshold.load(Ordering::Relaxed)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EngineState {
    #[default]
    Stopped,
    Starting,
    Running,
    Stopping,
    Error,
}

#[derive(Clone, Debug, Default)]
pub struct EngineStatusSnapshot {
    pub state: EngineState,
    pub message: String,
    pub input_device: String,
    pub output_device: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    pub passthrough_live_switchable: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceList {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TelemetrySnapshot {
    pub chunks: u64,
    pub inference_us: u64,
    pub input_rms: f32,
    pub output_rms: f32,
    pub input_overruns: u64,
    pub output_underruns: u64,
    pub output_dropped_samples: u64,
    pub output_buffer_samples: u64,
}

#[derive(Default)]
struct Telemetry {
    chunks: AtomicU64,
    inference_us: AtomicU64,
    input_rms_bits: AtomicU32,
    output_rms_bits: AtomicU32,
    input_overruns: AtomicU64,
    output_underruns: AtomicU64,
    output_dropped_samples: AtomicU64,
    output_buffer_samples: AtomicU64,
}

impl Telemetry {
    fn reset(&self) {
        self.chunks.store(0, Ordering::Relaxed);
        self.inference_us.store(0, Ordering::Relaxed);
        self.input_rms_bits.store(0, Ordering::Relaxed);
        self.output_rms_bits.store(0, Ordering::Relaxed);
        self.input_overruns.store(0, Ordering::Relaxed);
        self.output_underruns.store(0, Ordering::Relaxed);
        self.output_dropped_samples.store(0, Ordering::Relaxed);
        self.output_buffer_samples.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            chunks: self.chunks.load(Ordering::Relaxed),
            inference_us: self.inference_us.load(Ordering::Relaxed),
            input_rms: f32::from_bits(self.input_rms_bits.load(Ordering::Relaxed)),
            output_rms: f32::from_bits(self.output_rms_bits.load(Ordering::Relaxed)),
            input_overruns: self.input_overruns.load(Ordering::Relaxed),
            output_underruns: self.output_underruns.load(Ordering::Relaxed),
            output_dropped_samples: self.output_dropped_samples.load(Ordering::Relaxed),
            output_buffer_samples: self.output_buffer_samples.load(Ordering::Relaxed),
        }
    }
}

// Boxing the large `Apply` payload is intentionally declined: these commands
// flow at control-message cadence (model/config changes), not per audio block,
// so the size disparity costs nothing worth an extra heap allocation + indirection
// on every push. Kept inline so the worker's command path stays allocation-free.
#[allow(clippy::large_enum_variant)]
enum Command {
    Apply(RealtimeConfig),
    Stop,
    RefreshDevices(AudioBackend),
    Shutdown,
}

pub struct EngineController {
    tx: SyncSender<Command>,
    status: Arc<Mutex<EngineStatusSnapshot>>,
    devices: Arc<Mutex<DeviceList>>,
    telemetry: Arc<Telemetry>,
    live: Arc<AtomicLiveParams>,
    passthrough: Arc<AtomicBool>,
    control: Option<JoinHandle<()>>,
}

impl EngineController {
    pub fn new(initial_live: LiveParams) -> Self {
        // Standalone front-ends (GUI/CLI) may auto-download a missing Windows ML
        // catalog EP during model load; the VST3 plugin does not opt in.
        #[cfg(all(windows, feature = "windowsml"))]
        vc_core::windows_ml::set_ep_download_allowed(true);
        let (tx, rx) = mpsc::sync_channel(COMMAND_CAPACITY);
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let devices = Arc::new(Mutex::new(DeviceList::default()));
        let telemetry = Arc::new(Telemetry::default());
        let live = Arc::new(AtomicLiveParams::new(initial_live));
        let passthrough = Arc::new(AtomicBool::new(false));
        let control = {
            let status = Arc::clone(&status);
            let devices = Arc::clone(&devices);
            let telemetry = Arc::clone(&telemetry);
            let live = Arc::clone(&live);
            let passthrough = Arc::clone(&passthrough);
            thread::Builder::new()
                .name("vc-app-control".to_string())
                .stack_size(64 * 1024 * 1024)
                .spawn(move || control_loop(rx, status, devices, telemetry, live, passthrough))
                .expect("failed to spawn vc-app control thread")
        };
        Self {
            tx,
            status,
            devices,
            telemetry,
            live,
            passthrough,
            control: Some(control),
        }
    }

    pub fn apply_config(&self, config: RealtimeConfig) -> Result<()> {
        config.validate()?;
        self.try_command(Command::Apply(config))
    }

    pub fn stop(&self) -> Result<()> {
        self.try_command(Command::Stop)
    }

    pub fn refresh_devices(&self, backend: AudioBackend) -> Result<()> {
        self.try_command(Command::RefreshDevices(backend))
    }

    pub fn set_live_params(&self, params: LiveParams) {
        self.live.store(params);
    }

    pub fn set_passthrough(&self, enabled: bool) {
        self.passthrough.store(enabled, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (EngineStatusSnapshot, TelemetrySnapshot, DeviceList) {
        let status = self.status.lock().map(|s| s.clone()).unwrap_or_default();
        let devices = self.devices.lock().map(|d| d.clone()).unwrap_or_default();
        (status, self.telemetry.snapshot(), devices)
    }

    fn try_command(&self, command: Command) -> Result<()> {
        self.tx.try_send(command).map_err(|err| match err {
            TrySendError::Full(_) => anyhow!("engine command queue is full"),
            TrySendError::Disconnected(_) => anyhow!("engine control thread has stopped"),
        })
    }
}

impl Drop for EngineController {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(control) = self.control.take() {
            let _ = control.join();
        }
    }
}

fn control_loop(
    rx: Receiver<Command>,
    status: Arc<Mutex<EngineStatusSnapshot>>,
    devices: Arc<Mutex<DeviceList>>,
    telemetry: Arc<Telemetry>,
    live: Arc<AtomicLiveParams>,
    passthrough: Arc<AtomicBool>,
) {
    let mut session: Option<RealtimeSession> = None;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Command::Apply(config)) => {
                passthrough.store(config.passthrough, Ordering::Relaxed);
                set_status(&status, EngineState::Stopping, "Stopping previous session");
                drop(session.take());
                set_status(
                    &status,
                    EngineState::Starting,
                    "Loading model and audio devices",
                );
                // Surface a distinct status while a Windows ML EP downloads on
                // first use: the registration that triggers the (blocking,
                // possibly multi-minute) download happens inside the model load
                // below, so detect it here and update the message beforehand.
                #[cfg(all(windows, feature = "windowsml"))]
                if vc_core::windows_ml::provider_download_pending(config.provider) {
                    set_status(
                        &status,
                        EngineState::Starting,
                        "Downloading execution provider… first run can take a few minutes",
                    );
                }
                telemetry.reset();
                match RealtimeSession::start(
                    config,
                    Arc::clone(&telemetry),
                    Arc::clone(&live),
                    Arc::clone(&passthrough),
                ) {
                    Ok(new_session) => {
                        if let Ok(mut current) = status.lock() {
                            *current = new_session.status();
                        }
                        session = Some(new_session);
                    }
                    Err(err) => set_status(&status, EngineState::Error, format!("{err:#}")),
                }
            }
            Ok(Command::Stop) => {
                set_status(&status, EngineState::Stopping, "Stopping");
                drop(session.take());
                set_status(&status, EngineState::Stopped, "Stopped");
            }
            Ok(Command::RefreshDevices(backend)) => {
                let result = device_list(backend);
                if let Ok(mut current) = devices.lock() {
                    *current = result;
                }
            }
            Ok(Command::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if session
            .as_ref()
            .is_some_and(|s| !s.running.load(Ordering::Relaxed))
        {
            drop(session.take());
            set_status(&status, EngineState::Error, "Realtime worker stopped");
        }
    }
    drop(session);
}

fn set_status(
    status: &Mutex<EngineStatusSnapshot>,
    state: EngineState,
    message: impl Into<String>,
) {
    if let Ok(mut status) = status.lock() {
        status.state = state;
        status.message = message.into();
        if state != EngineState::Running {
            status.input_device.clear();
            status.output_device.clear();
            status.input_sample_rate = 0;
            status.output_sample_rate = 0;
            status.passthrough_live_switchable = false;
        }
    }
}

fn device_list(backend: AudioBackend) -> DeviceList {
    let result = match backend {
        AudioBackend::Cpal => audio::cpal_device_names(),
        AudioBackend::Wasapi => wasapi_device_names(),
    };
    match result {
        Ok((inputs, outputs)) => DeviceList {
            inputs,
            outputs,
            error: None,
        },
        Err(err) => DeviceList {
            error: Some(format!("{err:#}")),
            ..Default::default()
        },
    }
}

#[cfg(windows)]
fn wasapi_device_names() -> Result<(Vec<String>, Vec<String>)> {
    crate::audio::wasapi_audio::device_names()
}

#[cfg(not(windows))]
fn wasapi_device_names() -> Result<(Vec<String>, Vec<String>)> {
    bail!("WASAPI is only available on Windows")
}

enum PassthroughDenoiser {
    Off,
    Gate(dsp::NoiseGate),
    #[cfg(feature = "rnnoise")]
    Rnnoise(Box<vc_core::rnnoise::RnnoiseDenoiser>),
}

struct PassthroughProcessor {
    mode: DenoiserMode,
    shaping: NoiseGateShaping,
    input_rate: u32,
    output_rate: u32,
    denoiser: PassthroughDenoiser,
    resampler: dsp::StreamingResampleMono,
    input_scratch: Vec<f32>,
}

impl PassthroughProcessor {
    fn new(
        mode: DenoiserMode,
        shaping: NoiseGateShaping,
        input_rate: u32,
        output_rate: u32,
        live: &LiveParams,
    ) -> Result<Self> {
        let mut processor = Self {
            mode,
            shaping,
            input_rate,
            output_rate,
            denoiser: PassthroughDenoiser::Off,
            resampler: dsp::StreamingResampleMono::new(input_rate as usize, output_rate as usize)?,
            input_scratch: Vec::new(),
        };
        processor.reset(live)?;
        Ok(processor)
    }

    fn reset(&mut self, live: &LiveParams) -> Result<()> {
        self.resampler =
            dsp::StreamingResampleMono::new(self.input_rate as usize, self.output_rate as usize)?;
        self.denoiser = match self.mode {
            DenoiserMode::Rnnoise => {
                #[cfg(feature = "rnnoise")]
                {
                    PassthroughDenoiser::Rnnoise(Box::new(vc_core::rnnoise::RnnoiseDenoiser::new(
                        self.input_rate,
                    )?))
                }
                #[cfg(not(feature = "rnnoise"))]
                {
                    bail!("RNNoise support is not enabled in this build")
                }
            }
            DenoiserMode::Off | DenoiserMode::NoiseGate => PassthroughDenoiser::Off,
        };
        self.update_live_denoiser(live);
        Ok(())
    }

    fn update_live_denoiser(&mut self, live: &LiveParams) {
        if self.mode == DenoiserMode::Rnnoise {
            return;
        }
        if !live.noise_gate_enabled {
            self.denoiser = PassthroughDenoiser::Off;
            return;
        }
        match &mut self.denoiser {
            PassthroughDenoiser::Gate(gate) => gate.set_threshold(live.noise_gate_threshold),
            _ => {
                self.denoiser = PassthroughDenoiser::Gate(dsp::NoiseGate::new(
                    self.input_rate as f32,
                    live.noise_gate_threshold,
                    self.shaping.attack_ms,
                    self.shaping.release_ms,
                    self.shaping.floor,
                ));
            }
        }
    }

    fn process_chunk(
        &mut self,
        audio: &[f32],
        live: &LiveParams,
        prepared: &mut Vec<f32>,
    ) -> Result<ChunkStats> {
        self.update_live_denoiser(live);
        self.input_scratch.clear();
        self.input_scratch.extend(
            audio
                .iter()
                .map(|sample| (*sample * live.input_gain.max(0.0)).clamp(-1.0, 1.0)),
        );
        match &mut self.denoiser {
            PassthroughDenoiser::Off => {}
            PassthroughDenoiser::Gate(gate) => gate.process_in_place(&mut self.input_scratch),
            #[cfg(feature = "rnnoise")]
            PassthroughDenoiser::Rnnoise(denoiser) => {
                denoiser.process_in_place(&mut self.input_scratch)?
            }
        }
        let input_rms = dsp::rms(&self.input_scratch);
        prepared.clear();
        self.resampler.process_into(&self.input_scratch, prepared)?;
        let output_gain = live.output_gain.max(0.0);
        if (output_gain - 1.0).abs() > f32::EPSILON {
            for sample in prepared.iter_mut() {
                *sample = (*sample * output_gain).clamp(-1.0, 1.0);
            }
        }
        Ok(ChunkStats {
            silent: false,
            inference_time: Duration::ZERO,
            input_rms,
            output_rms: dsp::rms(prepared),
            model_output_samples: prepared.len(),
        })
    }
}

// Stateful processing remains on the inference worker. Passthrough-only
// sessions preserve the model-free diagnostic path; switchable sessions retain
// loaded RVC sessions but stop invoking them while passthrough is active.
#[allow(clippy::large_enum_variant)]
enum RuntimeModel {
    PassthroughOnly(PassthroughProcessor),
    Switchable {
        passthrough: PassthroughProcessor,
        rvc: ChunkConverter<RvcPipeline>,
        passthrough_active: bool,
    },
}

impl RuntimeModel {
    fn process_chunk(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        live: &LiveParams,
        passthrough_requested: bool,
        prepared: &mut Vec<f32>,
    ) -> Result<ChunkStats> {
        match self {
            Self::PassthroughOnly(passthrough) => passthrough.process_chunk(audio, live, prepared),
            Self::Switchable {
                passthrough,
                rvc,
                passthrough_active,
            } => {
                if passthrough_requested {
                    if !*passthrough_active {
                        passthrough.reset(live)?;
                        *passthrough_active = true;
                    }
                    return passthrough.process_chunk(audio, live, prepared);
                }
                if *passthrough_active {
                    // RVC was intentionally idle, so neither its rolling model
                    // context nor its output smoother may be joined to new input.
                    rvc.model_mut().reset_streaming_state()?;
                    rvc.reset_streaming_state();
                    *passthrough_active = false;
                }
                rvc.model_mut().apply_live(live);
                // Write the converted chunk straight into the worker-owned
                // `prepared` buffer (reused across chunks) instead of moving a
                // freshly allocated Vec out of the converter.
                rvc.process_chunk(audio, sample_rate, None, prepared)
            }
        }
    }
}

struct RealtimeSession {
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    input_stream: Option<AudioStream>,
    output_stream: Option<AudioStream>,
    status: EngineStatusSnapshot,
    debug_input_wav: Option<PathBuf>,
    debug_output_wav: Option<PathBuf>,
    debug_input: Arc<Mutex<Vec<f32>>>,
    debug_output: Arc<Mutex<Vec<f32>>>,
    input_rate: u32,
    output_rate: u32,
}

impl RealtimeSession {
    fn start(
        config: RealtimeConfig,
        telemetry: Arc<Telemetry>,
        live: Arc<AtomicLiveParams>,
        passthrough_live: Arc<AtomicBool>,
    ) -> Result<Self> {
        // Process-wide GPU scheduling priority (all backends). Applied here on
        // the controller thread, off the audio callback, and re-applied on every
        // reconfigure so a changed setting takes effect on the next session.
        set_process_gpu_priority(config.gpu_priority);
        // Disable CPU power throttling (EcoQoS) under High so inference stays on
        // performance cores at full clock even when the window loses focus,
        // removing the large foreground/background timing gap. Covers every
        // thread (ORT intra-op pool, TensorRT CUDA orchestration, worker), which
        // a per-thread override on the worker alone would not.
        set_process_power_throttling(config.gpu_priority == GpuPriority::High);
        let audio = RealtimeAudio::open(
            config.audio_backend,
            config.wasapi_input_exclusive,
            config.wasapi_output_exclusive,
            config.input_device.as_deref(),
            config.output_device.as_deref(),
            config.wasapi_buffer_ms,
        )?;
        let input_rate = audio.input_sample_rate();
        let output_rate = audio.output_sample_rate();
        let input_chunk = dsp::chunk_samples_for_rate(input_rate, config.chunk_ms);
        let output_chunk = dsp::chunk_samples_for_rate(output_rate, config.chunk_ms);
        let current_live = live.load();
        let debug_input = Arc::new(Mutex::new(Vec::new()));
        let debug_output = Arc::new(Mutex::new(Vec::new()));
        let passthrough_processor = PassthroughProcessor::new(
            config.denoiser_mode,
            config.noise_gate_shaping,
            input_rate,
            output_rate,
            &current_live,
        )?;
        let passthrough_live_switchable = config.has_complete_model_set();
        let model = if passthrough_live_switchable {
            let pipeline_config = config.pipeline_config(input_rate, input_chunk, &current_live);
            let pipeline = match config.denoiser_mode {
                DenoiserMode::Off | DenoiserMode::NoiseGate => RvcPipeline::load(pipeline_config)?,
                DenoiserMode::Rnnoise => {
                    #[cfg(feature = "rnnoise")]
                    {
                        RvcPipeline::load_with_rnnoise(pipeline_config)?
                    }
                    #[cfg(not(feature = "rnnoise"))]
                    {
                        bail!("RNNoise support is not enabled in this build")
                    }
                }
            };
            RuntimeModel::Switchable {
                passthrough: passthrough_processor,
                rvc: ChunkConverter::new(
                    pipeline,
                    ChunkOutputConfig {
                        kind: config.smoother.kind(),
                        output_sample_rate: output_rate,
                        output_chunk_samples: output_chunk,
                        crossfade_ms: config.crossfade_ms,
                        sola_search_ms: config.sola_search_ms,
                        tail_discard_ms: config.rvc_output_tail_discard_ms,
                    },
                ),
                passthrough_active: config.passthrough,
            }
        } else {
            RuntimeModel::PassthroughOnly(passthrough_processor)
        };

        let output_capacity = output_chunk * OUTPUT_QUEUE_CHUNKS;
        let running = Arc::new(AtomicBool::new(true));
        // Build the device streams before spawning the inference worker: a
        // stream failure then returns without ever starting (and stopping) the
        // worker and its model/CUDA context. The streams stay paused until
        // play(), so the worker can attach afterwards.
        let (input_stream, output_stream, mut input_consumer, mut output_producer) = build_streams(
            &audio,
            input_chunk * INPUT_QUEUE_CHUNKS,
            output_capacity,
            &running,
            &telemetry,
        )?;
        let worker_running = Arc::clone(&running);
        let worker_telemetry = Arc::clone(&telemetry);
        let worker_debug_input = Arc::clone(&debug_input);
        let worker_debug_output = Arc::clone(&debug_output);
        let capture_input = config.debug_input_wav.is_some();
        let capture_output = config.debug_output_wav.is_some();
        let mut worker = Some(
            thread::Builder::new()
                .name("vc-app-inference".to_string())
                .spawn(move || {
                    if let Err(err) = set_current_thread_priority(ThreadPriority::Max) {
                        tracing::warn!("failed to set inference worker thread priority: {err}");
                    }
                    let mut model = model;
                    let mut input_acc = Vec::<f32>::with_capacity(input_chunk * 2);
                    let mut prepared = Vec::<f32>::with_capacity(output_chunk * 2);
                    while worker_running.load(Ordering::SeqCst) {
                        let available = input_consumer
                            .slots()
                            .min(input_chunk.saturating_sub(input_acc.len()));
                        if available > 0 {
                            let old = input_acc.len();
                            input_acc.resize(old + available, 0.0);
                            if input_consumer
                                .pop_entire_slice(&mut input_acc[old..])
                                .is_err()
                            {
                                input_acc.truncate(old);
                            }
                        }
                        if input_acc.len() < input_chunk {
                            thread::sleep(Duration::from_millis(2));
                            continue;
                        }
                        if capture_input {
                            if let Ok(mut samples) = worker_debug_input.lock() {
                                samples.extend_from_slice(&input_acc[..input_chunk]);
                            }
                        }
                        let stats = model.process_chunk(
                            &input_acc[..input_chunk],
                            input_rate,
                            &live.load(),
                            passthrough_live.load(Ordering::Relaxed),
                            &mut prepared,
                        );
                        input_acc.clear();
                        let Ok(stats) = stats else {
                            worker_running.store(false, Ordering::SeqCst);
                            break;
                        };
                        worker_telemetry.chunks.fetch_add(1, Ordering::Relaxed);
                        worker_telemetry
                            .inference_us
                            .store(stats.inference_time.as_micros() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .input_rms_bits
                            .store(stats.input_rms.to_bits(), Ordering::Relaxed);
                        worker_telemetry
                            .output_rms_bits
                            .store(stats.output_rms.to_bits(), Ordering::Relaxed);
                        let output_silent = stats.silent;
                        if capture_output {
                            if let Ok(mut samples) = worker_debug_output.lock() {
                                samples.extend_from_slice(&prepared);
                            }
                        }
                        let should_queue = !output_silent
                            || should_queue_silent_output(
                                output_capacity - output_producer.slots(),
                                output_chunk,
                            );
                        if should_queue {
                            let (_, remainder) = output_producer.push_partial_slice(&prepared);
                            worker_telemetry
                                .output_dropped_samples
                                .fetch_add(remainder.len() as u64, Ordering::Relaxed);
                        }
                        worker_telemetry.output_buffer_samples.store(
                            (output_capacity - output_producer.slots()) as u64,
                            Ordering::Relaxed,
                        );
                    }
                })?,
        );

        if let Err(err) = output_stream.play().and_then(|_| input_stream.play()) {
            drop(input_stream);
            drop(output_stream);
            stop_startup_worker(&running, &mut worker);
            return Err(err);
        }

        Ok(Self {
            running,
            worker: worker.take(),
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
            status: EngineStatusSnapshot {
                state: EngineState::Running,
                message: format!("Running ({})", audio.backend_label()),
                input_device: audio.input_name().to_string(),
                output_device: audio.output_name().to_string(),
                input_sample_rate: input_rate,
                output_sample_rate: output_rate,
                passthrough_live_switchable,
            },
            debug_input_wav: config.debug_input_wav,
            debug_output_wav: config.debug_output_wav,
            debug_input,
            debug_output,
            input_rate,
            output_rate,
        })
    }

    fn status(&self) -> EngineStatusSnapshot {
        self.status.clone()
    }
}

impl Drop for RealtimeSession {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        drop(self.input_stream.take());
        drop(self.output_stream.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        if let Some(path) = &self.debug_input_wav {
            if let Ok(samples) = self.debug_input.lock() {
                let _ = write_wav_mono(path, &samples, self.input_rate);
            }
        }
        if let Some(path) = &self.debug_output_wav {
            if let Ok(samples) = self.debug_output.lock() {
                let _ = write_wav_mono(path, &samples, self.output_rate);
            }
        }
    }
}

fn should_queue_silent_output(buffered: usize, output_chunk: usize) -> bool {
    // Keep at most one generated-silence chunk queued. Filling the output ring
    // during quiet periods delays or drops the first converted speech when
    // input resumes.
    buffered <= output_chunk
}

/// Streams plus the worker-side ring-buffer ends created in the same attempt.
type StreamEndpoints = (
    AudioStream,
    AudioStream,
    rtrb::Consumer<f32>,
    rtrb::Producer<f32>,
);

fn build_streams(
    audio: &RealtimeAudio,
    input_capacity: usize,
    output_capacity: usize,
    running: &Arc<AtomicBool>,
    telemetry: &Arc<Telemetry>,
) -> Result<StreamEndpoints> {
    let (mut input_producer, input_consumer) = RingBuffer::<f32>::new(input_capacity);
    let (output_producer, mut output_consumer) = RingBuffer::<f32>::new(output_capacity);
    let input_running = Arc::clone(running);
    let input_telemetry = Arc::clone(telemetry);
    let input_stream = audio.build_input_stream(move |samples| {
        if !input_running.load(Ordering::Relaxed) {
            return;
        }
        let (_, remainder) = input_producer.push_partial_slice(samples);
        if !remainder.is_empty() {
            input_telemetry
                .input_overruns
                .fetch_add(1, Ordering::Relaxed);
        }
    })?;
    let output_running = Arc::clone(running);
    let output_telemetry = Arc::clone(telemetry);
    let output_stream = audio.build_output_stream(move |out| {
        if !output_running.load(Ordering::Relaxed) {
            out.fill(0.0);
            return;
        }
        let (_, remainder) = output_consumer.pop_partial_slice(out);
        if !remainder.is_empty() {
            remainder.fill(0.0);
            output_telemetry
                .output_underruns
                .fetch_add(1, Ordering::Relaxed);
        }
        output_telemetry
            .output_buffer_samples
            .store(output_consumer.cached_slots() as u64, Ordering::Relaxed);
    })?;
    Ok((input_stream, output_stream, input_consumer, output_producer))
}

fn stop_startup_worker(running: &AtomicBool, worker: &mut Option<JoinHandle<()>>) {
    // Stream playback can still fail after the inference worker starts. Always
    // stop and join it before returning so failed Apply attempts cannot leave a
    // model/CUDA context alive behind the next session.
    running.store(false, Ordering::SeqCst);
    if let Some(worker) = worker.take() {
        let _ = worker.join();
    }
}

/// Write mono `f32` samples to a 16-bit PCM WAV file. Shared by the realtime
/// debug capture here and the CLI's WAV-conversion output so both produce an
/// identical file format.
pub fn write_wav_mono(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in dsp::f32_to_i16(samples) {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_params_round_trip_through_atomics() {
        let params = LiveParams {
            pitch_shift: -3.5,
            speaker_id: 7,
            input_gain: 0.5,
            output_gain: 2.0,
            noise_gate_enabled: true,
            noise_gate_threshold: 0.025,
        };
        let atomic = AtomicLiveParams::new(params);
        let out = atomic.load();
        assert_eq!(out.pitch_shift, params.pitch_shift);
        assert_eq!(out.speaker_id, params.speaker_id);
        assert_eq!(out.input_gain, params.input_gain);
        assert_eq!(out.output_gain, params.output_gain);
        assert_eq!(out.noise_gate_enabled, params.noise_gate_enabled);
        assert_eq!(out.noise_gate_threshold, params.noise_gate_threshold);
    }

    #[test]
    fn validation_requires_models_unless_passthrough() {
        assert!(RealtimeConfig::default().validate().is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn complete_model_set_controls_live_passthrough_capability() {
        let model = PathBuf::from("model.onnx");
        let complete = RealtimeConfig {
            model: Some(model.clone()),
            embedder: Some(model.clone()),
            f0_model: Some(model),
            ..Default::default()
        };
        assert!(complete.has_complete_model_set());
        assert!(!RealtimeConfig {
            passthrough: true,
            model: Some(PathBuf::from("model.onnx")),
            ..Default::default()
        }
        .has_complete_model_set());
    }

    #[test]
    fn passthrough_processor_applies_input_and_output_gain() {
        let live = LiveParams {
            input_gain: 2.0,
            output_gain: 2.0,
            ..Default::default()
        };
        let mut processor = PassthroughProcessor::new(
            DenoiserMode::Off,
            NoiseGateShaping::default(),
            48_000,
            48_000,
            &live,
        )
        .unwrap();
        let mut prepared = Vec::new();

        let stats = processor
            .process_chunk(&[0.25; 480], &live, &mut prepared)
            .unwrap();

        assert!(prepared.iter().all(|sample| (*sample - 1.0).abs() < 1e-6));
        assert!((stats.input_rms - 0.5).abs() < 1e-6);
        assert!((stats.output_rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn passthrough_processor_applies_live_noise_gate() {
        let live = LiveParams {
            noise_gate_enabled: true,
            noise_gate_threshold: 0.5,
            ..Default::default()
        };
        let mut processor = PassthroughProcessor::new(
            DenoiserMode::NoiseGate,
            NoiseGateShaping {
                attack_ms: 0.0,
                release_ms: 0.0,
                floor: 0.0,
            },
            48_000,
            48_000,
            &live,
        )
        .unwrap();
        let mut prepared = Vec::new();

        processor
            .process_chunk(&[0.01; 480], &live, &mut prepared)
            .unwrap();

        assert!(dsp::rms(&prepared) < 1e-6);
    }

    #[cfg(feature = "rnnoise")]
    #[test]
    fn passthrough_processor_runs_rnnoise_and_preserves_chunk_length() {
        let live = LiveParams::default();
        let mut processor = PassthroughProcessor::new(
            DenoiserMode::Rnnoise,
            NoiseGateShaping::default(),
            48_000,
            48_000,
            &live,
        )
        .unwrap();
        let mut prepared = Vec::new();

        processor
            .process_chunk(&[0.0; 960], &live, &mut prepared)
            .unwrap();

        assert_eq!(prepared.len(), 960);
        assert!(prepared.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn silent_output_does_not_fill_the_output_ring() {
        assert!(should_queue_silent_output(0, 1_000));
        assert!(should_queue_silent_output(1_000, 1_000));
        assert!(!should_queue_silent_output(1_001, 1_000));
    }
}
