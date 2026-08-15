use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use rtrb::RingBuffer;
use thread_priority::{set_current_thread_priority, ThreadPriority};
use vc_core::dsp;
use vc_core::dynamic_tuning::{
    DynamicLanguageProfile, DynamicTuner, DynamicTuningMode, DynamicTuningObservation,
    DynamicTuningSnapshot,
};
use vc_core::model_rvc::{
    set_process_gpu_priority, set_process_power_throttling, ChunkConverter, ChunkOutputConfig,
    ChunkStats, F0Config, F0Mode, FeatureRetrievalConfig, GpuPriority, LiveParams, LoadProgress,
    NoiseGateShaping, OutputDynamicsConfig, RvcPipeline, RvcPipelineConfig,
};
use vc_core::sola::SmoothingKind;
use vc_core::validation::{
    validate_conversion_timing, validate_non_negative_f32, validate_unit_interval,
    ConversionTiming, CONVERSION_TIMING_LIMITS,
};
use vc_core::voice_calibration::{
    VoiceCalibrationAccumulator, VoiceCalibrationProfile, DEFAULT_VOICE_CALIBRATION_DURATION_MS,
};
use vc_core::Provider;

use crate::audio::{self, AudioStream, RealtimeAudio};

const INPUT_QUEUE_CHUNKS: usize = 4;
const OUTPUT_QUEUE_CHUNKS: usize = 4;
const COMMAND_CAPACITY: usize = 8;
const BASE_MODEL_REQUEST_ID: u64 = 0;
const FIRST_DYNAMIC_MODEL_REQUEST_ID: NonZeroU64 = NonZeroU64::MIN;
const CALIBRATION_IDLE: u8 = 0;
const CALIBRATION_REQUESTED: u8 = 1;
const CALIBRATION_COLLECTING: u8 = 2;
const CALIBRATION_READY: u8 = 3;

/// OS audio host / API. Modelled on cpal's `HostId` (the canonical serialized
/// tokens match: `wasapi`/`asio`/`coreaudio`/`alsa`/`jack`), so the same enum
/// works across platforms. Every variant is always defined — selecting one that
/// is unsupported on the running platform or build (e.g. ASIO without the `asio`
/// feature, or CoreAudio on Windows) errors at open/enumeration time rather than
/// being conditionally compiled away, which keeps frontend match arms and
/// value-enums stable across targets.
///
/// "Backend" in user-facing surfaces (CLI flags, GUI labels) maps onto these. The
/// shared-vs-exclusive choice is a separate option (`wasapi_*_exclusive`), not a
/// host: WASAPI shared routes through cpal, WASAPI exclusive through the bespoke
/// `wasapi_audio` path until cpal gains exclusive mode. See [`audio::cpal_host`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioHost {
    Wasapi,
    Asio,
    CoreAudio,
    Alsa,
    Jack,
}

// Not derivable: the default is platform-conditional (mirrors cpal::default_host),
// even though on any single target the active cfg branch collapses to one variant
// — which is what makes clippy think a derive would do.
#[allow(clippy::derivable_impls)]
impl Default for AudioHost {
    fn default() -> Self {
        // Mirror cpal::default_host() per platform so an unset/migrated value lands
        // on the native default.
        #[cfg(windows)]
        {
            AudioHost::Wasapi
        }
        #[cfg(target_os = "macos")]
        {
            AudioHost::CoreAudio
        }
        #[cfg(not(any(windows, target_os = "macos")))]
        {
            AudioHost::Alsa
        }
    }
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
    Gtcrn,
    WebRtc,
    DeepFilterNet3,
}

const DENOISER_MODE_BITS: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AppliedDenoiserSnapshot {
    generation: u64,
    mode: DenoiserMode,
}

/// Lock-free publication of the denoiser mode the worker actually applied.
/// Generation and mode share one atomic word so a background model loader can
/// never observe a new mode paired with an old generation (or vice versa).
struct AppliedDenoiserState {
    packed: AtomicU64,
}

impl AppliedDenoiserState {
    fn new(snapshot: AppliedDenoiserSnapshot) -> Self {
        Self {
            packed: AtomicU64::new(pack_applied_denoiser(snapshot)),
        }
    }

    fn load(&self) -> AppliedDenoiserSnapshot {
        unpack_applied_denoiser(self.packed.load(Ordering::Acquire))
    }

    fn store(&self, snapshot: AppliedDenoiserSnapshot) {
        self.packed
            .store(pack_applied_denoiser(snapshot), Ordering::Release);
    }
}

fn pack_applied_denoiser(snapshot: AppliedDenoiserSnapshot) -> u64 {
    let mode = match snapshot.mode {
        DenoiserMode::Off => 0,
        DenoiserMode::NoiseGate => 1,
        DenoiserMode::Rnnoise => 2,
        DenoiserMode::Gtcrn => 3,
        DenoiserMode::WebRtc => 4,
        DenoiserMode::DeepFilterNet3 => 5,
    };
    debug_assert!(snapshot.generation <= (u64::MAX >> DENOISER_MODE_BITS));
    (snapshot.generation << DENOISER_MODE_BITS) | mode
}

fn unpack_applied_denoiser(packed: u64) -> AppliedDenoiserSnapshot {
    let mode = match packed & ((1 << DENOISER_MODE_BITS) - 1) {
        0 => DenoiserMode::Off,
        1 => DenoiserMode::NoiseGate,
        2 => DenoiserMode::Rnnoise,
        3 => DenoiserMode::Gtcrn,
        4 => DenoiserMode::WebRtc,
        5 => DenoiserMode::DeepFilterNet3,
        _ => unreachable!("invalid packed denoiser mode"),
    };
    AppliedDenoiserSnapshot {
        generation: packed >> DENOISER_MODE_BITS,
        mode,
    }
}

#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    pub model: Option<PathBuf>,
    pub embedder: Option<PathBuf>,
    pub embedder_output: Option<String>,
    pub f0_model: Option<PathBuf>,
    pub f0_mode: F0Mode,
    pub fcpe_model: Option<PathBuf>,
    /// Optional target-speaker FAISS `added_IVF*_Flat_*.index`. It is immutable
    /// for a live session because decoding it is model-load work; `index_rate`
    /// and `protect` remain lock-free live parameters in [`LiveParams`].
    pub feature_index: Option<PathBuf>,
    pub provider: Provider,
    pub gpu_priority: GpuPriority,
    pub gpu_device_id: u32,
    // Host is per direction: input and output are independent streams (and
    // independent clock domains, already resampled between by the engine), so
    // e.g. input WASAPI + output ASIO is a valid combination.
    pub input_host: AudioHost,
    pub output_host: AudioHost,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    // Optional monitor output: a second device (on `output_host`, shared cpal
    // path) that plays the converted signal with its own live `monitor_gain`.
    // `monitor_output_device` is `None` = system default when enabled.
    pub monitor_output_enabled: bool,
    pub monitor_output_device: Option<String>,
    pub wasapi_input_exclusive: bool,
    pub wasapi_output_exclusive: bool,
    pub wasapi_buffer_ms: u32,
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub smoother: Smoother,
    pub rvc_output_tail_discard_ms: u32,
    pub extra_convert_ms: u32,
    /// Optional fixed generator frame count. Changing this reloads the model
    /// pipeline because fixed providers need a distinct engine/CUDA graph.
    pub rvc_frames: Option<usize>,
    pub f0: F0Config,
    pub denoiser_mode: DenoiserMode,
    // GTCRN model directory (holds gtcrn_stream.onnx). Required only when
    // `denoiser_mode == Gtcrn`; ignored otherwise.
    pub gtcrn_model_dir: Option<PathBuf>,
    /// WebRTC-style spectral suppression level. Used when `denoiser_mode` is
    /// `WebRtc`; the backend itself remains model-free and loads on the worker.
    pub webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel,
    /// Official DeepFilterNet3 `.tar.gz` model archive. Required only for the
    /// `DeepFilterNet3` mode and never embedded in a vc-rs distribution.
    pub deepfilternet3_model: Option<PathBuf>,
    pub dfn3_attenuation_limit_db: f32,
    pub dfn3_post_filter_beta: f32,
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
            f0_mode: F0Mode::Rmvpe,
            fcpe_model: None,
            feature_index: None,
            provider: Provider::Cpu,
            gpu_priority: GpuPriority::default(),
            gpu_device_id: 0,
            input_host: AudioHost::default(),
            output_host: AudioHost::default(),
            input_device: None,
            output_device: None,
            monitor_output_enabled: false,
            monitor_output_device: None,
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            wasapi_buffer_ms: 0,
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            smoother: Smoother::Sola,
            rvc_output_tail_discard_ms: 10,
            extra_convert_ms: 100,
            rvc_frames: None,
            f0: F0Config::default(),
            denoiser_mode: DenoiserMode::Off,
            gtcrn_model_dir: None,
            webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel::default(),
            deepfilternet3_model: None,
            dfn3_attenuation_limit_db: vc_core::denoise_config::DEFAULT_DFN3_ATTENUATION_LIMIT_DB,
            dfn3_post_filter_beta: vc_core::denoise_config::DEFAULT_DFN3_POST_FILTER_BETA,
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
        self.model.is_some()
            && self.embedder.is_some()
            && (!self.f0_mode.uses_rmvpe() || self.f0_model.is_some())
            && (!self.f0_mode.uses_fcpe() || self.fcpe_model.is_some())
    }

    pub fn validate(&self) -> Result<()> {
        if self.wasapi_input_exclusive && self.input_host != AudioHost::Wasapi {
            bail!("WASAPI exclusive input requires the WASAPI input host");
        }
        if self.wasapi_output_exclusive && self.output_host != AudioHost::Wasapi {
            bail!("WASAPI exclusive output requires the WASAPI output host");
        }
        if self.monitor_output_enabled && self.output_host == AudioHost::Asio {
            bail!("monitor output is not supported with the ASIO output host");
        }
        validate_conversion_timing(
            ConversionTiming {
                chunk_ms: self.chunk_ms,
                crossfade_ms: self.crossfade_ms,
                sola_search_ms: self.sola_search_ms,
                tail_discard_ms: self.rvc_output_tail_discard_ms,
                extra_convert_ms: self.extra_convert_ms,
            },
            CONVERSION_TIMING_LIMITS,
        )?;
        validate_unit_interval("RMS mix rate", self.output_dynamics.rms_mix_rate)?;
        validate_non_negative_f32("F0 threshold", self.f0.f0_threshold)?;
        validate_non_negative_f32("silence threshold", self.f0.silence_threshold)?;
        validate_non_negative_f32("noise gate attack (ms)", self.noise_gate_shaping.attack_ms)?;
        validate_non_negative_f32(
            "noise gate release (ms)",
            self.noise_gate_shaping.release_ms,
        )?;
        validate_unit_interval("noise gate floor", self.noise_gate_shaping.floor)?;
        validate_non_negative_f32("target output RMS", self.output_dynamics.target_output_rms)?;
        validate_non_negative_f32("max output gain", self.output_dynamics.max_output_gain)?;
        if !self.passthrough && (self.model.is_none() || self.embedder.is_none()) {
            bail!("model and embedder are required");
        }
        if !self.passthrough && self.f0_mode.uses_rmvpe() && self.f0_model.is_none() {
            bail!(
                "{} F0 mode requires an RMVPE model (f0_model)",
                self.f0_mode.label()
            );
        }
        if self.f0_mode.uses_fcpe() && self.fcpe_model.is_none() {
            bail!(
                "{} F0 mode requires an FCPE model (fcpe_model)",
                self.f0_mode.label()
            );
        }
        if self.denoiser_mode == DenoiserMode::Gtcrn && self.gtcrn_model_dir.is_none() {
            bail!("GTCRN denoiser requires a model directory (gtcrn_model_dir)");
        }
        if self.denoiser_mode == DenoiserMode::WebRtc {
            #[cfg(not(feature = "webrtc"))]
            bail!("WebRTC denoising support is not enabled in this build");
        }
        if self.denoiser_mode == DenoiserMode::DeepFilterNet3 {
            #[cfg(not(feature = "deepfilternet3"))]
            bail!("DeepFilterNet3 support is not enabled in this build");
            #[cfg(feature = "deepfilternet3")]
            {
                let model = self.deepfilternet3_model.as_deref().ok_or_else(|| {
                    anyhow!("DeepFilterNet3 requires a model archive (deepfilternet3_model)")
                })?;
                if !model.is_file() {
                    bail!(
                        "DeepFilterNet3 model archive not found: {}",
                        model.display()
                    );
                }
                if !self.dfn3_attenuation_limit_db.is_finite()
                    || !(0.0..=vc_core::denoise_config::MAX_DFN3_ATTENUATION_LIMIT_DB)
                        .contains(&self.dfn3_attenuation_limit_db)
                {
                    bail!(
                        "DeepFilterNet3 attenuation limit must be in 0..={} dB",
                        vc_core::denoise_config::MAX_DFN3_ATTENUATION_LIMIT_DB
                    );
                }
                if !self.dfn3_post_filter_beta.is_finite()
                    || !(0.0..=vc_core::denoise_config::MAX_DFN3_POST_FILTER_BETA)
                        .contains(&self.dfn3_post_filter_beta)
                {
                    bail!(
                        "DeepFilterNet3 post-filter beta must be in 0..={}",
                        vc_core::denoise_config::MAX_DFN3_POST_FILTER_BETA
                    );
                }
            }
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
        progress: Option<&'a dyn Fn(LoadProgress)>,
    ) -> RvcPipelineConfig<'a> {
        self.pipeline_config_with(
            self.model.as_ref().expect("validated"),
            sample_rate,
            chunk_samples,
            live,
            progress,
        )
    }

    /// Same as `pipeline_config`, but for an arbitrary model path — used by the
    /// background model-pool loader.
    fn pipeline_config_with<'a>(
        &'a self,
        model: &'a Path,
        sample_rate: u32,
        chunk_samples: usize,
        live: &LiveParams,
        progress: Option<&'a dyn Fn(LoadProgress)>,
    ) -> RvcPipelineConfig<'a> {
        let output_extra_ms = self
            .crossfade_ms
            .saturating_add(self.sola_search_ms)
            .saturating_add(self.rvc_output_tail_discard_ms);
        RvcPipelineConfig {
            model,
            embedder: self.embedder.as_ref().expect("validated"),
            embedder_output: self.embedder_output.as_deref(),
            f0_model: self.f0_model.as_deref(),
            f0_mode: self.f0_mode,
            fcpe_model: self.fcpe_model.as_deref(),
            provider: self.provider,
            gpu_priority: self.gpu_priority,
            gpu_device_id: self.gpu_device_id,
            sample_rate,
            chunk_samples,
            speaker_id: live.speaker_id,
            pitch_shift: live.pitch_shift,
            f0: self.f0.clone(),
            retrieval: FeatureRetrievalConfig {
                index_path: self.feature_index.as_deref(),
                index_rate: live.index_rate,
                protect: live.protect,
                protect_transition_ms: live.protect_transition_ms,
            },
            input_gain: live.input_gain,
            noise_gate_enabled: self.denoiser_mode == DenoiserMode::NoiseGate,
            silence_gate_enabled: live.silence_gate_enabled,
            noise_gate_threshold: live.noise_gate_threshold,
            denoiser_content_mix: live.denoiser_content_mix,
            denoiser_rmvpe_mix: live.denoiser_rmvpe_mix,
            noise_gate_shaping: self.noise_gate_shaping,
            output_extra_ms,
            volume_excluded_ms: self.crossfade_ms,
            extra_convert_ms: self.extra_convert_ms,
            rvc_frames: self.rvc_frames,
            output_gain: live.output_gain,
            output_dynamics: self.output_dynamics,
            progress,
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
    f0_threshold: AtomicU32,
    input_gain: AtomicU32,
    output_gain: AtomicU32,
    monitor_gain: AtomicU32,
    noise_gate_enabled: AtomicBool,
    silence_gate_enabled: AtomicBool,
    noise_gate_threshold: AtomicU32,
    index_rate: AtomicU32,
    protect: AtomicU32,
    protect_transition_ms: AtomicU32,
    denoiser_content_mix: AtomicU32,
    denoiser_rmvpe_mix: AtomicU32,
}

impl AtomicLiveParams {
    fn new(value: LiveParams) -> Self {
        let this = Self::default();
        this.store(value);
        this
    }

    fn store(&self, value: LiveParams) {
        // Keep malformed host/GUI automation out of the atomics. The worker
        // path is intentionally lock-free, so normalization happens before
        // publishing and does not require a recovery branch in the callback.
        let value = value.sanitized();
        self.pitch_shift
            .store(value.pitch_shift.to_bits(), Ordering::Relaxed);
        self.speaker_id.store(value.speaker_id, Ordering::Relaxed);
        self.f0_threshold
            .store(value.f0_threshold.to_bits(), Ordering::Relaxed);
        self.input_gain
            .store(value.input_gain.to_bits(), Ordering::Relaxed);
        self.output_gain
            .store(value.output_gain.to_bits(), Ordering::Relaxed);
        self.monitor_gain
            .store(value.monitor_gain.to_bits(), Ordering::Relaxed);
        self.noise_gate_enabled
            .store(value.noise_gate_enabled, Ordering::Relaxed);
        self.silence_gate_enabled
            .store(value.silence_gate_enabled, Ordering::Relaxed);
        self.noise_gate_threshold
            .store(value.noise_gate_threshold.to_bits(), Ordering::Relaxed);
        self.index_rate
            .store(value.index_rate.to_bits(), Ordering::Relaxed);
        self.protect
            .store(value.protect.to_bits(), Ordering::Relaxed);
        self.protect_transition_ms
            .store(value.protect_transition_ms, Ordering::Relaxed);
        self.denoiser_content_mix
            .store(value.denoiser_content_mix.to_bits(), Ordering::Relaxed);
        self.denoiser_rmvpe_mix
            .store(value.denoiser_rmvpe_mix.to_bits(), Ordering::Relaxed);
    }

    fn load(&self) -> LiveParams {
        LiveParams {
            pitch_shift: f32::from_bits(self.pitch_shift.load(Ordering::Relaxed)),
            speaker_id: self.speaker_id.load(Ordering::Relaxed),
            f0_threshold: f32::from_bits(self.f0_threshold.load(Ordering::Relaxed)),
            input_gain: f32::from_bits(self.input_gain.load(Ordering::Relaxed)),
            output_gain: f32::from_bits(self.output_gain.load(Ordering::Relaxed)),
            monitor_gain: f32::from_bits(self.monitor_gain.load(Ordering::Relaxed)),
            noise_gate_enabled: self.noise_gate_enabled.load(Ordering::Relaxed),
            silence_gate_enabled: self.silence_gate_enabled.load(Ordering::Relaxed),
            noise_gate_threshold: f32::from_bits(self.noise_gate_threshold.load(Ordering::Relaxed)),
            index_rate: f32::from_bits(self.index_rate.load(Ordering::Relaxed)),
            protect: f32::from_bits(self.protect.load(Ordering::Relaxed)),
            protect_transition_ms: self.protect_transition_ms.load(Ordering::Relaxed),
            denoiser_content_mix: f32::from_bits(self.denoiser_content_mix.load(Ordering::Relaxed)),
            denoiser_rmvpe_mix: f32::from_bits(self.denoiser_rmvpe_mix.load(Ordering::Relaxed)),
        }
        .sanitized()
    }

    fn set_noise_gate_enabled(&self, enabled: bool) {
        self.noise_gate_enabled.store(enabled, Ordering::Relaxed);
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
    pub detail: Option<String>,
    pub input_device: String,
    pub output_device: String,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
    // Monitor output device + sample rate; empty/0 when the monitor is disabled.
    pub monitor_device: String,
    pub monitor_sample_rate: u32,
    pub passthrough_live_switchable: bool,
    /// Speaker IDs accepted by the currently active RVC model. `None` means
    /// passthrough/no model or an exporter without a discoverable embedding
    /// initializer.
    pub speaker_count: Option<usize>,
    // Multi-model pool state: the active model's display name and the per-slot
    // load status (base model at slot 0, then any models added live).
    pub active_model: Option<String>,
    pub model_loads: Vec<ModelLoadStatus>,
}

/// Load state of one slot in the model pool, surfaced to the front-ends.
#[derive(Clone, Debug)]
pub enum ModelLoadState {
    Loading(String),
    Loaded,
    Error(String),
}

impl Default for ModelLoadState {
    fn default() -> Self {
        Self::Loading(String::new())
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelLoadStatus {
    pub path: String,
    pub state: ModelLoadState,
    /// Index into the worker's pool (dense, `Some` once loaded); the front-end
    /// switches to a model via this index.
    pub pool_index: Option<usize>,
    /// Monotonic load-request id (never reused), so a queued/loading model can
    /// be addressed even after earlier entries are removed from the list.
    pub request_id: u64,
}

fn model_load_request_is_live(status: &EngineStatusSnapshot, request_id: u64) -> bool {
    status.model_loads.iter().any(|entry| {
        entry.request_id == request_id && matches!(&entry.state, ModelLoadState::Loading(_))
    })
}

/// Allocate a dynamic-model request id while reserving zero permanently for
/// the base model status row. Exhaustion is practically unreachable and must
/// fail rather than wrap to zero or reuse an id that can still be referenced by
/// a background loader.
fn take_dynamic_model_request_id(next: &mut NonZeroU64) -> u64 {
    let request_id = next.get();
    *next = NonZeroU64::new(
        request_id
            .checked_add(1)
            .expect("dynamic model request ids exhausted"),
    )
    .expect("incrementing a nonzero request id cannot produce zero");
    request_id
}

/// Remove one non-base model status entry and remap dense pool indices after
/// its slot. The base row (`request_id == 0`, pool slot 0) is an invariant and
/// cannot be removed even by an internal/stale command.
fn remove_dynamic_model_status(
    status: &mut EngineStatusSnapshot,
    request_id: u64,
) -> Option<usize> {
    if request_id == BASE_MODEL_REQUEST_ID {
        return None;
    }
    let entry_index = status
        .model_loads
        .iter()
        .position(|entry| entry.request_id == request_id)?;
    let pool_slot = status.model_loads.remove(entry_index).pool_index;
    if let Some(pool_slot) = pool_slot {
        for entry in &mut status.model_loads {
            if entry.pool_index.is_some_and(|slot| slot > pool_slot) {
                entry.pool_index = entry.pool_index.map(|slot| slot - 1);
            }
        }
    }
    pool_slot
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
    pub embedder_us: u64,
    pub pitch_us: u64,
    pub rvc_us: u64,
    pub f0_voiced_ratio: f32,
    pub input_rms: f32,
    pub output_rms: f32,
    pub input_overruns: u64,
    pub output_underruns: u64,
    pub output_dropped_samples: u64,
    pub output_buffer_samples: u64,
    pub monitor_underruns: u64,
    pub monitor_dropped_samples: u64,
}

#[derive(Default)]
struct Telemetry {
    chunks: AtomicU64,
    inference_us: AtomicU64,
    embedder_us: AtomicU64,
    pitch_us: AtomicU64,
    rvc_us: AtomicU64,
    f0_voiced_ratio_bits: AtomicU32,
    input_rms_bits: AtomicU32,
    output_rms_bits: AtomicU32,
    input_overruns: AtomicU64,
    output_underruns: AtomicU64,
    output_dropped_samples: AtomicU64,
    output_buffer_samples: AtomicU64,
    monitor_underruns: AtomicU64,
    monitor_dropped_samples: AtomicU64,
}

impl Telemetry {
    fn reset(&self) {
        self.chunks.store(0, Ordering::Relaxed);
        self.inference_us.store(0, Ordering::Relaxed);
        self.embedder_us.store(0, Ordering::Relaxed);
        self.pitch_us.store(0, Ordering::Relaxed);
        self.rvc_us.store(0, Ordering::Relaxed);
        self.f0_voiced_ratio_bits.store(0, Ordering::Relaxed);
        self.input_rms_bits.store(0, Ordering::Relaxed);
        self.output_rms_bits.store(0, Ordering::Relaxed);
        self.input_overruns.store(0, Ordering::Relaxed);
        self.output_underruns.store(0, Ordering::Relaxed);
        self.output_dropped_samples.store(0, Ordering::Relaxed);
        self.output_buffer_samples.store(0, Ordering::Relaxed);
        self.monitor_underruns.store(0, Ordering::Relaxed);
        self.monitor_dropped_samples.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            chunks: self.chunks.load(Ordering::Relaxed),
            inference_us: self.inference_us.load(Ordering::Relaxed),
            embedder_us: self.embedder_us.load(Ordering::Relaxed),
            pitch_us: self.pitch_us.load(Ordering::Relaxed),
            rvc_us: self.rvc_us.load(Ordering::Relaxed),
            f0_voiced_ratio: f32::from_bits(self.f0_voiced_ratio_bits.load(Ordering::Relaxed)),
            input_rms: f32::from_bits(self.input_rms_bits.load(Ordering::Relaxed)),
            output_rms: f32::from_bits(self.output_rms_bits.load(Ordering::Relaxed)),
            input_overruns: self.input_overruns.load(Ordering::Relaxed),
            output_underruns: self.output_underruns.load(Ordering::Relaxed),
            output_dropped_samples: self.output_dropped_samples.load(Ordering::Relaxed),
            output_buffer_samples: self.output_buffer_samples.load(Ordering::Relaxed),
            monitor_underruns: self.monitor_underruns.load(Ordering::Relaxed),
            monitor_dropped_samples: self.monitor_dropped_samples.load(Ordering::Relaxed),
        }
    }
}

/// Lifecycle state for a short microphone calibration pass. The profile is
/// published only after the worker has consumed the requested number of input
/// samples, never from an audio callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VoiceCalibrationState {
    #[default]
    Idle,
    Requested,
    Collecting,
    Ready,
}

/// Frontend-readable state of the current or most recent calibration request.
/// The profile contains aggregate signal statistics only; raw microphone audio
/// remains worker-local and is dropped after each chunk.
#[derive(Clone, Copy, Debug, Default)]
pub struct VoiceCalibrationSnapshot {
    pub generation: u64,
    pub state: VoiceCalibrationState,
    pub captured_ms: u32,
    pub target_ms: u32,
    pub profile: Option<VoiceCalibrationProfile>,
}

/// Cross-thread control plane for one worker-owned calibration accumulator.
///
/// The GUI starts a generation through atomics; the inference worker notices it
/// between chunks, so the audio callbacks retain their lock-free, sample-moving
/// role. A mutex is used only once at completion to publish the tiny summary,
/// never while audio callbacks are executing.
#[derive(Default)]
struct VoiceCalibrationControl {
    next_generation: AtomicU64,
    requested_generation: AtomicU64,
    duration_ms: AtomicU32,
    state: AtomicU8,
    captured_ms: AtomicU32,
    profile: Mutex<Option<VoiceCalibrationProfile>>,
}

impl VoiceCalibrationControl {
    fn start(&self, duration_ms: u32) -> u64 {
        let duration_ms = duration_ms.clamp(
            vc_core::voice_calibration::MIN_VOICE_CALIBRATION_DURATION_MS,
            vc_core::voice_calibration::MAX_VOICE_CALIBRATION_DURATION_MS,
        );
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed) + 1;
        if let Ok(mut profile) = self.profile.lock() {
            *profile = None;
        }
        self.duration_ms.store(duration_ms, Ordering::Relaxed);
        self.captured_ms.store(0, Ordering::Relaxed);
        self.state.store(CALIBRATION_REQUESTED, Ordering::Relaxed);
        // Publish this last. The worker's Acquire load of the generation sees
        // the duration/result reset that belongs to this specific request.
        self.requested_generation
            .store(generation, Ordering::Release);
        generation
    }

    fn cancel(&self) {
        self.requested_generation.store(0, Ordering::Release);
        self.state.store(CALIBRATION_IDLE, Ordering::Relaxed);
        self.captured_ms.store(0, Ordering::Relaxed);
        if let Ok(mut profile) = self.profile.lock() {
            *profile = None;
        }
    }

    fn request(&self) -> Option<(u64, u32)> {
        let generation = self.requested_generation.load(Ordering::Acquire);
        (generation != 0).then(|| (generation, self.duration_ms.load(Ordering::Relaxed)))
    }

    fn mark_collecting(&self, generation: u64, captured_ms: u32) {
        if self.requested_generation.load(Ordering::Acquire) == generation {
            self.captured_ms.store(captured_ms, Ordering::Relaxed);
            self.state.store(CALIBRATION_COLLECTING, Ordering::Relaxed);
        }
    }

    fn finish(&self, generation: u64, profile: VoiceCalibrationProfile) {
        if self.requested_generation.load(Ordering::Acquire) != generation {
            return;
        }
        if let Ok(mut result) = self.profile.lock() {
            // Re-check while holding the result lock so a new request cannot be
            // overwritten by an old worker that was just finishing its chunk.
            if self.requested_generation.load(Ordering::Acquire) != generation {
                return;
            }
            *result = Some(profile);
            self.captured_ms
                .store(profile.captured_ms, Ordering::Relaxed);
            self.state.store(CALIBRATION_READY, Ordering::Release);
        }
    }

    fn snapshot(&self) -> VoiceCalibrationSnapshot {
        let generation = self.requested_generation.load(Ordering::Acquire);
        if generation == 0 {
            return VoiceCalibrationSnapshot::default();
        }
        let state = match self.state.load(Ordering::Acquire) {
            CALIBRATION_REQUESTED => VoiceCalibrationState::Requested,
            CALIBRATION_COLLECTING => VoiceCalibrationState::Collecting,
            CALIBRATION_READY => VoiceCalibrationState::Ready,
            _ => VoiceCalibrationState::Idle,
        };
        let profile = if state == VoiceCalibrationState::Ready {
            self.profile.lock().ok().and_then(|profile| *profile)
        } else {
            None
        };
        VoiceCalibrationSnapshot {
            generation,
            state,
            captured_ms: self.captured_ms.load(Ordering::Relaxed),
            target_ms: self.duration_ms.load(Ordering::Relaxed),
            profile,
        }
    }
}

/// Cross-thread control plane for the optional language-aware live overlay.
///
/// The mode is atomically sampled by the inference worker at chunk boundaries.
/// Its diagnostic snapshot is deliberately published with `try_lock()` only
/// from that worker: a slow GUI frame can lose one refresh, but must never add
/// blocking work to the audio callback or the conversion deadline.
#[derive(Default)]
struct DynamicTuningControl {
    mode: AtomicU8,
    latest: Mutex<DynamicTuningSnapshot>,
}

impl DynamicTuningControl {
    fn set_mode(&self, mode: DynamicTuningMode) {
        self.mode.store(mode.as_u8(), Ordering::Release);
        if let Ok(mut latest) = self.latest.lock() {
            *latest = DynamicTuningSnapshot {
                mode,
                profile: fixed_dynamic_profile(mode),
                confidence: if matches!(
                    mode,
                    DynamicTuningMode::Chinese
                        | DynamicTuningMode::English
                        | DynamicTuningMode::Japanese
                ) {
                    1.0
                } else {
                    0.0
                },
                ..DynamicTuningSnapshot::default()
            };
        }
    }

    fn mode(&self) -> DynamicTuningMode {
        DynamicTuningMode::from_u8(self.mode.load(Ordering::Acquire))
    }

    fn reset_snapshot(&self) {
        self.set_mode(self.mode());
    }

    fn publish(&self, snapshot: DynamicTuningSnapshot) {
        if let Ok(mut latest) = self.latest.try_lock() {
            *latest = snapshot;
        }
    }

    fn snapshot(&self) -> DynamicTuningSnapshot {
        self.latest
            .lock()
            .map(|snapshot| *snapshot)
            .unwrap_or_default()
    }
}

fn fixed_dynamic_profile(mode: DynamicTuningMode) -> DynamicLanguageProfile {
    match mode {
        DynamicTuningMode::Chinese => DynamicLanguageProfile::Chinese,
        DynamicTuningMode::English => DynamicLanguageProfile::English,
        DynamicTuningMode::Japanese => DynamicLanguageProfile::Japanese,
        DynamicTuningMode::Off | DynamicTuningMode::Auto => DynamicLanguageProfile::Neutral,
    }
}

// Boxing the large `Apply` payload is intentionally declined: these commands
// flow at control-message cadence (model/config changes), not per audio block,
// so the size disparity costs nothing worth an extra heap allocation + indirection
// on every push. Kept inline so the worker's command path stays allocation-free.
#[allow(clippy::large_enum_variant)]
enum Command {
    Apply(RealtimeConfig),
    // Live device reconfiguration: same-sample-rate swaps rebind the worker's
    // rings without a session restart; a sample-rate change falls back to a
    // full Apply-style restart (see `RealtimeSession::update_devices`).
    UpdateDevices(DeviceSpec),
    // Add an RVC model to the running session's pool. The load runs on a
    // background thread; success/failure returns as AddModelReady/AddModelFailed.
    // `request_id` is the monotonic id of the `model_loads` entry (see
    // `ModelLoadStatus`), stable across removals of earlier entries.
    AddModel(PathBuf),
    AddModelReady {
        request_id: u64,
        name: String,
        converter: ChunkConverter<RvcPipeline>,
        built_input_rate: u32,
        built_output_rate: u32,
        built_denoiser_generation: u64,
    },
    AddModelFailed {
        request_id: u64,
        error: String,
    },
    // Remove a model from the running pool by its `model_loads` request id
    // (used by the front-end's per-model delete button).
    RemoveModel {
        request_id: u64,
    },
    // Live denoiser hot-swap (off/gate/rnnoise/WebRTC on the worker; model
    // denoisers are pre-built off-thread before being swapped in).
    SetDenoiser(DenoiserMode),
    Stop,
    // (input_host, output_host): inputs are enumerated from the input host and
    // outputs from the output host.
    RefreshDevices(AudioHost, AudioHost),
    Shutdown,
}

/// Device endpoint selection, independent of the model/config side of
/// `RealtimeConfig`. Sent through `EngineController::set_devices` so the GUI can
/// swap devices live while a session is running.
#[derive(Clone, Debug, Default)]
pub struct DeviceSpec {
    pub input_host: AudioHost,
    pub output_host: AudioHost,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub monitor_output_enabled: bool,
    pub monitor_output_device: Option<String>,
    pub wasapi_input_exclusive: bool,
    pub wasapi_output_exclusive: bool,
    pub wasapi_buffer_ms: u32,
}

/// Commands sent from the control thread to the inference worker, drained at the
/// top of the worker loop. The worker is a normal thread — not an audio callback —
/// so an mpsc mailbox here is realtime-safe: the cpal stream callbacks never touch
/// it. Payloads are `Send` (the model/rings already move into the worker today).
#[allow(clippy::large_enum_variant)]
enum WorkerCommand {
    RebindRings {
        input_consumer: rtrb::Consumer<f32>,
        output_producer: rtrb::Producer<f32>,
        monitor_producer: Option<rtrb::Producer<f32>>,
    },
    // A model finished loading in the background; hand it to the worker to grow
    // the pool. `request_id` matches the `model_loads` entry the control loop
    // reserved (stable across removals of earlier entries).
    AddModel {
        converter: ChunkConverter<RvcPipeline>,
        name: String,
        request_id: u64,
        activate: bool,
        built_denoiser_generation: u64,
    },
    // Remove a pool slot by dense index. Base protection is enforced earlier by
    // stable request_id=0; a no-base session's first dynamic model can be slot 0.
    RemoveModel {
        slot: usize,
    },
    // Hot-swap the denoiser variant (off/gate/rnnoise/WebRTC built on the
    // worker). WebRTC level is static state, so it travels with the command
    // rather than reading mutable frontend config from the worker.
    SetDenoiser {
        mode: DenoiserMode,
        webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel,
        // Monotonic generation used to reject a stale async model swap.
        generation: u64,
    },
    // Pre-built GTCRN denoisers (one per model + passthrough), swapped in after
    // an off-thread engine load.
    #[cfg(feature = "gtcrn")]
    SwapGtcrn {
        model_denoisers: Vec<vc_core::denoise::GtcrnDenoiser>,
        passthrough_denoiser: Option<vc_core::denoise::GtcrnDenoiser>,
        generation: u64,
    },
    // Official DFN3 graph/archive construction is expensive and must never run
    // on the worker's audio deadline. Instances arrive from a loader thread and
    // transfer exclusive ownership to the worker.
    #[cfg(feature = "deepfilternet3")]
    SwapDeepFilterNet3 {
        model_denoisers: Vec<vc_core::denoise::DeepFilterNet3Denoiser>,
        passthrough_denoiser: Option<vc_core::denoise::DeepFilterNet3Denoiser>,
        generation: u64,
    },
}

/// Result of a live device reconfiguration: the worker rings were rebound in
/// place (`Swapped`), or the new endpoints change a sample rate and the session
/// must restart with the merged config (`RestartRequired`).
#[allow(clippy::large_enum_variant)]
enum UpdateDevicesOutcome {
    Swapped,
    RestartRequired(RealtimeConfig),
}

pub struct EngineController {
    tx: SyncSender<Command>,
    status: Arc<Mutex<EngineStatusSnapshot>>,
    devices: Arc<Mutex<DeviceList>>,
    telemetry: Arc<Telemetry>,
    voice_calibration: Arc<VoiceCalibrationControl>,
    dynamic_tuning: Arc<DynamicTuningControl>,
    live: Arc<AtomicLiveParams>,
    passthrough: Arc<AtomicBool>,
    active_model: Arc<AtomicUsize>,
    control: Option<JoinHandle<()>>,
}

impl EngineController {
    pub fn new(initial_live: LiveParams) -> Self {
        let (tx, rx) = mpsc::sync_channel(COMMAND_CAPACITY);
        let status = Arc::new(Mutex::new(EngineStatusSnapshot::default()));
        let devices = Arc::new(Mutex::new(DeviceList::default()));
        let telemetry = Arc::new(Telemetry::default());
        let voice_calibration = Arc::new(VoiceCalibrationControl::default());
        let dynamic_tuning = Arc::new(DynamicTuningControl::default());
        let live = Arc::new(AtomicLiveParams::new(initial_live));
        let passthrough = Arc::new(AtomicBool::new(false));
        let active_model = Arc::new(AtomicUsize::new(0));
        let control = {
            let tx = tx.clone();
            let status = Arc::clone(&status);
            let devices = Arc::clone(&devices);
            let telemetry = Arc::clone(&telemetry);
            let voice_calibration = Arc::clone(&voice_calibration);
            let dynamic_tuning = Arc::clone(&dynamic_tuning);
            let live = Arc::clone(&live);
            let passthrough = Arc::clone(&passthrough);
            let active_model = Arc::clone(&active_model);
            thread::Builder::new()
                .name("vc-app-control".to_string())
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    control_loop(
                        rx,
                        tx,
                        status,
                        devices,
                        telemetry,
                        voice_calibration,
                        dynamic_tuning,
                        live,
                        passthrough,
                        active_model,
                    )
                })
                .expect("failed to spawn vc-app control thread")
        };
        Self {
            tx,
            status,
            devices,
            telemetry,
            voice_calibration,
            dynamic_tuning,
            live,
            passthrough,
            active_model,
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

    pub fn refresh_devices(&self, input_host: AudioHost, output_host: AudioHost) -> Result<()> {
        self.try_command(Command::RefreshDevices(input_host, output_host))
    }

    /// Live device reconfiguration while a session is running. Same-sample-rate
    /// swaps rebind the worker rings without a restart; a sample-rate change
    /// restarts the session (see `RealtimeSession::update_devices`).
    pub fn set_devices(&self, spec: DeviceSpec) -> Result<()> {
        self.try_command(Command::UpdateDevices(spec))
    }

    /// Load an additional RVC model into the running session's pool (background).
    /// The new model becomes selectable via `set_active_model` once loaded.
    pub fn add_model(&self, path: PathBuf) -> Result<()> {
        self.try_command(Command::AddModel(path))
    }

    /// Switch the active pool model live. The worker picks the slot up on the
    /// next chunk (atomic write, no command round-trip).
    pub fn set_active_model(&self, slot: usize) {
        self.active_model.store(slot, Ordering::Relaxed);
    }

    /// Remove a model from the running session's pool by its `model_loads`
    /// `request_id` (used by the front-end's per-model delete button). Unloads
    /// the worker slot and drops the status entry; if it was active, the active
    /// slot falls back to the nearest remaining model.
    pub fn remove_model(&self, request_id: u64) -> Result<()> {
        if request_id == BASE_MODEL_REQUEST_ID {
            bail!("the base model cannot be removed");
        }
        self.try_command(Command::RemoveModel { request_id })
    }

    /// Hot-swap the denoiser live (off / noise-gate / rnnoise apply on the next
    /// worker iteration; gtcrn loads its engine in the background first). Also
    /// keeps the live gate flag coherent so the per-chunk live path does not
    /// fight the requested mode.
    pub fn set_denoiser(&self, mode: DenoiserMode) -> Result<()> {
        self.try_command(Command::SetDenoiser(mode))
    }

    pub fn set_live_params(&self, params: LiveParams) {
        self.live.store(params);
    }

    /// Choose the optional worker-side language-aware tuning overlay. The
    /// latest manual [`LiveParams`] remain the baseline and are overlaid only
    /// at inference-chunk boundaries.
    pub fn set_dynamic_tuning_mode(&self, mode: DynamicTuningMode) {
        self.dynamic_tuning.set_mode(mode);
    }

    /// Latest language-profile heuristic diagnostics. This contains no audio
    /// samples and can be queried freely by GUI/CLI frontends.
    pub fn dynamic_tuning_snapshot(&self) -> DynamicTuningSnapshot {
        self.dynamic_tuning.snapshot()
    }

    pub fn set_passthrough(&self, enabled: bool) {
        self.passthrough.store(enabled, Ordering::Relaxed);
    }

    /// Begin a bounded microphone calibration using the currently running
    /// session. The worker performs the analysis before model input gain or
    /// denoising; the callback remains untouched.
    pub fn start_voice_calibration(&self) -> Result<()> {
        let state = self
            .status
            .lock()
            .map(|status| status.state)
            .unwrap_or(EngineState::Error);
        if state != EngineState::Running {
            bail!("start the realtime engine before calibrating the microphone");
        }
        self.voice_calibration
            .start(DEFAULT_VOICE_CALIBRATION_DURATION_MS);
        Ok(())
    }

    /// Query progress/result for [`Self::start_voice_calibration`].
    pub fn voice_calibration_snapshot(&self) -> VoiceCalibrationSnapshot {
        self.voice_calibration.snapshot()
    }

    /// Discard a pending or published calibration result without touching the
    /// running audio session.
    pub fn cancel_voice_calibration(&self) {
        self.voice_calibration.cancel();
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

#[allow(clippy::too_many_arguments)]
fn control_loop(
    rx: Receiver<Command>,
    tx: SyncSender<Command>,
    status: Arc<Mutex<EngineStatusSnapshot>>,
    devices: Arc<Mutex<DeviceList>>,
    telemetry: Arc<Telemetry>,
    voice_calibration: Arc<VoiceCalibrationControl>,
    dynamic_tuning: Arc<DynamicTuningControl>,
    live: Arc<AtomicLiveParams>,
    passthrough: Arc<AtomicBool>,
    active_model: Arc<AtomicUsize>,
) {
    let mut session: Option<RealtimeSession> = None;
    // Every asynchronous denoiser load captures the generation at the time it
    // starts.  The control thread advances this value whenever the session or
    // selected denoiser changes, so a slow GTCRN/DFN3 loader cannot apply its
    // result after a newer request (or after a restart).  Keep this check off
    // the audio worker; it is only used by background/control threads.
    let denoiser_generation = Arc::new(AtomicU64::new(0));
    // Background model loads must use the mode the worker actually applied,
    // not merely the latest request (which may still be loading or may fail).
    let applied_denoiser = Arc::new(AppliedDenoiserState::new(AppliedDenoiserSnapshot {
        generation: 0,
        mode: DenoiserMode::Off,
    }));
    // Monotonic id for each model-load request, so a `model_loads` entry can be
    // addressed after earlier entries have been removed (per-model delete).
    let mut next_request_id = FIRST_DYNAMIC_MODEL_REQUEST_ID;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Command::Apply(config)) => {
                denoiser_generation.fetch_add(1, Ordering::AcqRel);
                restart_session(
                    &mut session,
                    config,
                    &status,
                    &telemetry,
                    &voice_calibration,
                    &dynamic_tuning,
                    &live,
                    &passthrough,
                    &active_model,
                    &denoiser_generation,
                    &applied_denoiser,
                    "Stopping previous session",
                );
            }
            Ok(Command::UpdateDevices(spec)) => {
                let outcome = session.as_mut().map(|session| session.update_devices(spec));
                match outcome {
                    Some(Ok(UpdateDevicesOutcome::Swapped)) => {
                        if let Some(session) = &session {
                            if let Ok(mut current) = status.lock() {
                                patch_device_status(&mut current, &session.status);
                            }
                        }
                    }
                    Some(Ok(UpdateDevicesOutcome::RestartRequired(config))) => {
                        denoiser_generation.fetch_add(1, Ordering::AcqRel);
                        restart_session(
                            &mut session,
                            config,
                            &status,
                            &telemetry,
                            &voice_calibration,
                            &dynamic_tuning,
                            &live,
                            &passthrough,
                            &active_model,
                            &denoiser_generation,
                            &applied_denoiser,
                            "Device sample rate changed; restarting",
                        );
                    }
                    Some(Err(err)) => set_recoverable_error(
                        &status,
                        "Device change failed; previous devices remain active",
                        &err,
                    ),
                    None => {}
                }
            }
            Ok(Command::AddModel(path)) => {
                handle_add_model(
                    session.as_ref(),
                    &tx,
                    &status,
                    &live,
                    applied_denoiser.load(),
                    path,
                    &mut next_request_id,
                );
            }
            Ok(Command::AddModelReady {
                request_id,
                name,
                converter,
                built_input_rate,
                built_output_rate,
                built_denoiser_generation,
            }) => {
                let Some(s) = session.as_ref() else {
                    update_model_load_status(
                        &status,
                        request_id,
                        ModelLoadState::Error("session stopped while loading".to_string()),
                    );
                    continue;
                };
                // The entry may have been deleted while it was loading — drop the
                // finished model instead of adding a ghost slot.
                let entry_present = status
                    .lock()
                    .map(|st| st.model_loads.iter().any(|m| m.request_id == request_id))
                    .unwrap_or(false);
                if !entry_present {
                    continue;
                }
                if built_input_rate != s.input_rate || built_output_rate != s.output_rate {
                    update_model_load_status(
                        &status,
                        request_id,
                        ModelLoadState::Error(
                            "device sample rates changed; model discarded".to_string(),
                        ),
                    );
                    continue;
                }
                if applied_denoiser.load().generation != built_denoiser_generation {
                    update_model_load_status(
                        &status,
                        request_id,
                        ModelLoadState::Error(
                            "denoiser changed while model was loading; retry the model load"
                                .to_string(),
                        ),
                    );
                    continue;
                }
                // Activate when this is the first loaded model (a passthrough-only
                // session grew its first RVC model).
                let activate = {
                    let st = status.lock().unwrap_or_else(|e| e.into_inner());
                    !st.model_loads
                        .iter()
                        .any(|m| matches!(m.state, ModelLoadState::Loaded))
                };
                if s.worker_tx
                    .send(WorkerCommand::AddModel {
                        converter,
                        name,
                        request_id,
                        activate,
                        built_denoiser_generation,
                    })
                    .is_err()
                {
                    update_model_load_status(
                        &status,
                        request_id,
                        ModelLoadState::Error("worker stopped; model discarded".to_string()),
                    );
                } else {
                    s.wake.wake();
                }
            }
            Ok(Command::AddModelFailed { request_id, error }) => {
                update_model_load_status(&status, request_id, ModelLoadState::Error(error));
            }
            Ok(Command::RemoveModel { request_id }) => {
                // Zero is permanently reserved for the base model. The public
                // controller rejects it, and this second check protects the
                // status/worker invariants from a stale or future internal
                // command that bypasses that API.
                if request_id == BASE_MODEL_REQUEST_ID {
                    continue;
                }
                let Some(s) = session.as_ref() else { continue };
                // Pull the entry out of the status list first, fixing up the pool
                // indices of entries after the removed slot.
                let pool_slot = {
                    let mut st = status.lock().unwrap_or_else(|e| e.into_inner());
                    remove_dynamic_model_status(&mut st, request_id)
                };
                // A loaded model must be dropped from the worker's pool too.
                if let Some(pool_slot) = pool_slot {
                    if s.worker_tx
                        .send(WorkerCommand::RemoveModel { slot: pool_slot })
                        .is_err()
                    {
                        set_error(&status, &anyhow!("worker command channel closed"));
                    } else {
                        s.wake.wake();
                    }
                }
            }
            Ok(Command::SetDenoiser(mode)) => {
                let generation = denoiser_generation.fetch_add(1, Ordering::AcqRel) + 1;
                handle_set_denoiser(
                    session.as_ref(),
                    &status,
                    mode,
                    generation,
                    &denoiser_generation,
                );
            }
            Ok(Command::Stop) => {
                denoiser_generation.fetch_add(1, Ordering::AcqRel);
                voice_calibration.cancel();
                dynamic_tuning.reset_snapshot();
                set_status(&status, EngineState::Stopping, "Stopping");
                drop(session.take());
                set_status(&status, EngineState::Stopped, "Stopped");
            }
            Ok(Command::RefreshDevices(input_host, output_host)) => {
                let result = device_list(input_host, output_host);
                if let Ok(mut current) = devices.lock() {
                    *current = result;
                }
            }
            Ok(Command::Shutdown) => {
                denoiser_generation.fetch_add(1, Ordering::AcqRel);
                break;
            }
            Err(RecvTimeoutError::Disconnected) => {
                denoiser_generation.fetch_add(1, Ordering::AcqRel);
                break;
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
        let stopped_message = session.as_ref().and_then(|session| {
            stopped_session_message_and_invalidate(
                &session.running,
                &session.endpoint_running,
                &denoiser_generation,
            )
        });
        if let Some(message) = stopped_message {
            voice_calibration.cancel();
            dynamic_tuning.reset_snapshot();
            drop(session.take());
            set_status(&status, EngineState::Error, message);
        }
    }
    voice_calibration.cancel();
    dynamic_tuning.reset_snapshot();
    drop(session);
}

/// Reserve a slot and spawn the background loader for a newly requested model.
/// The loader builds the pipeline + converter and reports back via
/// `Command::AddModelReady` / `AddModelFailed` through the control channel.
#[allow(clippy::too_many_arguments)]
fn handle_add_model(
    session: Option<&RealtimeSession>,
    tx: &SyncSender<Command>,
    status: &Arc<Mutex<EngineStatusSnapshot>>,
    live: &Arc<AtomicLiveParams>,
    applied_denoiser: AppliedDenoiserSnapshot,
    path: PathBuf,
    next_request_id: &mut NonZeroU64,
) {
    let Some(session) = session else {
        set_note(status, "Cannot add a model while stopped");
        return;
    };
    let path_string = path.to_string_lossy().into_owned();
    // Reject only when the NEW path duplicates an existing pool slot or the base
    // model — never compare existing entries against the base model (the base
    // model is always present as slot 0, which would make every add look like a
    // duplicate).
    let already_present = {
        let st = status.lock().unwrap_or_else(|e| e.into_inner());
        st.model_loads.iter().any(|m| m.path == path_string)
    } || session
        .config
        .model
        .as_deref()
        .is_some_and(|p| p.to_string_lossy() == path_string.as_str());
    if already_present {
        // Benign rejection — never tear down the running session over a
        // duplicate add. The GUI pre-checks and surfaces this as a UI note.
        set_note(status, "Model is already loaded or queued");
        return;
    }
    let ctx = session.model_load_context(path.clone(), &live.load(), applied_denoiser.mode);
    let request_id = take_dynamic_model_request_id(next_request_id);
    {
        let mut st = status.lock().unwrap_or_else(|e| e.into_inner());
        st.model_loads.push(ModelLoadStatus {
            path: path_string,
            state: ModelLoadState::Loading("queued".to_string()),
            pool_index: None,
            request_id,
        });
    }
    let tx = tx.clone();
    let loader_status = Arc::clone(status);
    let spawn = thread::Builder::new()
        .name("vc-app-model-loader".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let name = model_name_for(&path);
            let built_input_rate = ctx.sample_rate;
            let built_output_rate = ctx.output_rate;
            let result = load_model_converter(&ctx, |progress| {
                if let Ok(mut st) = loader_status.lock() {
                    if let Some(entry) = st
                        .model_loads
                        .iter_mut()
                        .find(|m| m.request_id == request_id)
                    {
                        entry.state = ModelLoadState::Loading(load_progress_message(progress));
                    }
                }
            });
            match result {
                Ok(converter) => {
                    let _ = tx.send(Command::AddModelReady {
                        request_id,
                        name,
                        converter,
                        built_input_rate,
                        built_output_rate,
                        built_denoiser_generation: applied_denoiser.generation,
                    });
                }
                Err(err) => {
                    let _ = tx.send(Command::AddModelFailed {
                        request_id,
                        error: format!("{err:#}"),
                    });
                }
            }
        });
    if let Err(err) = spawn {
        update_model_load_status(
            status,
            request_id,
            ModelLoadState::Error(format!("failed to spawn model loader: {err}")),
        );
    }
}

/// Live denoiser switch: off / noise-gate / RNNoise / WebRTC are sent straight
/// to the worker (their construction is bounded and model-free); GTCRN and DFN3
/// load on a background thread and hot-swap in when ready.
fn handle_set_denoiser(
    session: Option<&RealtimeSession>,
    status: &Arc<Mutex<EngineStatusSnapshot>>,
    mode: DenoiserMode,
    generation: u64,
    _current_generation: &Arc<AtomicU64>,
) {
    let Some(s) = session else {
        return;
    };
    match mode {
        DenoiserMode::Off
        | DenoiserMode::NoiseGate
        | DenoiserMode::Rnnoise
        | DenoiserMode::WebRtc => {
            if s.worker_tx
                .send(WorkerCommand::SetDenoiser {
                    mode,
                    webrtc_suppression_level: s.config.webrtc_suppression_level,
                    generation,
                })
                .is_err()
            {
                set_error(status, &anyhow!("worker command channel closed"));
            } else {
                s.wake.wake();
            }
        }
        DenoiserMode::Gtcrn => {
            #[cfg(feature = "gtcrn")]
            {
                let model_count = {
                    let st = status.lock().unwrap_or_else(|e| e.into_inner());
                    st.model_loads
                        .iter()
                        .filter(|m| matches!(m.state, ModelLoadState::Loaded))
                        .count()
                };
                let Some(model_dir) = s.config.gtcrn_model_dir.clone() else {
                    set_note(
                        status,
                        "GTCRN requires a model directory; set it and Apply first",
                    );
                    return;
                };
                let backend = vc_core::denoise::GtcrnBackend::for_provider(
                    s.config.provider,
                    s.config.gpu_priority,
                    s.config.gpu_device_id,
                );
                let input_rate = s.input_rate;
                let worker_tx = s.worker_tx.clone();
                let wake = Arc::clone(&s.wake);
                let running_message = s.status().message.clone();
                let loader_status = Arc::clone(status);
                let loader_generation = Arc::clone(_current_generation);
                let spawn = thread::Builder::new()
                    .name("vc-app-gtcrn-loader".to_string())
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let result =
                            load_gtcrn_denoisers(&model_dir, backend, input_rate, model_count);
                        // A newer mode/restart supersedes this load.  Do not
                        // publish status or enqueue a swap for stale results;
                        // dropping the freshly built state is safe because it
                        // never entered the real-time worker.
                        if loader_generation.load(Ordering::Acquire) != generation {
                            return;
                        }
                        if let Ok(mut st) = loader_status.lock() {
                            st.message = running_message.clone();
                        }
                        match result {
                            Ok((model_denoisers, passthrough_denoiser)) => {
                                if loader_generation.load(Ordering::Acquire) != generation {
                                    return;
                                }
                                if worker_tx
                                    .send(WorkerCommand::SwapGtcrn {
                                        model_denoisers,
                                        passthrough_denoiser,
                                        generation,
                                    })
                                    .is_err()
                                {
                                    if let Ok(mut st) = loader_status.lock() {
                                        st.detail =
                                            Some("worker stopped; GTCRN swap failed".to_string());
                                    }
                                } else {
                                    wake.wake();
                                }
                            }
                            Err(err) => {
                                if loader_generation.load(Ordering::Acquire) != generation {
                                    return;
                                }
                                if let Ok(mut st) = loader_status.lock() {
                                    st.detail = Some(format!("GTCRN load failed: {err:#}"));
                                }
                            }
                        }
                    });
                match spawn {
                    Ok(_) => {
                        set_status(status, EngineState::Running, "Loading GTCRN denoiser…");
                    }
                    Err(err) => set_error(status, &anyhow!("failed to spawn GTCRN loader: {err}")),
                }
            }
            #[cfg(not(feature = "gtcrn"))]
            {
                set_note(status, "GTCRN support is not enabled in this build");
            }
        }
        DenoiserMode::DeepFilterNet3 => {
            #[cfg(feature = "deepfilternet3")]
            {
                let model_count = {
                    let st = status.lock().unwrap_or_else(|e| e.into_inner());
                    st.model_loads
                        .iter()
                        .filter(|m| matches!(m.state, ModelLoadState::Loaded))
                        .count()
                };
                let Some(model_path) = s.config.deepfilternet3_model.clone() else {
                    set_note(
                        status,
                        "DeepFilterNet3 requires a model archive; set it and Apply first",
                    );
                    return;
                };
                let attenuation_limit_db = s.config.dfn3_attenuation_limit_db;
                let post_filter_beta = s.config.dfn3_post_filter_beta;
                let input_rate = s.input_rate;
                let worker_tx = s.worker_tx.clone();
                let wake = Arc::clone(&s.wake);
                let running_message = s.status().message.clone();
                let loader_status = Arc::clone(status);
                let loader_generation = Arc::clone(_current_generation);
                let spawn = thread::Builder::new()
                    .name("vc-app-dfn3-loader".to_string())
                    .stack_size(64 * 1024 * 1024)
                    .spawn(move || {
                        let result = load_deepfilternet3_denoisers(
                            &model_path,
                            attenuation_limit_db,
                            post_filter_beta,
                            input_rate,
                            model_count,
                        );
                        if loader_generation.load(Ordering::Acquire) != generation {
                            return;
                        }
                        if let Ok(mut st) = loader_status.lock() {
                            st.message = running_message.clone();
                        }
                        match result {
                            Ok((model_denoisers, passthrough_denoiser)) => {
                                if loader_generation.load(Ordering::Acquire) != generation {
                                    return;
                                }
                                if worker_tx
                                    .send(WorkerCommand::SwapDeepFilterNet3 {
                                        model_denoisers,
                                        passthrough_denoiser,
                                        generation,
                                    })
                                    .is_err()
                                {
                                    if let Ok(mut st) = loader_status.lock() {
                                        st.detail = Some(
                                            "worker stopped; DeepFilterNet3 swap failed"
                                                .to_string(),
                                        );
                                    }
                                } else {
                                    wake.wake();
                                }
                            }
                            Err(err) => {
                                if loader_generation.load(Ordering::Acquire) != generation {
                                    return;
                                }
                                if let Ok(mut st) = loader_status.lock() {
                                    st.detail =
                                        Some(format!("DeepFilterNet3 load failed: {err:#}"));
                                }
                            }
                        }
                    });
                match spawn {
                    Ok(_) => set_status(
                        status,
                        EngineState::Running,
                        "Loading DeepFilterNet3 denoiser...",
                    ),
                    Err(err) => set_error(
                        status,
                        &anyhow!("failed to spawn DeepFilterNet3 loader: {err}"),
                    ),
                }
            }
            #[cfg(not(feature = "deepfilternet3"))]
            set_note(
                status,
                "DeepFilterNet3 support is not enabled in this build",
            );
        }
    }
}

/// (model-side denoisers at the 16 kHz seam, device-rate passthrough instance).
#[cfg(feature = "gtcrn")]
type GtcrnDenoiserSet = (
    Vec<vc_core::denoise::GtcrnDenoiser>,
    Option<vc_core::denoise::GtcrnDenoiser>,
);

#[cfg(feature = "deepfilternet3")]
type DeepFilterNet3DenoiserSet = (
    Vec<vc_core::denoise::DeepFilterNet3Denoiser>,
    Option<vc_core::denoise::DeepFilterNet3Denoiser>,
);

/// Build a private DFN3 instance for every RVC model plus passthrough. Each
/// instance contains mutable recurrent state and is therefore never shared by
/// the model pool or the two processing routes.
#[cfg(feature = "deepfilternet3")]
fn load_deepfilternet3_denoisers(
    model_path: &Path,
    attenuation_limit_db: f32,
    post_filter_beta: f32,
    input_rate: u32,
    model_count: usize,
) -> Result<DeepFilterNet3DenoiserSet> {
    let config = vc_core::denoise::DeepFilterNet3Config {
        model_path,
        attenuation_limit_db,
        post_filter_beta,
    };
    let mut model_denoisers = Vec::with_capacity(model_count);
    for _ in 0..model_count {
        model_denoisers.push(vc_core::denoise::DeepFilterNet3Denoiser::new(
            config, input_rate,
        )?);
    }
    let passthrough_denoiser = Some(vc_core::denoise::DeepFilterNet3Denoiser::new(
        config, input_rate,
    )?);
    Ok((model_denoisers, passthrough_denoiser))
}

/// Build one GTCRN denoiser per loaded model (at the 16 kHz model seam) plus one
/// device-rate instance for the passthrough path. Runs off the worker thread —
/// engine load can take seconds.
#[cfg(feature = "gtcrn")]
fn load_gtcrn_denoisers(
    model_dir: &Path,
    backend: vc_core::denoise::GtcrnBackend,
    input_rate: u32,
    model_count: usize,
) -> Result<GtcrnDenoiserSet> {
    let mut model_denoisers = Vec::with_capacity(model_count);
    for _ in 0..model_count {
        model_denoisers.push(vc_core::denoise::GtcrnDenoiser::new(
            vc_core::denoise::GtcrnConfig { model_dir, backend },
            vc_core::model_rvc::EMBEDDER_SAMPLE_RATE,
        )?);
    }
    let passthrough_denoiser = Some(vc_core::denoise::GtcrnDenoiser::new(
        vc_core::denoise::GtcrnConfig { model_dir, backend },
        input_rate,
    )?);
    Ok((model_denoisers, passthrough_denoiser))
}

/// Update only the load state of one pool slot, preserving the rest of the
/// status (device fields, message, state). Never goes through `set_status`.
fn update_model_load_status(
    status: &Mutex<EngineStatusSnapshot>,
    request_id: u64,
    state: ModelLoadState,
) {
    if let Ok(mut st) = status.lock() {
        if let Some(entry) = st
            .model_loads
            .iter_mut()
            .find(|m| m.request_id == request_id)
        {
            entry.state = state;
        }
    }
}

/// Static denoiser selection shared by the RVC and passthrough worker paths.
///
/// It deliberately owns model paths so a background model-pool load cannot
/// borrow the live session config. Construction and graph loading stay off the
/// audio callback; this type is only copied into worker-owned processors.
#[derive(Clone, Debug)]
struct DenoiserLoadSettings {
    mode: DenoiserMode,
    #[cfg_attr(not(feature = "gtcrn"), allow(dead_code))]
    gtcrn_model_dir: Option<PathBuf>,
    #[cfg_attr(not(feature = "webrtc"), allow(dead_code))]
    webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel,
    #[cfg_attr(not(feature = "deepfilternet3"), allow(dead_code))]
    deepfilternet3_model: Option<PathBuf>,
    #[cfg_attr(not(feature = "deepfilternet3"), allow(dead_code))]
    dfn3_attenuation_limit_db: f32,
    #[cfg_attr(not(feature = "deepfilternet3"), allow(dead_code))]
    dfn3_post_filter_beta: f32,
    #[cfg(feature = "gtcrn")]
    gtcrn_backend: vc_core::denoise::GtcrnBackend,
}

impl DenoiserLoadSettings {
    fn from_realtime(config: &RealtimeConfig, mode: DenoiserMode) -> Self {
        Self {
            mode,
            gtcrn_model_dir: config.gtcrn_model_dir.clone(),
            webrtc_suppression_level: config.webrtc_suppression_level,
            deepfilternet3_model: config.deepfilternet3_model.clone(),
            dfn3_attenuation_limit_db: config.dfn3_attenuation_limit_db,
            dfn3_post_filter_beta: config.dfn3_post_filter_beta,
            #[cfg(feature = "gtcrn")]
            gtcrn_backend: vc_core::denoise::GtcrnBackend::for_provider(
                config.provider,
                config.gpu_priority,
                config.gpu_device_id,
            ),
        }
    }
}

/// Owned snapshot of the model-side config needed to build a `ChunkConverter`
/// for a new voice model off the worker thread (the loader has no access to the
/// running session's borrowed config).
struct ModelLoadContext {
    model: PathBuf,
    embedder: PathBuf,
    embedder_output: Option<String>,
    f0_model: Option<PathBuf>,
    f0_mode: F0Mode,
    fcpe_model: Option<PathBuf>,
    feature_index: Option<PathBuf>,
    provider: Provider,
    gpu_priority: GpuPriority,
    gpu_device_id: u32,
    sample_rate: u32,
    chunk_samples: usize,
    speaker_id: i64,
    pitch_shift: f32,
    index_rate: f32,
    protect: f32,
    protect_transition_ms: u32,
    denoiser_content_mix: f32,
    denoiser_rmvpe_mix: f32,
    input_gain: f32,
    output_gain: f32,
    f0: F0Config,
    noise_gate_enabled: bool,
    silence_gate_enabled: bool,
    noise_gate_threshold: f32,
    noise_gate_shaping: NoiseGateShaping,
    output_extra_ms: u32,
    volume_excluded_ms: u32,
    extra_convert_ms: u32,
    rvc_frames: Option<usize>,
    output_dynamics: OutputDynamicsConfig,
    smoother_kind: SmoothingKind,
    output_rate: u32,
    output_chunk: usize,
    crossfade_ms: u32,
    sola_search_ms: u32,
    tail_discard_ms: u32,
    denoiser: DenoiserLoadSettings,
}

impl ModelLoadContext {
    fn build_pipeline_config<'a>(
        &'a self,
        progress: &'a dyn Fn(LoadProgress),
    ) -> RvcPipelineConfig<'a> {
        RvcPipelineConfig {
            model: &self.model,
            embedder: &self.embedder,
            embedder_output: self.embedder_output.as_deref(),
            f0_model: self.f0_model.as_deref(),
            f0_mode: self.f0_mode,
            fcpe_model: self.fcpe_model.as_deref(),
            provider: self.provider,
            gpu_priority: self.gpu_priority,
            gpu_device_id: self.gpu_device_id,
            sample_rate: self.sample_rate,
            chunk_samples: self.chunk_samples,
            speaker_id: self.speaker_id,
            pitch_shift: self.pitch_shift,
            f0: self.f0.clone(),
            retrieval: FeatureRetrievalConfig {
                index_path: self.feature_index.as_deref(),
                index_rate: self.index_rate,
                protect: self.protect,
                protect_transition_ms: self.protect_transition_ms,
            },
            input_gain: self.input_gain,
            noise_gate_enabled: self.noise_gate_enabled,
            silence_gate_enabled: self.silence_gate_enabled,
            noise_gate_threshold: self.noise_gate_threshold,
            denoiser_content_mix: self.denoiser_content_mix,
            denoiser_rmvpe_mix: self.denoiser_rmvpe_mix,
            noise_gate_shaping: self.noise_gate_shaping,
            output_extra_ms: self.output_extra_ms,
            volume_excluded_ms: self.volume_excluded_ms,
            extra_convert_ms: self.extra_convert_ms,
            rvc_frames: self.rvc_frames,
            output_gain: self.output_gain,
            output_dynamics: self.output_dynamics,
            progress: Some(progress),
        }
    }
}

impl DenoiserLoadSettings {
    /// Load an `RvcPipeline` honoring this static denoiser selection. Shared by
    /// initial session startup and the background model-pool loader so every
    /// front-end route keeps identical denoiser behavior and delay accounting.
    fn load_pipeline(&self, config: RvcPipelineConfig<'_>) -> Result<RvcPipeline> {
        match self.mode {
            DenoiserMode::Off | DenoiserMode::NoiseGate => RvcPipeline::load(config),
            DenoiserMode::Rnnoise => {
                #[cfg(feature = "rnnoise")]
                {
                    RvcPipeline::load_with_rnnoise(config)
                }
                #[cfg(not(feature = "rnnoise"))]
                {
                    bail!("RNNoise support is not enabled in this build")
                }
            }
            DenoiserMode::WebRtc => {
                #[cfg(feature = "webrtc")]
                {
                    RvcPipeline::load_with_webrtc(config, self.webrtc_suppression_level)
                }
                #[cfg(not(feature = "webrtc"))]
                {
                    bail!("WebRTC denoising support is not enabled in this build")
                }
            }
            DenoiserMode::DeepFilterNet3 => {
                #[cfg(feature = "deepfilternet3")]
                {
                    let model_path = self.deepfilternet3_model.as_deref().ok_or_else(|| {
                        anyhow!("DeepFilterNet3 requires a model archive (deepfilternet3_model)")
                    })?;
                    RvcPipeline::load_with_deepfilternet3(
                        config,
                        vc_core::denoise::DeepFilterNet3Config {
                            model_path,
                            attenuation_limit_db: self.dfn3_attenuation_limit_db,
                            post_filter_beta: self.dfn3_post_filter_beta,
                        },
                    )
                }
                #[cfg(not(feature = "deepfilternet3"))]
                {
                    bail!("DeepFilterNet3 support is not enabled in this build")
                }
            }
            DenoiserMode::Gtcrn => {
                #[cfg(feature = "gtcrn")]
                {
                    let model_dir = self.gtcrn_model_dir.as_deref().ok_or_else(|| {
                        anyhow!("GTCRN denoiser requires a model directory (gtcrn_model_dir)")
                    })?;
                    RvcPipeline::load_with_gtcrn(
                        config,
                        vc_core::denoise::GtcrnConfig {
                            model_dir,
                            backend: self.gtcrn_backend,
                        },
                    )
                }
                #[cfg(not(feature = "gtcrn"))]
                {
                    bail!("GTCRN support is not enabled in this build")
                }
            }
        }
    }
}

/// Build a `ChunkConverter` for a new voice model from a `ModelLoadContext`.
/// Runs on the background loader thread (64 MB stack — engine cache probes
/// recurse deeply).
fn load_model_converter(
    ctx: &ModelLoadContext,
    progress: impl Fn(LoadProgress),
) -> Result<ChunkConverter<RvcPipeline>> {
    let pipeline = ctx
        .denoiser
        .load_pipeline(ctx.build_pipeline_config(&progress))?;
    Ok(ChunkConverter::new(
        pipeline,
        ChunkOutputConfig {
            kind: ctx.smoother_kind,
            output_sample_rate: ctx.output_rate,
            output_chunk_samples: ctx.output_chunk,
            crossfade_ms: ctx.crossfade_ms,
            sola_search_ms: ctx.sola_search_ms,
            tail_discard_ms: ctx.tail_discard_ms,
        },
    ))
}

/// Tear down the current session (if any) and start a fresh one with `config`.
/// Shared by `Command::Apply` and the device-swap restart fallback so the
/// teardown/start/status sequence stays in one place.
#[allow(clippy::too_many_arguments)]
fn restart_session(
    session: &mut Option<RealtimeSession>,
    config: RealtimeConfig,
    status: &Arc<Mutex<EngineStatusSnapshot>>,
    telemetry: &Arc<Telemetry>,
    voice_calibration: &Arc<VoiceCalibrationControl>,
    dynamic_tuning: &Arc<DynamicTuningControl>,
    live: &Arc<AtomicLiveParams>,
    passthrough: &Arc<AtomicBool>,
    active_model: &Arc<AtomicUsize>,
    denoiser_generation: &Arc<AtomicU64>,
    applied_denoiser: &Arc<AppliedDenoiserState>,
    stopping_message: &str,
) {
    // A calibration belongs to one microphone/session timeline. Do not let a
    // worker completing just as a device/model restart occurs publish a profile
    // measured from the old stream into the new configuration.
    voice_calibration.cancel();
    // A newly spawned worker owns a fresh DynamicTuner. Reset its frontend
    // diagnostics at the same lifecycle boundary so an old room-noise estimate
    // is never shown as if it belonged to the new input device/session.
    dynamic_tuning.reset_snapshot();
    passthrough.store(config.passthrough, Ordering::Relaxed);
    set_status(status, EngineState::Stopping, stopping_message);
    drop(session.take());
    set_status(status, EngineState::Starting, "Validating configuration");
    telemetry.reset();
    let initial_denoiser_mode = config.denoiser_mode;
    match RealtimeSession::start(
        config,
        Arc::clone(telemetry),
        Arc::clone(voice_calibration),
        Arc::clone(dynamic_tuning),
        Arc::clone(live),
        Arc::clone(passthrough),
        Arc::clone(active_model),
        Arc::clone(denoiser_generation),
        Arc::clone(applied_denoiser),
        status,
    ) {
        Ok(new_session) => {
            applied_denoiser.store(AppliedDenoiserSnapshot {
                generation: denoiser_generation.load(Ordering::Acquire),
                mode: initial_denoiser_mode,
            });
            live.set_noise_gate_enabled(initial_denoiser_mode == DenoiserMode::NoiseGate);
            if let Ok(mut current) = status.lock() {
                *current = new_session.status();
            }
            *session = Some(new_session);
        }
        Err(err) => set_error(status, &err),
    }
}

fn set_status(
    status: &Mutex<EngineStatusSnapshot>,
    state: EngineState,
    message: impl Into<String>,
) {
    if let Ok(mut status) = status.lock() {
        status.state = state;
        status.message = message.into();
        status.detail = None;
        if state != EngineState::Running {
            status.input_device.clear();
            status.output_device.clear();
            status.input_sample_rate = 0;
            status.output_sample_rate = 0;
            status.monitor_device.clear();
            status.monitor_sample_rate = 0;
            status.passthrough_live_switchable = false;
        }
    }
}

fn set_error(status: &Mutex<EngineStatusSnapshot>, error: &anyhow::Error) {
    if let Ok(mut status) = status.lock() {
        status.state = EngineState::Error;
        status.message = error.to_string();
        status.detail = Some(format!("{error:#}"));
        status.input_device.clear();
        status.output_device.clear();
        status.input_sample_rate = 0;
        status.output_sample_rate = 0;
        status.monitor_device.clear();
        status.monitor_sample_rate = 0;
        status.passthrough_live_switchable = false;
    }
}

/// Copy only endpoint-owned status fields after a same-rate device swap. Model
/// load entries and the active model are worker-owned and may have changed
/// since the session's startup snapshot was created, so replacing the complete
/// shared status here would roll live pool state backwards.
fn patch_device_status(current: &mut EngineStatusSnapshot, device_status: &EngineStatusSnapshot) {
    current.message.clone_from(&device_status.message);
    current.detail.clone_from(&device_status.detail);
    current.input_device.clone_from(&device_status.input_device);
    current
        .output_device
        .clone_from(&device_status.output_device);
    current.input_sample_rate = device_status.input_sample_rate;
    current.output_sample_rate = device_status.output_sample_rate;
    current
        .monitor_device
        .clone_from(&device_status.monitor_device);
    current.monitor_sample_rate = device_status.monitor_sample_rate;
}

/// Publish a control-path failure without tearing down a still-healthy running
/// session. Device reconfiguration is transactional: when the candidate fails,
/// the old streams, worker, model pool, and device fields remain authoritative.
fn set_recoverable_error(
    status: &Mutex<EngineStatusSnapshot>,
    message: impl Into<String>,
    error: &anyhow::Error,
) {
    if let Ok(mut status) = status.lock() {
        status.message = message.into();
        status.detail = Some(format!("{error:#}"));
    }
}

/// Detect an unexpected worker/endpoint stop and invalidate asynchronous
/// denoiser loads before the session is dropped. Otherwise a loader started by
/// the failed session could race a later restart and install stale state.
fn stopped_session_message_and_invalidate(
    worker_running: &AtomicBool,
    endpoint_running: &AtomicBool,
    denoiser_generation: &AtomicU64,
) -> Option<&'static str> {
    let message = if !worker_running.load(Ordering::Acquire) {
        Some("Realtime worker stopped")
    } else if !endpoint_running.load(Ordering::Acquire) {
        Some("Realtime audio endpoint stopped")
    } else {
        None
    };
    if message.is_some() {
        denoiser_generation.fetch_add(1, Ordering::AcqRel);
    }
    message
}

fn load_progress_message(progress: LoadProgress) -> String {
    match progress {
        LoadProgress::Idle => "Idle".to_string(),
        LoadProgress::ValidatingConfig => "Validating configuration".to_string(),
        LoadProgress::PreparingProvider => "Preparing execution provider".to_string(),
        LoadProgress::DownloadingProvider => "Downloading execution provider".to_string(),
        LoadProgress::BuildingEngine { role } => {
            format!("Building {} TensorRT engine", role.label())
        }
        LoadProgress::LoadingModel { role } => format!("Loading {} model", role.label()),
        LoadProgress::OpeningAudioDevices => "Opening audio devices".to_string(),
        LoadProgress::Running => "Running".to_string(),
        LoadProgress::Failed => "Failed".to_string(),
    }
}

/// Non-fatal status note: sets the status message without changing the engine
/// state or clearing device fields. Used for benign rejections (e.g. a duplicate
/// model add) that must not tear down a running session.
fn set_note(status: &Mutex<EngineStatusSnapshot>, message: impl Into<String>) {
    if let Ok(mut st) = status.lock() {
        st.message = message.into();
    }
}

fn device_list(input_host: AudioHost, output_host: AudioHost) -> DeviceList {
    // Inputs and outputs can come from different hosts, so enumerate each
    // direction from its own host and surface the first error if either fails.
    // Enumeration always goes through cpal (it lists every host's devices,
    // including WASAPI shared — the bespoke WASAPI path is only the exclusive-mode
    // opener, not a separate device list).
    let inputs = audio::cpal_input_names(input_host);
    let outputs = audio::cpal_output_names(output_host);
    let error = inputs
        .as_ref()
        .err()
        .or(outputs.as_ref().err())
        .map(|err| format!("{err:#}"));
    DeviceList {
        inputs: inputs.unwrap_or_default(),
        outputs: outputs.unwrap_or_default(),
        error,
    }
}

enum PassthroughDenoiser {
    Off,
    Gate(dsp::NoiseGate),
    #[cfg(feature = "rnnoise")]
    Rnnoise(Box<vc_core::denoise::RnnoiseDenoiser>),
    #[cfg(feature = "webrtc")]
    WebRtc(Box<vc_core::denoise::WebRtcDenoiser>),
    // Device-rate GTCRN instance, independent of the RVC-path 16 kHz one; its
    // adapter resamplers engage (device <-> 16 kHz). Boxed like the others.
    #[cfg(feature = "gtcrn")]
    Gtcrn(Box<vc_core::denoise::GtcrnDenoiser>),
    #[cfg(feature = "deepfilternet3")]
    DeepFilterNet3(Box<vc_core::denoise::DeepFilterNet3Denoiser>),
}

struct PassthroughProcessor {
    mode: DenoiserMode,
    shaping: NoiseGateShaping,
    input_rate: u32,
    output_rate: u32,
    // GTCRN model dir, used only when `mode == Gtcrn` to (re)build the denoiser.
    // Read only on the `gtcrn`-gated reset arm; inert without that feature.
    #[cfg_attr(not(feature = "gtcrn"), allow(dead_code))]
    gtcrn_model_dir: Option<PathBuf>,
    // DFN3 archive and controls are retained for session construction only.
    // Live DFN3 switching uses pre-built worker-owned instances instead of
    // parsing this archive on the audio deadline.
    #[cfg_attr(not(feature = "deepfilternet3"), allow(dead_code))]
    deepfilternet3_model: Option<PathBuf>,
    #[cfg_attr(not(feature = "deepfilternet3"), allow(dead_code))]
    dfn3_attenuation_limit_db: f32,
    #[cfg_attr(not(feature = "deepfilternet3"), allow(dead_code))]
    dfn3_post_filter_beta: f32,
    webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel,
    #[cfg(feature = "gtcrn")]
    gtcrn_backend: vc_core::denoise::GtcrnBackend,
    denoiser: PassthroughDenoiser,
    resampler: dsp::StreamingResampleMono,
    input_scratch: Vec<f32>,
}

impl PassthroughProcessor {
    fn new(
        shaping: NoiseGateShaping,
        input_rate: u32,
        output_rate: u32,
        denoiser_settings: DenoiserLoadSettings,
        live: &LiveParams,
    ) -> Result<Self> {
        let mut processor = Self {
            mode: denoiser_settings.mode,
            shaping,
            input_rate,
            output_rate,
            gtcrn_model_dir: denoiser_settings.gtcrn_model_dir,
            deepfilternet3_model: denoiser_settings.deepfilternet3_model,
            dfn3_attenuation_limit_db: denoiser_settings.dfn3_attenuation_limit_db,
            dfn3_post_filter_beta: denoiser_settings.dfn3_post_filter_beta,
            webrtc_suppression_level: denoiser_settings.webrtc_suppression_level,
            #[cfg(feature = "gtcrn")]
            gtcrn_backend: denoiser_settings.gtcrn_backend,
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
        // Route re-entry resets state but must not rebuild a loaded DFN3 graph
        // (or an already initialized recurrent denoiser) on the worker. A new
        // mode takes the construction path below; a continuing mode reuses its
        // private instance and merely clears its streaming history.
        match (&mut self.denoiser, self.mode) {
            #[cfg(feature = "rnnoise")]
            (PassthroughDenoiser::Rnnoise(denoiser), DenoiserMode::Rnnoise) => {
                denoiser.reset()?;
                self.update_live_denoiser(live);
                return Ok(());
            }
            #[cfg(feature = "webrtc")]
            (PassthroughDenoiser::WebRtc(denoiser), DenoiserMode::WebRtc) => {
                denoiser.reset()?;
                self.update_live_denoiser(live);
                return Ok(());
            }
            #[cfg(feature = "gtcrn")]
            (PassthroughDenoiser::Gtcrn(denoiser), DenoiserMode::Gtcrn) => {
                denoiser.reset()?;
                self.update_live_denoiser(live);
                return Ok(());
            }
            #[cfg(feature = "deepfilternet3")]
            (PassthroughDenoiser::DeepFilterNet3(denoiser), DenoiserMode::DeepFilterNet3) => {
                denoiser.reset()?;
                self.update_live_denoiser(live);
                return Ok(());
            }
            _ => {}
        }
        self.denoiser = match self.mode {
            DenoiserMode::Rnnoise => {
                #[cfg(feature = "rnnoise")]
                {
                    PassthroughDenoiser::Rnnoise(Box::new(vc_core::denoise::RnnoiseDenoiser::new(
                        self.input_rate,
                    )?))
                }
                #[cfg(not(feature = "rnnoise"))]
                {
                    bail!("RNNoise support is not enabled in this build")
                }
            }
            DenoiserMode::Gtcrn => {
                #[cfg(feature = "gtcrn")]
                {
                    let model_dir = self.gtcrn_model_dir.as_deref().ok_or_else(|| {
                        anyhow!("GTCRN denoiser requires a model directory (gtcrn_model_dir)")
                    })?;
                    PassthroughDenoiser::Gtcrn(Box::new(vc_core::denoise::GtcrnDenoiser::new(
                        vc_core::denoise::GtcrnConfig {
                            model_dir,
                            backend: self.gtcrn_backend,
                        },
                        self.input_rate,
                    )?))
                }
                #[cfg(not(feature = "gtcrn"))]
                {
                    bail!("GTCRN support is not enabled in this build")
                }
            }
            DenoiserMode::WebRtc => {
                #[cfg(feature = "webrtc")]
                {
                    PassthroughDenoiser::WebRtc(Box::new(vc_core::denoise::WebRtcDenoiser::new(
                        self.input_rate,
                        self.webrtc_suppression_level,
                    )?))
                }
                #[cfg(not(feature = "webrtc"))]
                {
                    bail!("WebRTC denoising support is not enabled in this build")
                }
            }
            DenoiserMode::DeepFilterNet3 => {
                #[cfg(feature = "deepfilternet3")]
                {
                    let model_path = self.deepfilternet3_model.as_deref().ok_or_else(|| {
                        anyhow!("DeepFilterNet3 requires a model archive (deepfilternet3_model)")
                    })?;
                    PassthroughDenoiser::DeepFilterNet3(Box::new(
                        vc_core::denoise::DeepFilterNet3Denoiser::new(
                            vc_core::denoise::DeepFilterNet3Config {
                                model_path,
                                attenuation_limit_db: self.dfn3_attenuation_limit_db,
                                post_filter_beta: self.dfn3_post_filter_beta,
                            },
                            self.input_rate,
                        )?,
                    ))
                }
                #[cfg(not(feature = "deepfilternet3"))]
                {
                    bail!("DeepFilterNet3 support is not enabled in this build")
                }
            }
            DenoiserMode::Off | DenoiserMode::NoiseGate => PassthroughDenoiser::Off,
        };
        self.update_live_denoiser(live);
        Ok(())
    }

    /// Hot-swap the passthrough denoiser variant (off / gate / rnnoise). Unlike
    /// `reset`, the resampler is left untouched — rebuilding it would drop the
    /// phase of the audio already flowing through passthrough.
    fn set_denoiser_mode(
        &mut self,
        mode: DenoiserMode,
        live: &LiveParams,
        webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel,
    ) -> Result<()> {
        self.mode = mode;
        self.webrtc_suppression_level = webrtc_suppression_level;
        self.denoiser = match mode {
            DenoiserMode::Off => PassthroughDenoiser::Off,
            DenoiserMode::NoiseGate => PassthroughDenoiser::Gate(dsp::NoiseGate::new(
                self.input_rate as f32,
                live.noise_gate_threshold,
                self.shaping.attack_ms,
                self.shaping.release_ms,
                self.shaping.floor,
            )),
            DenoiserMode::Rnnoise => {
                #[cfg(feature = "rnnoise")]
                {
                    PassthroughDenoiser::Rnnoise(Box::new(vc_core::denoise::RnnoiseDenoiser::new(
                        self.input_rate,
                    )?))
                }
                #[cfg(not(feature = "rnnoise"))]
                {
                    bail!("RNNoise support is not enabled in this build")
                }
            }
            DenoiserMode::WebRtc => {
                #[cfg(feature = "webrtc")]
                {
                    PassthroughDenoiser::WebRtc(Box::new(vc_core::denoise::WebRtcDenoiser::new(
                        self.input_rate,
                        self.webrtc_suppression_level,
                    )?))
                }
                #[cfg(not(feature = "webrtc"))]
                {
                    bail!("WebRTC denoising support is not enabled in this build")
                }
            }
            // Graph-backed modes attach via their pre-built swap command; clear
            // the device-rate stage while their loader owns construction.
            DenoiserMode::Gtcrn | DenoiserMode::DeepFilterNet3 => PassthroughDenoiser::Off,
        };
        self.update_live_denoiser(live);
        Ok(())
    }

    /// Hot-swap the device-rate GTCRN instance (`None` leaves gtcrn mode).
    #[cfg(feature = "gtcrn")]
    fn set_gtcrn(&mut self, denoiser: Option<vc_core::denoise::GtcrnDenoiser>) {
        self.mode = if denoiser.is_some() {
            DenoiserMode::Gtcrn
        } else {
            DenoiserMode::Off
        };
        self.denoiser = denoiser
            .map(|d| PassthroughDenoiser::Gtcrn(Box::new(d)))
            .unwrap_or(PassthroughDenoiser::Off);
    }

    /// Hot-swap the device-rate DFN3 instance constructed on a loader thread.
    #[cfg(feature = "deepfilternet3")]
    fn set_deepfilternet3(&mut self, denoiser: Option<vc_core::denoise::DeepFilterNet3Denoiser>) {
        self.mode = if denoiser.is_some() {
            DenoiserMode::DeepFilterNet3
        } else {
            DenoiserMode::Off
        };
        self.denoiser = denoiser
            .map(|d| PassthroughDenoiser::DeepFilterNet3(Box::new(d)))
            .unwrap_or(PassthroughDenoiser::Off);
    }

    fn update_live_denoiser(&mut self, live: &LiveParams) {
        // Stateful denoisers are static load-time choices; a live gate toggle must
        // never tear them down.
        if matches!(
            self.mode,
            DenoiserMode::Rnnoise
                | DenoiserMode::Gtcrn
                | DenoiserMode::WebRtc
                | DenoiserMode::DeepFilterNet3
        ) {
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
            #[cfg(feature = "webrtc")]
            PassthroughDenoiser::WebRtc(denoiser) => {
                denoiser.process_in_place(&mut self.input_scratch)?
            }
            #[cfg(feature = "gtcrn")]
            PassthroughDenoiser::Gtcrn(denoiser) => {
                denoiser.process_in_place(&mut self.input_scratch)?
            }
            #[cfg(feature = "deepfilternet3")]
            PassthroughDenoiser::DeepFilterNet3(denoiser) => {
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
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms,
            output_rms: dsp::rms(prepared),
            model_output_samples: prepared.len(),
            voiced_ratio: 0.0,
            pitch_variation_semitones: 0.0,
            pitch_frames: 0,
        })
    }
}

// Stateful processing remains on the inference worker. Passthrough-only
// sessions preserve the model-free diagnostic path; switchable sessions retain
// loaded RVC sessions but stop invoking them while passthrough is active; a
// pool is a switchable session that grew past one RVC model (added live), with
// the active slot selected by an atomic written from the front-end.
fn remap_slot_after_removal(
    requested_before: usize,
    removed_slot: usize,
    remaining_models: usize,
) -> usize {
    if remaining_models == 0 {
        return 0;
    }
    if requested_before == removed_slot {
        removed_slot.min(remaining_models - 1)
    } else if requested_before > removed_slot {
        requested_before - 1
    } else {
        requested_before
    }
}

fn remove_aligned_model_slot<T>(models: &mut Vec<T>, names: &mut Vec<String>, slot: usize) -> bool {
    if slot >= models.len() || models.len() != names.len() {
        return false;
    }
    models.remove(slot);
    names.remove(slot);
    true
}

/// Remap the frontend-requested dense slot only if it still contains the value
/// observed before deletion. A concurrent `set_active_model` wins the CAS; the
/// process path's existing bounds clamp handles that newer request against the
/// shortened pool on its next chunk.
fn try_remap_requested_slot_after_removal(
    active_requested: &AtomicUsize,
    requested_before: usize,
    removed_slot: usize,
    remaining_models: usize,
) -> bool {
    let remapped = remap_slot_after_removal(requested_before, removed_slot, remaining_models);
    active_requested
        .compare_exchange(
            requested_before,
            remapped,
            Ordering::Relaxed,
            Ordering::Relaxed,
        )
        .is_ok()
}

/// Publish the controller-selected first-model activation without overwriting
/// a newer frontend selection. Adding a non-active model does not change any
/// existing dense index, so it must not write the atomic at all.
fn try_publish_active_slot_after_add(
    active_requested: &AtomicUsize,
    requested_before: usize,
    active_after: usize,
    activate: bool,
) -> bool {
    activate
        && active_requested
            .compare_exchange(
                requested_before,
                active_after,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
}

fn keep_first_reset_error(first_error: &mut Option<anyhow::Error>, result: Result<()>) {
    if first_error.is_none() {
        if let Err(err) = result {
            *first_error = Some(err);
        }
    }
}

fn reset_rvc_converter_history(
    converter: &mut ChunkConverter<RvcPipeline>,
    first_error: &mut Option<anyhow::Error>,
) {
    let model_reset = converter.model_mut().reset_streaming_state();
    // The smoother/tail is independent from the model reset result and must
    // always be cleared before audio from a newly-bound device can arrive.
    converter.reset_streaming_state();
    keep_first_reset_error(first_error, model_reset);
}

#[allow(clippy::large_enum_variant)]
enum RuntimeModel {
    PassthroughOnly(PassthroughProcessor),
    Switchable {
        passthrough: PassthroughProcessor,
        rvc: ChunkConverter<RvcPipeline>,
        passthrough_active: bool,
        name: String,
    },
    Pool {
        passthrough: PassthroughProcessor,
        models: Vec<ChunkConverter<RvcPipeline>>,
        names: Vec<String>,
        active: usize,
        passthrough_active: bool,
        active_requested: Arc<AtomicUsize>,
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
                ..
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
                if rvc.model_mut().apply_live(live) {
                    rvc.reset_streaming_state();
                }
                // Write the converted chunk straight into the worker-owned
                // `prepared` buffer (reused across chunks) instead of moving a
                // freshly allocated Vec out of the converter.
                rvc.process_chunk(audio, sample_rate, None, prepared)
            }
            Self::Pool {
                passthrough,
                models,
                active,
                passthrough_active,
                active_requested,
                ..
            } => {
                if passthrough_requested {
                    if !*passthrough_active {
                        passthrough.reset(live)?;
                        *passthrough_active = true;
                    }
                    return passthrough.process_chunk(audio, live, prepared);
                }
                // `remove_model` normally converts an empty pool back to the
                // PassthroughOnly variant. Keep this guard as a defensive
                // invariant at the realtime seam: a stale command or future
                // pool mutation must never turn `len() - 1` into an underflow
                // or index an empty vector on the worker thread.
                if models.is_empty() {
                    return passthrough.process_chunk(audio, live, prepared);
                }
                let requested = active_requested
                    .load(Ordering::Relaxed)
                    .min(models.len() - 1);
                if *passthrough_active || requested != *active {
                    // The model being activated was idle, so neither its rolling
                    // context nor its output smoother may be joined to the input
                    // (mirrors the passthrough transition above).
                    models[requested].model_mut().reset_streaming_state()?;
                    models[requested].reset_streaming_state();
                    *active = requested;
                    *passthrough_active = false;
                }
                let model = &mut models[*active];
                if model.model_mut().apply_live(live) {
                    model.reset_streaming_state();
                }
                model.process_chunk(audio, sample_rate, None, prepared)
            }
        }
    }

    /// Grow into (or extend) a model pool with a freshly-loaded converter. The
    /// active slot is seeded from `activate` (used when the first model is added
    /// to a passthrough-only session); `active_requested` is kept in sync with
    /// the returned slot.
    fn into_pool_with(
        self,
        converter: ChunkConverter<RvcPipeline>,
        name: String,
        activate: bool,
        active_requested: Arc<AtomicUsize>,
    ) -> RuntimeModel {
        let requested_before = active_requested.load(Ordering::Relaxed);
        let (passthrough, mut models, mut names, mut active, passthrough_active) = match self {
            Self::PassthroughOnly(passthrough) => (passthrough, Vec::new(), Vec::new(), 0, false),
            Self::Switchable {
                passthrough,
                rvc,
                passthrough_active,
                name: existing,
            } => (
                passthrough,
                vec![rvc],
                vec![existing],
                0,
                passthrough_active,
            ),
            Self::Pool {
                passthrough,
                models,
                names,
                active,
                passthrough_active,
                active_requested: _,
            } => (passthrough, models, names, active, passthrough_active),
        };
        let slot = models.len();
        models.push(converter);
        names.push(name);
        if activate {
            active = slot;
        }
        let _ = try_publish_active_slot_after_add(
            &active_requested,
            requested_before,
            active,
            activate,
        );
        RuntimeModel::Pool {
            passthrough,
            models,
            names,
            active,
            passthrough_active,
            active_requested,
        }
    }

    fn model_count(&self) -> usize {
        match self {
            Self::PassthroughOnly(_) => 0,
            Self::Switchable { .. } => 1,
            Self::Pool { models, .. } => models.len(),
        }
    }

    /// Drop a pool slot by its dense index. Base-model protection belongs to the
    /// control layer's stable request id: in a passthrough-only session the first
    /// live-added (and removable) model legitimately occupies dense slot 0. If
    /// the active/requested model is removed, it falls back to the nearest
    /// remaining model; later indices shift down. A compare-exchange preserves a
    /// newer concurrent frontend request instead of overwriting it.
    fn remove_model(mut self, slot: usize) -> RuntimeModel {
        let mut empty_pool_requested_before = None;
        if let Self::Pool {
            models,
            names,
            active,
            active_requested,
            ..
        } = &mut self
        {
            if models.is_empty() {
                empty_pool_requested_before = Some(active_requested.load(Ordering::Relaxed));
            } else {
                let requested_before = active_requested.load(Ordering::Relaxed);
                if remove_aligned_model_slot(models, names, slot) {
                    *active = remap_slot_after_removal(*active, slot, models.len());
                    let _ = try_remap_requested_slot_after_removal(
                        active_requested,
                        requested_before,
                        slot,
                        models.len(),
                    );
                }
            }
        }
        // A model-free session has a distinct runtime variant. Apart from
        // avoiding an empty-pool index, this keeps status/passthrough semantics
        // coherent after deleting the final live-loaded model.
        if matches!(&self, Self::Pool { models, .. } if models.is_empty()) {
            if let Self::Pool {
                passthrough,
                active_requested,
                ..
            } = self
            {
                let requested_before = empty_pool_requested_before
                    .unwrap_or_else(|| active_requested.load(Ordering::Relaxed));
                let _ = active_requested.compare_exchange(
                    requested_before,
                    0,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                return Self::PassthroughOnly(passthrough);
            }
        }
        self
    }

    fn active_model_name(&self) -> Option<String> {
        match self {
            Self::PassthroughOnly(_) => None,
            Self::Switchable { name, .. } => Some(name.clone()),
            Self::Pool { names, active, .. } => names.get(*active).cloned(),
        }
    }

    fn active_speaker_count(&mut self) -> Option<usize> {
        match self {
            Self::PassthroughOnly(_) => None,
            Self::Switchable { rvc, .. } => rvc.model_mut().speaker_count(),
            Self::Pool { models, active, .. } => models
                .get_mut(*active)
                .and_then(|model| model.model_mut().speaker_count()),
        }
    }

    /// Reset every stream-derived state after a same-rate device rebind. This
    /// runs only on the inference worker: callbacks keep queue-only ownership,
    /// while all loaded (including inactive) models, passthrough denoisers and
    /// converter smoothers start the new device on a clean timeline.
    fn reset_for_device_rebind(&mut self, live: &LiveParams) -> Result<()> {
        let mut first_error = None;
        match self {
            Self::PassthroughOnly(passthrough) => {
                keep_first_reset_error(&mut first_error, passthrough.reset(live));
            }
            Self::Switchable {
                passthrough, rvc, ..
            } => {
                keep_first_reset_error(&mut first_error, passthrough.reset(live));
                reset_rvc_converter_history(rvc, &mut first_error);
            }
            Self::Pool {
                passthrough,
                models,
                ..
            } => {
                keep_first_reset_error(&mut first_error, passthrough.reset(live));
                for model in models {
                    reset_rvc_converter_history(model, &mut first_error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Hot-swap the denoiser variant across the passthrough processor and every
    /// loaded RVC model. Runs on the worker thread (rnnoise build is cheap).
    fn set_denoiser_mode(
        &mut self,
        mode: DenoiserMode,
        input_rate: u32,
        live: &LiveParams,
        webrtc_suppression_level: vc_core::denoise_config::WebRtcSuppressionLevel,
    ) -> Result<()> {
        let core_mode: vc_core::model_rvc::InputDenoiserMode = mode.into();
        match self {
            Self::PassthroughOnly(passthrough) => {
                passthrough.set_denoiser_mode(mode, live, webrtc_suppression_level)
            }
            Self::Switchable {
                passthrough, rvc, ..
            } => {
                passthrough.set_denoiser_mode(mode, live, webrtc_suppression_level)?;
                rvc.model_mut().set_denoiser_mode(
                    core_mode,
                    input_rate,
                    webrtc_suppression_level,
                )?;
                rvc.reset_streaming_state();
                Ok(())
            }
            Self::Pool {
                passthrough,
                models,
                ..
            } => {
                passthrough.set_denoiser_mode(mode, live, webrtc_suppression_level)?;
                for model in models.iter_mut() {
                    model.model_mut().set_denoiser_mode(
                        core_mode,
                        input_rate,
                        webrtc_suppression_level,
                    )?;
                    model.reset_streaming_state();
                }
                Ok(())
            }
        }
    }

    /// Hot-swap pre-built GTCRN denoisers (one per model at 16 kHz, plus one
    /// device-rate instance for passthrough).
    #[cfg(feature = "gtcrn")]
    fn set_gtcrn(
        &mut self,
        passthrough_denoiser: Option<vc_core::denoise::GtcrnDenoiser>,
        model_denoisers: Vec<vc_core::denoise::GtcrnDenoiser>,
    ) -> bool {
        if !denoiser_set_matches_model_count(model_denoisers.len(), self.model_count()) {
            return false;
        }
        match self {
            Self::PassthroughOnly(passthrough) => passthrough.set_gtcrn(passthrough_denoiser),
            Self::Switchable {
                passthrough, rvc, ..
            } => {
                passthrough.set_gtcrn(passthrough_denoiser);
                rvc.model_mut()
                    .set_gtcrn(model_denoisers.into_iter().next());
                rvc.reset_streaming_state();
            }
            Self::Pool {
                passthrough,
                models,
                ..
            } => {
                passthrough.set_gtcrn(passthrough_denoiser);
                for (model, denoiser) in models.iter_mut().zip(model_denoisers) {
                    model.model_mut().set_gtcrn(Some(denoiser));
                    model.reset_streaming_state();
                }
            }
        }
        true
    }

    /// Hot-swap pre-built DeepFilterNet3 denoisers (one device-rate stream per
    /// RVC model and one for passthrough). The DFN3 graph is built by the
    /// background loader, never by this audio-deadline worker.
    #[cfg(feature = "deepfilternet3")]
    fn set_deepfilternet3(
        &mut self,
        passthrough_denoiser: Option<vc_core::denoise::DeepFilterNet3Denoiser>,
        model_denoisers: Vec<vc_core::denoise::DeepFilterNet3Denoiser>,
    ) -> bool {
        if !denoiser_set_matches_model_count(model_denoisers.len(), self.model_count()) {
            return false;
        }
        match self {
            Self::PassthroughOnly(passthrough) => {
                passthrough.set_deepfilternet3(passthrough_denoiser)
            }
            Self::Switchable {
                passthrough, rvc, ..
            } => {
                passthrough.set_deepfilternet3(passthrough_denoiser);
                rvc.model_mut()
                    .set_deepfilternet3(model_denoisers.into_iter().next());
                rvc.reset_streaming_state();
            }
            Self::Pool {
                passthrough,
                models,
                ..
            } => {
                passthrough.set_deepfilternet3(passthrough_denoiser);
                for (model, denoiser) in models.iter_mut().zip(model_denoisers) {
                    model.model_mut().set_deepfilternet3(Some(denoiser));
                    model.reset_streaming_state();
                }
            }
        }
        true
    }
}

fn denoiser_set_matches_model_count(supplied: usize, current: usize) -> bool {
    supplied == current
}

impl From<DenoiserMode> for vc_core::model_rvc::InputDenoiserMode {
    fn from(mode: DenoiserMode) -> Self {
        match mode {
            DenoiserMode::Off => vc_core::model_rvc::InputDenoiserMode::Off,
            DenoiserMode::NoiseGate => vc_core::model_rvc::InputDenoiserMode::Gate,
            DenoiserMode::Rnnoise => vc_core::model_rvc::InputDenoiserMode::Rnnoise,
            DenoiserMode::Gtcrn => vc_core::model_rvc::InputDenoiserMode::Gtcrn,
            DenoiserMode::WebRtc => vc_core::model_rvc::InputDenoiserMode::WebRtc,
            DenoiserMode::DeepFilterNet3 => vc_core::model_rvc::InputDenoiserMode::DeepFilterNet3,
        }
    }
}

/// Minimal worker-wakeup channel: the audio input callback calls [`wake`] after
/// queuing samples so the inference worker can `park`/`unpark` instead of
/// polling on a fixed sleep. `OnceLock` defers capturing the worker `Thread`
/// until the worker spawns and registers itself, preserving the
/// build-streams-then-spawn-then-play startup order (a stream-build failure must
/// not have started the worker). A fresh `WorkerWake` is created per session, so
/// the one-shot registration is never reused across worker restarts.
///
/// [`wake`]: WorkerWake::wake
#[derive(Default)]
struct WorkerWake {
    thread: OnceLock<Thread>,
}

impl WorkerWake {
    /// Called once by the worker on spawn. First registration wins; the worker
    /// is the only registrant.
    fn register_current(&self) {
        let _ = self.thread.set(thread::current());
    }

    /// Unpark the worker if it has registered. Wait-free: `unpark` only stores a
    /// token (or performs one OS wakeup when the worker is actually parked), so
    /// it is safe to call from the realtime audio callback.
    fn wake(&self) {
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
    }
}

/// Pull from the input ring into `input_acc` until it holds one inference chunk
/// or the ring is drained, returning whether a full chunk is ready.
///
/// Kept free of model/telemetry/IO so the accumulation logic is unit testable
/// and the frame-grid/alignment-sensitive work stays in the worker. Only the
/// chunk-sized deficit is read, so a callback block larger than `input_chunk`
/// leaves its remainder in the ring for the next chunk, and sub-chunk blocks
/// coalesce across calls.
fn accumulate_input_chunk(
    consumer: &mut rtrb::Consumer<f32>,
    input_acc: &mut Vec<f32>,
    input_chunk: usize,
) -> bool {
    let available = consumer
        .slots()
        .min(input_chunk.saturating_sub(input_acc.len()));
    if available > 0 {
        let old = input_acc.len();
        input_acc.resize(old + available, 0.0);
        if consumer.pop_entire_slice(&mut input_acc[old..]).is_err() {
            input_acc.truncate(old);
        }
    }
    input_acc.len() >= input_chunk
}

struct RealtimeSession {
    /// Worker/session lifetime. Device endpoint failures are tracked by the
    /// separately swappable `endpoint_running` flag so a candidate rebind can
    /// fail without stopping the current worker and streams.
    running: Arc<AtomicBool>,
    endpoint_running: Arc<AtomicBool>,
    wake: Arc<WorkerWake>,
    worker: Option<JoinHandle<()>>,
    input_stream: Option<AudioStream>,
    output_stream: Option<AudioStream>,
    monitor_stream: Option<AudioStream>,
    status: EngineStatusSnapshot,
    debug_input_wav: Option<PathBuf>,
    debug_output_wav: Option<PathBuf>,
    debug_input: Arc<Mutex<Vec<f32>>>,
    debug_output: Arc<Mutex<Vec<f32>>>,
    input_rate: u32,
    output_rate: u32,
    // Retained for live device swaps: the worker command mailbox, the config
    // (reused when a device sample-rate change forces a full restart), the ring
    // capacities (fixed while rates are unchanged), and telemetry for stream
    // rebuilds.
    worker_tx: Sender<WorkerCommand>,
    config: RealtimeConfig,
    input_capacity: usize,
    output_capacity: usize,
    monitor_capacity: usize,
    telemetry: Arc<Telemetry>,
}

impl RealtimeSession {
    // These are independently owned lifecycle handles (device telemetry,
    // calibration, dynamic tuning, live controls, routing, and status). Keep
    // them explicit at the one session-construction boundary rather than
    // inventing a short-lived aggregate that obscures ownership on restart.
    #[allow(clippy::too_many_arguments)]
    fn start(
        config: RealtimeConfig,
        telemetry: Arc<Telemetry>,
        voice_calibration: Arc<VoiceCalibrationControl>,
        dynamic_tuning: Arc<DynamicTuningControl>,
        live: Arc<AtomicLiveParams>,
        passthrough_live: Arc<AtomicBool>,
        active_model: Arc<AtomicUsize>,
        denoiser_generation: Arc<AtomicU64>,
        applied_denoiser: Arc<AppliedDenoiserState>,
        status: &Arc<Mutex<EngineStatusSnapshot>>,
    ) -> Result<Self> {
        // Reset the live active-model slot to the base model on every session.
        active_model.store(0, Ordering::Relaxed);
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
        set_status(status, EngineState::Starting, "Opening audio devices");
        let audio = RealtimeAudio::open(
            config.input_host,
            config.output_host,
            config.wasapi_input_exclusive,
            config.wasapi_output_exclusive,
            config.input_device.as_deref(),
            config.output_device.as_deref(),
            config.wasapi_buffer_ms,
            config.monitor_output_enabled,
            config.monitor_output_device.as_deref(),
        )?;
        let input_rate = audio.input_sample_rate();
        let output_rate = audio.output_sample_rate();
        let input_chunk = dsp::chunk_samples_for_rate(input_rate, config.chunk_ms);
        let output_chunk = dsp::chunk_samples_for_rate(output_rate, config.chunk_ms);
        // Monitor output: a second device playing the converted signal at its
        // own rate, resampled from the primary output rate on the worker.
        let monitor_rate = audio.monitor_sample_rate();
        let monitor_chunk = if config.monitor_output_enabled {
            dsp::chunk_samples_for_rate(monitor_rate, config.chunk_ms)
        } else {
            0
        };
        let monitor_capacity = if config.monitor_output_enabled {
            monitor_chunk * OUTPUT_QUEUE_CHUNKS
        } else {
            0
        };
        let mut monitor_resampler = if config.monitor_output_enabled {
            Some(dsp::StreamingResampleMono::new(
                output_rate as usize,
                monitor_rate as usize,
            )?)
        } else {
            None
        };
        let current_live = live.load();
        let denoiser_settings = DenoiserLoadSettings::from_realtime(&config, config.denoiser_mode);
        let debug_input = Arc::new(Mutex::new(Vec::new()));
        let debug_output = Arc::new(Mutex::new(Vec::new()));
        let passthrough_processor = PassthroughProcessor::new(
            config.noise_gate_shaping,
            input_rate,
            output_rate,
            denoiser_settings.clone(),
            &current_live,
        )?;
        let passthrough_live_switchable = config.has_complete_model_set();
        let mut model = if passthrough_live_switchable {
            let report_progress = |progress| {
                set_status(
                    status,
                    EngineState::Starting,
                    load_progress_message(progress),
                );
            };
            let pipeline_config = config.pipeline_config(
                input_rate,
                input_chunk,
                &current_live,
                Some(&report_progress),
            );
            let pipeline = denoiser_settings.load_pipeline(pipeline_config)?;
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
                name: config
                    .model
                    .as_deref()
                    .map(model_name_for)
                    .unwrap_or_else(|| "<base model>".to_string()),
            }
        } else {
            RuntimeModel::PassthroughOnly(passthrough_processor)
        };

        let speaker_count = model.active_speaker_count();
        let output_capacity = output_chunk * OUTPUT_QUEUE_CHUNKS;
        let input_capacity = input_chunk * INPUT_QUEUE_CHUNKS;
        let running = Arc::new(AtomicBool::new(true));
        let endpoint_running = Arc::new(AtomicBool::new(true));
        let wake = Arc::new(WorkerWake::default());
        // Worker command mailbox (control thread → worker). Unbounded so large
        // payloads (later: whole model converters) never block the sender.
        let (worker_tx, worker_rx) = mpsc::channel();
        // Build the device streams before spawning the inference worker: a
        // stream failure then returns without ever starting (and stopping) the
        // worker and its model/CUDA context. The streams stay paused until
        // play(), so the worker can attach afterwards.
        let endpoints = build_streams(
            &audio,
            input_capacity,
            output_capacity,
            config.monitor_output_enabled.then_some(monitor_capacity),
            &endpoint_running,
            &wake,
            &telemetry,
        )?;
        let input_stream = endpoints.input_stream;
        let output_stream = endpoints.output_stream;
        let monitor_stream = endpoints.monitor_stream;
        let mut input_consumer = endpoints.input_consumer;
        let mut output_producer = endpoints.output_producer;
        let mut monitor_producer = endpoints.monitor_producer;
        let worker_running = Arc::clone(&running);
        let worker_wake = Arc::clone(&wake);
        let worker_telemetry = Arc::clone(&telemetry);
        let worker_voice_calibration = Arc::clone(&voice_calibration);
        let worker_dynamic_tuning = Arc::clone(&dynamic_tuning);
        let worker_denoiser_generation = Arc::clone(&denoiser_generation);
        let worker_applied_denoiser = Arc::clone(&applied_denoiser);
        let worker_debug_input = Arc::clone(&debug_input);
        let worker_debug_output = Arc::clone(&debug_output);
        // The worker owns the model pool, so it is the source of truth for pool
        // load state: it updates the shared status when a model is added.
        let worker_status = Arc::clone(status);
        let worker_active_model = Arc::clone(&active_model);
        let capture_input = config.debug_input_wav.is_some();
        let capture_output = config.debug_output_wav.is_some();
        let mut worker = Some(
            thread::Builder::new()
                .name("vc-app-inference".to_string())
                .spawn(move || {
                    // Register before the first park so the input callback's
                    // wake() can reach this thread. A wake() that races ahead of
                    // registration is recovered by the self-wake below plus the
                    // park timeout; the ring is the source of truth either way.
                    worker_wake.register_current();
                    worker_wake.wake();
                    if let Err(err) = set_current_thread_priority(ThreadPriority::Max) {
                        tracing::warn!("failed to set inference worker thread priority: {err}");
                    }
                    let mut model = model;
                    let mut input_acc = Vec::<f32>::with_capacity(input_chunk * 2);
                    let mut prepared = Vec::<f32>::with_capacity(output_chunk * 2);
                    let mut monitor_prepared = Vec::<f32>::with_capacity(monitor_chunk * 2);
                    // The accumulator is worker-owned and contains only fixed
                    // histogram/F0 counters. Never move this into a device
                    // callback: calibration is a setup task, not RT I/O.
                    let mut calibration_generation = 0u64;
                    let mut calibration: Option<VoiceCalibrationAccumulator> = None;
                    // This overlay is owned entirely by the inference worker.
                    // It samples the user's base controls once per chunk and
                    // never reaches into an audio callback or an editor lock.
                    let mut dynamic_tuner = DynamicTuner::default();
                    // Last pool slot reported to the shared status, so a live
                    // model switch is propagated without re-looking-up the name
                    // (and allocating) on every chunk.
                    let mut last_active_slot = 0usize;
                    // Async denoiser commands carry a generation.  The worker
                    // is the final authority because loader/control sends can
                    // race; stale swaps are dropped before touching state.
                    let mut applied_denoiser_generation =
                        worker_denoiser_generation.load(Ordering::Acquire);
                    while worker_running.load(Ordering::SeqCst) {
                        match worker_voice_calibration.request() {
                            Some((generation, duration_ms))
                                if generation != calibration_generation =>
                            {
                                calibration_generation = generation;
                                calibration =
                                    Some(VoiceCalibrationAccumulator::new(input_rate, duration_ms));
                                worker_voice_calibration.mark_collecting(generation, 0);
                            }
                            Some(_) => {}
                            None => {
                                calibration_generation = 0;
                                calibration = None;
                            }
                        }
                        // Drain worker commands at the top of every iteration so a
                        // live change (device rebind, later model/denoiser swaps)
                        // applies promptly even while parked waiting for input.
                        while let Ok(command) = worker_rx.try_recv() {
                            match command {
                                WorkerCommand::RebindRings {
                                    input_consumer: ic,
                                    output_producer: op,
                                    monitor_producer: mp,
                                } => {
                                    let live_params = live.load();
                                    if let Err(err) = model.reset_for_device_rebind(&live_params) {
                                        // Continuing would concatenate the new
                                        // microphone with partially-reset model
                                        // or denoiser history. Stop the worker;
                                        // the control loop will publish Error.
                                        tracing::warn!(
                                            "failed to reset model history for device rebind: {err:#}"
                                        );
                                        worker_running.store(false, Ordering::SeqCst);
                                        break;
                                    }
                                    let next_monitor_resampler = if mp.is_some() {
                                        match dsp::StreamingResampleMono::new(
                                            output_rate as usize,
                                            monitor_rate as usize,
                                        ) {
                                            Ok(resampler) => Some(resampler),
                                            Err(err) => {
                                                tracing::warn!(
                                                    "failed to reset monitor resampler for device rebind: {err:#}"
                                                );
                                                worker_running.store(false, Ordering::SeqCst);
                                                break;
                                            }
                                        }
                                    } else {
                                        None
                                    };
                                    input_consumer = ic;
                                    output_producer = op;
                                    monitor_producer = mp;
                                    monitor_resampler = next_monitor_resampler;
                                    // Discard every worker-owned partial buffer
                                    // and adaptive observation from the previous
                                    // device. The new rings begin one clean
                                    // model/denoiser/monitor timeline.
                                    input_acc.clear();
                                    prepared.clear();
                                    monitor_prepared.clear();
                                    calibration_generation = 0;
                                    calibration = None;
                                    dynamic_tuner = DynamicTuner::default();
                                    worker_dynamic_tuning.reset_snapshot();
                                }
                                WorkerCommand::AddModel {
                                    converter,
                                    name,
                                    request_id,
                                    activate,
                                    built_denoiser_generation,
                                } => {
                                    // The control thread can remove a loading
                                    // request before its background builder
                                    // finishes.  Remove has no pool slot in that
                                    // window, so the worker must re-check the
                                    // tombstone here or the completed converter
                                    // would become a ghost model after deletion.
                                    let request_is_live = worker_status
                                        .lock()
                                        .map(|status| model_load_request_is_live(&status, request_id))
                                        .unwrap_or(false);
                                    if !request_is_live {
                                        continue;
                                    }
                                    if built_denoiser_generation != applied_denoiser_generation {
                                        update_model_load_status(
                                            &worker_status,
                                            request_id,
                                            ModelLoadState::Error(
                                                "denoiser changed before model activation; retry the model load"
                                                    .to_string(),
                                            ),
                                        );
                                        continue;
                                    }
                                    // A passthrough-only session gains live
                                    // passthrough switching once it has a model.
                                    let was_passthrough_only =
                                        matches!(model, RuntimeModel::PassthroughOnly(_));
                                    model = model.into_pool_with(
                                        converter,
                                        name,
                                        activate,
                                        Arc::clone(&worker_active_model),
                                    );
                                    if let Ok(mut st) = worker_status.lock() {
                                        if let Some(entry) = st
                                            .model_loads
                                            .iter_mut()
                                            .find(|m| m.request_id == request_id)
                                        {
                                            entry.state = ModelLoadState::Loaded;
                                            entry.pool_index = Some(model.model_count() - 1);
                                        }
                                        st.active_model = model.active_model_name();
                                        st.speaker_count = model.active_speaker_count();
                                        if was_passthrough_only {
                                            st.passthrough_live_switchable = true;
                                        }
                                    }
                                }
                                WorkerCommand::RemoveModel { slot } => {
                                    model = model.remove_model(slot);
                                    if model.model_count() == 0 {
                                        // The passthrough processor may have been
                                        // idle while the deleted RVC model was
                                        // active. Reset its resampler/denoiser
                                        // timeline before exposing the model-free
                                        // route, just like a normal RVC ->
                                        // passthrough transition.
                                        let live_params = live.load();
                                        if let RuntimeModel::PassthroughOnly(passthrough) = &mut model
                                        {
                                            if let Err(err) = passthrough.reset(&live_params) {
                                                tracing::warn!(
                                                    "failed to reset passthrough after final model removal: {err}"
                                                );
                                            }
                                        }
                                    }
                                    // Removing the active pool slot may fall back to a
                                    // model with a different embedding table (for example
                                    // MXGF 308-speaker -> stock 109-speaker). Publish the
                                    // new range with the active model name atomically from
                                    // the worker so GUI/host automation cannot use stale
                                    // bounds after a pool edit.
                                    if let Ok(mut st) = worker_status.lock() {
                                        st.active_model = model.active_model_name();
                                        st.speaker_count = model.active_speaker_count();
                                        st.passthrough_live_switchable = model.model_count() > 0;
                                    }
                                }
                                WorkerCommand::SetDenoiser {
                                    mode,
                                    webrtc_suppression_level,
                                    generation,
                                } => {
                                    if generation
                                        == worker_denoiser_generation.load(Ordering::Acquire)
                                        && generation >= applied_denoiser_generation
                                    {
                                        let live_params = live.load();
                                        match model.set_denoiser_mode(
                                            mode,
                                            input_rate,
                                            &live_params,
                                            webrtc_suppression_level,
                                        ) {
                                            Ok(()) => {
                                                applied_denoiser_generation = generation;
                                                worker_applied_denoiser.store(
                                                    AppliedDenoiserSnapshot { generation, mode },
                                                );
                                                live.set_noise_gate_enabled(
                                                    mode == DenoiserMode::NoiseGate,
                                                );
                                            }
                                            Err(err) => {
                                                tracing::warn!(
                                                    "failed to switch denoiser: {err}"
                                                );
                                                if let Ok(mut status) = worker_status.lock() {
                                                    status.detail = Some(format!(
                                                        "Denoiser switch failed: {err:#}"
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                                #[cfg(feature = "gtcrn")]
                                WorkerCommand::SwapGtcrn {
                                    model_denoisers,
                                    passthrough_denoiser,
                                    generation,
                                } => {
                                    if generation
                                        == worker_denoiser_generation.load(Ordering::Acquire)
                                        && generation >= applied_denoiser_generation
                                    {
                                        if model.set_gtcrn(
                                            passthrough_denoiser,
                                            model_denoisers,
                                        ) {
                                            applied_denoiser_generation = generation;
                                            worker_applied_denoiser.store(
                                                AppliedDenoiserSnapshot {
                                                    generation,
                                                    mode: DenoiserMode::Gtcrn,
                                                },
                                            );
                                            live.set_noise_gate_enabled(false);
                                        } else if let Ok(mut status) = worker_status.lock() {
                                            status.detail = Some(
                                                "GTCRN swap discarded because the model pool changed during loading; retry denoiser selection"
                                                    .to_string(),
                                            );
                                        }
                                    }
                                }
                                #[cfg(feature = "deepfilternet3")]
                                WorkerCommand::SwapDeepFilterNet3 {
                                    model_denoisers,
                                    passthrough_denoiser,
                                    generation,
                                } => {
                                    if generation
                                        == worker_denoiser_generation.load(Ordering::Acquire)
                                        && generation >= applied_denoiser_generation
                                    {
                                        if model.set_deepfilternet3(
                                            passthrough_denoiser,
                                            model_denoisers,
                                        ) {
                                            applied_denoiser_generation = generation;
                                            worker_applied_denoiser.store(
                                                AppliedDenoiserSnapshot {
                                                    generation,
                                                    mode: DenoiserMode::DeepFilterNet3,
                                                },
                                            );
                                            live.set_noise_gate_enabled(false);
                                        } else if let Ok(mut status) = worker_status.lock() {
                                            status.detail = Some(
                                                "DeepFilterNet3 swap discarded because the model pool changed during loading; retry denoiser selection"
                                                    .to_string(),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        if !worker_running.load(Ordering::SeqCst) {
                            break;
                        }
                        if !accumulate_input_chunk(&mut input_consumer, &mut input_acc, input_chunk)
                        {
                            // Re-check the stop flag before parking so a stop
                            // requested between the loop head and here exits
                            // without waiting for the timeout or a final wake().
                            if !worker_running.load(Ordering::SeqCst) {
                                break;
                            }
                            // Wait for the input callback's wake() (sent after it
                            // queues samples) or the safety timeout. Replaces the
                            // fixed 2 ms poll: no wasted tail latency when input
                            // arrives, no idle spin when the input stream stops.
                            thread::park_timeout(Duration::from_millis(100));
                            continue;
                        }
                        if capture_input {
                            if let Ok(mut samples) = worker_debug_input.lock() {
                                samples.extend_from_slice(&input_acc[..input_chunk]);
                            }
                        }
                        // Capture raw device input before `LiveParams::input_gain`
                        // or a denoiser changes its level. The return count makes
                        // the final partial chunk's F0 contribution proportional
                        // to the actual microphone duration that was sampled.
                        let calibration_observation = calibration.as_mut().map(|collector| {
                            let observed = collector.observe_audio(&input_acc[..input_chunk]);
                            worker_voice_calibration
                                .mark_collecting(calibration_generation, collector.captured_ms());
                            (observed, collector.is_complete())
                        });
                        let dynamic_mode = worker_dynamic_tuning.mode();
                        // Measure raw device audio before input gain or a
                        // denoiser. The completed RVC chunk supplies the F0
                        // metrics below, so this becomes an allocation-free
                        // full observation after `process_chunk` returns.
                        let dynamic_observation = dynamic_mode.is_enabled().then(|| {
                            DynamicTuningObservation::from_audio(
                                &input_acc[..input_chunk],
                                0.0,
                                0.0,
                            )
                        });
                        let live_params = dynamic_tuner.live_params(dynamic_mode, live.load());
                        let stats = model.process_chunk(
                            &input_acc[..input_chunk],
                            input_rate,
                            &live_params,
                            passthrough_live.load(Ordering::Relaxed),
                            &mut prepared,
                        );
                        input_acc.clear();
                        let Ok(stats) = stats else {
                            worker_running.store(false, Ordering::SeqCst);
                            break;
                        };
                        if let Some(mut observation) = dynamic_observation {
                            observation.voiced_ratio = stats.voiced_ratio;
                            observation.pitch_variation_semitones = stats.pitch_variation_semitones;
                            dynamic_tuner.observe(dynamic_mode, observation);
                        }
                        // A mode switch to Off still refreshes the UI on the
                        // next processed chunk. `try_lock` makes diagnostic
                        // publication lossy rather than stalling inference.
                        worker_dynamic_tuning.publish(dynamic_tuner.snapshot());
                        if let Some((observed_samples, complete)) = calibration_observation {
                            let profile = if observed_samples > 0 {
                                calibration.as_mut().and_then(|collector| {
                                    collector.observe_f0(
                                        stats.voiced_ratio,
                                        stats.pitch_frames,
                                        observed_samples as f32 / input_chunk as f32,
                                    );
                                    complete.then(|| collector.finish())
                                })
                            } else {
                                None
                            };
                            if let Some(profile) = profile {
                                worker_voice_calibration.finish(calibration_generation, profile);
                                calibration = None;
                            }
                        }
                        // Reflect a live model switch in the shared status.
                        // `EngineController::set_active_model` writes the
                        // `active_model` atomic that `Pool::process_chunk` picks
                        // up; without this the GUI panel's highlight (and its
                        // Switch-button logic) would stay on the old model.
                        let requested = worker_active_model.load(Ordering::Relaxed);
                        if requested != last_active_slot {
                            last_active_slot = requested;
                            if let Ok(mut st) = worker_status.lock() {
                                st.active_model = model.active_model_name();
                                st.speaker_count = model.active_speaker_count();
                            }
                        }
                        worker_telemetry.chunks.fetch_add(1, Ordering::Relaxed);
                        worker_telemetry
                            .inference_us
                            .store(stats.inference_time.as_micros() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .embedder_us
                            .store(stats.embedder_time.as_micros() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .pitch_us
                            .store(stats.pitch_time.as_micros() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .rvc_us
                            .store(stats.rvc_time.as_micros() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .f0_voiced_ratio_bits
                            .store(stats.voiced_ratio.to_bits(), Ordering::Relaxed);
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
                        let output_push_len = queued_output_len(
                            output_silent,
                            output_capacity - output_producer.slots(),
                            output_chunk,
                            prepared.len(),
                        );
                        if output_push_len > 0 {
                            let (_, remainder) =
                                output_producer.push_partial_slice(&prepared[..output_push_len]);
                            worker_telemetry
                                .output_dropped_samples
                                .fetch_add(remainder.len() as u64, Ordering::Relaxed);
                        }
                        worker_telemetry.output_buffer_samples.store(
                            (output_capacity - output_producer.slots()) as u64,
                            Ordering::Relaxed,
                        );
                        // Monitor output: resample the converted signal to the
                        // monitor device's rate, apply the live monitor gain, and
                        // queue it. The resampler is fed every chunk (even when
                        // the silence gate below suppresses the push) so its
                        // stateful phase stays aligned with the primary timeline.
                        if let (Some(producer), Some(resampler)) =
                            (monitor_producer.as_mut(), monitor_resampler.as_mut())
                        {
                            monitor_prepared.clear();
                            if resampler
                                .process_into(&prepared, &mut monitor_prepared)
                                .is_err()
                            {
                                monitor_prepared.clear();
                            }
                            let monitor_gain = live_params.monitor_gain.max(0.0);
                            if (monitor_gain - 1.0).abs() > f32::EPSILON {
                                for sample in monitor_prepared.iter_mut() {
                                    *sample = (*sample * monitor_gain).clamp(-1.0, 1.0);
                                }
                            }
                            let monitor_push_len = queued_output_len(
                                output_silent,
                                monitor_capacity - producer.slots(),
                                monitor_chunk,
                                monitor_prepared.len(),
                            );
                            if monitor_push_len > 0 {
                                let (_, remainder) = producer
                                    .push_partial_slice(&monitor_prepared[..monitor_push_len]);
                                worker_telemetry
                                    .monitor_dropped_samples
                                    .fetch_add(remainder.len() as u64, Ordering::Relaxed);
                            }
                        }
                    }
                })?,
        );

        if let Err(err) = activate_candidate_endpoints(
            &endpoint_running,
            || output_stream.play(),
            || match &monitor_stream {
                Some(stream) => stream.play(),
                None => Ok(()),
            },
            || input_stream.play(),
            || Ok(()),
        ) {
            drop(input_stream);
            drop(output_stream);
            drop(monitor_stream);
            stop_startup_worker(&running, &wake, &mut worker);
            return Err(err);
        }

        Ok(Self {
            running,
            endpoint_running,
            wake,
            worker: worker.take(),
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
            monitor_stream,
            status: EngineStatusSnapshot {
                state: EngineState::Running,
                message: format!(
                    "Running (in: {} / out: {})",
                    audio.input_host_label(),
                    audio.output_host_label()
                ),
                detail: None,
                input_device: audio.input_name().to_string(),
                output_device: audio.output_name().to_string(),
                input_sample_rate: input_rate,
                output_sample_rate: output_rate,
                monitor_device: if config.monitor_output_enabled {
                    audio.monitor_name().to_string()
                } else {
                    String::new()
                },
                monitor_sample_rate: if config.monitor_output_enabled {
                    monitor_rate
                } else {
                    0
                },
                passthrough_live_switchable,
                speaker_count,
                active_model: if passthrough_live_switchable {
                    Some(
                        config
                            .model
                            .as_deref()
                            .map(model_name_for)
                            .unwrap_or_else(|| "<base model>".to_string()),
                    )
                } else {
                    None
                },
                // The base model occupies pool slot 0; models added live follow.
                model_loads: if passthrough_live_switchable {
                    vec![ModelLoadStatus {
                        path: config
                            .model
                            .as_deref()
                            .map(|p| p.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        state: ModelLoadState::Loaded,
                        pool_index: Some(0),
                        request_id: BASE_MODEL_REQUEST_ID,
                    }]
                } else {
                    Vec::new()
                },
            },
            // Clone the debug paths so `config` itself can move into the session
            // (retained for live device restarts).
            debug_input_wav: config.debug_input_wav.clone(),
            debug_output_wav: config.debug_output_wav.clone(),
            debug_input,
            debug_output,
            input_rate,
            output_rate,
            worker_tx,
            config,
            input_capacity,
            output_capacity,
            monitor_capacity,
            telemetry,
        })
    }

    fn status(&self) -> EngineStatusSnapshot {
        self.status.clone()
    }

    /// Reconfigure the audio devices without tearing down the worker or model
    /// when the new endpoints resolve to the same sample rates (input/output/
    /// monitor). A rate change invalidates the model-layer objects sized against
    /// the old rates, so it returns `RestartRequired` with a merged config that
    /// keeps the model side and only swaps the device fields.
    fn update_devices(&mut self, dev: DeviceSpec) -> Result<UpdateDevicesOutcome> {
        let audio = RealtimeAudio::open(
            dev.input_host,
            dev.output_host,
            dev.wasapi_input_exclusive,
            dev.wasapi_output_exclusive,
            dev.input_device.as_deref(),
            dev.output_device.as_deref(),
            dev.wasapi_buffer_ms,
            dev.monitor_output_enabled,
            dev.monitor_output_device.as_deref(),
        )?;
        let new_input_rate = audio.input_sample_rate();
        let new_output_rate = audio.output_sample_rate();
        let new_monitor_rate = audio.monitor_sample_rate();
        let monitor_ok = match (dev.monitor_output_enabled, self.monitor_stream.is_some()) {
            (true, true) => new_monitor_rate == self.status.monitor_sample_rate,
            (false, false) => true,
            // Enabling or disabling the monitor changes the rate triple; the
            // worker's monitor resampler is worker-local and sized to it, so a
            // rate change means a full restart.
            _ => false,
        };
        if new_input_rate != self.input_rate || new_output_rate != self.output_rate || !monitor_ok {
            let mut cfg = self.config.clone();
            apply_device_spec(&mut cfg, &dev);
            return Ok(UpdateDevicesOutcome::RestartRequired(cfg));
        }
        let candidate_running = Arc::new(AtomicBool::new(true));
        let StreamEndpoints {
            input_stream,
            output_stream,
            monitor_stream,
            input_consumer,
            output_producer,
            monitor_producer,
        } = build_streams(
            &audio,
            self.input_capacity,
            self.output_capacity,
            dev.monitor_output_enabled.then_some(self.monitor_capacity),
            &candidate_running,
            &self.wake,
            &self.telemetry,
        )?;
        // Candidate callbacks have a private health flag and bounded rings.
        // Start every stream before publishing its ring ends to the worker; a
        // play/send failure then drops only the candidate while the old device
        // timeline remains fully connected.
        activate_candidate_endpoints(
            &candidate_running,
            || output_stream.play(),
            || match &monitor_stream {
                Some(stream) => stream.play(),
                None => Ok(()),
            },
            || input_stream.play(),
            || {
                self.worker_tx
                    .send(WorkerCommand::RebindRings {
                        input_consumer,
                        output_producer,
                        monitor_producer,
                    })
                    .map_err(|_| anyhow!("worker command channel closed"))
            },
        )?;
        self.wake.wake();

        // Sending RebindRings is the commit point: no fallible work follows.
        // Swap health ownership first so a late callback error from an old
        // stream cannot stop the new endpoint set, then retire the old handles.
        let old_endpoint_running = std::mem::replace(&mut self.endpoint_running, candidate_running);
        old_endpoint_running.store(false, Ordering::SeqCst);
        let old_input_stream = self.input_stream.replace(input_stream);
        let old_output_stream = self.output_stream.replace(output_stream);
        let old_monitor_stream = std::mem::replace(&mut self.monitor_stream, monitor_stream);
        drop(old_input_stream);
        drop(old_output_stream);
        drop(old_monitor_stream);
        apply_device_spec(&mut self.config, &dev);
        // Update the session status in place; the state stays Running.
        let status = &mut self.status;
        status.detail = None;
        status.input_device = audio.input_name().to_string();
        status.output_device = audio.output_name().to_string();
        status.input_sample_rate = new_input_rate;
        status.output_sample_rate = new_output_rate;
        status.monitor_device = if dev.monitor_output_enabled {
            audio.monitor_name().to_string()
        } else {
            String::new()
        };
        status.monitor_sample_rate = if dev.monitor_output_enabled {
            new_monitor_rate
        } else {
            0
        };
        status.message = format!(
            "Running (in: {} / out: {})",
            audio.input_host_label(),
            audio.output_host_label()
        );
        Ok(UpdateDevicesOutcome::Swapped)
    }

    /// Owned snapshot of everything the background model loader needs to build a
    /// `ChunkConverter` for a new voice model: the session's model-side config,
    /// current rates/chunks, and the live parameter seed. `denoiser_mode` is the
    /// live value (the user may have hot-swapped it after Apply), so models added
    /// to the pool load with the currently-active denoiser.
    fn model_load_context(
        &self,
        new_model: PathBuf,
        live: &LiveParams,
        denoiser_mode: DenoiserMode,
    ) -> ModelLoadContext {
        let output_extra_ms = self
            .config
            .crossfade_ms
            .saturating_add(self.config.sola_search_ms)
            .saturating_add(self.config.rvc_output_tail_discard_ms);
        ModelLoadContext {
            model: new_model,
            embedder: self.config.embedder.clone().expect("validated"),
            embedder_output: self.config.embedder_output.clone(),
            f0_model: self.config.f0_model.clone(),
            f0_mode: self.config.f0_mode,
            fcpe_model: self.config.fcpe_model.clone(),
            feature_index: self.config.feature_index.clone(),
            provider: self.config.provider,
            gpu_priority: self.config.gpu_priority,
            gpu_device_id: self.config.gpu_device_id,
            sample_rate: self.input_rate,
            chunk_samples: dsp::chunk_samples_for_rate(self.input_rate, self.config.chunk_ms),
            speaker_id: live.speaker_id,
            pitch_shift: live.pitch_shift,
            index_rate: live.index_rate,
            protect: live.protect,
            protect_transition_ms: live.protect_transition_ms,
            denoiser_content_mix: live.denoiser_content_mix,
            denoiser_rmvpe_mix: live.denoiser_rmvpe_mix,
            input_gain: live.input_gain,
            output_gain: live.output_gain,
            f0: self.config.f0.clone(),
            noise_gate_enabled: denoiser_mode == DenoiserMode::NoiseGate,
            silence_gate_enabled: live.silence_gate_enabled,
            noise_gate_threshold: live.noise_gate_threshold,
            noise_gate_shaping: self.config.noise_gate_shaping,
            output_extra_ms,
            volume_excluded_ms: self.config.crossfade_ms,
            extra_convert_ms: self.config.extra_convert_ms,
            rvc_frames: self.config.rvc_frames,
            output_dynamics: self.config.output_dynamics,
            smoother_kind: self.config.smoother.kind(),
            output_rate: self.output_rate,
            output_chunk: dsp::chunk_samples_for_rate(self.output_rate, self.config.chunk_ms),
            crossfade_ms: self.config.crossfade_ms,
            sola_search_ms: self.config.sola_search_ms,
            tail_discard_ms: self.config.rvc_output_tail_discard_ms,
            denoiser: DenoiserLoadSettings::from_realtime(&self.config, denoiser_mode),
        }
    }
}

impl Drop for RealtimeSession {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.endpoint_running.store(false, Ordering::SeqCst);
        // Wake a parked worker so it observes the cleared running flag and exits
        // even when the input stream has already stopped delivering wake()s.
        self.wake.wake();
        drop(self.input_stream.take());
        drop(self.output_stream.take());
        drop(self.monitor_stream.take());
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

fn apply_device_spec(config: &mut RealtimeConfig, dev: &DeviceSpec) {
    config.input_host = dev.input_host;
    config.output_host = dev.output_host;
    config.input_device.clone_from(&dev.input_device);
    config.output_device.clone_from(&dev.output_device);
    config.monitor_output_enabled = dev.monitor_output_enabled;
    config
        .monitor_output_device
        .clone_from(&dev.monitor_output_device);
    config.wasapi_input_exclusive = dev.wasapi_input_exclusive;
    config.wasapi_output_exclusive = dev.wasapi_output_exclusive;
    config.wasapi_buffer_ms = dev.wasapi_buffer_ms;
}

/// Start a candidate endpoint set in dependency order and publish its worker
/// rings only after every stream is healthy. The caller owns the old endpoint
/// flag, so clearing only `candidate_running` makes all pre-commit failures
/// rollback without disturbing the active session.
fn activate_candidate_endpoints<PlayOutput, PlayMonitor, PlayInput, Rebind>(
    candidate_running: &AtomicBool,
    play_output: PlayOutput,
    play_monitor: PlayMonitor,
    play_input: PlayInput,
    rebind: Rebind,
) -> Result<()>
where
    PlayOutput: FnOnce() -> Result<()>,
    PlayMonitor: FnOnce() -> Result<()>,
    PlayInput: FnOnce() -> Result<()>,
    Rebind: FnOnce() -> Result<()>,
{
    let ensure_running = || {
        if candidate_running.load(Ordering::Acquire) {
            Ok(())
        } else {
            bail!("candidate audio endpoint stopped during startup")
        }
    };
    let result = (|| {
        play_output()?;
        ensure_running()?;
        play_monitor()?;
        ensure_running()?;
        play_input()?;
        ensure_running()?;
        // A successful rebind is the commit point. Do not add a fallible step
        // after it: the worker may consume the new ring ends immediately.
        rebind()
    })();
    if result.is_err() {
        candidate_running.store(false, Ordering::SeqCst);
    }
    result
}

/// Short display name for a model path (the file name; falls back to the full
/// path). Used for the model pool's per-slot labels.
fn model_name_for(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

fn queued_output_len(
    output_silent: bool,
    buffered: usize,
    output_chunk: usize,
    prepared_len: usize,
) -> usize {
    if output_silent {
        // Keep exactly one chunk's worth of generated silence buffered. Queue
        // only the missing prefix so quiet processing cannot build latency or
        // displace the first converted speech when input resumes.
        output_chunk.saturating_sub(buffered).min(prepared_len)
    } else {
        prepared_len
    }
}

/// Streams plus the worker-side ring-buffer ends created in the same attempt.
/// The monitor ends are optional (present only when a monitor output is
/// configured); the monitor consumer lives inside the monitor stream callback.
struct StreamEndpoints {
    input_stream: AudioStream,
    output_stream: AudioStream,
    monitor_stream: Option<AudioStream>,
    input_consumer: rtrb::Consumer<f32>,
    output_producer: rtrb::Producer<f32>,
    monitor_producer: Option<rtrb::Producer<f32>>,
}

fn build_streams(
    audio: &RealtimeAudio,
    input_capacity: usize,
    output_capacity: usize,
    monitor_capacity: Option<usize>,
    running: &Arc<AtomicBool>,
    wake: &Arc<WorkerWake>,
    telemetry: &Arc<Telemetry>,
) -> Result<StreamEndpoints> {
    let (mut input_producer, input_consumer) = RingBuffer::<f32>::new(input_capacity);
    let (output_producer, mut output_consumer) = RingBuffer::<f32>::new(output_capacity);
    let input_running = Arc::clone(running);
    let input_wake = Arc::clone(wake);
    let input_telemetry = Arc::clone(telemetry);
    let input_stream = audio.build_input_stream_with_running(running, move |samples| {
        if !input_running.load(Ordering::Relaxed) {
            return;
        }
        let (pushed, remainder) = input_producer.push_partial_slice(samples);
        if !pushed.is_empty() {
            // Notify the worker that input is queued instead of letting it find
            // the data on a fixed poll. Only the input callback wakes the worker;
            // the output callback (consuming the output ring) is not a reason to
            // run inference.
            input_wake.wake();
        }
        if !remainder.is_empty() {
            input_telemetry
                .input_overruns
                .fetch_add(1, Ordering::Relaxed);
        }
    })?;
    let output_running = Arc::clone(running);
    let output_telemetry = Arc::clone(telemetry);
    let output_stream = audio.build_output_stream_with_running(running, move |out| {
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
    let (monitor_stream, monitor_producer) = match monitor_capacity {
        Some(capacity) => {
            let (monitor_producer, mut monitor_consumer) = RingBuffer::<f32>::new(capacity);
            let monitor_running = Arc::clone(running);
            let monitor_telemetry = Arc::clone(telemetry);
            let stream = audio.build_monitor_stream_with_running(running, move |out| {
                if !monitor_running.load(Ordering::Relaxed) {
                    out.fill(0.0);
                    return;
                }
                let (_, remainder) = monitor_consumer.pop_partial_slice(out);
                if !remainder.is_empty() {
                    remainder.fill(0.0);
                    monitor_telemetry
                        .monitor_underruns
                        .fetch_add(1, Ordering::Relaxed);
                }
            })?;
            (Some(stream), Some(monitor_producer))
        }
        None => (None, None),
    };
    Ok(StreamEndpoints {
        input_stream,
        output_stream,
        monitor_stream,
        input_consumer,
        output_producer,
        monitor_producer,
    })
}

fn stop_startup_worker(
    running: &AtomicBool,
    wake: &WorkerWake,
    worker: &mut Option<JoinHandle<()>>,
) {
    // Stream playback can still fail after the inference worker starts. Always
    // stop and join it before returning so failed Apply attempts cannot leave a
    // model/CUDA context alive behind the next session. Wake it first: with no
    // playing input stream it may already be parked waiting for input.
    running.store(false, Ordering::SeqCst);
    wake.wake();
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

    fn test_denoiser_settings(mode: DenoiserMode) -> DenoiserLoadSettings {
        DenoiserLoadSettings::from_realtime(&RealtimeConfig::default(), mode)
    }

    #[test]
    fn live_params_round_trip_through_atomics() {
        let params = LiveParams {
            pitch_shift: -3.5,
            speaker_id: 7,
            f0_threshold: 0.04,
            input_gain: 0.5,
            output_gain: 2.0,
            monitor_gain: 1.5,
            noise_gate_enabled: true,
            silence_gate_enabled: true,
            noise_gate_threshold: 0.025,
            index_rate: 0.7,
            protect: 0.33,
            protect_transition_ms: 20,
            denoiser_content_mix: 0.42,
            denoiser_rmvpe_mix: 0.58,
        };
        let atomic = AtomicLiveParams::new(params);
        let out = atomic.load();
        assert_eq!(out.pitch_shift, params.pitch_shift);
        assert_eq!(out.speaker_id, params.speaker_id);
        assert_eq!(out.f0_threshold, params.f0_threshold);
        assert_eq!(out.input_gain, params.input_gain);
        assert_eq!(out.output_gain, params.output_gain);
        assert_eq!(out.silence_gate_enabled, params.silence_gate_enabled);
        assert_eq!(out.monitor_gain, params.monitor_gain);
        assert_eq!(out.noise_gate_enabled, params.noise_gate_enabled);
        assert_eq!(out.noise_gate_threshold, params.noise_gate_threshold);
        assert_eq!(out.index_rate, params.index_rate);
        assert_eq!(out.protect, params.protect);
        assert_eq!(out.protect_transition_ms, params.protect_transition_ms);
        assert_eq!(out.denoiser_content_mix, params.denoiser_content_mix);
        assert_eq!(out.denoiser_rmvpe_mix, params.denoiser_rmvpe_mix);
    }

    #[test]
    fn atomic_live_params_reject_non_finite_automation() {
        let atomic = AtomicLiveParams::new(LiveParams {
            pitch_shift: f32::NAN,
            input_gain: f32::INFINITY,
            output_gain: -1.0,
            monitor_gain: f32::NEG_INFINITY,
            noise_gate_threshold: 2.0,
            ..LiveParams::default()
        });
        let out = atomic.load();
        assert_eq!(out.pitch_shift, 0.0);
        assert_eq!(out.input_gain, 1.0);
        assert_eq!(out.output_gain, 0.0);
        assert_eq!(out.monitor_gain, 1.0);
        assert_eq!(
            out.noise_gate_threshold,
            vc_core::model_rvc::MAX_NOISE_GATE_THRESHOLD
        );
    }

    #[test]
    fn applied_denoiser_snapshot_round_trips_generation_and_mode_atomically() {
        for mode in [
            DenoiserMode::Off,
            DenoiserMode::NoiseGate,
            DenoiserMode::Rnnoise,
            DenoiserMode::Gtcrn,
            DenoiserMode::WebRtc,
            DenoiserMode::DeepFilterNet3,
        ] {
            let snapshot = AppliedDenoiserSnapshot {
                generation: 123_456,
                mode,
            };
            assert_eq!(
                unpack_applied_denoiser(pack_applied_denoiser(snapshot)),
                snapshot
            );
            let state = AppliedDenoiserState::new(snapshot);
            assert_eq!(state.load(), snapshot);
        }
    }

    #[test]
    fn dynamic_model_ids_never_collide_with_the_base_id() {
        let mut next = FIRST_DYNAMIC_MODEL_REQUEST_ID;
        assert_eq!(take_dynamic_model_request_id(&mut next), 1);
        assert_eq!(take_dynamic_model_request_id(&mut next), 2);
        assert_ne!(next.get(), BASE_MODEL_REQUEST_ID);
    }

    #[test]
    fn model_status_removal_protects_base_and_supports_dynamic_slot_zero() {
        let mut with_base = EngineStatusSnapshot {
            model_loads: vec![
                ModelLoadStatus {
                    path: "base.onnx".to_string(),
                    state: ModelLoadState::Loaded,
                    pool_index: Some(0),
                    request_id: BASE_MODEL_REQUEST_ID,
                },
                ModelLoadStatus {
                    path: "dynamic.onnx".to_string(),
                    state: ModelLoadState::Loaded,
                    pool_index: Some(1),
                    request_id: 1,
                },
            ],
            ..EngineStatusSnapshot::default()
        };
        assert_eq!(
            remove_dynamic_model_status(&mut with_base, BASE_MODEL_REQUEST_ID),
            None
        );
        assert_eq!(with_base.model_loads.len(), 2);
        assert_eq!(remove_dynamic_model_status(&mut with_base, 1), Some(1));
        assert_eq!(with_base.model_loads.len(), 1);
        assert_eq!(with_base.model_loads[0].request_id, BASE_MODEL_REQUEST_ID);

        let mut without_base = EngineStatusSnapshot {
            model_loads: vec![ModelLoadStatus {
                path: "first-dynamic.onnx".to_string(),
                state: ModelLoadState::Loaded,
                pool_index: Some(0),
                request_id: 1,
            }],
            ..EngineStatusSnapshot::default()
        };
        assert_eq!(remove_dynamic_model_status(&mut without_base, 1), Some(0));
        assert!(without_base.model_loads.is_empty());
    }

    #[test]
    fn dense_pool_removal_allows_the_only_dynamic_slot_zero() {
        let mut models = vec![7_u8];
        let mut names = vec!["dynamic".to_string()];
        assert!(remove_aligned_model_slot(&mut models, &mut names, 0));
        assert!(models.is_empty());
        assert!(names.is_empty());
        assert!(!remove_aligned_model_slot(&mut models, &mut names, 0));
    }

    #[test]
    fn pool_slot_remap_preserves_a_concurrent_frontend_selection() {
        assert_eq!(remap_slot_after_removal(0, 1, 2), 0);
        assert_eq!(remap_slot_after_removal(1, 1, 2), 1);
        assert_eq!(remap_slot_after_removal(2, 1, 2), 1);

        let requested = AtomicUsize::new(2);
        assert!(try_remap_requested_slot_after_removal(&requested, 2, 1, 2));
        assert_eq!(requested.load(Ordering::Relaxed), 1);

        requested.store(2, Ordering::Relaxed);
        assert!(!try_remap_requested_slot_after_removal(&requested, 1, 1, 2));
        assert_eq!(requested.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn adding_a_non_active_model_does_not_overwrite_requested_slot() {
        let requested = AtomicUsize::new(2);
        assert!(!try_publish_active_slot_after_add(&requested, 2, 0, false));
        assert_eq!(requested.load(Ordering::Relaxed), 2);

        assert!(try_publish_active_slot_after_add(&requested, 2, 3, true));
        assert_eq!(requested.load(Ordering::Relaxed), 3);
        requested.store(4, Ordering::Relaxed);
        assert!(!try_publish_active_slot_after_add(&requested, 3, 1, true));
        assert_eq!(requested.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn removing_the_final_pool_model_restores_passthrough_variant() {
        let live = LiveParams::default();
        let passthrough = PassthroughProcessor::new(
            NoiseGateShaping::default(),
            48_000,
            48_000,
            test_denoiser_settings(DenoiserMode::Off),
            &live,
        )
        .unwrap();
        let active_requested = Arc::new(AtomicUsize::new(0));
        let model = RuntimeModel::Pool {
            passthrough,
            models: Vec::new(),
            names: Vec::new(),
            active: 0,
            passthrough_active: false,
            active_requested: Arc::clone(&active_requested),
        };
        let model = model.remove_model(0);
        assert!(matches!(model, RuntimeModel::PassthroughOnly(_)));
        assert_eq!(active_requested.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cancelled_model_load_is_not_live_for_worker_add() {
        let mut status = EngineStatusSnapshot::default();
        status.model_loads.push(ModelLoadStatus {
            path: "pending.onnx".to_string(),
            state: ModelLoadState::Loading("loading".to_string()),
            pool_index: None,
            request_id: 41,
        });
        assert!(model_load_request_is_live(&status, 41));

        status.model_loads.clear();
        assert!(!model_load_request_is_live(&status, 41));
    }

    #[test]
    fn calibration_control_rejects_a_result_from_a_replaced_request() {
        let control = VoiceCalibrationControl::default();
        let first = control.start(2_000);
        control.mark_collecting(first, 750);
        assert_eq!(control.snapshot().state, VoiceCalibrationState::Collecting);

        let second = control.start(2_000);
        let stale = VoiceCalibrationProfile {
            captured_ms: 2_000,
            speech_rms: 0.05,
            ..VoiceCalibrationProfile::default()
        };
        control.finish(first, stale);
        let pending = control.snapshot();
        assert_eq!(pending.generation, second);
        assert_eq!(pending.state, VoiceCalibrationState::Requested);
        assert!(pending.profile.is_none());

        control.finish(second, stale);
        let complete = control.snapshot();
        assert_eq!(complete.state, VoiceCalibrationState::Ready);
        assert_eq!(complete.profile, Some(stale));

        control.cancel();
        assert_eq!(control.snapshot().state, VoiceCalibrationState::Idle);
    }

    #[test]
    fn dynamic_tuning_control_resets_diagnostics_without_changing_mode() {
        let control = DynamicTuningControl::default();
        control.set_mode(DynamicTuningMode::Japanese);
        assert_eq!(control.mode(), DynamicTuningMode::Japanese);
        assert_eq!(control.snapshot().profile, DynamicLanguageProfile::Japanese);

        control.publish(DynamicTuningSnapshot {
            mode: DynamicTuningMode::Japanese,
            profile: DynamicLanguageProfile::Japanese,
            confidence: 1.0,
            noise_floor_rms: 0.02,
            estimated_snr_db: 8.0,
        });
        control.reset_snapshot();

        let snapshot = control.snapshot();
        assert_eq!(snapshot.mode, DynamicTuningMode::Japanese);
        assert_eq!(snapshot.profile, DynamicLanguageProfile::Japanese);
        assert_eq!(snapshot.confidence, 1.0);
        assert_eq!(snapshot.noise_floor_rms, 0.0);
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
    fn monitor_output_rejected_with_asio_output_host() {
        assert!(RealtimeConfig {
            passthrough: true,
            monitor_output_enabled: true,
            output_host: AudioHost::Asio,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            monitor_output_enabled: true,
            ..Default::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn validation_rejects_out_of_range_conversion_timing() {
        assert!(RealtimeConfig {
            passthrough: true,
            chunk_ms: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            chunk_ms: CONVERSION_TIMING_LIMITS.min_chunk_ms - 1,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            chunk_ms: CONVERSION_TIMING_LIMITS.max_chunk_ms + 1,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            extra_convert_ms: CONVERSION_TIMING_LIMITS.min_extra_convert_ms - 1,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(RealtimeConfig {
            passthrough: true,
            extra_convert_ms: CONVERSION_TIMING_LIMITS.max_extra_convert_ms + 1,
            ..Default::default()
        }
        .validate()
        .is_err());
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
    fn hybrid_model_set_requires_fcpe_before_live_switching() {
        let model = PathBuf::from("model.onnx");
        let hybrid_without_fcpe = RealtimeConfig {
            model: Some(model.clone()),
            embedder: Some(model.clone()),
            f0_model: Some(model.clone()),
            f0_mode: F0Mode::Hybrid,
            ..Default::default()
        };
        assert!(!hybrid_without_fcpe.has_complete_model_set());
        assert!(hybrid_without_fcpe.validate().is_err());

        let hybrid = RealtimeConfig {
            fcpe_model: Some(model.clone()),
            ..hybrid_without_fcpe
        };
        assert!(hybrid.has_complete_model_set());
        assert!(hybrid.validate().is_ok());
    }

    #[test]
    fn fcpe_model_set_does_not_require_rmvpe() {
        let model = PathBuf::from("model.onnx");
        let fcpe = RealtimeConfig {
            model: Some(model.clone()),
            embedder: Some(model.clone()),
            f0_model: None,
            f0_mode: F0Mode::Fcpe,
            fcpe_model: Some(model.clone()),
            ..Default::default()
        };
        assert!(fcpe.has_complete_model_set());
        assert!(fcpe.validate().is_ok());

        let missing_fcpe = RealtimeConfig {
            fcpe_model: None,
            ..fcpe
        };
        assert!(!missing_fcpe.has_complete_model_set());
        assert!(missing_fcpe.validate().is_err());
    }

    #[test]
    fn passthrough_processor_applies_input_and_output_gain() {
        let live = LiveParams {
            input_gain: 2.0,
            output_gain: 2.0,
            ..Default::default()
        };
        let mut processor = PassthroughProcessor::new(
            NoiseGateShaping::default(),
            48_000,
            48_000,
            test_denoiser_settings(DenoiserMode::Off),
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
            NoiseGateShaping {
                attack_ms: 0.0,
                release_ms: 0.0,
                floor: 0.0,
            },
            48_000,
            48_000,
            test_denoiser_settings(DenoiserMode::NoiseGate),
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
            NoiseGateShaping::default(),
            48_000,
            48_000,
            test_denoiser_settings(DenoiserMode::Rnnoise),
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
    fn silent_output_tops_up_but_never_exceeds_one_chunk() {
        assert_eq!(queued_output_len(true, 0, 1_000, 1_000), 1_000);
        assert_eq!(queued_output_len(true, 400, 1_000, 1_000), 600);
        assert_eq!(queued_output_len(true, 1_000, 1_000, 1_000), 0);
        assert_eq!(queued_output_len(true, 1_001, 1_000, 1_000), 0);
        assert_eq!(queued_output_len(false, 1_000, 1_000, 1_000), 1_000);
    }

    #[test]
    fn denoiser_swap_requires_an_exact_model_count() {
        assert!(denoiser_set_matches_model_count(0, 0));
        assert!(denoiser_set_matches_model_count(3, 3));
        assert!(!denoiser_set_matches_model_count(2, 3));
        assert!(!denoiser_set_matches_model_count(4, 3));
    }

    #[test]
    fn device_status_patch_preserves_worker_owned_model_state() {
        let mut current = EngineStatusSnapshot {
            state: EngineState::Running,
            message: "old devices".to_string(),
            input_device: "old input".to_string(),
            output_device: "old output".to_string(),
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            speaker_count: Some(308),
            active_model: Some("dynamic.onnx".to_string()),
            model_loads: vec![ModelLoadStatus {
                path: "dynamic.onnx".to_string(),
                state: ModelLoadState::Loaded,
                pool_index: Some(1),
                request_id: 7,
            }],
            ..EngineStatusSnapshot::default()
        };
        let device = EngineStatusSnapshot {
            state: EngineState::Error,
            message: "new devices".to_string(),
            input_device: "new input".to_string(),
            output_device: "new output".to_string(),
            input_sample_rate: 48_000,
            output_sample_rate: 48_000,
            ..EngineStatusSnapshot::default()
        };

        patch_device_status(&mut current, &device);

        assert_eq!(current.state, EngineState::Running);
        assert_eq!(current.input_device, "new input");
        assert_eq!(current.output_device, "new output");
        assert_eq!(current.speaker_count, Some(308));
        assert_eq!(current.active_model.as_deref(), Some("dynamic.onnx"));
        assert_eq!(current.model_loads.len(), 1);
        assert_eq!(current.model_loads[0].request_id, 7);
    }

    #[test]
    fn failed_device_candidate_keeps_running_session_status() {
        let status = Mutex::new(EngineStatusSnapshot {
            state: EngineState::Running,
            input_device: "working input".to_string(),
            output_device: "working output".to_string(),
            model_loads: vec![ModelLoadStatus {
                path: "base.onnx".to_string(),
                state: ModelLoadState::Loaded,
                pool_index: Some(0),
                request_id: BASE_MODEL_REQUEST_ID,
            }],
            ..EngineStatusSnapshot::default()
        });
        set_recoverable_error(&status, "candidate failed", &anyhow!("test playback error"));
        let status = status.into_inner().unwrap();
        assert_eq!(status.state, EngineState::Running);
        assert_eq!(status.input_device, "working input");
        assert_eq!(status.output_device, "working output");
        assert_eq!(status.model_loads.len(), 1);
        assert_eq!(status.message, "candidate failed");
        assert!(status.detail.unwrap().contains("test playback error"));
    }

    #[test]
    fn spontaneous_endpoint_stop_invalidates_async_denoiser_loads() {
        let worker_running = AtomicBool::new(true);
        let endpoint_running = AtomicBool::new(false);
        let generation = AtomicU64::new(9);
        assert_eq!(
            stopped_session_message_and_invalidate(&worker_running, &endpoint_running, &generation,),
            Some("Realtime audio endpoint stopped")
        );
        assert_eq!(generation.load(Ordering::Acquire), 10);
    }

    #[test]
    fn candidate_endpoints_publish_rings_only_after_all_streams_start() {
        let running = AtomicBool::new(true);
        let order = std::cell::RefCell::new(Vec::new());
        activate_candidate_endpoints(
            &running,
            || {
                order.borrow_mut().push("output");
                Ok(())
            },
            || {
                order.borrow_mut().push("monitor");
                Ok(())
            },
            || {
                order.borrow_mut().push("input");
                Ok(())
            },
            || {
                order.borrow_mut().push("rebind");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(order.into_inner(), ["output", "monitor", "input", "rebind"]);
        assert!(running.load(Ordering::Acquire));
    }

    #[test]
    fn failed_candidate_start_does_not_publish_rings() {
        let running = AtomicBool::new(true);
        let rebind_called = AtomicBool::new(false);
        let result = activate_candidate_endpoints(
            &running,
            || Ok(()),
            || Err(anyhow!("monitor failed")),
            || Ok(()),
            || {
                rebind_called.store(true, Ordering::Relaxed);
                Ok(())
            },
        );
        assert!(result.is_err());
        assert!(!running.load(Ordering::Acquire));
        assert!(!rebind_called.load(Ordering::Relaxed));
    }

    #[test]
    fn worker_wake_token_persists_when_unpark_precedes_park() {
        // A wake() that lands before the worker parks must not be lost: the next
        // park returns immediately on the stored token.
        let wake = WorkerWake::default();
        wake.register_current();
        wake.wake();
        let start = std::time::Instant::now();
        thread::park_timeout(Duration::from_secs(5));
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn accumulate_input_chunk_coalesces_subchunk_blocks() {
        let (mut producer, mut consumer) = RingBuffer::<f32>::new(16);
        let mut acc = Vec::new();
        let _ = producer.push_partial_slice(&[1.0, 2.0]);
        assert!(!accumulate_input_chunk(&mut consumer, &mut acc, 4));
        assert_eq!(acc, vec![1.0, 2.0]);
        let _ = producer.push_partial_slice(&[3.0, 4.0]);
        assert!(accumulate_input_chunk(&mut consumer, &mut acc, 4));
        assert_eq!(acc, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn accumulate_input_chunk_carries_oversized_block_remainder() {
        // A callback block larger than one chunk must not lose its tail: only the
        // chunk-sized deficit is read, the rest stays in the ring for next time.
        let (mut producer, mut consumer) = RingBuffer::<f32>::new(16);
        let mut acc = Vec::new();
        let _ = producer.push_partial_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(accumulate_input_chunk(&mut consumer, &mut acc, 4));
        assert_eq!(acc, vec![1.0, 2.0, 3.0, 4.0]);
        acc.clear();
        assert!(!accumulate_input_chunk(&mut consumer, &mut acc, 4));
        assert_eq!(acc, vec![5.0, 6.0]);
    }

    #[test]
    fn accumulate_input_chunk_processes_backlog_without_waiting() {
        // Two chunks queued at once must both complete on back-to-back calls so
        // the worker drains a backlog instead of parking between chunks.
        let (mut producer, mut consumer) = RingBuffer::<f32>::new(16);
        let mut acc = Vec::new();
        let _ = producer.push_partial_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(accumulate_input_chunk(&mut consumer, &mut acc, 4));
        acc.clear();
        assert!(accumulate_input_chunk(&mut consumer, &mut acc, 4));
        assert_eq!(acc, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn stop_wakes_a_parked_worker_within_bounded_time() {
        // With no input arriving, a stop request must still terminate the worker:
        // running=false + wake() releases the park.
        let running = Arc::new(AtomicBool::new(true));
        let wake = Arc::new(WorkerWake::default());
        let worker_running = Arc::clone(&running);
        let worker_wake = Arc::clone(&wake);
        let mut worker = Some(thread::spawn(move || {
            worker_wake.register_current();
            worker_wake.wake();
            while worker_running.load(Ordering::SeqCst) {
                // No ring here: emulate the "no full chunk" branch directly.
                if !worker_running.load(Ordering::SeqCst) {
                    break;
                }
                thread::park_timeout(Duration::from_secs(5));
            }
        }));
        let start = std::time::Instant::now();
        stop_startup_worker(&running, &wake, &mut worker);
        assert!(start.elapsed() < Duration::from_secs(4));
    }
}
