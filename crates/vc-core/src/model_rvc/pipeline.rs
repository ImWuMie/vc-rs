use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracing::{debug, info};

use crate::dsp;
use crate::Provider;

use super::api::{ModelOutput, VoiceModel};
use super::f0_hybrid::HybridF0Fusion;
use super::f0_postprocess::{F0PostprocessConfig, F0Postprocessor};
use super::feature::FeatureTensor;
use super::feature_index::FeatureIndex;
use super::inspect::{inspect_contentvec_input_name, inspect_rvc_model};
use super::native_tensorrt::native_engine_is_cached;
#[cfg(feature = "gtcrn")]
use super::native_tensorrt::native_gtcrn_engine_is_cached;
use super::pitch::{
    align_pitchf_to_features_into, center_crop_pitchf_to_features_into, coarse_pitch_into,
    pitchf_tail_for_output_into, voiced_ratio,
};
use super::sessions::{
    FcpePitchSession, HubertEmbedderSession, RmvpePitchSession, RvcModelSession,
};
use super::shape::{
    extra_convert_samples_from_ms, keep_tail_in_place, ms_to_samples,
    onnx_silence_front_feature_frames, resolve_rvc_context_samples_16k,
    rmvpe_model_input_samples_for_context_16k, tensor_rt_model_input_samples_16k, RVC_SAMPLE_RATE,
};
use super::speech_activity::{OutputSilenceEnvelope, SpeechActivityDetector};
use super::stream::{RvcStreamState, SampleDelay, StreamInputTiming};

/// RMVPE confidence threshold used by upstream realtime RVC and MXGF. The old
/// vc-rs default of 0.3 discarded substantially more low-confidence voiced
/// frames and made large upward pitch shifts sound intermittent or grainy.
pub const DEFAULT_F0_THRESHOLD: f32 = 0.03;
/// Standard RVC protect value. `0.33` keeps roughly two thirds of the original
/// ContentVec signal on unvoiced frames while leaving voiced frames fully
/// retrieved; this preserves fricatives and breaths when index retrieval is on.
pub const DEFAULT_PROTECT: f32 = 0.33;
/// Standard RVC's `protect` upper bound. A value of `0.5` disables consonant
/// protection; smaller values retain progressively more original ContentVec on
/// unvoiced frames.
pub const MAX_PROTECT: f32 = 0.5;
/// Optional vc-rs extension: smooth the retrieved/original ContentVec blend on
/// voiced frames beside an unvoiced boundary. Zero preserves RVC's binary
/// protect mask exactly.
pub const DEFAULT_PROTECT_TRANSITION_MS: u32 = 0;
/// Cap the optional protect-boundary ramp so live automation has a bounded
/// worker cost and cannot spread consonant protection across an entire chunk.
pub const MAX_PROTECT_TRANSITION_MS: u32 = 100;
/// Keep live pitch automation within a finite, musically useful range. The
/// frontends expose +/-24 semitones, while the wider +/-48 range preserves
/// compatibility with existing CLI configurations without allowing `powf` in
/// F0 processing to overflow on arbitrary host input.
pub const MIN_PITCH_SHIFT_SEMITONES: f32 = -48.0;
pub const MAX_PITCH_SHIFT_SEMITONES: f32 = 48.0;
/// Live gain values are applied to normalized audio and then hard-clipped. A
/// 64x ceiling corresponds to roughly +36 dB (the VST parameter limit) and
/// prevents malformed automation from producing non-finite intermediate data.
pub const MAX_LIVE_GAIN: f32 = 64.0;
/// Audio RMS and gate thresholds operate on normalized samples, so values
/// above one have no meaningful interpretation.
pub const MAX_NOISE_GATE_THRESHOLD: f32 = 1.0;
const RVC_FEATURE_FRAME_MS: u32 = 10;
/// Default base share of the denoised signal mixed into ContentVec when a
/// denoiser is active. The worker can reduce it briefly for speech transients;
/// RMVPE has its own smaller independent adaptation.
pub const DEFAULT_DENOISER_CONTENT_MIX: f32 = 0.25;
/// Keep the residual blend bounded: 0 is exact raw ContentVec input, while 1 is
/// the maximum cleaned share before speech-transient protection is applied.
pub const MAX_DENOISER_CONTENT_MIX: f32 = 1.0;
/// Default base share of the denoised branch sent to RMVPE. Keep the base at
/// full denoise for compatibility; low-ZCR voiced onsets may retain a small raw
/// share so the denoiser cannot erase their first pitch periods.
pub const DEFAULT_DENOISER_RMVPE_MIX: f32 = 1.0;
pub const MAX_DENOISER_RMVPE_MIX: f32 = 1.0;

/// F0 backend selection. Sessions and fixed GPU resources are selected and
/// built only while loading the pipeline; switching modes requires a reload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum F0Mode {
    #[default]
    Rmvpe,
    Fcpe,
    Hybrid,
}

impl F0Mode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Rmvpe => "rmvpe",
            Self::Fcpe => "fcpe",
            Self::Hybrid => "hybrid",
        }
    }

    pub const fn uses_rmvpe(self) -> bool {
        matches!(self, Self::Rmvpe | Self::Hybrid)
    }

    pub const fn uses_fcpe(self) -> bool {
        matches!(self, Self::Fcpe | Self::Hybrid)
    }
}
use super::tensorrt::{
    derive_rvc_feature_len, provider_uses_fixed_shape, tensor_rt_model_cache_key,
    validate_rvc_static_profile_frames, ModelRole, TensorRtRunMode, TensorRtSessionProfile,
    TensorRtSessionPurpose, CUDA_GRAPH_ENV,
};
#[cfg(feature = "ort")]
use super::tensorrt::{tensor_rt_warmup_feature_len, TensorRtSharedWaveform};

/// Device-rate input denoising stage, applied after `input_gain` to the cleaned
/// branch. ContentVec and RMVPE each keep a separately delayed raw branch and
/// receive their own configured denoised share. Keep future denoisers behind
/// variants and match arms here so the timing-sensitive call site in `process()`
/// stays centralized.
enum InputDenoiser {
    Off,
    Gate(dsp::NoiseGate),
    #[cfg(feature = "rnnoise")]
    Rnnoise(Box<crate::denoise::RnnoiseDenoiser>),
    #[cfg(feature = "webrtc")]
    WebRtc(Box<crate::denoise::WebRtcDenoiser>),
    #[cfg(feature = "deepfilternet3")]
    DeepFilterNet3(Box<crate::denoise::DeepFilterNet3Denoiser>),
}

/// Input denoiser variant for live hot-swapping on `RvcPipeline`. Distinct from
/// vc-app's front-end `DenoiserMode` (which also covers gate shaping), so the
/// engine can convert between them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputDenoiserMode {
    Off,
    Gate,
    Rnnoise,
    Gtcrn,
    WebRtc,
    DeepFilterNet3,
}

impl InputDenoiser {
    fn is_stateful(&self) -> bool {
        match self {
            Self::Off | Self::Gate(_) => false,
            #[cfg(feature = "rnnoise")]
            Self::Rnnoise(_) => true,
            #[cfg(feature = "webrtc")]
            Self::WebRtc(_) => true,
            #[cfg(feature = "deepfilternet3")]
            Self::DeepFilterNet3(_) => true,
        }
    }

    fn process_in_place(&mut self, buf: &mut [f32]) -> Result<()> {
        match self {
            InputDenoiser::Off => {}
            InputDenoiser::Gate(gate) => gate.process_in_place(buf),
            #[cfg(feature = "rnnoise")]
            InputDenoiser::Rnnoise(denoiser) => denoiser.process_in_place(buf)?,
            #[cfg(feature = "webrtc")]
            InputDenoiser::WebRtc(denoiser) => denoiser.process_in_place(buf)?,
            #[cfg(feature = "deepfilternet3")]
            InputDenoiser::DeepFilterNet3(denoiser) => denoiser.process_in_place(buf)?,
        }
        Ok(())
    }

    fn reset(&mut self, sample_rate: f32, threshold: f32, shaping: NoiseGateShaping) -> Result<()> {
        *self = match self {
            InputDenoiser::Off => InputDenoiser::Off,
            InputDenoiser::Gate(_) => InputDenoiser::Gate(dsp::NoiseGate::new(
                sample_rate,
                threshold,
                shaping.attack_ms,
                shaping.release_ms,
                shaping.floor,
            )),
            #[cfg(feature = "rnnoise")]
            InputDenoiser::Rnnoise(denoiser) => {
                denoiser.reset()?;
                return Ok(());
            }
            #[cfg(feature = "webrtc")]
            InputDenoiser::WebRtc(denoiser) => {
                denoiser.reset()?;
                return Ok(());
            }
            #[cfg(feature = "deepfilternet3")]
            InputDenoiser::DeepFilterNet3(denoiser) => {
                denoiser.reset()?;
                return Ok(());
            }
        };
        Ok(())
    }

    /// Gate output is sample-aligned; model denoisers are emitted through
    /// fixed-delay adapters. The raw ContentVec branch must be delayed by the
    /// same amount before residual mixing, otherwise it contains a second,
    /// earlier voice.
    fn content_delay_samples(&self) -> usize {
        match self {
            Self::Off | Self::Gate(_) => 0,
            #[cfg(feature = "rnnoise")]
            Self::Rnnoise(denoiser) => denoiser.latency_samples(),
            #[cfg(feature = "webrtc")]
            Self::WebRtc(denoiser) => denoiser.latency_samples(),
            #[cfg(feature = "deepfilternet3")]
            Self::DeepFilterNet3(denoiser) => denoiser.latency_samples(),
        }
    }
}

fn build_input_denoiser(config: &RvcPipelineConfig<'_>) -> InputDenoiser {
    if config.noise_gate_enabled {
        InputDenoiser::Gate(dsp::NoiseGate::new(
            config.sample_rate as f32,
            normalized_noise_gate_threshold(config.noise_gate_threshold),
            config.noise_gate_shaping.attack_ms,
            config.noise_gate_shaping.release_ms,
            config.noise_gate_shaping.floor,
        ))
    } else {
        InputDenoiser::Off
    }
}

pub struct RvcPipeline {
    embedder: HubertEmbedderSession,
    f0_mode: F0Mode,
    /// Pitch sessions are optional individually but their presence is fixed by
    /// `f0_mode` at load time. Never create or swap them from `process`: fixed
    /// profiles, bindings, and CUDA Graphs belong outside the realtime seam.
    pitch: Option<RmvpePitchSession>,
    fcpe: Option<FcpePitchSession>,
    hybrid_fusion: HybridF0Fusion,
    rvc: RvcModelSession,
    #[cfg(feature = "ort")]
    shared_waveform: Option<TensorRtSharedWaveform>,
    // Derived once from the generator's `emb_g.weight` initializer. Live host
    // automation is clamped against it before inference so an out-of-range ID
    // cannot tear down a realtime session.
    speaker_count: Option<usize>,
    speaker_id: i64,
    pitch_shift: f32,
    /// Optional RVC target-speaker retrieval index. It is loaded before the
    /// pipeline enters the worker and queried only from that worker, never from
    /// an audio callback.
    feature_index: Option<FeatureIndex>,
    index_rate: f32,
    protect: f32,
    protect_transition_ms: u32,
    f0_threshold: f32,
    silence_threshold: f32,
    input_gain: f32,
    // Input noise reduction. `input_denoiser` is the active stage (Off when
    // disabled); the remaining fields let `set_noise_gate` rebuild the gate
    // when it is toggled back on without a full pipeline reload.
    input_denoiser: InputDenoiser,
    // This output-side silence suppressor intentionally does not alter model
    // input or inference scheduling. RVC can emit a target-voice noise floor
    // for a quiet source, and stateful input denoisers cannot be replaced by the
    // legacy input gate at runtime. Keep this as an independent live flag.
    silence_gate_enabled: bool,
    // Adaptive source-activity state for the output-only silence suppressor.
    // It stays alongside the model stream state on the conversion worker and
    // never feeds a callback or changes inference scheduling.
    speech_activity: SpeechActivityDetector,
    // Model output needs a cross-chunk ramp after the activity decision flips.
    // Keep it on this shared worker path so frontend output rings never receive
    // a click-causing hard zero and every host gets identical behavior.
    output_silence_envelope: OutputSilenceEnvelope,
    // Device-rate counterpart to the stream state's GTCRN delay. It aligns raw
    // speech with model-denoiser output before both branches are resampled.
    content_raw_delay: SampleDelay,
    denoiser_content_mix: f32,
    denoiser_rmvpe_mix: f32,
    noise_gate_threshold: f32,
    noise_gate_attack_ms: f32,
    noise_gate_release_ms: f32,
    noise_gate_floor: f32,
    noise_gate_sample_rate: f32,
    output_extra_ms: u32,
    volume_excluded_ms: u32,
    // The RVC model's native output sample rate (metadata `samplingRate`, default
    // RVC_SAMPLE_RATE). All convert/output-window sizing and the reported output
    // rate use this so non-48 kHz models (e.g. 32 kHz) are not mis-sized.
    rvc_sample_rate: u32,
    extra_convert_samples: usize,
    /// Load-time selected generator length, checked against the actual
    /// ContentVec output before every custom-T inference. This is a guardrail
    /// for nonstandard embedders; fixed TensorRT backends additionally validate
    /// it while constructing their profile and CUDA graph.
    requested_rvc_frames: Option<usize>,
    rmvpe_input_samples_16k: usize,
    output_gain: f32,
    volume_envelope: bool,
    rms_mix_rate: f32,
    auto_output_gain: bool,
    target_output_rms: f32,
    max_output_gain: f32,
    stream_state: RvcStreamState,
    // Absolute generator-frame timeline for exported `rnd` inputs. The end
    // coordinate advances only by newly appended 10 ms frames; replayed rolling
    // context therefore receives the same latent value on every inference.
    rnd_timeline: RvcNoiseTimeline,
    // Reused per-chunk buffer for the gain-scaled / denoised input, so `process`
    // does not allocate a fresh Vec every chunk when input_gain != 1.0 or a
    // denoiser is active. Empty when the zero-copy (gain==1.0, denoiser-off) path
    // is taken.
    input_scratch: Vec<f32>,
    // Reused denoised copy for the RMVPE branch. Keeping the raw ContentVec
    // buffer and this copy separate is the point of the dual-path design. The
    // raw buffer may be delayed in place to match a fixed-delay denoiser.
    denoiser_scratch: Vec<f32>,
    // Reused embedder output tensor, refilled in place each chunk by
    // `extract_into` so the per-chunk ContentVec output is not reallocated.
    feature_tensor: FeatureTensor,
    // Original ContentVec features saved only while RVC `protect` is active.
    // It follows the exact same trim/repeat operations as `feature_tensor`, so
    // its frames remain aligned with the generator's 10 ms F0 grid.
    original_feature_tensor: FeatureTensor,
    input_reference_scratch: Vec<f32>,
    rms_mix_scratch: dsp::RmsMixScratch,
    // Natural RMVPE F0 after the optional worker-side waveform periodicity
    // check. This feeds the rolling raw-F0 timeline before any continuity or
    // pitch shift, so retrieval, Protect, VAD, and synthesis agree on voicing.
    pitchf_validated_scratch: Vec<f32>,
    pitchf_untrimmed_scratch: Vec<f32>,
    pitchf_scratch: Vec<f32>,
    pitch_scratch: Vec<i64>,
    f0_postprocess: F0Postprocessor,
    // Non-destructive output of `process_pitchf_into`: `pitchf_scratch` holds the
    // aligned raw F0 input, so the post-processed result needs its own buffer.
    pitchf_postprocessed_scratch: Vec<f32>,
}

/// Post-conversion output level shaping, applied after inference.
///
/// Grouped because every front-end carries the same five knobs verbatim and
/// passes them straight through `RealtimeConfig` into `RvcPipelineConfig`;
/// keeping them as one unit means adding an output-level knob touches the
/// struct, not each front-end's field-by-field config mapping. `output_gain`
/// is deliberately *not* here: it is a live (per-block) parameter, not static
/// load-time config.
#[derive(Clone, Copy, Debug)]
pub struct OutputDynamicsConfig {
    /// Match the converted output's short-term envelope to the input's.
    pub volume_envelope: bool,
    /// Blend ratio (0..=1) for mixing input RMS back into the output level.
    pub rms_mix_rate: f32,
    /// Automatically scale output toward `target_output_rms`.
    pub auto_output_gain: bool,
    pub target_output_rms: f32,
    pub max_output_gain: f32,
}

impl Default for OutputDynamicsConfig {
    fn default() -> Self {
        Self {
            volume_envelope: false,
            rms_mix_rate: 0.0,
            auto_output_gain: false,
            target_output_rms: 0.03,
            max_output_gain: 512.0,
        }
    }
}

/// Static (load-time) shaping for the input noise gate.
///
/// Same rationale as [`OutputDynamicsConfig`]: every front-end carries these
/// three knobs verbatim and passes them straight through, so grouping them
/// means adding a gate-shaping knob touches the struct, not each front-end's
/// field-by-field mapping. The gate's `enabled`/`threshold` are deliberately
/// *not* here: they are live (per-block) parameters, kept as separate
/// initial-value fields on `RvcPipelineConfig`. Attack/release/floor shape the
/// smoothing coefficients fixed when the gate is constructed.
#[derive(Clone, Copy, Debug)]
pub struct NoiseGateShaping {
    pub attack_ms: f32,
    pub release_ms: f32,
    pub floor: f32,
}

impl Default for NoiseGateShaping {
    fn default() -> Self {
        Self {
            attack_ms: 5.0,
            release_ms: 50.0,
            floor: 0.0,
        }
    }
}

/// Static (load-time) F0 configuration.
///
/// Same grouping rationale as [`OutputDynamicsConfig`]: these knobs travel
/// together from every front-end through `RealtimeConfig` into
/// `RvcPipelineConfig`. Front-ends expose the narrowly scoped continuity mode;
/// the remaining corrective filters stay available to callers through the
/// shared config without becoming front-end-specific conversion paths.
#[derive(Clone, Debug)]
pub struct F0Config {
    /// RMVPE voiced/unvoiced confidence threshold.
    pub f0_threshold: f32,
    /// Input RMS below which a chunk is treated as silence.
    pub silence_threshold: f32,
    pub postprocess: F0PostprocessConfig,
}

/// RVC's optional target-speaker feature retrieval and consonant protection.
///
/// The index path is immutable for one pipeline because opening/decoding a
/// FAISS file is load-time work. `index_rate` and `protect` are live values so
/// front ends can A/B them without reloading models. Both are clamped in the
/// worker-side `apply_live` path, so hostile automation cannot turn a malformed
/// float into invalid feature data. `index_rate` is the base retrieval share;
/// the shared worker may reduce it per frame when distance, voicing, or a sharp
/// ContentVec boundary indicates that source features are more reliable.
#[derive(Clone, Debug)]
pub struct FeatureRetrievalConfig<'a> {
    pub index_path: Option<&'a Path>,
    pub index_rate: f32,
    pub protect: f32,
    /// Optional vc-rs-only smoothing around RVC protect boundaries. This is
    /// live-adjustable after loading; zero keeps upstream's binary behavior.
    pub protect_transition_ms: u32,
}

impl Default for FeatureRetrievalConfig<'_> {
    fn default() -> Self {
        Self {
            index_path: None,
            index_rate: 0.0,
            protect: DEFAULT_PROTECT,
            protect_transition_ms: DEFAULT_PROTECT_TRANSITION_MS,
        }
    }
}

impl Default for F0Config {
    fn default() -> Self {
        Self {
            f0_threshold: DEFAULT_F0_THRESHOLD,
            silence_threshold: 0.0001,
            postprocess: F0PostprocessConfig::continuity_with_stabilization(true, true),
        }
    }
}

/// Live (per-block) conversion parameters: the knobs a host can change between
/// chunks without reloading the pipeline. Applied through
/// [`RvcPipeline::apply_live`], which is the single live-update entry point
/// shared by every front-end (the standalone worker and the VST3 host callback).
///
/// `noise_gate_enabled`/`noise_gate_threshold` control the legacy input gate,
/// whose attack/release/floor and denoiser *variant* selection (off / gate /
/// rnnoise) are static load-time config. `silence_gate_enabled` is separate: it
/// retains the input denoiser and mutes generated output after sustained source
/// silence, so it is safe to use with stateful denoisers (see `set_noise_gate`).
#[derive(Clone, Copy, Debug)]
pub struct LiveParams {
    pub pitch_shift: f32,
    pub speaker_id: i64,
    /// RMVPE voiced/unvoiced confidence threshold. This is live so the dynamic
    /// tuner can react to speech/noise conditions without rebuilding inference
    /// sessions or interrupting the rolling F0 timeline.
    pub f0_threshold: f32,
    pub input_gain: f32,
    pub output_gain: f32,
    // Monitor output gain, applied by the vc-app realtime worker when routing
    // the converted signal to the separate monitor output device. The RVC
    // pipeline itself ignores it (it is a vc-app routing concern, but it rides
    // on LiveParams so it flows through the atomic live-params channel to the
    // worker alongside the other live knobs).
    pub monitor_gain: f32,
    pub noise_gate_enabled: bool,
    /// Source-activity output mute that can coexist with any input denoiser.
    /// It uses the F0/silence threshold from [`F0Config`] as its source
    /// reference; the input Noise Gate threshold is intentionally independent.
    pub silence_gate_enabled: bool,
    pub noise_gate_threshold: f32,
    /// Base share of the denoised signal mixed into ContentVec. `0.25` keeps a
    /// raw residual; worker-side transient protection can reduce a nonzero base
    /// briefly, while zero remains exact raw input.
    pub denoiser_content_mix: f32,
    /// Base share of the denoised signal sent to RMVPE. `0.0` is exact aligned
    /// raw input; low-ZCR voiced onsets can slightly reduce a nonzero base.
    pub denoiser_rmvpe_mix: f32,
    /// Base RVC index blend amount. Meaningful only if an index was selected
    /// while the pipeline was loaded; the worker adapts this value per frame,
    /// and zero gives an exact no-retrieval fast path.
    pub index_rate: f32,
    /// Amount of indexed ContentVec retained on unvoiced (F0 <= 0) frames.
    /// Standard RVC accepts `0.0..=0.5`; `0.5` disables protection.
    pub protect: f32,
    /// Optional vc-rs-only smoothing width around voiced/unvoiced Protect
    /// boundaries. It is quantized to RVC's 10 ms feature grid; zero keeps the
    /// standard RVC binary mask.
    pub protect_transition_ms: u32,
}

impl Default for LiveParams {
    fn default() -> Self {
        Self {
            pitch_shift: 0.0,
            speaker_id: 0,
            f0_threshold: DEFAULT_F0_THRESHOLD,
            input_gain: 1.0,
            output_gain: 1.0,
            monitor_gain: 1.0,
            noise_gate_enabled: false,
            silence_gate_enabled: false,
            noise_gate_threshold: 0.01,
            denoiser_content_mix: DEFAULT_DENOISER_CONTENT_MIX,
            denoiser_rmvpe_mix: DEFAULT_DENOISER_RMVPE_MIX,
            index_rate: 0.0,
            protect: DEFAULT_PROTECT,
            protect_transition_ms: DEFAULT_PROTECT_TRANSITION_MS,
        }
    }
}

impl LiveParams {
    /// Return a finite, bounded snapshot suitable for the lock-free worker
    /// channel. Frontends historically passed this plain struct directly, so
    /// sanitizing at the shared boundary protects both realtime and VST3 paths
    /// without changing the public fields or requiring a fallible API.
    pub fn sanitized(self) -> Self {
        Self {
            pitch_shift: normalized_pitch_shift(self.pitch_shift),
            speaker_id: self.speaker_id,
            f0_threshold: if self.f0_threshold.is_finite() {
                self.f0_threshold.clamp(0.001, 0.5)
            } else {
                DEFAULT_F0_THRESHOLD
            },
            input_gain: normalized_live_gain(self.input_gain),
            output_gain: normalized_live_gain(self.output_gain),
            monitor_gain: normalized_live_gain(self.monitor_gain),
            noise_gate_enabled: self.noise_gate_enabled,
            silence_gate_enabled: self.silence_gate_enabled,
            noise_gate_threshold: normalized_noise_gate_threshold(self.noise_gate_threshold),
            index_rate: normalized_unit_value(self.index_rate),
            protect: normalized_protect(self.protect),
            protect_transition_ms: normalized_protect_transition_ms(self.protect_transition_ms),
            denoiser_content_mix: normalized_denoiser_content_mix(self.denoiser_content_mix),
            denoiser_rmvpe_mix: normalized_denoiser_rmvpe_mix(self.denoiser_rmvpe_mix),
        }
    }
}

pub struct RvcPipelineConfig<'a> {
    pub model: &'a Path,
    pub embedder: &'a Path,
    pub embedder_output: Option<&'a str>,
    /// RMVPE ONNX path. Required for `Rmvpe` and `Hybrid`, unused for `Fcpe`.
    pub f0_model: Option<&'a Path>,
    pub f0_mode: F0Mode,
    /// FCPE ONNX path. Required for `Fcpe` and `Hybrid`, unused for `Rmvpe`.
    pub fcpe_model: Option<&'a Path>,
    pub provider: Provider,
    pub gpu_priority: super::GpuPriority,
    pub gpu_device_id: u32,
    pub sample_rate: u32,
    pub chunk_samples: usize,
    pub speaker_id: i64,
    pub pitch_shift: f32,
    pub f0: F0Config,
    pub retrieval: FeatureRetrievalConfig<'a>,
    pub input_gain: f32,
    pub noise_gate_enabled: bool,
    /// Keep converted output quiet after sustained source silence without
    /// replacing a configured stateful input denoiser.
    pub silence_gate_enabled: bool,
    pub noise_gate_threshold: f32,
    pub denoiser_content_mix: f32,
    pub denoiser_rmvpe_mix: f32,
    pub noise_gate_shaping: NoiseGateShaping,
    pub output_extra_ms: u32,
    pub volume_excluded_ms: u32,
    pub extra_convert_ms: u32,
    /// Optional fixed RVC generator frame count. This is resolved while loading
    /// into one ContentVec window, TensorRT profile, engine, and CUDA graph;
    /// it is intentionally not a live parameter.
    pub rvc_frames: Option<usize>,
    pub output_gain: f32,
    pub output_dynamics: OutputDynamicsConfig,
    /// Optional load-time progress callback. It is invoked only while building
    /// the pipeline, never from inference or an audio callback.
    pub progress: Option<&'a dyn Fn(LoadProgress)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadModelRole {
    ContentVec,
    Rmvpe,
    Fcpe,
    Rvc,
    Gtcrn,
}

impl LoadModelRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::ContentVec => "ContentVec",
            Self::Rmvpe => "RMVPE",
            Self::Fcpe => "FCPE",
            Self::Rvc => "RVC",
            Self::Gtcrn => "GTCRN",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoadProgress {
    Idle,
    ValidatingConfig,
    PreparingProvider,
    DownloadingProvider,
    BuildingEngine { role: LoadModelRole },
    LoadingModel { role: LoadModelRole },
    OpeningAudioDevices,
    Running,
    Failed,
}

fn report_progress(config: &RvcPipelineConfig<'_>, progress: LoadProgress) {
    if let Some(report) = config.progress {
        report(progress);
    }
}

/// Validate the profile selected at load time against both the ONNX contract
/// and an optional user-selected T. This must stay on the loading path: the
/// realtime worker is allowed to reuse the resulting fixed buffers only.
fn validate_rvc_profile_frames(
    static_onnx_frames: Option<usize>,
    requested_frames: Option<usize>,
    actual_frames: usize,
) -> Result<()> {
    validate_rvc_static_profile_frames(static_onnx_frames, actual_frames)?;
    if let Some(requested_frames) = requested_frames {
        if requested_frames != actual_frames {
            bail!(
                "RVC custom T={requested_frames}, but the selected ContentVec profile produces T={actual_frames}; use auto or a T compatible with this embedder"
            );
        }
    }
    Ok(())
}

fn resolve_f0_models<'a>(
    mode: F0Mode,
    rmvpe_model: Option<&'a Path>,
    fcpe_model: Option<&'a Path>,
) -> Result<(Option<&'a Path>, Option<&'a Path>)> {
    let rmvpe_model = if mode.uses_rmvpe() {
        Some(rmvpe_model.ok_or_else(|| {
            anyhow::anyhow!(
                "F0 mode '{}' requires an RMVPE ONNX model path (f0_model)",
                mode.label()
            )
        })?)
    } else {
        None
    };
    let fcpe_model = if mode.uses_fcpe() {
        Some(fcpe_model.ok_or_else(|| {
            anyhow::anyhow!(
                "F0 mode '{}' requires an FCPE ONNX model path (fcpe_model)",
                mode.label()
            )
        })?)
    } else {
        None
    };
    Ok((rmvpe_model, fcpe_model))
}

fn report_native_load_progress(
    config: &RvcPipelineConfig<'_>,
    profile: &TensorRtSessionProfile,
    role: LoadModelRole,
) {
    if let Some(progress) = native_engine_build_progress(native_engine_is_cached(profile), role) {
        report_progress(config, progress);
    }
    report_progress(config, LoadProgress::LoadingModel { role });
}

fn native_engine_build_progress(cached: bool, role: LoadModelRole) -> Option<LoadProgress> {
    (!cached).then_some(LoadProgress::BuildingEngine { role })
}

fn normalize_speaker_id(speaker_id: i64, speaker_count: Option<usize>) -> i64 {
    let max = speaker_count
        .and_then(|count| count.checked_sub(1))
        .and_then(|max| i64::try_from(max).ok())
        .unwrap_or(i64::MAX);
    speaker_id.clamp(0, max)
}

fn normalized_unit_value(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn normalized_pitch_shift(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(MIN_PITCH_SHIFT_SEMITONES, MAX_PITCH_SHIFT_SEMITONES)
    } else {
        0.0
    }
}

fn normalized_live_gain(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_LIVE_GAIN)
    } else {
        1.0
    }
}

fn normalized_noise_gate_threshold(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_NOISE_GATE_THRESHOLD)
    } else {
        0.01
    }
}

/// Tracks the right edge of the generator window rather than assuming a fixed
/// chunk/window size. This also handles a larger or smaller offline tail window:
/// its start moves by both the appended-frame count and the window-size delta.
#[derive(Default)]
struct RvcNoiseTimeline {
    window_end_frame: Option<i64>,
}

impl RvcNoiseTimeline {
    fn reset(&mut self) {
        self.window_end_frame = None;
    }

    fn next_window_start(&mut self, frame_len: usize, new_feature_frames: usize) -> Result<i64> {
        let frame_len = i64::try_from(frame_len).context("RVC frame length does not fit i64")?;
        let new_feature_frames = i64::try_from(new_feature_frames)
            .context("new RVC feature-frame count does not fit i64")?;
        let window_end_frame = match self.window_end_frame {
            Some(previous_end) => previous_end
                .checked_add(new_feature_frames)
                .context("RVC rnd timeline overflow")?,
            // Anchor the first (possibly left-padded) window at zero. Later
            // windows still share coordinates because only new frames advance
            // this right edge.
            None => frame_len,
        };
        let window_start_frame = window_end_frame
            .checked_sub(frame_len)
            .context("RVC rnd timeline underflow")?;
        self.window_end_frame = Some(window_end_frame);
        Ok(window_start_frame)
    }
}

fn normalized_protect(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_PROTECT)
    } else {
        DEFAULT_PROTECT
    }
}

fn normalized_protect_transition_ms(value: u32) -> u32 {
    value.min(MAX_PROTECT_TRANSITION_MS)
}

fn normalized_denoiser_content_mix(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_DENOISER_CONTENT_MIX)
    } else {
        DEFAULT_DENOISER_CONTENT_MIX
    }
}

fn normalized_denoiser_rmvpe_mix(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, MAX_DENOISER_RMVPE_MIX)
    } else {
        DEFAULT_DENOISER_RMVPE_MIX
    }
}

fn output_silence_threshold(configured_threshold: f32) -> f32 {
    // The output activity gate has its own F0/silence reference.  Do not reuse
    // the input Noise Gate threshold here: input denoisers commonly use values
    // around 0.01, while a quiet but valid microphone phrase can be far below
    // that level.  Coupling the two made enabling input denoising swallow the
    // first syllable after an idle period.
    if configured_threshold.is_finite() {
        configured_threshold.max(0.0)
    } else {
        0.0
    }
}

fn protect_transition_frames(transition_ms: u32) -> usize {
    // The generator's F0 and repeated ContentVec tensors are both on a 10 ms
    // grid. Round a nonzero user value up so requesting any transition cannot
    // silently become the exact binary path.
    usize::try_from(
        normalized_protect_transition_ms(transition_ms).saturating_add(RVC_FEATURE_FRAME_MS - 1)
            / RVC_FEATURE_FRAME_MS,
    )
    .expect("protect transition frame count fits usize")
}

fn retrieval_is_active(feature_index: &Option<FeatureIndex>, index_rate: f32) -> bool {
    feature_index.is_some() && index_rate > 0.0
}

fn protect_is_active(feature_index: &Option<FeatureIndex>, index_rate: f32, protect: f32) -> bool {
    retrieval_is_active(feature_index, index_rate) && protect < MAX_PROTECT
}

/// Apply the timing conversion required by RVC's generator to one ContentVec
/// tensor. The saved original tensor must take this same path before `protect`
/// blends it back; changing only one side would silently misalign consonants.
fn prepare_rvc_feature_frames(
    feature_tensor: &mut FeatureTensor,
    silence_front_frames: usize,
    feature_len_before_trim: usize,
) -> Result<()> {
    if silence_front_frames > 0 && silence_front_frames < feature_len_before_trim {
        if silence_front_frames.is_multiple_of(2) {
            // `silence_front_frames` is on RVC's repeated 10 ms grid. Drop the
            // equivalent ContentVec frames before repeat so discarded context is
            // not duplicated and shifted every chunk.
            feature_tensor.trim_front_frames(silence_front_frames / 2)?;
            feature_tensor.repeat_frames(2)?;
        } else {
            feature_tensor.repeat_frames(2)?;
            feature_tensor.trim_front_frames(silence_front_frames)?;
        }
    } else {
        feature_tensor.repeat_frames(2)?;
    }
    Ok(())
}

fn load_feature_index(
    path: Option<&Path>,
    expected_dimensions: usize,
) -> Result<Option<FeatureIndex>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.as_os_str().is_empty() {
        return Ok(None);
    }
    let index = FeatureIndex::load(path, expected_dimensions)?;
    let summary = index.summary();
    info!(
        "loaded RVC feature index: {} dimensions={} vectors={} lists={} nprobe={}",
        path.display(),
        summary.dimensions,
        summary.vectors,
        summary.lists,
        summary.probes
    );
    Ok(Some(index))
}

impl RvcPipeline {
    #[cfg(feature = "rnnoise")]
    pub fn load_with_rnnoise(config: RvcPipelineConfig<'_>) -> Result<Self> {
        if config.noise_gate_enabled {
            bail!("RNNoise and the input noise gate are mutually exclusive");
        }
        let sample_rate = config.sample_rate;
        let mut pipeline = Self::load(config)?;
        pipeline.input_denoiser =
            InputDenoiser::Rnnoise(Box::new(crate::denoise::RnnoiseDenoiser::new(sample_rate)?));
        pipeline
            .content_raw_delay
            .configure(pipeline.input_denoiser.content_delay_samples());
        Ok(pipeline)
    }

    /// Load with the in-tree WebRTC-style device-rate noise suppressor. It uses
    /// the same fixed-delay contract as RNNoise, so ContentVec residual mixing
    /// remains sample-aligned while RMVPE receives the fully enhanced branch.
    #[cfg(feature = "webrtc")]
    pub fn load_with_webrtc(
        config: RvcPipelineConfig<'_>,
        level: crate::denoise::WebRtcSuppressionLevel,
    ) -> Result<Self> {
        if config.noise_gate_enabled {
            bail!("WebRTC denoising and the input noise gate are mutually exclusive");
        }
        let sample_rate = config.sample_rate;
        let mut pipeline = Self::load(config)?;
        pipeline.input_denoiser = InputDenoiser::WebRtc(Box::new(
            crate::denoise::WebRtcDenoiser::new(sample_rate, level)?,
        ));
        pipeline
            .content_raw_delay
            .configure(pipeline.input_denoiser.content_delay_samples());
        Ok(pipeline)
    }

    /// Load with the official DeepFilterNet3 streaming runtime. Model loading is
    /// intentionally a load-time operation; callers build this pipeline on a
    /// worker/background loader before it can receive audio.
    #[cfg(feature = "deepfilternet3")]
    pub fn load_with_deepfilternet3(
        config: RvcPipelineConfig<'_>,
        dfn3: crate::denoise::DeepFilterNet3Config<'_>,
    ) -> Result<Self> {
        if config.noise_gate_enabled {
            bail!("DeepFilterNet3 and the input noise gate are mutually exclusive");
        }
        let sample_rate = config.sample_rate;
        let mut pipeline = Self::load(config)?;
        pipeline.input_denoiser = InputDenoiser::DeepFilterNet3(Box::new(
            crate::denoise::DeepFilterNet3Denoiser::new(dfn3, sample_rate)?,
        ));
        pipeline
            .content_raw_delay
            .configure(pipeline.input_denoiser.content_delay_samples());
        Ok(pipeline)
    }

    /// Load with GTCRN input denoising at the 16 kHz RVC seam. Unlike RNNoise
    /// (a device-rate `InputDenoiser`), GTCRN lives in the stream state and
    /// denoises the new 16 kHz increment inside `generate_input`.
    #[cfg(feature = "gtcrn")]
    pub fn load_with_gtcrn(
        config: RvcPipelineConfig<'_>,
        gtcrn: crate::denoise::GtcrnConfig<'_>,
    ) -> Result<Self> {
        if config.noise_gate_enabled {
            bail!("GTCRN and the input noise gate are mutually exclusive");
        }
        if let crate::denoise::GtcrnBackend::NativeTensorRt { gpu_device_id, .. } = gtcrn.backend {
            let model = crate::denoise::model_file_for_cache_probe(gtcrn.model_dir)?;
            if !native_gtcrn_engine_is_cached(&model, gpu_device_id) {
                report_progress(
                    &config,
                    LoadProgress::BuildingEngine {
                        role: LoadModelRole::Gtcrn,
                    },
                );
            }
            report_progress(
                &config,
                LoadProgress::LoadingModel {
                    role: LoadModelRole::Gtcrn,
                },
            );
        }
        // Build the adapter at 16 kHz so its resamplers are bypass — only the
        // hop FIFO and fixed delay run on the increment fed by `resampler_16k`.
        let denoiser =
            crate::denoise::GtcrnDenoiser::new(gtcrn, super::shape::EMBEDDER_SAMPLE_RATE)?;
        let mut pipeline = Self::load(config)?;
        pipeline.stream_state.set_gtcrn(Some(denoiser));
        Ok(pipeline)
    }

    pub fn load(config: RvcPipelineConfig<'_>) -> Result<Self> {
        report_progress(&config, LoadProgress::ValidatingConfig);
        let (rmvpe_model, fcpe_model) =
            resolve_f0_models(config.f0_mode, config.f0_model, config.fcpe_model)?;
        if provider_needs_fixed_shape_profile(config.provider) {
            return Self::load_fixed_shape(config);
        }

        report_progress(&config, LoadProgress::PreparingProvider);
        // Resolve the generator's I/O names (vcclient vs RVC WebUI / converter
        // export aliases) and the model's native sample rate before sizing the
        // convert/output windows, so the ORT bind sites use the right names and
        // all sizing math runs in the model's actual rate domain.
        let rvc_info = inspect_rvc_model(config.model)?;
        let rvc_sample_rate = rvc_info.rvc_sample_rate.unwrap_or(RVC_SAMPLE_RATE);
        let speaker_count = rvc_info.speaker_count;
        let expected_feat_channels_usize = usize::try_from(rvc_info.expected_feat_channels)
            .context("RVC expected feature channel count does not fit usize")?;
        let feature_index =
            load_feature_index(config.retrieval.index_path, expected_feat_channels_usize)?;
        let speaker_id = normalize_speaker_id(config.speaker_id, speaker_count);
        if speaker_id != config.speaker_id {
            info!(
                "clamped initial speaker ID from {} to {} for model speaker_count={}",
                config.speaker_id,
                speaker_id,
                speaker_count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
        }
        // CLI-facing configuration is milliseconds for consistency with other latency knobs.
        // The RVC shape and trimming code below use the model's sample-rate domain, so keep the
        // conversion at load time and leave the per-chunk processing path in samples.
        let extra_convert_samples =
            extra_convert_samples_from_ms(config.extra_convert_ms, rvc_sample_rate);
        let automatic_input_samples_16k = tensor_rt_model_input_samples_16k(
            config.chunk_samples,
            config.sample_rate,
            config.output_extra_ms,
            extra_convert_samples,
            rvc_sample_rate,
        );
        let input_samples_16k = resolve_rvc_context_samples_16k(
            automatic_input_samples_16k,
            config.rvc_frames,
            rvc_info.static_feature_frames,
            extra_convert_samples,
            rvc_sample_rate,
        )?;
        let rmvpe_input_samples_16k = rmvpe_model_input_samples_for_context_16k(
            config.chunk_samples,
            config.sample_rate,
            input_samples_16k,
        );
        report_progress(
            &config,
            LoadProgress::LoadingModel {
                role: LoadModelRole::Rvc,
            },
        );
        let rvc = RvcModelSession::load(
            config.model,
            config.provider,
            None,
            Some(rvc_info.expected_feat_channels),
            TensorRtRunMode::PinnedCpu,
            TensorRtSessionPurpose::Main,
            rvc_info.io_names,
        )?;
        let expected_feat_channels = rvc.expected_feat_channels;
        report_progress(
            &config,
            LoadProgress::LoadingModel {
                role: LoadModelRole::ContentVec,
            },
        );
        let embedder = HubertEmbedderSession::load(
            config.embedder,
            config.provider,
            expected_feat_channels,
            config.embedder_output,
            None,
            TensorRtRunMode::PinnedCpu,
            TensorRtSessionPurpose::Main,
        )?;
        let pitch = if let Some(rmvpe_model) = rmvpe_model {
            report_progress(
                &config,
                LoadProgress::LoadingModel {
                    role: LoadModelRole::Rmvpe,
                },
            );
            Some(RmvpePitchSession::load(
                rmvpe_model,
                config.provider,
                None,
                TensorRtRunMode::PinnedCpu,
                TensorRtSessionPurpose::Main,
            )?)
        } else {
            None
        };
        let fcpe = if let Some(fcpe_model) = fcpe_model {
            report_progress(
                &config,
                LoadProgress::LoadingModel {
                    role: LoadModelRole::Fcpe,
                },
            );
            Some(FcpePitchSession::load(
                fcpe_model,
                config.provider,
                None,
                TensorRtRunMode::PinnedCpu,
                TensorRtSessionPurpose::Main,
            )?)
        } else {
            None
        };
        let mut stream_state = RvcStreamState::new(rvc_sample_rate);
        stream_state.set_contentvec_context_samples_16k(input_samples_16k);
        Ok(Self {
            embedder,
            f0_mode: config.f0_mode,
            pitch,
            fcpe,
            hybrid_fusion: HybridF0Fusion::default(),
            rvc,
            #[cfg(feature = "ort")]
            shared_waveform: None,
            speaker_count,
            speaker_id,
            pitch_shift: normalized_pitch_shift(config.pitch_shift),
            feature_index,
            index_rate: normalized_unit_value(config.retrieval.index_rate),
            protect: normalized_protect(config.retrieval.protect),
            protect_transition_ms: normalized_protect_transition_ms(
                config.retrieval.protect_transition_ms,
            ),
            f0_threshold: if config.f0.f0_threshold.is_finite() {
                config.f0.f0_threshold.clamp(0.001, 0.5)
            } else {
                DEFAULT_F0_THRESHOLD
            },
            silence_threshold: config.f0.silence_threshold,
            input_gain: normalized_live_gain(config.input_gain),
            input_denoiser: build_input_denoiser(&config),
            silence_gate_enabled: config.silence_gate_enabled,
            speech_activity: SpeechActivityDetector::default(),
            output_silence_envelope: OutputSilenceEnvelope::default(),
            content_raw_delay: SampleDelay::default(),
            denoiser_content_mix: normalized_denoiser_content_mix(config.denoiser_content_mix),
            denoiser_rmvpe_mix: normalized_denoiser_rmvpe_mix(config.denoiser_rmvpe_mix),
            noise_gate_threshold: normalized_noise_gate_threshold(config.noise_gate_threshold),
            noise_gate_attack_ms: config.noise_gate_shaping.attack_ms,
            noise_gate_release_ms: config.noise_gate_shaping.release_ms,
            noise_gate_floor: config.noise_gate_shaping.floor,
            noise_gate_sample_rate: config.sample_rate as f32,
            output_extra_ms: config.output_extra_ms,
            volume_excluded_ms: config.volume_excluded_ms,
            rvc_sample_rate,
            extra_convert_samples,
            requested_rvc_frames: config.rvc_frames,
            rmvpe_input_samples_16k,
            output_gain: normalized_live_gain(config.output_gain),
            volume_envelope: config.output_dynamics.volume_envelope,
            rms_mix_rate: config.output_dynamics.rms_mix_rate,
            auto_output_gain: config.output_dynamics.auto_output_gain,
            target_output_rms: config.output_dynamics.target_output_rms,
            max_output_gain: config.output_dynamics.max_output_gain,
            stream_state,
            rnd_timeline: RvcNoiseTimeline::default(),
            input_scratch: Vec::new(),
            denoiser_scratch: Vec::new(),
            feature_tensor: FeatureTensor::default(),
            original_feature_tensor: FeatureTensor::default(),
            input_reference_scratch: Vec::new(),
            rms_mix_scratch: dsp::RmsMixScratch::default(),
            pitchf_validated_scratch: Vec::new(),
            pitchf_untrimmed_scratch: Vec::new(),
            pitchf_scratch: Vec::new(),
            pitch_scratch: Vec::new(),
            f0_postprocess: F0Postprocessor::new(config.f0.postprocess.clone()),
            pitchf_postprocessed_scratch: Vec::new(),
        })
    }

    fn load_fixed_shape(config: RvcPipelineConfig<'_>) -> Result<Self> {
        report_progress(&config, LoadProgress::PreparingProvider);
        let (rmvpe_model, fcpe_model) =
            resolve_f0_models(config.f0_mode, config.f0_model, config.fcpe_model)?;
        // Windows ML catalog providers may also use fixed-shape profiles, but
        // their adapter selection is owned by Windows ML. Only explicit CUDA
        // backends consume the user-selected CUDA device ID.
        let gpu_device_id = if config.provider.is_cuda() || config.provider.is_tensorrt() {
            config.gpu_device_id
        } else {
            0
        };
        // NvTensorRtRtx (explicit, or auto-selected by `Provider::WindowsMl`) uses
        // the pinned-CPU run mode: it has no CUDA IoBinding/CUDA-graph path, so the
        // CUDA device-I/O modes from the env must not be applied to it.
        let tensor_rt_run_mode =
            if config.provider.is_tensorrt() || provider_drives_nvtrtx(config.provider) {
                TensorRtRunMode::PinnedCpu
            } else {
                TensorRtRunMode::cuda_from_env()
            };
        info!(
            "{} run mode selected mode={} cuda_graph={} device_io={} env_var={}",
            config.provider.label(),
            tensor_rt_run_mode.label(),
            tensor_rt_run_mode.cuda_graph(),
            tensor_rt_run_mode.device_io(),
            if config.provider.is_tensorrt() {
                "native-tensorrt"
            } else {
                CUDA_GRAPH_ENV
            }
        );
        let rvc_info = inspect_rvc_model(config.model)?;
        let speaker_count = rvc_info.speaker_count;
        let speaker_id = normalize_speaker_id(config.speaker_id, speaker_count);
        if speaker_id != config.speaker_id {
            info!(
                "clamped initial speaker ID from {} to {} for model speaker_count={}",
                config.speaker_id,
                speaker_id,
                speaker_count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
        }
        let expected_feat_channels = rvc_info.expected_feat_channels;
        let expected_feat_channels_usize = usize::try_from(expected_feat_channels)
            .context("RVC expected feature channel count does not fit in usize")?;
        let feature_index =
            load_feature_index(config.retrieval.index_path, expected_feat_channels_usize)?;
        let rvc_sample_rate = rvc_info.rvc_sample_rate.unwrap_or(RVC_SAMPLE_RATE);
        let extra_convert_samples =
            extra_convert_samples_from_ms(config.extra_convert_ms, rvc_sample_rate);
        let automatic_input_samples_16k = tensor_rt_model_input_samples_16k(
            config.chunk_samples,
            config.sample_rate,
            config.output_extra_ms,
            extra_convert_samples,
            rvc_sample_rate,
        );
        let input_samples_16k = resolve_rvc_context_samples_16k(
            automatic_input_samples_16k,
            config.rvc_frames,
            rvc_info.static_feature_frames,
            extra_convert_samples,
            rvc_sample_rate,
        )?;
        let rmvpe_input_samples_16k = rmvpe_model_input_samples_for_context_16k(
            config.chunk_samples,
            config.sample_rate,
            input_samples_16k,
        );
        let (
            contentvec_model_cache_key,
            rmvpe_model_cache_key,
            fcpe_model_cache_key,
            rvc_model_cache_key,
        ) = if provider_needs_fixed_shape_profile(config.provider) {
            (
                Some(tensor_rt_model_cache_key(config.embedder)?),
                rmvpe_model.map(tensor_rt_model_cache_key).transpose()?,
                fcpe_model.map(tensor_rt_model_cache_key).transpose()?,
                Some(tensor_rt_model_cache_key(config.model)?),
            )
        } else {
            (None, None, None, None)
        };
        // Fixed-shape GPU profiles must use the model's exported input name.
        // Keep this CPU-only probe at load time; the realtime path relies on
        // the resulting profile for CUDA/TensorRT validation and IoBinding.
        let contentvec_input_name = inspect_contentvec_input_name(
            config.embedder,
            expected_feat_channels,
            config.embedder_output,
        )?;
        let contentvec_profile = TensorRtSessionProfile::single_input(
            ModelRole::ContentVec,
            contentvec_input_name,
            input_samples_16k,
        )
        .with_gpu_priority(config.gpu_priority)
        .with_gpu_device_id(gpu_device_id)
        .with_optional_model_cache_key(contentvec_model_cache_key);
        let rmvpe_profile = rmvpe_model.map(|_| {
            TensorRtSessionProfile::single_input(
                ModelRole::Rmvpe,
                "waveform",
                rmvpe_input_samples_16k,
            )
            .with_gpu_priority(config.gpu_priority)
            .with_gpu_device_id(gpu_device_id)
            .with_optional_model_cache_key(rmvpe_model_cache_key)
        });
        let fcpe_profile = fcpe_model.map(|_| {
            TensorRtSessionProfile::new(
                ModelRole::Fcpe,
                vec![super::tensorrt::TensorRtInputShape {
                    name: "audio".to_string(),
                    dims: vec![1, rmvpe_input_samples_16k, 1],
                }],
            )
            .with_gpu_priority(config.gpu_priority)
            .with_gpu_device_id(gpu_device_id)
            .with_optional_model_cache_key(fcpe_model_cache_key)
        });
        #[cfg(feature = "ort")]
        let shared_waveform_shape = [1usize, input_samples_16k];
        #[cfg(feature = "ort")]
        let mut shared_waveform: Option<TensorRtSharedWaveform> = None;

        let (embedder, pitch, fcpe, rvc) = if tensor_rt_run_mode.cuda_graph() {
            #[cfg(not(feature = "ort"))]
            {
                unreachable!("cuda_graph run mode requires the `ort` feature")
            }
            #[cfg(feature = "ort")]
            {
                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::ContentVec,
                    },
                );
                let mut embedder_probe = HubertEmbedderSession::load(
                    config.embedder,
                    config.provider,
                    expected_feat_channels,
                    config.embedder_output,
                    Some(contentvec_profile.clone()),
                    TensorRtRunMode::PinnedCpu,
                    TensorRtSessionPurpose::Probe,
                )?;
                let warmup = tensor_rt_warmup_feature_len(
                    &mut embedder_probe,
                    input_samples_16k,
                    extra_convert_samples,
                    rvc_sample_rate,
                )?;
                drop(embedder_probe);
                let feature_len = warmup.rvc_feature_len;
                validate_rvc_profile_frames(
                    rvc_info.static_feature_frames,
                    config.rvc_frames,
                    feature_len,
                )?;
                let rvc_profile = TensorRtSessionProfile::rvc(
                    feature_len,
                    expected_feat_channels_usize,
                    &rvc_info.io_names,
                )
                .with_gpu_priority(config.gpu_priority)
                .with_gpu_device_id(gpu_device_id)
                .with_optional_model_cache_key(rvc_model_cache_key.clone());
                info!(
                    "fixed runtime profiles backend={} sample_rate={} chunk_samples={} contentvec={} rmvpe={} fcpe={} rvc={}",
                    config.provider.label(),
                    config.sample_rate,
                    config.chunk_samples,
                    contentvec_profile.profile_shapes,
                    rmvpe_profile
                        .as_ref()
                        .map(|profile| profile.profile_shapes.as_str())
                        .unwrap_or("disabled"),
                    fcpe_profile
                        .as_ref()
                        .map(|profile| profile.profile_shapes.as_str())
                        .unwrap_or("disabled"),
                    rvc_profile.profile_shapes
                );

                let rmvpe_output_shape = if let (Some(rmvpe_model), Some(rmvpe_profile)) =
                    (rmvpe_model, rmvpe_profile.as_ref())
                {
                    report_progress(
                        &config,
                        LoadProgress::LoadingModel {
                            role: LoadModelRole::Rmvpe,
                        },
                    );
                    let mut pitch_probe = RmvpePitchSession::load(
                        rmvpe_model,
                        config.provider,
                        Some(rmvpe_profile.clone()),
                        TensorRtRunMode::PinnedCpu,
                        TensorRtSessionPurpose::Probe,
                    )?;
                    let shape = pitch_probe
                        .warmup_output_shape(rmvpe_input_samples_16k, config.f0.f0_threshold)?;
                    drop(pitch_probe);
                    Some(shape)
                } else {
                    None
                };

                let fcpe_output_shape = if let (Some(fcpe_model), Some(fcpe_profile)) =
                    (fcpe_model, fcpe_profile.as_ref())
                {
                    report_progress(
                        &config,
                        LoadProgress::LoadingModel {
                            role: LoadModelRole::Fcpe,
                        },
                    );
                    let mut fcpe_probe = FcpePitchSession::load(
                        fcpe_model,
                        config.provider,
                        Some(fcpe_profile.clone()),
                        TensorRtRunMode::PinnedCpu,
                        TensorRtSessionPurpose::Probe,
                    )?;
                    let shape = fcpe_probe.warmup_output_shape(rmvpe_input_samples_16k)?;
                    drop(fcpe_probe);
                    Some(shape)
                } else {
                    None
                };

                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::Rvc,
                    },
                );
                let mut rvc_probe = RvcModelSession::load(
                    config.model,
                    config.provider,
                    Some(rvc_profile.clone()),
                    Some(rvc_info.expected_feat_channels),
                    TensorRtRunMode::PinnedCpu,
                    TensorRtSessionPurpose::Probe,
                    rvc_info.io_names.clone(),
                )?;
                let rvc_output_shape = rvc_probe.warmup_output_shape(
                    feature_len,
                    rvc_info.expected_feat_channels,
                    speaker_id,
                )?;
                drop(rvc_probe);

                let mut embedder = HubertEmbedderSession::load(
                    config.embedder,
                    config.provider,
                    expected_feat_channels,
                    config.embedder_output,
                    Some(contentvec_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                shared_waveform = if tensor_rt_run_mode.device_io() {
                    Some(TensorRtSharedWaveform::new(
                        &embedder.session,
                        &shared_waveform_shape,
                        gpu_device_id,
                    )?)
                } else {
                    None
                };
                embedder.enable_tensorrt_binding(
                    &warmup.contentvec_output_shape,
                    shared_waveform.as_ref(),
                )?;

                let pitch = if let (Some(rmvpe_model), Some(rmvpe_profile), Some(output_shape)) =
                    (rmvpe_model, rmvpe_profile, rmvpe_output_shape.as_deref())
                {
                    let mut pitch = RmvpePitchSession::load(
                        rmvpe_model,
                        config.provider,
                        Some(rmvpe_profile),
                        tensor_rt_run_mode,
                        TensorRtSessionPurpose::Final,
                    )?;
                    pitch.enable_tensorrt_binding(output_shape, config.f0.f0_threshold, None)?;
                    Some(pitch)
                } else {
                    None
                };

                let fcpe = if let (Some(fcpe_model), Some(fcpe_profile), Some(output_shape)) =
                    (fcpe_model, fcpe_profile, fcpe_output_shape.as_deref())
                {
                    let mut fcpe = FcpePitchSession::load(
                        fcpe_model,
                        config.provider,
                        Some(fcpe_profile),
                        tensor_rt_run_mode,
                        TensorRtSessionPurpose::Final,
                    )?;
                    fcpe.enable_tensorrt_binding(output_shape)?;
                    Some(fcpe)
                } else {
                    None
                };

                let mut rvc = RvcModelSession::load(
                    config.model,
                    config.provider,
                    Some(rvc_profile),
                    Some(rvc_info.expected_feat_channels),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                    rvc_info.io_names.clone(),
                )?;
                rvc.enable_tensorrt_binding(&rvc_output_shape, speaker_id)?;
                (embedder, pitch, fcpe, rvc)
            }
        } else if config.provider.is_tensorrt() {
            // Native TensorRT engines self-report their fixed output shapes after
            // deserialize, so there is no warmup inference here: the RVC
            // `feature_len` is derived arithmetically from the ContentVec engine's
            // output frame count. Engine builds run in an isolated helper process
            // (native_tensorrt.rs has no in-process Builder), so the historical
            // "build RVC before other TensorRT runtimes in the same process"
            // ordering no longer applies and ContentVec can load first.
            report_native_load_progress(&config, &contentvec_profile, LoadModelRole::ContentVec);
            let embedder = HubertEmbedderSession::load(
                config.embedder,
                config.provider,
                expected_feat_channels,
                config.embedder_output,
                Some(contentvec_profile),
                tensor_rt_run_mode,
                TensorRtSessionPurpose::Final,
            )?;
            let contentvec_frames = match embedder.native_contentvec_output_frames() {
                Some(frames) => frames?,
                None => bail!("native TensorRT embedder is missing its engine"),
            };
            let feature_len =
                derive_rvc_feature_len(contentvec_frames, extra_convert_samples, rvc_sample_rate)?;
            validate_rvc_profile_frames(
                rvc_info.static_feature_frames,
                config.rvc_frames,
                feature_len,
            )?;
            let rvc_profile = TensorRtSessionProfile::rvc(
                feature_len,
                expected_feat_channels_usize,
                &rvc_info.io_names,
            )
            .with_gpu_priority(config.gpu_priority)
            .with_gpu_device_id(gpu_device_id)
            .with_optional_model_cache_key(rvc_model_cache_key.clone());
            info!(
                "fixed runtime profiles backend={} sample_rate={} chunk_samples={} contentvec={} rmvpe={} fcpe={} rvc={}",
                config.provider.label(),
                config.sample_rate,
                config.chunk_samples,
                embedder
                    .tensor_rt_profile
                    .as_ref()
                    .map(|profile| profile.profile_shapes.as_str())
                    .unwrap_or("none"),
                rmvpe_profile
                    .as_ref()
                    .map(|profile| profile.profile_shapes.as_str())
                    .unwrap_or("disabled"),
                fcpe_profile
                    .as_ref()
                    .map(|profile| profile.profile_shapes.as_str())
                    .unwrap_or("disabled"),
                rvc_profile.profile_shapes
            );
            report_native_load_progress(&config, &rvc_profile, LoadModelRole::Rvc);
            let mut rvc = RvcModelSession::load(
                config.model,
                config.provider,
                Some(rvc_profile),
                Some(rvc_info.expected_feat_channels),
                tensor_rt_run_mode,
                TensorRtSessionPurpose::Final,
                rvc_info.io_names.clone(),
            )?;
            // Validates the engine frame/channel counts against the runtime
            // profile; native engines self-report their output shape and use no
            // ORT IoBinding, so the returned shape is intentionally discarded.
            rvc.warmup_output_shape(feature_len, rvc_info.expected_feat_channels, speaker_id)?;

            let pitch =
                if let (Some(rmvpe_model), Some(rmvpe_profile)) = (rmvpe_model, rmvpe_profile) {
                    report_native_load_progress(&config, &rmvpe_profile, LoadModelRole::Rmvpe);
                    let mut pitch = RmvpePitchSession::load(
                        rmvpe_model,
                        config.provider,
                        Some(rmvpe_profile),
                        tensor_rt_run_mode,
                        TensorRtSessionPurpose::Final,
                    )?;
                    pitch.warmup_output_shape(rmvpe_input_samples_16k, config.f0.f0_threshold)?;
                    Some(pitch)
                } else {
                    None
                };

            let fcpe = if let (Some(fcpe_model), Some(fcpe_profile)) = (fcpe_model, fcpe_profile) {
                report_native_load_progress(&config, &fcpe_profile, LoadModelRole::Fcpe);
                let mut fcpe = FcpePitchSession::load(
                    fcpe_model,
                    config.provider,
                    Some(fcpe_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                fcpe.warmup_output_shape(rmvpe_input_samples_16k)?;
                Some(fcpe)
            } else {
                None
            };

            (embedder, pitch, fcpe, rvc)
        } else {
            #[cfg(not(feature = "ort"))]
            {
                bail!(
                    "provider {} requires the `ort` feature; this build supports native TensorRT only",
                    config.provider.label()
                )
            }
            #[cfg(feature = "ort")]
            {
                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::ContentVec,
                    },
                );
                let mut embedder = HubertEmbedderSession::load(
                    config.embedder,
                    config.provider,
                    expected_feat_channels,
                    config.embedder_output,
                    Some(contentvec_profile),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                )?;
                let warmup = tensor_rt_warmup_feature_len(
                    &mut embedder,
                    input_samples_16k,
                    extra_convert_samples,
                    rvc_sample_rate,
                )?;
                let feature_len = warmup.rvc_feature_len;
                validate_rvc_profile_frames(
                    rvc_info.static_feature_frames,
                    config.rvc_frames,
                    feature_len,
                )?;
                shared_waveform = if tensor_rt_run_mode.device_io() {
                    Some(TensorRtSharedWaveform::new(
                        &embedder.session,
                        &shared_waveform_shape,
                        gpu_device_id,
                    )?)
                } else {
                    None
                };
                embedder.enable_tensorrt_binding(
                    &warmup.contentvec_output_shape,
                    shared_waveform.as_ref(),
                )?;
                let rvc_profile = TensorRtSessionProfile::rvc(
                    feature_len,
                    expected_feat_channels_usize,
                    &rvc_info.io_names,
                )
                .with_gpu_priority(config.gpu_priority)
                .with_gpu_device_id(gpu_device_id)
                .with_optional_model_cache_key(rvc_model_cache_key.clone());
                info!(
                    "fixed runtime profiles backend={} sample_rate={} chunk_samples={} contentvec={} rmvpe={} fcpe={} rvc={}",
                    config.provider.label(),
                    config.sample_rate,
                    config.chunk_samples,
                    embedder
                        .tensor_rt_profile
                        .as_ref()
                        .map(|profile| profile.profile_shapes.as_str())
                        .unwrap_or("none"),
                    rmvpe_profile
                        .as_ref()
                        .map(|profile| profile.profile_shapes.as_str())
                        .unwrap_or("disabled"),
                    fcpe_profile
                        .as_ref()
                        .map(|profile| profile.profile_shapes.as_str())
                        .unwrap_or("disabled"),
                    rvc_profile.profile_shapes
                );

                let pitch = if let (Some(rmvpe_model), Some(rmvpe_profile)) =
                    (rmvpe_model, rmvpe_profile)
                {
                    report_progress(
                        &config,
                        LoadProgress::LoadingModel {
                            role: LoadModelRole::Rmvpe,
                        },
                    );
                    let mut pitch = RmvpePitchSession::load(
                        rmvpe_model,
                        config.provider,
                        Some(rmvpe_profile),
                        tensor_rt_run_mode,
                        TensorRtSessionPurpose::Final,
                    )?;
                    let rmvpe_output_shape = pitch
                        .warmup_output_shape(rmvpe_input_samples_16k, config.f0.f0_threshold)?;
                    pitch.enable_tensorrt_binding(
                        &rmvpe_output_shape,
                        config.f0.f0_threshold,
                        None,
                    )?;
                    Some(pitch)
                } else {
                    None
                };

                let fcpe =
                    if let (Some(fcpe_model), Some(fcpe_profile)) = (fcpe_model, fcpe_profile) {
                        report_progress(
                            &config,
                            LoadProgress::LoadingModel {
                                role: LoadModelRole::Fcpe,
                            },
                        );
                        let mut fcpe = FcpePitchSession::load(
                            fcpe_model,
                            config.provider,
                            Some(fcpe_profile),
                            tensor_rt_run_mode,
                            TensorRtSessionPurpose::Final,
                        )?;
                        let output_shape = fcpe.warmup_output_shape(rmvpe_input_samples_16k)?;
                        fcpe.enable_tensorrt_binding(&output_shape)?;
                        Some(fcpe)
                    } else {
                        None
                    };

                report_progress(
                    &config,
                    LoadProgress::LoadingModel {
                        role: LoadModelRole::Rvc,
                    },
                );
                let mut rvc = RvcModelSession::load(
                    config.model,
                    config.provider,
                    Some(rvc_profile),
                    Some(rvc_info.expected_feat_channels),
                    tensor_rt_run_mode,
                    TensorRtSessionPurpose::Final,
                    rvc_info.io_names.clone(),
                )?;
                let rvc_output_shape = rvc.warmup_output_shape(
                    feature_len,
                    rvc_info.expected_feat_channels,
                    speaker_id,
                )?;
                rvc.enable_tensorrt_binding(&rvc_output_shape, speaker_id)?;
                (embedder, pitch, fcpe, rvc)
            }
        };

        let mut stream_state = RvcStreamState::new(rvc_sample_rate);
        stream_state.set_contentvec_context_samples_16k(input_samples_16k);
        Ok(Self {
            embedder,
            f0_mode: config.f0_mode,
            pitch,
            fcpe,
            hybrid_fusion: HybridF0Fusion::default(),
            rvc,
            #[cfg(feature = "ort")]
            shared_waveform,
            speaker_count,
            speaker_id,
            pitch_shift: normalized_pitch_shift(config.pitch_shift),
            feature_index,
            index_rate: normalized_unit_value(config.retrieval.index_rate),
            protect: normalized_protect(config.retrieval.protect),
            protect_transition_ms: normalized_protect_transition_ms(
                config.retrieval.protect_transition_ms,
            ),
            f0_threshold: if config.f0.f0_threshold.is_finite() {
                config.f0.f0_threshold.clamp(0.001, 0.5)
            } else {
                DEFAULT_F0_THRESHOLD
            },
            silence_threshold: config.f0.silence_threshold,
            input_gain: normalized_live_gain(config.input_gain),
            input_denoiser: build_input_denoiser(&config),
            silence_gate_enabled: config.silence_gate_enabled,
            speech_activity: SpeechActivityDetector::default(),
            output_silence_envelope: OutputSilenceEnvelope::default(),
            content_raw_delay: SampleDelay::default(),
            denoiser_content_mix: normalized_denoiser_content_mix(config.denoiser_content_mix),
            denoiser_rmvpe_mix: normalized_denoiser_rmvpe_mix(config.denoiser_rmvpe_mix),
            noise_gate_threshold: normalized_noise_gate_threshold(config.noise_gate_threshold),
            noise_gate_attack_ms: config.noise_gate_shaping.attack_ms,
            noise_gate_release_ms: config.noise_gate_shaping.release_ms,
            noise_gate_floor: config.noise_gate_shaping.floor,
            noise_gate_sample_rate: config.sample_rate as f32,
            output_extra_ms: config.output_extra_ms,
            volume_excluded_ms: config.volume_excluded_ms,
            rvc_sample_rate,
            extra_convert_samples,
            requested_rvc_frames: config.rvc_frames,
            rmvpe_input_samples_16k,
            output_gain: normalized_live_gain(config.output_gain),
            volume_envelope: config.output_dynamics.volume_envelope,
            rms_mix_rate: config.output_dynamics.rms_mix_rate,
            auto_output_gain: config.output_dynamics.auto_output_gain,
            target_output_rms: config.output_dynamics.target_output_rms,
            max_output_gain: config.output_dynamics.max_output_gain,
            stream_state,
            rnd_timeline: RvcNoiseTimeline::default(),
            input_scratch: Vec::new(),
            denoiser_scratch: Vec::new(),
            feature_tensor: FeatureTensor::default(),
            original_feature_tensor: FeatureTensor::default(),
            input_reference_scratch: Vec::new(),
            rms_mix_scratch: dsp::RmsMixScratch::default(),
            pitchf_validated_scratch: Vec::new(),
            pitchf_untrimmed_scratch: Vec::new(),
            pitchf_scratch: Vec::new(),
            pitch_scratch: Vec::new(),
            f0_postprocess: F0Postprocessor::new(config.f0.postprocess.clone()),
            pitchf_postprocessed_scratch: Vec::new(),
        })
    }

    /// Runtime-mutable conversion parameters. These mirror the matching
    /// `RvcPipelineConfig` fields and let a host (e.g. the VST3 plugin) drive
    /// them from automation between chunks without reloading the pipeline.
    pub fn set_pitch_shift(&mut self, pitch_shift: f32) {
        self.pitch_shift = normalized_pitch_shift(pitch_shift);
    }

    pub fn set_speaker_id(&mut self, speaker_id: i64) {
        // This runs once per worker chunk for live automation. Keep it a pure
        // integer clamp: no allocation, logging, or model inspection belongs on
        // the latency-sensitive inference path.
        self.speaker_id = normalize_speaker_id(speaker_id, self.speaker_count);
    }

    pub fn set_f0_threshold(&mut self, f0_threshold: f32) {
        // A bad live value must not poison RMVPE's threshold for later chunks.
        // Keep the accepted range aligned with the frontend controls while
        // allowing a dynamically selected lower confidence cutoff for tones.
        self.f0_threshold = if f0_threshold.is_finite() {
            f0_threshold.clamp(0.001, 0.5)
        } else {
            DEFAULT_F0_THRESHOLD
        };
    }

    pub fn speaker_count(&self) -> Option<usize> {
        self.speaker_count
    }

    pub fn set_input_gain(&mut self, input_gain: f32) {
        self.input_gain = normalized_live_gain(input_gain);
    }

    pub fn set_index_rate(&mut self, index_rate: f32) {
        self.index_rate = normalized_unit_value(index_rate);
    }

    pub fn set_protect(&mut self, protect: f32) {
        self.protect = normalized_protect(protect);
    }

    pub fn set_protect_transition_ms(&mut self, protect_transition_ms: u32) {
        self.protect_transition_ms = normalized_protect_transition_ms(protect_transition_ms);
    }

    pub fn set_denoiser_content_mix(&mut self, denoiser_content_mix: f32) {
        self.denoiser_content_mix = normalized_denoiser_content_mix(denoiser_content_mix);
    }

    pub fn set_denoiser_rmvpe_mix(&mut self, denoiser_rmvpe_mix: f32) {
        self.denoiser_rmvpe_mix = normalized_denoiser_rmvpe_mix(denoiser_rmvpe_mix);
    }

    /// Live-update the input noise gate. Toggling on (re)builds the gate from
    /// the stored attack/release/floor; while it stays on, only the threshold
    /// changes so the envelope/gain state carries across chunks. Attack and
    /// release are not live-adjustable (they shape the smoothing coefficients
    /// fixed at construction).
    /// Returns true when the gate mode changed and the caller must also discard
    /// its `ChunkConverter` smoother history. Threshold-only automation keeps
    /// both the gate envelope and the conversion timeline intact.
    pub fn set_noise_gate(&mut self, enabled: bool, threshold: f32) -> bool {
        self.noise_gate_threshold = normalized_noise_gate_threshold(threshold);
        // Standalone live-parameter updates must not replace a configured
        // stateful denoiser. Feature-gated arms avoid referencing a backend in a
        // smaller distribution build while keeping the guard at one call site.
        if self.input_denoiser.is_stateful() {
            return false;
        }
        // GTCRN owns the 16 kHz input-denoise seam in `stream_state`; a live gate
        // toggle must never build a device-rate gate that would fight it.
        #[cfg(feature = "gtcrn")]
        if self.stream_state.gtcrn.is_some() {
            return false;
        }
        if !enabled {
            if matches!(self.input_denoiser, InputDenoiser::Off) {
                return false;
            }
            self.input_denoiser = InputDenoiser::Off;
            self.content_raw_delay.configure(0);
            self.clear_conversion_history();
            return true;
        }
        match &mut self.input_denoiser {
            InputDenoiser::Gate(gate) => {
                gate.set_threshold(threshold);
                false
            }
            _ => {
                self.input_denoiser = InputDenoiser::Gate(dsp::NoiseGate::new(
                    self.noise_gate_sample_rate,
                    threshold,
                    self.noise_gate_attack_ms,
                    self.noise_gate_release_ms,
                    self.noise_gate_floor,
                ));
                self.content_raw_delay.configure(0);
                self.clear_conversion_history();
                true
            }
        }
    }

    /// Live-toggle source-activity output muting without modifying the input
    /// denoiser or its delay. It is intentionally independent from the legacy
    /// input gate so WebRTC, GTCRN, RNNoise, and DeepFilterNet3 can still keep
    /// their noise estimators warm during quiet periods.
    pub fn set_silence_gate(&mut self, enabled: bool) {
        if self.silence_gate_enabled != enabled {
            // Do not carry a learned room floor or a pending hangover across an
            // explicit user toggle. The first enabled chunk is deliberately
            // protected by the detector's startup grace, preventing a toggle in
            // the middle of a quiet syllable from creating an abrupt mute.
            self.speech_activity.reset();
            self.output_silence_envelope.reset();
            self.stream_state.prev_silence = false;
        }
        self.silence_gate_enabled = enabled;
    }

    /// Hot-swap the inexpensive input denoiser variants live. GTCRN and
    /// DeepFilterNet3 attach through their dedicated pre-built swap methods;
    /// switching away from them clears their separate seams. Model loading never
    /// occurs on the audio callback, and the gate/stateful stages are mutually
    /// exclusive by construction.
    pub fn set_denoiser_mode(
        &mut self,
        mode: InputDenoiserMode,
        device_rate: u32,
        webrtc_level: crate::denoise_config::WebRtcSuppressionLevel,
    ) -> Result<()> {
        #[cfg(not(feature = "webrtc"))]
        let _ = webrtc_level;
        // Construct the next state before dropping the active one so a failed
        // RNNoise build leaves the current stream usable.
        let input_denoiser = match mode {
            InputDenoiserMode::Off => InputDenoiser::Off,
            InputDenoiserMode::Gate => InputDenoiser::Gate(dsp::NoiseGate::new(
                device_rate as f32,
                self.noise_gate_threshold,
                self.noise_gate_attack_ms,
                self.noise_gate_release_ms,
                self.noise_gate_floor,
            )),
            InputDenoiserMode::Rnnoise => {
                #[cfg(feature = "rnnoise")]
                {
                    InputDenoiser::Rnnoise(Box::new(crate::denoise::RnnoiseDenoiser::new(
                        device_rate,
                    )?))
                }
                #[cfg(not(feature = "rnnoise"))]
                {
                    bail!("RNNoise support is not enabled in this build")
                }
            }
            InputDenoiserMode::WebRtc => {
                #[cfg(feature = "webrtc")]
                {
                    InputDenoiser::WebRtc(Box::new(crate::denoise::WebRtcDenoiser::new(
                        device_rate,
                        webrtc_level,
                    )?))
                }
                #[cfg(not(feature = "webrtc"))]
                {
                    bail!("WebRTC denoising support is not enabled in this build")
                }
            }
            // GTCRN attaches via `set_gtcrn`; clear the device-rate stage so the
            // two never coexist.
            InputDenoiserMode::Gtcrn | InputDenoiserMode::DeepFilterNet3 => InputDenoiser::Off,
        };
        #[cfg(feature = "gtcrn")]
        self.stream_state.set_gtcrn(None);
        self.input_denoiser = input_denoiser;
        self.noise_gate_sample_rate = device_rate as f32;
        self.content_raw_delay
            .configure(self.input_denoiser.content_delay_samples());
        // Denoiser output timelines are not interchangeable. Starting a new
        // mode with old ContentVec/F0 history would concatenate differently
        // delayed signals inside one inference window.
        self.clear_conversion_history();
        Ok(())
    }

    /// Hot-swap the GTCRN stage (16 kHz input seam). `None` leaves gtcrn mode.
    /// The caller builds the denoiser off the worker thread (engine load takes
    /// seconds); clearing the gate/rnnoise stage here keeps the stages mutually
    /// exclusive.
    #[cfg(feature = "gtcrn")]
    pub fn set_gtcrn(&mut self, denoiser: Option<crate::denoise::GtcrnDenoiser>) {
        self.input_denoiser = InputDenoiser::Off;
        self.content_raw_delay.configure(0);
        self.stream_state.set_gtcrn(None);
        self.clear_conversion_history();
        self.stream_state.set_gtcrn(denoiser);
    }

    /// Hot-swap a pre-built DeepFilterNet3 device-rate instance. `DfTract`
    /// construction parses model archives and builds three graphs, so callers
    /// must do it in the background loader before invoking this worker method.
    #[cfg(feature = "deepfilternet3")]
    pub fn set_deepfilternet3(&mut self, denoiser: Option<crate::denoise::DeepFilterNet3Denoiser>) {
        self.input_denoiser = denoiser
            .map(|denoiser| InputDenoiser::DeepFilterNet3(Box::new(denoiser)))
            .unwrap_or(InputDenoiser::Off);
        self.content_raw_delay
            .configure(self.input_denoiser.content_delay_samples());
        #[cfg(feature = "gtcrn")]
        self.stream_state.set_gtcrn(None);
        self.clear_conversion_history();
    }

    pub fn set_output_gain(&mut self, output_gain: f32) {
        self.output_gain = normalized_live_gain(output_gain);
    }

    /// Apply a full [`LiveParams`] snapshot. The single per-chunk live-update
    /// path: the standalone worker and the VST3 host callback both build a
    /// `LiveParams` and call this, so the live knobs stay wired identically
    /// across front-ends. `set_noise_gate` keeps the rnnoise guard, while the
    /// separate silence gate never touches a configured input-denoiser stage.
    /// Returns true when live automation changed the input-denoiser timeline.
    /// Front-ends must then reset the owning `ChunkConverter` so SOLA/PSOLA does
    /// not join post-switch output against a pre-switch tail.
    pub fn apply_live(&mut self, live: &LiveParams) -> bool {
        // Normalize once at the shared pipeline boundary. This protects direct
        // VST3/embedding callers as well as vc-app's atomic mirror, and keeps
        // every downstream calculation finite without a fallible realtime API.
        let live = live.sanitized();
        self.set_pitch_shift(live.pitch_shift);
        self.set_speaker_id(live.speaker_id);
        self.set_f0_threshold(live.f0_threshold);
        self.set_input_gain(live.input_gain);
        self.set_output_gain(live.output_gain);
        let denoiser_changed =
            self.set_noise_gate(live.noise_gate_enabled, live.noise_gate_threshold);
        self.set_silence_gate(live.silence_gate_enabled);
        self.set_denoiser_content_mix(live.denoiser_content_mix);
        self.set_denoiser_rmvpe_mix(live.denoiser_rmvpe_mix);
        self.set_index_rate(live.index_rate);
        self.set_protect(live.protect);
        self.set_protect_transition_ms(live.protect_transition_ms);
        denoiser_changed
    }

    /// Clear every model-domain buffer tied to the current input timeline while
    /// retaining loaded inference sessions and any already-installed GTCRN
    /// engine. Callers changing a denoiser must reset the outer chunk smoother
    /// too; that history belongs to `ChunkConverter`, not this pipeline.
    fn clear_conversion_history(&mut self) {
        // Audio/F0 history is disposable, but a fixed ContentVec context is
        // load-time shape state shared with TensorRT's profile. Carry it over
        // when replacing the stream state or a denoiser toggle can make the
        // next fixed-profile inference use a different T.
        let contentvec_context_samples_16k = self.stream_state.contentvec_context_samples_16k();
        #[cfg(feature = "gtcrn")]
        let gtcrn = self.stream_state.gtcrn.take();
        self.stream_state = RvcStreamState::new(self.rvc_sample_rate);
        if let Some(samples) = contentvec_context_samples_16k {
            self.stream_state
                .set_contentvec_context_samples_16k(samples);
        }
        self.rnd_timeline.reset();
        #[cfg(feature = "gtcrn")]
        self.stream_state.set_gtcrn(gtcrn);
        self.input_scratch.clear();
        self.denoiser_scratch.clear();
        self.feature_tensor = FeatureTensor::default();
        self.original_feature_tensor = FeatureTensor::default();
        self.input_reference_scratch.clear();
        self.rms_mix_scratch = dsp::RmsMixScratch::default();
        self.pitchf_validated_scratch.clear();
        self.pitchf_untrimmed_scratch.clear();
        self.pitchf_scratch.clear();
        self.pitch_scratch.clear();
        self.pitchf_postprocessed_scratch.clear();
        self.speech_activity.reset();
        self.output_silence_envelope.reset();
    }

    /// Discards rolling audio/F0 context while retaining loaded inference
    /// sessions.
    ///
    /// Standalone passthrough can pause RVC processing for an arbitrary time.
    /// On resume, old audio context must not be concatenated with the new live
    /// input. Denoiser state is reset too because RNNoise owns fixed-delay
    /// buffers that may otherwise emit audio captured before the pause.
    pub fn reset_streaming_state(&mut self) -> Result<()> {
        self.input_denoiser.reset(
            self.noise_gate_sample_rate,
            self.noise_gate_threshold,
            NoiseGateShaping {
                attack_ms: self.noise_gate_attack_ms,
                release_ms: self.noise_gate_release_ms,
                floor: self.noise_gate_floor,
            },
        )?;
        self.content_raw_delay
            .configure(self.input_denoiser.content_delay_samples());
        // Preserve the loaded GTCRN denoiser across a context reset, but reset its
        // fixed-delay/cache state (mirroring the RNNoise reset above) so it does
        // not emit audio captured before the pause.
        #[cfg(feature = "gtcrn")]
        if let Some(gtcrn) = self.stream_state.gtcrn.as_mut() {
            gtcrn.reset()?;
        }
        self.clear_conversion_history();
        Ok(())
    }
}

impl VoiceModel for RvcPipeline {
    fn process(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        out_audio: &mut Vec<f32>,
        out_pitchf: &mut Vec<f32>,
    ) -> Result<ModelOutput> {
        let total_start = Instant::now();
        let input_gain = self.input_gain.max(0.0);
        let apply_gain = (input_gain - 1.0).abs() > f32::EPSILON;
        // GTCRN is attached to the 16 kHz stream rather than the device-rate
        // `InputDenoiser` enum. Include it here so the worker builds a separate
        // pitch branch; stream state aligns raw ContentVec samples to GTCRN's
        // fixed delay before their residual mix.
        let gtcrn_active = {
            #[cfg(feature = "gtcrn")]
            {
                self.stream_state.gtcrn.is_some()
            }
            #[cfg(not(feature = "gtcrn"))]
            {
                false
            }
        };
        let denoiser_active = !matches!(self.input_denoiser, InputDenoiser::Off) || gtcrn_active;
        // Only gain≠1.0 or an active denoiser need an owned buffer; the no-op
        // path keeps `audio` as a zero-copy borrow. When an owned buffer is
        // needed, reuse `input_scratch` instead of allocating a fresh Vec every
        // chunk. Take it into a local so the borrow does not collide with the
        // later `&mut self.stream_state` call; it is written back at the end of
        // the function to retain the allocation.
        let mut input_scratch = std::mem::take(&mut self.input_scratch);
        let mut denoiser_scratch = std::mem::take(&mut self.denoiser_scratch);
        if apply_gain || denoiser_active {
            input_scratch.clear();
            if apply_gain {
                input_scratch.extend(
                    audio
                        .iter()
                        .map(|sample| (*sample * input_gain).clamp(-1.0, 1.0)),
                );
            } else {
                input_scratch.extend_from_slice(audio);
            }
            // Preserve the gain-scaled source for ContentVec, then denoise a
            // reusable copy for RMVPE. This avoids deleting fricatives from the
            // content branch while keeping pitch detection noise-robust.
            if denoiser_active && !gtcrn_active {
                denoiser_scratch.clear();
                denoiser_scratch.extend_from_slice(&input_scratch);
                self.input_denoiser
                    .process_in_place(&mut denoiser_scratch)?;
                // RNNoise emits a fixed-latency signal. Delay the raw branch
                // before resampling so residual ContentVec mixing compares the
                // same instant of speech rather than creating an echo.
                self.content_raw_delay.process_in_place(&mut input_scratch);
            } else if gtcrn_active {
                // GTCRN runs after the 16 kHz resampler in stream state. Keep
                // this device-rate copy raw; it is only a placeholder source
                // for the pitch branch and is replaced by GTCRN at 16 kHz.
                denoiser_scratch.clear();
                denoiser_scratch.extend_from_slice(&input_scratch);
            }
        }
        let input_audio: &[f32] = if apply_gain || denoiser_active {
            &input_scratch
        } else {
            audio
        };
        let output_extra_len = ms_to_samples(self.rvc_sample_rate, self.output_extra_ms);
        let volume_excluded_len = ms_to_samples(self.rvc_sample_rate, self.volume_excluded_ms);
        let stream_input = if denoiser_active {
            self.stream_state.generate_input_with_pitch(
                input_audio,
                &denoiser_scratch,
                StreamInputTiming {
                    sample_rate,
                    crossfade_and_search_samples: output_extra_len,
                    volume_excluded_samples: volume_excluded_len,
                    extra_convert_samples: self.extra_convert_samples,
                    denoiser_content_mix: self.denoiser_content_mix,
                    denoiser_rmvpe_mix: self.denoiser_rmvpe_mix,
                },
            )?
        } else {
            self.stream_state.generate_input(
                input_audio,
                sample_rate,
                output_extra_len,
                volume_excluded_len,
                self.extra_convert_samples,
            )?
        };
        // `input_audio` is no longer borrowed past this point; return the buffer
        // so its capacity is reused on the next chunk.
        self.input_scratch = input_scratch;
        self.denoiser_scratch = denoiser_scratch;

        if stream_input.stream_restarted {
            // `generate_input` owns resampler/path resets. Keep the adaptive
            // detector on exactly the same timeline so a previous device or
            // denoiser branch cannot donate a stale noise floor to the new one.
            self.speech_activity.reset();
            self.output_silence_envelope.reset();
            self.rnd_timeline.reset();
        }

        // input_rms/silence follow the configured RMVPE branch. With denoising
        // off this is the same single-resampled signal as ContentVec; with
        // denoising on each branch can retain a different raw share. The final
        // activity decision waits for raw RMVPE F0 below, so interpolation in F0
        // post-processing cannot manufacture speech evidence during silence.
        let input_rms = stream_input.input_rms;
        let source_silence_threshold = output_silence_threshold(self.silence_threshold);

        // Features
        let embedder_start = Instant::now();
        #[cfg(feature = "ort")]
        if let Some(shared_waveform) = self.shared_waveform.as_mut() {
            // Shared CUDA input is charged to embedder_time because the public
            // metrics do not have a separate transfer bucket. RMVPE uses a
            // separate upstream-RVC bucket window, so this stable device address
            // is now owned by ContentVec only.
            let h2d_us = shared_waveform.copy_from_slice(&self.stream_state.audio_16k_buffer)?;
            debug!(
                "shared waveform h2d backend={} samples={} consumers=contentvec h2d_us={}",
                self.embedder.provider.label(),
                self.stream_state.audio_16k_buffer.len(),
                h2d_us
            );
        }
        self.embedder.extract_into(
            &self.stream_state.audio_16k_buffer,
            &mut self.feature_tensor,
        )?;
        let raw_feature_len = self
            .feature_tensor
            .shape
            .get(1)
            .copied()
            .context("embedder output must be rank-3 [1, frames, channels]")?;
        if raw_feature_len <= 0 {
            bail!("embedder produced zero frames");
        }
        let raw_feature_len = usize::try_from(raw_feature_len)
            .context("embedder frame length does not fit in usize")?;
        let raw_feature_dimensions = self
            .feature_tensor
            .shape
            .get(2)
            .copied()
            .and_then(|dimensions| usize::try_from(dimensions).ok())
            .context("embedder output must be rank-3 [1, frames, channels]")?;
        let feature_len_before_trim = raw_feature_len
            .checked_mul(2)
            .context("repeated embedder frame length overflowed")?;

        let protect_active = protect_is_active(&self.feature_index, self.index_rate, self.protect);
        if protect_active {
            self.original_feature_tensor.copy_from(&self.feature_tensor);
        }
        let embedder_time = embedder_start.elapsed();
        // Pitch
        let pitch_start = Instant::now();
        // Extract natural F0. RMVPE receives zero pitch shift so every backend
        // enters the same downstream continuity/shift path. The selected
        // sessions and fixed buffers were established at load time; this match
        // must remain inference-only on the worker path.
        let audio_16k_len = self.stream_state.pitch_16k_buffer.len();
        let rmvpe_input_samples_16k = self.rmvpe_input_samples_16k.min(audio_16k_len);
        let rmvpe_window_start_samples = audio_16k_len - rmvpe_input_samples_16k;
        let rmvpe_audio_16k = &self.stream_state.pitch_16k_buffer[rmvpe_window_start_samples..];
        let pitchf_raw = match self.f0_mode {
            F0Mode::Rmvpe => self
                .pitch
                .as_mut()
                .context("RMVPE mode loaded without an RMVPE session")?
                .extract(rmvpe_audio_16k, 0.0, self.f0_threshold)?,
            F0Mode::Fcpe => self
                .fcpe
                .as_mut()
                .context("FCPE mode loaded without an FCPE session")?
                .extract(rmvpe_audio_16k)?,
            F0Mode::Hybrid => {
                let rmvpe_pitch = self
                    .pitch
                    .as_mut()
                    .context("hybrid F0 mode loaded without an RMVPE session")?
                    .extract(rmvpe_audio_16k, 0.0, self.f0_threshold)?;
                let fcpe_pitch = self
                    .fcpe
                    .as_mut()
                    .context("hybrid F0 mode loaded without an FCPE session")?
                    .extract(rmvpe_audio_16k)?;
                self.hybrid_fusion
                    .fuse(rmvpe_pitch, fcpe_pitch, rmvpe_audio_16k)?
            }
        };
        let rmvpe_audio_16k_len = rmvpe_audio_16k.len();
        let pitchf_raw_len = pitchf_raw.len();
        self.f0_postprocess.validate_raw_pitchf_into(
            pitchf_raw,
            rmvpe_audio_16k,
            &mut self.pitchf_validated_scratch,
        );
        self.stream_state.update_pitchf_from_estimator_window(
            &self.pitchf_validated_scratch,
            rmvpe_window_start_samples,
        );
        let raw_voiced_ratio = voiced_ratio(
            self.stream_state
                .newest_raw_pitchf(stream_input.new_feature_frames),
        );
        let adaptive_source_active = if self.silence_gate_enabled {
            // Feed exactly the fresh RMVPE branch tail. Reusing the rolling
            // window would let a previous phrase hold the silence gate open;
            // running only while enabled keeps the optional neural VAD out of
            // the normal conversion CPU budget.
            let speech_audio = self
                .stream_state
                .newest_pitch_audio(stream_input.speech_features.samples);
            self.speech_activity.observe(
                stream_input.speech_features,
                speech_audio,
                raw_voiced_ratio,
                source_silence_threshold,
            )
        } else {
            false
        };
        // The detector is only authoritative while the output-only suppressor
        // is enabled. Preserve the old static bookkeeping otherwise so a future
        // live toggle has a defined two-chunk confirmation boundary.
        let is_silent = if self.silence_gate_enabled {
            !adaptive_source_active
        } else {
            source_silence_threshold > 0.0 && input_rms < source_silence_threshold
        };
        let output_silent = self.silence_gate_enabled
            && is_silent
            && self.stream_state.prev_silence
            // Never discard a whole conversion increment solely because the
            // aggregate detector missed a short onset.  The guard is scalar
            // and worker-owned; it adds no callback allocation or inference
            // branch and lets the next chunk close the gate once the signal is
            // unambiguously ambient.
            && self.speech_activity.safe_to_mute(stream_input.speech_features);
        self.stream_state.prev_silence = is_silent;
        let pitch_frames = self.stream_state.pitchf_buffer.len();
        let pitch_time = pitch_start.elapsed();
        // RMVPE's center-padded STFT and ContentVec's convolutional frontend do
        // not expose the same frame count for the same waveform. First center
        // crop to the untrimmed ContentVec grid so a 183->180 case uses
        // pitchf[1..181], then apply the existing tail crop for silence_front.
        center_crop_pitchf_to_features_into(
            &self.stream_state.pitchf_buffer,
            feature_len_before_trim,
            &mut self.pitchf_untrimmed_scratch,
        );

        // Retrieval needs natural F0 and untrimmed ContentVec neighbors on the
        // same timeline. Running it here keeps voiced/unvoiced and boundary
        // decisions independent of the later pitch shift/post-processing.
        if let Some(feature_index) = self.feature_index.as_mut() {
            feature_index.blend_frames_adaptive_in_place(
                &mut self.feature_tensor.data,
                raw_feature_len,
                raw_feature_dimensions,
                self.index_rate,
                &self.pitchf_untrimmed_scratch,
            )?;
        }

        let silence_front_frames =
            onnx_silence_front_feature_frames(self.extra_convert_samples, self.rvc_sample_rate);
        // If the requested front context is larger than this window, the
        // preparation helper intentionally keeps the whole tensor; in that
        // case there is no offset to apply when looking up adaptive scales.
        let effective_silence_front_frames =
            if silence_front_frames > 0 && silence_front_frames < feature_len_before_trim {
                silence_front_frames
            } else {
                0
            };
        prepare_rvc_feature_frames(
            &mut self.feature_tensor,
            silence_front_frames,
            feature_len_before_trim,
        )?;
        if protect_active {
            prepare_rvc_feature_frames(
                &mut self.original_feature_tensor,
                silence_front_frames,
                feature_len_before_trim,
            )?;
        }
        let feature_len = self
            .feature_tensor
            .shape
            .get(1)
            .copied()
            .and_then(|len| usize::try_from(len).ok())
            .context("trimmed embedder frame length does not fit in usize")?;
        if let Some(requested_frames) = self.requested_rvc_frames {
            if feature_len != requested_frames {
                bail!(
                    "RVC custom T={requested_frames}, but the selected ContentVec window produced T={feature_len}; use auto or a T compatible with this embedder"
                )
            }
        }
        align_pitchf_to_features_into(
            &self.pitchf_untrimmed_scratch,
            feature_len,
            &mut self.pitchf_scratch,
        );
        if protect_active {
            // Use natural (unshifted, unpostprocessed) F0. Pitch correction may
            // fill short gaps for synthesis, but it must not classify an
            // originally unvoiced consonant as voiced for feature retrieval.
            let adaptive_scales = self
                .feature_index
                .as_ref()
                .map(|feature_index| feature_index.adaptive_protect_scales());
            self.feature_tensor.protect_unvoiced_frames_with_adaptive(
                &self.original_feature_tensor,
                &self.pitchf_scratch,
                self.protect,
                protect_transition_frames(self.protect_transition_ms),
                adaptive_scales,
                effective_silence_front_frames,
            )?;
        }
        // Guardrail: pitchf_buffer / pitchf_scratch hold raw (un-transposed) F0
        // because extract() above is called with 0.0 shift. Post-process the
        // RVC-aligned natural F0 and apply pitch_shift exactly once at the end.
        // Always run this, even when post-processing is disabled: the shift is
        // applied here, so skipping it would drop pitch shift entirely.
        // voiced_ratio and coarse_pitch_into below must use this post-processed
        // pitchf so the RVC inputs stay consistent (island/gap edits change the
        // voiced frame count).
        self.f0_postprocess.process_pitchf_into(
            &self.pitchf_scratch,
            self.pitch_shift,
            &mut self.pitchf_postprocessed_scratch,
        );
        let pitchf = self.pitchf_postprocessed_scratch.as_slice();
        debug!(
            "pitch update: audio_16k_samples={}, rmvpe_input_samples={}, rmvpe_window_start_samples={}, pitchf_raw_len={}, pitchf_buffer_len={}, feature_len={}",
            self.stream_state.audio_16k_buffer.len(),
            rmvpe_audio_16k_len,
            rmvpe_window_start_samples,
            pitchf_raw_len,
            self.stream_state.pitchf_buffer.len(),
            feature_len,
        );
        let voiced_ratio = voiced_ratio(pitchf);
        coarse_pitch_into(pitchf, &mut self.pitch_scratch);
        let pitch = self.pitch_scratch.as_slice();

        // RVC. The converted samples are written straight into the caller-owned
        // `out_audio` buffer (reused across chunks) and all post-processing runs
        // in place on it; the output pitchf goes into `out_pitchf`.
        let rvc_start = Instant::now();
        let rnd_window_start_frame = self
            .rnd_timeline
            .next_window_start(feature_len, stream_input.new_feature_frames)?;
        self.rvc.infer(
            &self.feature_tensor.data,
            &self.feature_tensor.shape,
            feature_len,
            pitch,
            pitchf,
            self.speaker_id,
            rnd_window_start_frame,
            out_audio,
        )?;
        let rvc_time = rvc_start.elapsed();
        let raw_output_samples = out_audio.len();
        keep_tail_in_place(out_audio, stream_input.out_size);
        pitchf_tail_for_output_into(pitchf, out_audio.len(), self.rvc_sample_rate, out_pitchf);
        let output_envelope = if self.volume_envelope {
            stream_input.volume.sqrt().clamp(0.0, 1.0)
        } else {
            1.0
        };
        dsp::clamp_scale_in_place(out_audio, output_envelope);
        if self.rms_mix_rate < 1.0 {
            // Captured before apply_rms_mix mutates `out_audio`, but only used
            // in the debug! below; skip the extra RMS pass when debug is off.
            let output_rms_before_mix = if tracing::enabled!(tracing::Level::DEBUG) {
                dsp::rms(out_audio)
            } else {
                0.0
            };
            // `out_audio` has already been trimmed to the same tail that SOLA
            // will search over. Use the input buffer tail with the same
            // duration; taking the head would compare against past context
            // added only to stabilize the model.
            // Reference the residual-preserving ContentVec rolling signal
            // (resampled 16 kHz -> RVC rate), so output leveling follows the
            // articulation branch rather than the fully denoised RMVPE branch.
            let input_reference = self.stream_state.output_reference_audio(
                super::shape::EMBEDDER_SAMPLE_RATE,
                self.rvc_sample_rate,
                out_audio.len(),
                &mut self.input_reference_scratch,
            )?;
            dsp::apply_rms_mix_with_scratch(
                input_reference,
                out_audio,
                self.rvc_sample_rate as usize,
                self.rms_mix_rate,
                &mut self.rms_mix_scratch,
            );
            debug!(
                "rms_mix_rate={:.3} input_ref_rms={:.8} output_rms_before_mix={:.8} output_rms_after_mix={:.8}",
                self.rms_mix_rate,
                dsp::rms(input_reference),
                output_rms_before_mix,
                dsp::rms(out_audio)
            );
        }
        let output_rms_before_gain = dsp::rms(out_audio);
        // Do not skip the model on silence: RVC's rolling ContentVec/F0 context
        // must continue advancing, especially with large extra-convert windows.
        // Once two source-silent chunks establish a stable boundary, mute only
        // the generated audio so model noise cannot leak into an idle mic.
        let mute_silent_output = self.silence_gate_enabled && output_silent;
        let applied_output_gain = self.applied_output_gain(output_rms_before_gain);
        let output_rms_after_gain = if (applied_output_gain - 1.0).abs() > f32::EPSILON {
            dsp::apply_gain_and_rms(out_audio, applied_output_gain)
        } else {
            output_rms_before_gain
        };
        let output_rms = if self.silence_gate_enabled {
            self.output_silence_envelope.apply_and_rms(
                out_audio,
                !mute_silent_output,
                self.rvc_sample_rate,
            )
        } else {
            output_rms_after_gain
        };
        // `silent` controls queue policy in realtime frontends. A fade-out must
        // remain queueable until every sample has reached zero; otherwise a
        // nearly-full output ring can cut the envelope abruptly and click.
        let fully_silent = mute_silent_output && self.output_silence_envelope.is_silent();

        Ok(ModelOutput {
            sample_rate: self.rvc_sample_rate,
            inference_time: total_start.elapsed(),
            embedder_time,
            pitch_time,
            rvc_time,
            input_rms,
            voiced_ratio,
            raw_output_samples,
            output_rms,
            applied_output_gain,
            feature_frames: feature_len,
            pitch_frames,
            silent: fully_silent,
            convert_size: stream_input.convert_size,
            out_size: stream_input.out_size,
            model_input_samples: self.stream_state.audio_buffer.len(),
            volume: stream_input.volume,
        })
    }
}

fn provider_needs_fixed_shape_profile(provider: Provider) -> bool {
    provider_uses_fixed_shape(provider) || provider_drives_nvtrtx(provider)
}

/// True when sessions for this provider run through the NvTensorRtRtx
/// (TensorRT-RTX) Windows ML catalog EP — either because it was requested
/// explicitly, or because the "Auto" Windows ML provider (`Provider::WindowsMl`)
/// resolves to it as the best available catalog EP on this machine.
///
/// That EP rejects dynamic shapes, so it needs the same fixed-shape profile and
/// pinned-CPU run mode as the explicit provider. Without this, `Provider::WindowsMl`
/// on a machine with TensorRT-RTX installed takes the dynamic-shape `load` path,
/// passes no profile, and the session build fails with
/// "Windows ML NvTensorRtRtx requires a fixed-shape profile". The catalog lookup
/// is cached (OnceLock) and matches what `load_session` selects later, so the
/// load-time routing decision and the session build stay in agreement.
fn provider_drives_nvtrtx(provider: Provider) -> bool {
    if provider == Provider::WindowsMlNvTensorRtRtx {
        return true;
    }
    #[cfg(all(windows, feature = "windowsml"))]
    if provider == Provider::WindowsMl {
        return matches!(
            crate::windows_ml::try_register_best_catalog_ep(),
            Ok(Some(
                crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx
            ))
        );
    }
    false
}

impl RvcPipeline {
    fn applied_output_gain(&self, output_rms: f32) -> f32 {
        let manual_gain = self.output_gain.max(0.0);
        if !self.auto_output_gain || output_rms <= 1e-8 {
            return manual_gain;
        }
        let auto_gain = (self.target_output_rms.max(0.0) / output_rms)
            .clamp(1.0, self.max_output_gain.max(1.0));
        manual_gain * auto_gain
    }
}

impl std::fmt::Debug for RvcPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RvcPipeline")
            .field("f0_mode", &self.f0_mode)
            .field("speaker_count", &self.speaker_count)
            .field("speaker_id", &self.speaker_id)
            .field("pitch_shift", &self.pitch_shift)
            .field("f0_threshold", &self.f0_threshold)
            .field("silence_threshold", &self.silence_threshold)
            .field("input_gain", &self.input_gain)
            .field("output_extra_ms", &self.output_extra_ms)
            .field("volume_excluded_ms", &self.volume_excluded_ms)
            .field("extra_convert_samples", &self.extra_convert_samples)
            .field("rmvpe_input_samples_16k", &self.rmvpe_input_samples_16k)
            .field("output_gain", &self.output_gain)
            .field("volume_envelope", &self.volume_envelope)
            .field("rms_mix_rate", &self.rms_mix_rate)
            .field("auto_output_gain", &self.auto_output_gain)
            .field("target_output_rms", &self.target_output_rms)
            .field("max_output_gain", &self.max_output_gain)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn f0_modes_resolve_only_the_sessions_they_use() {
        let rmvpe = Path::new("rmvpe.onnx");
        let fcpe = Path::new("fcpe.onnx");

        assert_eq!(
            resolve_f0_models(F0Mode::Rmvpe, Some(rmvpe), Some(fcpe)).unwrap(),
            (Some(rmvpe), None)
        );
        assert_eq!(
            resolve_f0_models(F0Mode::Fcpe, Some(rmvpe), Some(fcpe)).unwrap(),
            (None, Some(fcpe))
        );
        assert_eq!(
            resolve_f0_models(F0Mode::Hybrid, Some(rmvpe), Some(fcpe)).unwrap(),
            (Some(rmvpe), Some(fcpe))
        );
    }

    #[test]
    fn f0_modes_reject_missing_required_models() {
        let rmvpe = Path::new("rmvpe.onnx");
        let fcpe = Path::new("fcpe.onnx");

        assert!(resolve_f0_models(F0Mode::Rmvpe, None, Some(fcpe)).is_err());
        assert!(resolve_f0_models(F0Mode::Fcpe, Some(rmvpe), None).is_err());
        assert!(resolve_f0_models(F0Mode::Hybrid, Some(rmvpe), None).is_err());
        assert!(resolve_f0_models(F0Mode::Hybrid, None, Some(fcpe)).is_err());
    }

    #[test]
    fn native_engine_build_progress_is_only_reported_for_cache_miss() {
        assert_eq!(native_engine_build_progress(true, LoadModelRole::Rvc), None);
        assert_eq!(
            native_engine_build_progress(false, LoadModelRole::Rvc),
            Some(LoadProgress::BuildingEngine {
                role: LoadModelRole::Rvc
            })
        );
    }

    #[test]
    fn speaker_id_is_clamped_to_exported_embedding_table() {
        assert_eq!(normalize_speaker_id(-1, Some(308)), 0);
        assert_eq!(normalize_speaker_id(307, Some(308)), 307);
        assert_eq!(normalize_speaker_id(308, Some(308)), 307);
        assert_eq!(normalize_speaker_id(500, None), 500);
    }

    #[test]
    fn protect_transition_quantizes_up_to_the_rvc_feature_grid() {
        assert_eq!(protect_transition_frames(0), 0);
        assert_eq!(protect_transition_frames(1), 1);
        assert_eq!(protect_transition_frames(10), 1);
        assert_eq!(protect_transition_frames(11), 2);
        assert_eq!(protect_transition_frames(20), 2);
        assert_eq!(protect_transition_frames(MAX_PROTECT_TRANSITION_MS), 10);
        assert_eq!(protect_transition_frames(MAX_PROTECT_TRANSITION_MS + 1), 10);
    }

    #[test]
    fn rmvpe_denoiser_mix_defaults_to_full_cleaned_base() {
        assert_eq!(
            LiveParams::default().denoiser_rmvpe_mix,
            DEFAULT_DENOISER_RMVPE_MIX
        );
        assert_eq!(normalized_denoiser_rmvpe_mix(-1.0), 0.0);
        assert_eq!(normalized_denoiser_rmvpe_mix(2.0), MAX_DENOISER_RMVPE_MIX);
        assert_eq!(
            normalized_denoiser_rmvpe_mix(f32::NAN),
            DEFAULT_DENOISER_RMVPE_MIX
        );
    }

    #[test]
    fn live_params_sanitize_non_finite_and_out_of_range_values() {
        let params = LiveParams {
            pitch_shift: f32::INFINITY,
            f0_threshold: f32::NAN,
            input_gain: -1.0,
            output_gain: 1000.0,
            monitor_gain: f32::NEG_INFINITY,
            noise_gate_threshold: 2.0,
            index_rate: f32::NAN,
            protect: f32::INFINITY,
            denoiser_content_mix: -1.0,
            denoiser_rmvpe_mix: 2.0,
            ..LiveParams::default()
        }
        .sanitized();

        assert_eq!(params.pitch_shift, 0.0);
        assert_eq!(params.f0_threshold, DEFAULT_F0_THRESHOLD);
        assert_eq!(params.input_gain, 0.0);
        assert_eq!(params.output_gain, MAX_LIVE_GAIN);
        assert_eq!(params.monitor_gain, 1.0);
        assert_eq!(params.noise_gate_threshold, MAX_NOISE_GATE_THRESHOLD);
        assert_eq!(params.index_rate, 0.0);
        assert_eq!(params.protect, DEFAULT_PROTECT);
        assert_eq!(params.denoiser_content_mix, 0.0);
        assert_eq!(params.denoiser_rmvpe_mix, MAX_DENOISER_RMVPE_MIX);
    }

    #[test]
    fn pitch_and_gain_normalizers_keep_finite_values_bounded() {
        assert_eq!(normalized_pitch_shift(-100.0), MIN_PITCH_SHIFT_SEMITONES);
        assert_eq!(normalized_pitch_shift(100.0), MAX_PITCH_SHIFT_SEMITONES);
        assert_eq!(normalized_live_gain(-1.0), 0.0);
        assert_eq!(normalized_live_gain(100.0), MAX_LIVE_GAIN);
        assert_eq!(normalized_noise_gate_threshold(-1.0), 0.0);
        assert_eq!(
            normalized_noise_gate_threshold(2.0),
            MAX_NOISE_GATE_THRESHOLD
        );
    }

    #[test]
    fn silence_suppressor_threshold_is_independent_from_input_gate() {
        assert_eq!(output_silence_threshold(0.0001), 0.0001);
        assert_eq!(output_silence_threshold(0.03), 0.03);
        assert_eq!(output_silence_threshold(f32::NAN), 0.0);
    }

    #[test]
    fn rnd_timeline_tracks_overlap_and_variable_window_sizes() {
        let mut timeline = RvcNoiseTimeline::default();
        assert_eq!(timeline.next_window_start(100, 20).unwrap(), 0);
        assert_eq!(timeline.next_window_start(100, 20).unwrap(), 20);
        // Shrinking the rolling window moves its left edge forward by the new
        // frames plus the removed prefix; growing it can move the edge back.
        assert_eq!(timeline.next_window_start(80, 20).unwrap(), 60);
        assert_eq!(timeline.next_window_start(110, 20).unwrap(), 50);
    }

    #[test]
    fn rnd_timeline_reset_repeats_the_first_window() {
        let mut timeline = RvcNoiseTimeline::default();
        assert_eq!(timeline.next_window_start(96, 16).unwrap(), 0);
        assert_eq!(timeline.next_window_start(96, 16).unwrap(), 16);
        timeline.reset();
        assert_eq!(timeline.next_window_start(96, 16).unwrap(), 0);
    }
}
