//! Shared application runtime used by the CLI and standalone GUI.
//!
//! Frontends communicate with [`EngineController`]. Audio callbacks only touch
//! preallocated lock-free sample rings and atomics; they must never be coupled
//! to GUI rendering, model loading, or other blocking work.

pub mod audio;
mod realtime;

pub use realtime::{
    write_wav_mono, AudioHost, DenoiserMode, DeviceList, DeviceSpec, EngineController, EngineState,
    EngineStatusSnapshot, ModelLoadState, ModelLoadStatus, RealtimeConfig, Smoother,
    TelemetrySnapshot, VoiceCalibrationSnapshot, VoiceCalibrationState,
};
pub use vc_core::dynamic_tuning::{
    DynamicLanguageProfile, DynamicTuningMode, DynamicTuningSnapshot,
};
pub use vc_core::model_rvc::{
    F0Config, F0Mode, F0PostprocessConfig, LiveParams, NoiseGateShaping, OutputDynamicsConfig,
    DEFAULT_DENOISER_CONTENT_MIX, DEFAULT_DENOISER_RMVPE_MIX, DEFAULT_F0_THRESHOLD,
    DEFAULT_PROTECT, DEFAULT_PROTECT_TRANSITION_MS, MAX_DENOISER_CONTENT_MIX,
    MAX_DENOISER_RMVPE_MIX, MAX_LIVE_GAIN, MAX_NOISE_GATE_THRESHOLD, MAX_PITCH_SHIFT_SEMITONES,
    MAX_PROTECT, MAX_PROTECT_TRANSITION_MS, MIN_PITCH_SHIFT_SEMITONES,
};
