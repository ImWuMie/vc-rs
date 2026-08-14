//! Input-denoiser family for the standalone front-ends.
//!
//! All models share one streaming contract: a model-specific [`FrameDenoiser`]
//! (native rate + fixed frame size + per-frame processing) wrapped by the
//! model-agnostic [`FixedDelayAdapter`], which owns resampling, frame
//! accumulation, and the fixed-delay output timeline. New models (GTCRN,
//! DeepFilterNet3) slot in as additional `FrameDenoiser` impls — they must reuse
//! this seam, not invent a parallel structure.

mod adapter;
#[cfg(feature = "rnnoise")]
mod rnnoise;
// GTCRN's STFT/iSTFT is ort-free DSP so it builds and tests without the Windows
// ML runtime; `gtcrn.rs` is the only ort-touching GTCRN file.
#[cfg(feature = "deepfilternet3")]
mod deepfilternet3;
#[cfg(feature = "gtcrn")]
mod gtcrn;
#[cfg(feature = "gtcrn")]
mod stft;
#[cfg(feature = "webrtc")]
mod webrtc;

// `FrameDenoiser` / `FixedDelayAdapter` are crate-internal building blocks; only
// concrete denoisers are exported. The `allow` keeps them from tripping
// dead-code lints in feature combinations that compile the adapter without a
// model impl (none today, but future gtcrn/deepfilternet gating relies on it).
#[allow(unused_imports)]
pub(crate) use adapter::{FixedDelayAdapter, FrameDenoiser};

#[cfg(feature = "rnnoise")]
pub use rnnoise::RnnoiseDenoiser;

#[cfg(feature = "gtcrn")]
pub(crate) use gtcrn::model_file_for_cache_probe;
#[cfg(feature = "gtcrn")]
pub use gtcrn::{GtcrnBackend, GtcrnConfig, GtcrnDenoiser};

#[cfg(feature = "webrtc")]
pub use crate::denoise_config::WebRtcSuppressionLevel;
#[cfg(feature = "webrtc")]
pub use webrtc::WebRtcDenoiser;

#[cfg(feature = "deepfilternet3")]
pub use crate::denoise_config::{
    DEFAULT_DFN3_ATTENUATION_LIMIT_DB, DEFAULT_DFN3_POST_FILTER_BETA,
    MAX_DFN3_ATTENUATION_LIMIT_DB, MAX_DFN3_POST_FILTER_BETA,
};
#[cfg(feature = "deepfilternet3")]
pub use deepfilternet3::{DeepFilterNet3Config, DeepFilterNet3Denoiser};
