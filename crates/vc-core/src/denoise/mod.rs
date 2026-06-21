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

// `FrameDenoiser` / `FixedDelayAdapter` are crate-internal building blocks; only
// concrete denoisers are exported. The `allow` keeps them from tripping
// dead-code lints in feature combinations that compile the adapter without a
// model impl (none today, but future gtcrn/deepfilternet gating relies on it).
#[allow(unused_imports)]
pub(crate) use adapter::{FixedDelayAdapter, FrameDenoiser};

#[cfg(feature = "rnnoise")]
pub use rnnoise::RnnoiseDenoiser;
