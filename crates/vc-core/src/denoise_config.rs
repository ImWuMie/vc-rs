//! Feature-independent configuration shared by optional input denoisers.
//!
//! Front ends persist these values even in a reduced package that does not ship
//! a particular backend. The runtime implementation stays feature-gated in
//! [`crate::denoise`], where selecting an unavailable mode produces a clear
//! error instead of making config parsing or the basic build fail.

/// Strength of the in-tree WebRTC-style noise suppressor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum WebRtcSuppressionLevel {
    Low,
    #[default]
    Moderate,
    High,
    VeryHigh,
}

/// Default maximum DeepFilterNet3 attenuation. This conservative value leaves
/// enough residual articulation for the RVC ContentVec branch.
pub const DEFAULT_DFN3_ATTENUATION_LIMIT_DB: f32 = 18.0;
pub const MAX_DFN3_ATTENUATION_LIMIT_DB: f32 = 100.0;
pub const DEFAULT_DFN3_POST_FILTER_BETA: f32 = 0.0;
pub const MAX_DFN3_POST_FILTER_BETA: f32 = 0.1;
