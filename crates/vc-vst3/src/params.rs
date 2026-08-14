//! Plugin parameters and persisted state.
//!
//! - The live, automatable knobs (`pitch`/`speaker`/gains) are `#[id]` params
//!   owned by the DAW (the host persists/automates them per project).
//! - `settings` (model paths, provider, chunking, thresholds) and the editor
//!   window size are non-parameter state persisted via `#[persist]`, so they
//!   are saved/restored with the project too.

use std::sync::{Arc, RwLock};

use nice_plug::prelude::*;
use nice_plug_egui::EguiState;
use vc_core::model_rvc::{
    DEFAULT_DENOISER_CONTENT_MIX, DEFAULT_DENOISER_RMVPE_MIX, DEFAULT_PROTECT,
    DEFAULT_PROTECT_TRANSITION_MS, MAX_PROTECT, MAX_PROTECT_TRANSITION_MS,
};

use crate::config::PluginConfig;

#[derive(Params)]
pub struct VcRvcParams {
    #[id = "pitch"]
    pub pitch_shift: FloatParam,
    #[id = "speaker"]
    pub speaker_id: IntParam,
    #[id = "ingain"]
    pub input_gain_db: FloatParam,
    #[id = "outgain"]
    pub output_gain_db: FloatParam,
    #[id = "ngate"]
    pub noise_gate: BoolParam,
    /// Gate threshold in dB; converted to a linear amplitude before it reaches
    /// the core gate (`util::db_to_gain`).
    #[id = "ngthr"]
    pub noise_gate_threshold_db: FloatParam,
    /// Share of the cleaned input mixed into ContentVec. This is
    /// live/automatable and does not rebuild the model.
    #[id = "dcontent"]
    pub denoiser_content_mix: FloatParam,
    /// Share of the cleaned input sent to RMVPE. This is live/automatable and
    /// defaults to the historical fully denoised path.
    #[id = "drmvpe"]
    pub denoiser_rmvpe_mix: FloatParam,
    /// RVC index blend and consonant protection are host-automatable because
    /// vc-core consumes them from the worker before each conversion chunk.
    /// Selecting/replacing the index itself remains a reload-scoped setting.
    #[id = "idxrate"]
    pub index_rate: FloatParam,
    #[id = "protect"]
    pub protect: FloatParam,
    /// vc-rs-only easing around Protect boundaries. This stays a DAW parameter
    /// so hosts can A/B it without a pipeline reload; zero is stock RVC.
    #[id = "protecttrans"]
    pub protect_transition_ms: FloatParam,

    /// Model paths and conversion settings. Set via the GUI / config seed and
    /// persisted with the project.
    #[persist = "settings"]
    pub settings: RwLock<PluginConfig>,

    /// Editor window size, persisted so the host remembers it.
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,
}

impl Default for VcRvcParams {
    fn default() -> Self {
        Self {
            pitch_shift: FloatParam::new(
                "Pitch",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_unit(" st")
            .with_step_size(0.5),
            // RVC v2's stock table has 109 rows; MXGF-f048k exports 308. Keep
            // the DAW parameter range at the larger static contract and let
            // vc-core clamp IDs for smaller or unusual models at load time.
            speaker_id: IntParam::new("Speaker", 0, IntRange::Linear { min: 0, max: 307 }),
            input_gain_db: FloatParam::new(
                "Input Gain",
                0.0,
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB"),
            output_gain_db: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: -36.0,
                    max: 36.0,
                },
            )
            .with_unit(" dB"),
            noise_gate: BoolParam::new("Noise Gate", false),
            noise_gate_threshold_db: FloatParam::new(
                "Gate Threshold",
                -40.0,
                FloatRange::Linear {
                    min: -80.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB"),
            denoiser_content_mix: FloatParam::new(
                "Denoiser -> ContentVec",
                DEFAULT_DENOISER_CONTENT_MIX,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            denoiser_rmvpe_mix: FloatParam::new(
                "Denoiser -> RMVPE",
                DEFAULT_DENOISER_RMVPE_MIX,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            index_rate: FloatParam::new(
                "Index Rate",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            ),
            protect: FloatParam::new(
                "Protect",
                DEFAULT_PROTECT,
                FloatRange::Linear {
                    min: 0.0,
                    max: MAX_PROTECT,
                },
            ),
            protect_transition_ms: FloatParam::new(
                "Protect Transition",
                DEFAULT_PROTECT_TRANSITION_MS as f32,
                FloatRange::Linear {
                    min: 0.0,
                    max: MAX_PROTECT_TRANSITION_MS as f32,
                },
            )
            .with_unit(" ms")
            .with_step_size(10.0),
            settings: RwLock::new(PluginConfig::default()),
            editor_state: EguiState::from_size(480, 520),
        }
    }
}
