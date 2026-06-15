use std::time::Duration;

use anyhow::Result;

use crate::sola::{self, ChunkSmoother, ChunkSmootherConfig, JoinDiagnostics, SmoothingKind};

use super::{ModelOutput, VoiceModel};

#[derive(Clone, Copy, Debug)]
pub struct ChunkOutputConfig {
    pub kind: SmoothingKind,
    pub output_sample_rate: u32,
    pub output_chunk_samples: usize,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub tail_discard_ms: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkStats {
    pub silent: bool,
    pub inference_time: Duration,
    pub input_rms: f32,
    pub output_rms: f32,
    pub model_output_samples: usize,
}

/// Owns the stateful model-to-fixed-output conversion shared by WAV and the
/// worker-side realtime paths.
///
/// Keep this off audio callbacks: model processing, smoothing, and resampling
/// may allocate. Output settings are intentionally fixed for the converter
/// lifetime; rebuild the converter together with the model when a stream's chunk
/// size or output format changes.
pub struct ChunkConverter<M> {
    model: M,
    output: ChunkOutputConfig,
    smoother: Option<(u32, ChunkSmoother)>,
    // Reused per-chunk buffers for the model's converted audio and output
    // pitchf, so `process_chunk` does not allocate them every chunk.
    model_audio: Vec<f32>,
    model_pitchf: Vec<f32>,
}

impl<M: VoiceModel> ChunkConverter<M> {
    pub fn new(model: M, output: ChunkOutputConfig) -> Self {
        Self {
            model,
            output,
            smoother: None,
            model_audio: Vec::new(),
            model_pitchf: Vec::new(),
        }
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Diagnostics for the most recent [`Self::process_chunk`] / [`Self::prime`]
    /// join, or `None` before the first chunk builds the smoother. Diagnostics
    /// only (offline join analysis); realtime callers can ignore it.
    pub fn last_join_diagnostics(&self) -> Option<JoinDiagnostics> {
        self.smoother
            .as_ref()
            .map(|(_, smoother)| smoother.last_diagnostics())
    }

    /// The configured crossfade window in model-domain samples, or `None` before
    /// the first chunk builds the smoother. Diagnostics: comparing this against a
    /// chunk's `crossfade_len` shows when the output chunk was shorter than the
    /// crossfade window (the join then can't use the full overlap).
    pub fn join_crossfade_samples(&self) -> Option<usize> {
        self.smoother
            .as_ref()
            .map(|(_, smoother)| smoother.crossfade_samples())
    }

    /// Discards output-joining history without rebuilding the owned model.
    ///
    /// Realtime callers use this after a period where model processing was
    /// paused. Reusing the old smoother history would join fresh model output
    /// against audio emitted before the pause.
    pub fn reset_streaming_state(&mut self) {
        self.smoother = None;
    }

    pub fn process_chunk(
        &mut self,
        input: &[f32],
        input_sample_rate: u32,
        final_tail: Option<&mut Vec<f32>>,
        out: &mut Vec<f32>,
    ) -> Result<ChunkStats> {
        let meta = self.model.process(
            input,
            input_sample_rate,
            &mut self.model_audio,
            &mut self.model_pitchf,
        )?;
        let stats = chunk_stats(&meta, self.model_audio.len());
        let model_sample_rate = meta.sample_rate;
        let output_sample_rate = self.output.output_sample_rate;
        let output_chunk_samples = self.output.output_chunk_samples;
        self.ensure_smoother(model_sample_rate);
        // Disjoint field borrows: the smoother and the model output buffers are
        // separate fields, so this does not conflict.
        let smoother = &mut self.smoother.as_mut().expect("smoother set above").1;
        sola::prepare_model_output(
            &self.model_audio,
            &self.model_pitchf,
            model_sample_rate,
            output_sample_rate,
            output_chunk_samples,
            smoother,
            final_tail,
            out,
        )?;
        Ok(stats)
    }

    /// Runs the same model/smoother initialization path as a normal chunk but
    /// emits no audio. WAV conversion uses this for its historical silent
    /// preroll; realtime paths deliberately start from their first real chunk.
    pub fn prime(&mut self, input: &[f32], input_sample_rate: u32) -> Result<ChunkStats> {
        let meta = self.model.process(
            input,
            input_sample_rate,
            &mut self.model_audio,
            &mut self.model_pitchf,
        )?;
        let stats = chunk_stats(&meta, self.model_audio.len());
        self.ensure_smoother(meta.sample_rate);
        let smoother = &mut self.smoother.as_mut().expect("smoother set above").1;
        smoother.prime_model_output(&self.model_audio, &self.model_pitchf);
        Ok(stats)
    }

    /// Ensures `self.smoother` matches `model_sample_rate`, rebuilding it on a
    /// rate change. Split out of the per-chunk path so callers can then take a
    /// disjoint borrow of the smoother alongside the model output buffers.
    fn ensure_smoother(&mut self, model_sample_rate: u32) {
        if self.smoother.as_ref().map(|(rate, _)| *rate) != Some(model_sample_rate) {
            self.smoother = Some((
                model_sample_rate,
                sola::model_domain_chunk_smoother(ChunkSmootherConfig {
                    kind: self.output.kind,
                    output_chunk_samples: self.output.output_chunk_samples,
                    output_sample_rate: self.output.output_sample_rate,
                    model_sample_rate,
                    crossfade_ms: self.output.crossfade_ms,
                    sola_search_ms: self.output.sola_search_ms,
                    tail_discard_ms: self.output.tail_discard_ms,
                }),
            ));
        }
    }
}

fn chunk_stats(meta: &ModelOutput, model_output_samples: usize) -> ChunkStats {
    ChunkStats {
        silent: meta.silent,
        inference_time: meta.inference_time,
        input_rms: meta.input_rms,
        output_rms: meta.output_rms,
        // This is the length immediately before smoothing, not the pipeline's
        // separately reported `raw_output_samples` diagnostic.
        model_output_samples,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use anyhow::{anyhow, Result};

    use super::*;

    struct FakeModel {
        // Each entry pairs the model-domain audio the fake emits with its
        // metadata; `process` writes the audio into the caller's out buffer.
        outputs: VecDeque<Result<(Vec<f32>, ModelOutput)>>,
        calls: usize,
    }

    impl FakeModel {
        fn new(outputs: impl IntoIterator<Item = Result<(Vec<f32>, ModelOutput)>>) -> Self {
            Self {
                outputs: outputs.into_iter().collect(),
                calls: 0,
            }
        }
    }

    impl VoiceModel for FakeModel {
        fn process(
            &mut self,
            _audio: &[f32],
            _sample_rate: u32,
            out_audio: &mut Vec<f32>,
            out_pitchf: &mut Vec<f32>,
        ) -> Result<ModelOutput> {
            self.calls += 1;
            let (audio, meta) = self.outputs.pop_front().expect("fake output")?;
            out_audio.clear();
            out_audio.extend_from_slice(&audio);
            out_pitchf.clear();
            Ok(meta)
        }
    }

    fn config() -> ChunkOutputConfig {
        ChunkOutputConfig {
            kind: SmoothingKind::Sola,
            output_sample_rate: 1_000,
            output_chunk_samples: 4,
            crossfade_ms: 2,
            sola_search_ms: 2,
            tail_discard_ms: 0,
        }
    }

    fn output(audio: Vec<f32>, sample_rate: u32) -> (Vec<f32>, ModelOutput) {
        let meta = ModelOutput {
            raw_output_samples: 999,
            output_rms: 0.75,
            convert_size: audio.len(),
            out_size: audio.len(),
            model_input_samples: audio.len(),
            sample_rate,
            inference_time: Duration::from_micros(123),
            embedder_time: Duration::ZERO,
            pitch_time: Duration::ZERO,
            rvc_time: Duration::ZERO,
            input_rms: 0.25,
            voiced_ratio: 0.0,
            applied_output_gain: 1.0,
            feature_frames: 0,
            pitch_frames: 0,
            silent: true,
            volume: 0.0,
        };
        (audio, meta)
    }

    #[test]
    fn processes_once_and_returns_fixed_audio_and_stats() {
        let mut converter =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![1.0; 8], 1_000))]), config());

        let mut out = Vec::new();
        let stats = converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut out)
            .unwrap();

        assert_eq!(converter.model_mut().calls, 1);
        assert_eq!(out.len(), 4);
        assert!(stats.silent);
        assert_eq!(stats.inference_time, Duration::from_micros(123));
        assert_eq!(stats.input_rms, 0.25);
        assert_eq!(stats.output_rms, 0.75);
        assert_eq!(stats.model_output_samples, 8);
    }

    #[test]
    fn smoother_persists_until_model_rate_changes() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
            Ok(output(vec![3.0; 16], 2_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());

        let mut first = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut first)
            .unwrap();
        let mut second = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut second)
            .unwrap();
        let mut changed_rate = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut changed_rate)
            .unwrap();

        assert_eq!(first, vec![0.0; 4]);
        assert_ne!(second, vec![0.0; 4]);
        assert_eq!(changed_rate, vec![0.0; 4]);
    }

    #[test]
    fn reset_streaming_state_discards_smoother_history() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
            Ok(output(vec![3.0; 8], 1_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());

        let mut scratch = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut scratch)
            .unwrap();
        let mut joined = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut joined)
            .unwrap();
        assert_ne!(joined, vec![0.0; 4]);

        converter.reset_streaming_state();
        let mut reset = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut reset)
            .unwrap();
        assert_eq!(reset, vec![0.0; 4]);
    }

    #[test]
    fn prime_initializes_smoother_without_emitting_audio() {
        let prime_audio = vec![0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0];
        let real_audio = vec![0.1, 0.2, 1.0, 0.5, 6.0, 7.0, 8.0, 9.0];
        let mut converter = ChunkConverter::new(
            FakeModel::new([
                Ok(output(prime_audio, 1_000)),
                Ok(output(real_audio.clone(), 1_000)),
            ]),
            config(),
        );
        let mut without_prime =
            ChunkConverter::new(FakeModel::new([Ok(output(real_audio, 1_000))]), config());

        let stats = converter.prime(&[0.0; 4], 1_000).unwrap();
        let mut primed = Vec::new();
        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut primed)
            .unwrap();
        let mut unprimed = Vec::new();
        without_prime
            .process_chunk(&[0.0; 4], 1_000, None, &mut unprimed)
            .unwrap();

        assert_eq!(stats.model_output_samples, 8);
        assert_ne!(primed, unprimed);
    }

    #[test]
    fn final_tail_is_only_updated_when_requested() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());
        let mut tail = vec![9.0];
        let mut out = Vec::new();

        converter
            .process_chunk(&[0.0; 4], 1_000, None, &mut out)
            .unwrap();
        assert_eq!(tail, vec![9.0]);
        converter
            .process_chunk(&[0.0; 4], 1_000, Some(&mut tail), &mut out)
            .unwrap();
        assert_ne!(tail, vec![9.0]);
    }

    #[test]
    fn model_and_output_errors_are_returned() {
        let mut out = Vec::new();
        let mut model_error =
            ChunkConverter::new(FakeModel::new([Err(anyhow!("model failed"))]), config());
        assert!(model_error
            .process_chunk(&[0.0; 4], 1_000, None, &mut out)
            .is_err());

        let mut output_error =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![1.0; 8], 0))]), config());
        assert!(output_error
            .process_chunk(&[0.0; 4], 1_000, None, &mut out)
            .is_err());
    }
}
