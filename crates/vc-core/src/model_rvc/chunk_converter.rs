use std::time::Duration;

use anyhow::Result;

use crate::sola::{self, ChunkSmoother, ChunkSmootherConfig, SmoothingKind};

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

pub struct ConvertedChunk {
    pub audio: Vec<f32>,
    pub stats: ChunkStats,
}

/// Owns the stateful model-to-fixed-output conversion shared by WAV and the
/// worker-side realtime paths.
///
/// Keep this off audio callbacks: model processing, smoothing, resampling, and
/// the returned `Vec` may all allocate. Output settings are intentionally fixed
/// for the converter lifetime; rebuild the converter together with the model
/// when a stream's chunk size or output format changes.
pub struct ChunkConverter<M> {
    model: M,
    output: ChunkOutputConfig,
    smoother: Option<(u32, ChunkSmoother)>,
}

impl<M: VoiceModel> ChunkConverter<M> {
    pub fn new(model: M, output: ChunkOutputConfig) -> Self {
        Self {
            model,
            output,
            smoother: None,
        }
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
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
    ) -> Result<ConvertedChunk> {
        let out = self.model.process(input, input_sample_rate)?;
        let stats = chunk_stats(&out);
        let model_sample_rate = out.sample_rate;
        let output_sample_rate = self.output.output_sample_rate;
        let output_chunk_samples = self.output.output_chunk_samples;
        let smoother = self.smoother_for(model_sample_rate);
        let prepared = sola::prepare_model_output(
            out,
            output_sample_rate,
            output_chunk_samples,
            smoother,
            final_tail,
        )?;
        Ok(ConvertedChunk {
            audio: prepared.audio,
            stats,
        })
    }

    /// Runs the same model/smoother initialization path as a normal chunk but
    /// emits no audio. WAV conversion uses this for its historical silent
    /// preroll; realtime paths deliberately start from their first real chunk.
    pub fn prime(&mut self, input: &[f32], input_sample_rate: u32) -> Result<ChunkStats> {
        let out = self.model.process(input, input_sample_rate)?;
        let stats = chunk_stats(&out);
        self.smoother_for(out.sample_rate)
            .prime_model_output(&out.audio, &out.pitchf);
        Ok(stats)
    }

    fn smoother_for(&mut self, model_sample_rate: u32) -> &mut ChunkSmoother {
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
        &mut self.smoother.as_mut().expect("smoother set above").1
    }
}

fn chunk_stats(out: &ModelOutput) -> ChunkStats {
    ChunkStats {
        silent: out.silent,
        inference_time: out.inference_time,
        input_rms: out.input_rms,
        output_rms: out.output_rms,
        // This is the length immediately before smoothing, not the pipeline's
        // separately reported `raw_output_samples` diagnostic.
        model_output_samples: out.audio.len(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use anyhow::{anyhow, Result};

    use super::*;

    struct FakeModel {
        outputs: VecDeque<Result<ModelOutput>>,
        calls: usize,
    }

    impl FakeModel {
        fn new(outputs: impl IntoIterator<Item = Result<ModelOutput>>) -> Self {
            Self {
                outputs: outputs.into_iter().collect(),
                calls: 0,
            }
        }
    }

    impl VoiceModel for FakeModel {
        fn process(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<ModelOutput> {
            self.calls += 1;
            self.outputs.pop_front().expect("fake output")
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

    fn output(audio: Vec<f32>, sample_rate: u32) -> ModelOutput {
        ModelOutput {
            raw_output_samples: 999,
            output_rms: 0.75,
            convert_size: audio.len(),
            out_size: audio.len(),
            model_input_samples: audio.len(),
            audio,
            pitchf: Vec::new(),
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
        }
    }

    #[test]
    fn processes_once_and_returns_fixed_audio_and_stats() {
        let mut converter =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![1.0; 8], 1_000))]), config());

        let converted = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();

        assert_eq!(converter.model_mut().calls, 1);
        assert_eq!(converted.audio.len(), 4);
        assert!(converted.stats.silent);
        assert_eq!(converted.stats.inference_time, Duration::from_micros(123));
        assert_eq!(converted.stats.input_rms, 0.25);
        assert_eq!(converted.stats.output_rms, 0.75);
        assert_eq!(converted.stats.model_output_samples, 8);
    }

    #[test]
    fn smoother_persists_until_model_rate_changes() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
            Ok(output(vec![3.0; 16], 2_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());

        let first = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        let second = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        let changed_rate = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();

        assert_eq!(first.audio, vec![0.0; 4]);
        assert_ne!(second.audio, vec![0.0; 4]);
        assert_eq!(changed_rate.audio, vec![0.0; 4]);
    }

    #[test]
    fn reset_streaming_state_discards_smoother_history() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
            Ok(output(vec![3.0; 8], 1_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());

        converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        let joined = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        assert_ne!(joined.audio, vec![0.0; 4]);

        converter.reset_streaming_state();
        let reset = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        assert_eq!(reset.audio, vec![0.0; 4]);
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
        let primed = converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        let unprimed = without_prime.process_chunk(&[0.0; 4], 1_000, None).unwrap();

        assert_eq!(stats.model_output_samples, 8);
        assert_ne!(primed.audio, unprimed.audio);
    }

    #[test]
    fn final_tail_is_only_updated_when_requested() {
        let outputs = [
            Ok(output(vec![1.0; 8], 1_000)),
            Ok(output(vec![2.0; 8], 1_000)),
        ];
        let mut converter = ChunkConverter::new(FakeModel::new(outputs), config());
        let mut tail = vec![9.0];

        converter.process_chunk(&[0.0; 4], 1_000, None).unwrap();
        assert_eq!(tail, vec![9.0]);
        converter
            .process_chunk(&[0.0; 4], 1_000, Some(&mut tail))
            .unwrap();
        assert_ne!(tail, vec![9.0]);
    }

    #[test]
    fn model_and_output_errors_are_returned() {
        let mut model_error =
            ChunkConverter::new(FakeModel::new([Err(anyhow!("model failed"))]), config());
        assert!(model_error.process_chunk(&[0.0; 4], 1_000, None).is_err());

        let mut output_error =
            ChunkConverter::new(FakeModel::new([Ok(output(vec![1.0; 8], 0))]), config());
        assert!(output_error.process_chunk(&[0.0; 4], 1_000, None).is_err());
    }
}
