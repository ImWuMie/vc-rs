use std::path::Path;
#[cfg(feature = "ort")]
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
#[cfg(feature = "ort")]
use ort::ep;
#[cfg(feature = "ort")]
use ort::memory::Allocator;
#[cfg(feature = "ort")]
use ort::session::builder::GraphOptimizationLevel;
#[cfg(feature = "ort")]
use ort::session::{IoBinding, Session};
#[cfg(feature = "ort")]
use ort::value::{Tensor, TensorRef, ValueType};
#[cfg(feature = "ort")]
use tracing::debug;
use tracing::info;

use crate::Provider;

use super::feature::FeatureTensor;
use super::native_tensorrt::{
    NativeContentVecEngine, NativeFcpeEngine, NativeRmvpeEngine, NativeRvcEngine,
};
#[cfg(feature = "ort")]
use super::noise::{GaussianNoise, RVC_RND_SEED};
use super::onnx_meta::{read_model_io, RvcIoNames};
use super::tensorrt::*;

/// Validate and copy an embedder output into a reused `FeatureTensor`.
///
/// Model outputs are an external trust boundary.  Keep this check on the
/// worker-side inference path so malformed exports return a normal error
/// before `FeatureTensor::repeat_frames`/`copy_within` can observe a bad
/// batch, channel count, data length, or non-finite value.  The success path
/// only scans the values that are copied already and performs no allocation.
fn validate_feature_tensor(shape: &[i64], data: &[f32], expected_channels: usize) -> Result<()> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] <= 0 {
        bail!("embedder output must have shape [1, frames>0, channels], got {shape:?}");
    }
    let channels = usize::try_from(shape[2]).context("embedder output has invalid channels")?;
    if channels != expected_channels {
        bail!(
            "embedder output channel count changed: got {}, expected {}",
            channels,
            expected_channels
        );
    }
    let frames = usize::try_from(shape[1]).context("embedder output has invalid frames")?;
    let expected_len = frames
        .checked_mul(channels)
        .ok_or_else(|| anyhow!("embedder output shape volume overflows usize"))?;
    if data.len() != expected_len {
        bail!(
            "embedder output contains {} values, expected {} for shape {shape:?}",
            data.len(),
            expected_len
        );
    }
    if data.iter().any(|value| !value.is_finite()) {
        bail!("embedder output contains non-finite values");
    }
    Ok(())
}

fn fill_feature_tensor(
    out: &mut FeatureTensor,
    shape: &[i64],
    data: &[f32],
    expected_channels: usize,
) -> Result<()> {
    validate_feature_tensor(shape, data, expected_channels)?;
    out.data.clear();
    out.data.extend_from_slice(data);
    out.shape.clear();
    out.shape.extend_from_slice(shape);
    Ok(())
}

fn validate_rvc_audio_shape(shape: &[i64], data_len: Option<usize>) -> Result<()> {
    // Community exporters differ only in whether they preserve the two
    // singleton axes around mono audio. Treat those layouts as equivalent,
    // while retaining an exact-volume check before any downstream copy or
    // smoothing code can observe an inconsistent tensor.
    let samples = match shape {
        [samples] if *samples > 0 => *samples,
        [batch, samples] if *batch == 1 && *samples > 0 => *samples,
        [batch, channels, samples] if *batch == 1 && *channels == 1 && *samples > 0 => *samples,
        _ => {
            bail!(
                "RVC audio output must have shape [samples>0], [1, samples>0], or [1, 1, samples>0]; got {shape:?}"
            )
        }
    };
    let samples = usize::try_from(samples).context("RVC audio output has invalid sample count")?;
    if let Some(data_len) = data_len {
        if data_len != samples {
            bail!(
                "RVC audio output contains {} values, expected {} for shape {shape:?}",
                data_len,
                samples
            );
        }
    }
    Ok(())
}

fn validate_rvc_audio_data(data: &[f32]) -> Result<()> {
    if data.iter().any(|sample| !sample.is_finite()) {
        bail!("RVC audio output contains non-finite values");
    }
    Ok(())
}

#[cfg(test)]
mod output_contract_tests {
    use super::{
        validate_feature_tensor, validate_rmvpe_pitch_data, validate_rmvpe_pitch_shape,
        validate_rvc_audio_data, validate_rvc_audio_shape,
    };

    #[test]
    fn validates_contentvec_output_shape_volume_channels_and_values() {
        validate_feature_tensor(&[1, 2, 3], &[0.0; 6], 3).unwrap();

        for (shape, data, channels) in [
            (&[2_i64, 2, 3][..], &[0.0; 12][..], 3),
            (&[1, 0, 3][..], &[][..], 3),
            (&[1, 2, 4][..], &[0.0; 8][..], 3),
            (&[1, 2, 3][..], &[0.0; 5][..], 3),
        ] {
            assert!(
                validate_feature_tensor(shape, data, channels).is_err(),
                "shape={shape:?} data_len={} channels={channels}",
                data.len()
            );
        }
        assert!(validate_feature_tensor(&[1, 1, 2], &[0.0, f32::NAN], 2).is_err());
    }

    #[test]
    fn validates_common_rmvpe_output_layouts_and_values() {
        for shape in [&[32_i64][..], &[1, 32][..], &[1, 32, 1][..]] {
            validate_rmvpe_pitch_shape(shape, Some(32)).unwrap();
        }
        for shape in [
            &[][..],
            &[0_i64][..],
            &[2, 32][..],
            &[1, 32, 2][..],
            &[1, 1, 32, 1][..],
        ] {
            assert!(
                validate_rmvpe_pitch_shape(shape, None).is_err(),
                "shape={shape:?}"
            );
        }
        assert!(validate_rmvpe_pitch_shape(&[1, 32], Some(31)).is_err());
        assert!(validate_rmvpe_pitch_data(&[1.0, f32::INFINITY]).is_err());
        validate_rmvpe_pitch_data(&[0.0, 220.0]).unwrap();
    }

    #[test]
    fn accepts_common_mono_rvc_output_layouts() {
        for shape in [&[320_i64][..], &[1, 320][..], &[1, 1, 320][..]] {
            validate_rvc_audio_shape(shape, Some(320)).unwrap();
        }
    }

    #[test]
    fn rejects_invalid_rvc_output_layout_or_volume() {
        for shape in [
            &[][..],
            &[0_i64][..],
            &[2, 320][..],
            &[1, 2, 320][..],
            &[1, 1, 0][..],
            &[1, 1, 1, 320][..],
        ] {
            assert!(
                validate_rvc_audio_shape(shape, None).is_err(),
                "shape={shape:?}"
            );
        }
        assert!(validate_rvc_audio_shape(&[1, 320], Some(319)).is_err());
    }

    #[test]
    fn rejects_non_finite_rvc_audio() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_rvc_audio_data(&[0.0, value]).is_err());
        }
        validate_rvc_audio_data(&[-1.0, 0.0, 1.0]).unwrap();
    }
}

fn validate_rmvpe_pitch_shape(shape: &[i64], data_len: Option<usize>) -> Result<()> {
    let samples = match shape {
        [frames] if *frames > 0 => {
            usize::try_from(*frames).context("RMVPE frame count overflow")?
        }
        [batch, frames] if *batch == 1 && *frames > 0 => {
            usize::try_from(*frames).context("RMVPE frame count overflow")?
        }
        [batch, frames, channels] if *batch == 1 && *channels == 1 && *frames > 0 => {
            usize::try_from(*frames).context("RMVPE frame count overflow")?
        }
        _ => {
            bail!(
                "RMVPE pitchf output must have shape [frames], [1, frames], or [1, frames, 1]; got {shape:?}"
            )
        }
    };
    if let Some(data_len) = data_len {
        if data_len != samples {
            bail!(
                "RMVPE pitchf output contains {} values, expected {} for shape {shape:?}",
                data_len,
                samples
            );
        }
    }
    Ok(())
}

fn validate_rmvpe_pitch_data(data: &[f32]) -> Result<()> {
    if data.iter().any(|value| !value.is_finite()) {
        bail!("RMVPE pitchf output contains non-finite values");
    }
    Ok(())
}

pub(super) struct HubertEmbedderSession {
    // Present only in the ORT build, where it backs CPU/CUDA inference. The
    // native TensorRT-only build drops ORT entirely and runs through `native`.
    #[cfg(feature = "ort")]
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<HubertTensorRtBinding>,
    native: Option<NativeContentVecEngine>,
    pub(super) input_name: String,
    pub(super) output_name: String,
    expected_channels: usize,
}

impl HubertEmbedderSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        expected_channels: i64,
        requested_output: Option<&str>,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        // Names/validation come from the provider-neutral ONNX reader so the
        // native TensorRT path needs no ORT session.
        let io = read_model_io(path)?;
        let input_name = io.single_input_name()?.to_string();
        let output_name = io.select_embedder_output(expected_channels, requested_output)?;
        let expected_channels_i64 = expected_channels;
        let expected_channels = usize::try_from(expected_channels)
            .context("ContentVec expected channel count does not fit usize")?;
        if expected_channels == 0 {
            bail!("ContentVec expected channel count must be positive");
        }
        let native = if provider.is_tensorrt() {
            let profile = tensor_rt_profile.as_ref().ok_or_else(|| {
                anyhow!("native TensorRT ContentVec requires a fixed-shape profile")
            })?;
            Some(NativeContentVecEngine::load(
                path,
                profile,
                input_name.as_str(),
                output_name.as_str(),
                expected_channels_i64,
            )?)
        } else {
            None
        };
        // In the ORT build the session backs CPU/CUDA inference; the native
        // TensorRT path keeps a CPU session only as an unused placeholder there
        // (it is compiled out of the TensorRT-only build).
        #[cfg(feature = "ort")]
        let session = {
            let session_provider = if provider.is_tensorrt() {
                Provider::Cpu
            } else {
                provider
            };
            load_session(
                path,
                session_provider,
                ModelRole::ContentVec,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?
        };
        info!(
            "loaded embedder: {} input={} output={}",
            path.display(),
            input_name,
            output_name
        );
        Ok(Self {
            #[cfg(feature = "ort")]
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            #[cfg(feature = "ort")]
            tensor_rt_binding: None,
            native,
            input_name,
            output_name,
            expected_channels,
        })
    }

    /// ContentVec output frame count from the native TensorRT engine, when this
    /// embedder is backed by one. `None` for ORT-backed sessions. The engine
    /// self-reports its fixed output length, so this needs no warmup inference.
    pub(super) fn native_contentvec_output_frames(&self) -> Option<Result<usize>> {
        self.native.as_ref().map(|native| native.output_frames())
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        shared_waveform: Option<&TensorRtSharedWaveform>,
    ) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) {
            return Ok(());
        }
        if self.native.is_some() {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("ContentVec IoBinding requires a fixed-shape profile"))?;
        let input_shape = profile.fixed_input_dims(self.input_name.as_str())?;
        let output_shape = i64_shape_to_usize(output_shape, "contentvec output")?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                HubertTensorRtBinding::Pinned(HubertTensorRtPinnedBinding::new(
                    &self.session,
                    self.input_name.as_str(),
                    input_shape,
                    self.output_name.as_str(),
                    &output_shape,
                    profile.gpu_device_id,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let mut binding = HubertTensorRtGraphBinding::new(
                    &self.session,
                    self.input_name.as_str(),
                    input_shape,
                    self.output_name.as_str(),
                    &output_shape,
                    shared_waveform,
                    profile.gpu_device_id,
                )?;
                binding.warmup_capture(
                    &mut self.session,
                    self.output_name.as_str(),
                    ModelRole::ContentVec,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                HubertTensorRtBinding::CudaGraph(binding)
            }
        };
        let shared_waveform_input = match &binding {
            HubertTensorRtBinding::Pinned(_) => false,
            HubertTensorRtBinding::CudaGraph(binding) => binding.shared_waveform_input,
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input={} input_shape={} output={} output_shape={} shared_waveform_input={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::ContentVec.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            self.input_name,
            format_usize_shape(input_shape),
            self.output_name,
            format_usize_shape(&output_shape),
            shared_waveform_input,
            self.tensor_rt_run_mode.bound_input_memory(),
            self.tensor_rt_run_mode.bound_output_memory()
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    /// Runs ContentVec, writing the rank-3 feature tensor into `out` (a
    /// caller-owned buffer reused across chunks) instead of allocating a fresh
    /// `FeatureTensor` each call.
    pub(super) fn extract_into(
        &mut self,
        audio_16k: &[f32],
        out: &mut FeatureTensor,
    ) -> Result<()> {
        let input_shape = [1usize, audio_16k.len()];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            self.input_name.as_str(),
            &input_shape,
        )?;
        #[cfg(feature = "ort")]
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k, out);
        }
        let expected_channels = self.expected_channels;
        if let Some(native) = self.native.as_mut() {
            native.extract_into(audio_16k, out)?;
            validate_feature_tensor(&out.shape, &out.data, expected_channels)?;
            return Ok(());
        }
        #[cfg(feature = "ort")]
        {
            self.extract_with_session_run(audio_16k, &input_shape, out)
        }
        #[cfg(not(feature = "ort"))]
        {
            let _ = out;
            bail!("ContentVec session inference requires the `ort` feature; this build supports native TensorRT only")
        }
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_session_run(
        &mut self,
        audio_16k: &[f32],
        input_shape: &[usize; 2],
        out: &mut FeatureTensor,
    ) -> Result<()> {
        // Borrow the worker-owned input slice for synchronous ORT runs. Using
        // Tensor::from_array here would allocate and copy the full waveform on
        // every realtime chunk.
        let input = TensorRef::from_array_view((*input_shape, audio_16k))?;
        let run_start = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;
        debug!(
            "embedder session.run backend={} input={} shape={} elapsed_us={}",
            self.provider.label(),
            self.input_name,
            format_usize_shape(input_shape),
            run_start.elapsed().as_micros()
        );
        let value = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| anyhow!("embedder output '{}' not found", self.output_name))?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        if shape.len() != 3 {
            bail!("embedder output must be rank-3 [1, frames, channels], got {shape}");
        }
        fill_feature_tensor(out, shape, data, self.expected_channels)
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_binding(
        &mut self,
        audio_16k: &[f32],
        out: &mut FeatureTensor,
    ) -> Result<()> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT ContentVec IoBinding is not initialized"))?;
        match binding {
            HubertTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.audio, audio_16k, "audio")?;
                binding
                    .binding
                    .bind_input(self.input_name.as_str(), &binding.audio)
                    .with_context(|| {
                        format!(
                            "failed to bind TensorRT ContentVec input '{}'",
                            self.input_name
                        )
                    })?;
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT ContentVec bound output")?;
                debug!(
                    "embedder session.run_binding backend={} cuda_graph=false device_io=false input={} shape={} output={} output_shape={} elapsed_us={}",
                    self.provider.label(),
                    self.input_name,
                    format_usize_shape(&binding.input_shape),
                    self.output_name,
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                if shape.len() != 3 {
                    bail!("embedder output must be rank-3 [1, frames, channels], got {shape}");
                }
                let actual_shape = i64_shape_to_usize(shape, "contentvec output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT ContentVec bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                fill_feature_tensor(out, shape, data, self.expected_channels)
            }
            HubertTensorRtBinding::CudaGraph(binding) => {
                let h2d_us = binding
                    .copy_audio_to_device_if_owned(audio_16k, self.input_name.as_str())?
                    .unwrap_or(0);
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                let d2h_start = Instant::now();
                copy_f32_tensor_to_host(
                    &binding.device_output,
                    &mut binding.host_output,
                    self.output_name.as_str(),
                )?;
                let d2h_us = d2h_start.elapsed().as_micros();
                debug!(
                    "embedder session.run_binding(device_io=true) backend={} cuda_graph={} shared_waveform_input={} input={} shape={} output={} output_shape={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    binding.shared_waveform_input,
                    self.input_name,
                    format_usize_shape(&binding.input_shape),
                    self.output_name,
                    format_usize_shape(&binding.output_shape),
                    h2d_us,
                    run_us,
                    d2h_us,
                    h2d_us + run_us + d2h_us
                );
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                if shape.len() != 3 {
                    bail!("embedder output must be rank-3 [1, frames, channels], got {shape}");
                }
                let actual_shape = i64_shape_to_usize(shape, "contentvec output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT ContentVec bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                fill_feature_tensor(out, shape, data, self.expected_channels)
            }
        }
    }
}

pub(super) struct RmvpePitchSession {
    #[cfg(feature = "ort")]
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<RmvpeTensorRtBinding>,
    native: Option<NativeRmvpeEngine>,
    // Reused buffer for the pitch-shift-scaled F0 output, so `extract` does not
    // allocate a fresh Vec every chunk. Returned as a borrowed slice.
    pitchf_scratch: Vec<f32>,
}

impl RmvpePitchSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        let io = read_model_io(path)?;
        io.validate_rmvpe_contract()?;
        let native = if provider.is_tensorrt() {
            let profile = tensor_rt_profile
                .as_ref()
                .ok_or_else(|| anyhow!("native TensorRT RMVPE requires a fixed-shape profile"))?;
            Some(NativeRmvpeEngine::load(path, profile)?)
        } else {
            None
        };
        #[cfg(feature = "ort")]
        let session = {
            let session_provider = if provider.is_tensorrt() {
                Provider::Cpu
            } else {
                provider
            };
            load_session(
                path,
                session_provider,
                ModelRole::Rmvpe,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?
        };
        info!("loaded RMVPE f0 model: {}", path.display());
        Ok(Self {
            #[cfg(feature = "ort")]
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            #[cfg(feature = "ort")]
            tensor_rt_binding: None,
            native,
            pitchf_scratch: Vec::new(),
        })
    }

    pub(super) fn warmup_output_shape(
        &mut self,
        audio_16k_samples: usize,
        threshold: f32,
    ) -> Result<Vec<i64>> {
        let waveform_shape = [1usize, audio_16k_samples];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "waveform",
            &waveform_shape,
        )?;
        if let Some(native) = self.native.as_ref() {
            return Ok(native.warmup_output_shape());
        }
        #[cfg(feature = "ort")]
        {
            let waveform = Tensor::from_array((waveform_shape, vec![0.0f32; audio_16k_samples]))?;
            let threshold = Tensor::from_array(([1usize], vec![threshold]))?;
            let run_start = Instant::now();
            let outputs = self.session.run(ort::inputs![
                "waveform" => waveform,
                "threshold" => threshold,
            ])?;
            debug!(
                "rmvpe warmup session.run backend={} input=waveform shape={} elapsed_us={}",
                self.provider.label(),
                format_usize_shape(&waveform_shape),
                run_start.elapsed().as_micros()
            );
            let value = outputs
                .get("pitchf")
                .ok_or_else(|| anyhow!("RMVPE output 'pitchf' not found"))?;
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            validate_rmvpe_pitch_shape(shape, Some(data.len()))?;
            validate_rmvpe_pitch_data(data)?;
            Ok(shape.to_vec())
        }
        #[cfg(not(feature = "ort"))]
        bail!("RMVPE warmup requires the `ort` feature; native TensorRT reports its own shape")
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        threshold: f32,
        shared_waveform: Option<&TensorRtSharedWaveform>,
    ) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) {
            return Ok(());
        }
        if self.native.is_some() {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("RMVPE IoBinding requires a fixed-shape profile"))?;
        let waveform_shape = profile.fixed_input_dims("waveform")?;
        let output_shape = i64_shape_to_usize(output_shape, "rmvpe output")?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                RmvpeTensorRtBinding::Pinned(RmvpeTensorRtPinnedBinding::new(
                    &self.session,
                    waveform_shape,
                    &output_shape,
                    threshold,
                    profile.gpu_device_id,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let mut binding = RmvpeTensorRtGraphBinding::new(
                    &self.session,
                    waveform_shape,
                    &output_shape,
                    threshold,
                    shared_waveform,
                    profile.gpu_device_id,
                )?;
                binding.warmup_capture(
                    &mut self.session,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                RmvpeTensorRtBinding::CudaGraph(binding)
            }
        };
        let shared_waveform_input = match &binding {
            RmvpeTensorRtBinding::Pinned(_) => false,
            RmvpeTensorRtBinding::CudaGraph(binding) => binding.shared_waveform_input,
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input=waveform input_shape={} output=pitchf output_shape={} shared_waveform_input={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::Rmvpe.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            format_usize_shape(waveform_shape),
            format_usize_shape(&output_shape),
            shared_waveform_input,
            self.tensor_rt_run_mode.bound_input_memory(),
            self.tensor_rt_run_mode.bound_output_memory()
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    /// Extracts F0, returning a borrowed slice into a reused internal buffer so
    /// the realtime path does not allocate a fresh Vec each chunk. The result is
    /// valid until the next `extract` call.
    pub(super) fn extract(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
    ) -> Result<&[f32]> {
        let waveform_shape = [1usize, audio_16k.len()];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "waveform",
            &waveform_shape,
        )?;
        #[cfg(feature = "ort")]
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k, pitch_shift, threshold);
        }
        if let Some(native) = self.native.as_mut() {
            self.pitchf_scratch.clear();
            native.extract_into(audio_16k, pitch_shift, threshold, &mut self.pitchf_scratch)?;
            if self.pitchf_scratch.is_empty() {
                bail!("native TensorRT RMVPE returned an empty pitchf output");
            }
            validate_rmvpe_pitch_data(&self.pitchf_scratch)?;
            return Ok(self.pitchf_scratch.as_slice());
        }
        #[cfg(feature = "ort")]
        {
            self.extract_with_session_run(audio_16k, pitch_shift, threshold, &waveform_shape)
        }
        #[cfg(not(feature = "ort"))]
        bail!("RMVPE session inference requires the `ort` feature; this build supports native TensorRT only")
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_session_run(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
        waveform_shape: &[usize; 2],
    ) -> Result<&[f32]> {
        let threshold_value = [threshold];
        let waveform = TensorRef::from_array_view((*waveform_shape, audio_16k))?;
        let threshold = TensorRef::from_array_view(([1usize], threshold_value.as_slice()))?;
        let run_start = Instant::now();
        let outputs = self.session.run(ort::inputs![
            "waveform" => waveform,
            "threshold" => threshold,
        ])?;
        debug!(
            "rmvpe session.run backend={} input=waveform shape={} elapsed_us={}",
            self.provider.label(),
            format_usize_shape(waveform_shape),
            run_start.elapsed().as_micros()
        );
        let value = outputs
            .get("pitchf")
            .ok_or_else(|| anyhow!("RMVPE output 'pitchf' not found"))?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        validate_rmvpe_pitch_shape(shape, Some(data.len()))?;
        validate_rmvpe_pitch_data(data)?;
        let factor = 2.0f32.powf(pitch_shift / 12.0);
        // `data` borrows `outputs` (i.e. `self.session`); `pitchf_scratch` is a
        // disjoint field, so scaling into it here is a valid split borrow.
        self.pitchf_scratch.clear();
        self.pitchf_scratch
            .extend(data.iter().map(|f0| f0 * factor));
        Ok(self.pitchf_scratch.as_slice())
    }

    #[cfg(feature = "ort")]
    pub(super) fn extract_with_binding(
        &mut self,
        audio_16k: &[f32],
        pitch_shift: f32,
        threshold: f32,
    ) -> Result<&[f32]> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT RMVPE IoBinding is not initialized"))?;
        match binding {
            RmvpeTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.waveform, audio_16k, "waveform")?;
                binding
                    .binding
                    .bind_input("waveform", &binding.waveform)
                    .context("failed to bind TensorRT RMVPE input 'waveform'")?;
                binding.bind_threshold_if_changed(threshold)?;
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT RMVPE bound output")?;
                debug!(
                    "rmvpe session.run_binding backend={} cuda_graph=false device_io=false input=waveform shape={} output=pitchf output_shape={} elapsed_us={}",
                    self.provider.label(),
                    format_usize_shape(&binding.waveform_shape),
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                validate_rmvpe_pitch_shape(shape, Some(data.len()))?;
                validate_rmvpe_pitch_data(data)?;
                let actual_shape = i64_shape_to_usize(shape, "rmvpe output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RMVPE bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                let factor = 2.0f32.powf(pitch_shift / 12.0);
                // `data` borrows the bound output (self.tensor_rt_binding); the
                // scratch is a disjoint field, so scaling into it is valid here.
                self.pitchf_scratch.clear();
                self.pitchf_scratch
                    .extend(data.iter().map(|f0| f0 * factor));
                Ok(self.pitchf_scratch.as_slice())
            }
            RmvpeTensorRtBinding::CudaGraph(binding) => {
                let h2d_start = Instant::now();
                let waveform_h2d_us = binding
                    .copy_waveform_to_device_if_owned(audio_16k)?
                    .unwrap_or(0);
                binding.copy_threshold_if_changed(threshold)?;
                let h2d_us = h2d_start.elapsed().as_micros();
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                let d2h_start = Instant::now();
                copy_f32_tensor_to_host(
                    &binding.device_output,
                    &mut binding.host_output,
                    "pitchf",
                )?;
                let d2h_us = d2h_start.elapsed().as_micros();
                debug!(
                    "rmvpe session.run_binding(device_io=true) backend={} cuda_graph={} shared_waveform_input={} input=waveform shape={} output=pitchf output_shape={} waveform_h2d_us={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    binding.shared_waveform_input,
                    format_usize_shape(&binding.waveform_shape),
                    format_usize_shape(&binding.output_shape),
                    waveform_h2d_us,
                    h2d_us,
                    run_us,
                    d2h_us,
                    h2d_us + run_us + d2h_us
                );
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                validate_rmvpe_pitch_shape(shape, Some(data.len()))?;
                validate_rmvpe_pitch_data(data)?;
                let actual_shape = i64_shape_to_usize(shape, "rmvpe output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RMVPE bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                let factor = 2.0f32.powf(pitch_shift / 12.0);
                // `data` borrows the bound output (self.tensor_rt_binding); the
                // scratch is a disjoint field, so scaling into it is valid here.
                self.pitchf_scratch.clear();
                self.pitchf_scratch
                    .extend(data.iter().map(|f0| f0 * factor));
                Ok(self.pitchf_scratch.as_slice())
            }
        }
    }
}

/// FCPE pitch session. The model contract is intentionally kept separate from
/// RMVPE: FCPE consumes `[1, samples, 1]` and returns `[1, frames, 1]` on a
/// 160-sample (10 ms) grid, with no threshold input. Fixed-shape providers
/// allocate/bind this exact window during load so inference never changes a
/// TensorRT profile or CUDA-graph address.
pub(super) struct FcpePitchSession {
    #[cfg(feature = "ort")]
    pub(super) session: Session,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<HubertTensorRtBinding>,
    native: Option<NativeFcpeEngine>,
    output_scratch: Vec<f32>,
}

impl FcpePitchSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
    ) -> Result<Self> {
        let io = read_model_io(path)?;
        let contract = io.validate_fcpe_input_contract()?;
        if let (Some(profile), Some(static_samples)) =
            (tensor_rt_profile.as_ref(), contract.static_input_samples)
        {
            let profile_samples = profile
                .fixed_input_dims("audio")?
                .get(1)
                .copied()
                .ok_or_else(|| {
                    anyhow!("FCPE TensorRT profile audio shape must be [1, samples, 1]")
                })?;
            if profile_samples != static_samples {
                bail!(
                    "FCPE static ONNX input has {} samples but the selected fixed profile requires {}; use a matching FCPE export or a dynamic FCPE model",
                    static_samples,
                    profile_samples
                );
            }
        }
        let native = if provider.is_tensorrt() {
            let profile = tensor_rt_profile
                .as_ref()
                .ok_or_else(|| anyhow!("native TensorRT FCPE requires a fixed-shape profile"))?;
            Some(NativeFcpeEngine::load(path, profile)?)
        } else {
            None
        };
        #[cfg(feature = "ort")]
        let session = {
            // Native TensorRT owns inference for Provider::TensorRt; retain a
            // CPU ORT session only as a compile-time placeholder in combined
            // builds, exactly like the existing RMVPE session.
            let session_provider = if provider.is_tensorrt() {
                Provider::Cpu
            } else {
                provider
            };
            load_session(
                path,
                session_provider,
                ModelRole::Fcpe,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?
        };
        info!(
            "loaded FCPE f0 model: {} input=audio output=f0_hz static_input_samples={:?} static_output_frames={:?}",
            path.display(),
            contract.static_input_samples,
            contract.static_output_frames
        );
        Ok(Self {
            #[cfg(feature = "ort")]
            session,
            provider,
            tensor_rt_profile,
            tensor_rt_run_mode,
            #[cfg(feature = "ort")]
            tensor_rt_binding: None,
            native,
            output_scratch: Vec::new(),
        })
    }

    pub(super) fn warmup_output_shape(&mut self, audio_samples: usize) -> Result<Vec<i64>> {
        let input_shape = [1usize, audio_samples, 1usize];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "audio",
            &input_shape,
        )?;
        if let Some(native) = self.native.as_ref() {
            let shape = native.warmup_output_shape();
            validate_fcpe_output_shape(&shape, audio_samples)?;
            return Ok(shape);
        }
        #[cfg(feature = "ort")]
        {
            let audio = Tensor::from_array((input_shape, vec![0.0f32; audio_samples]))?;
            let outputs = self.session.run(ort::inputs!["audio" => audio])?;
            let value = outputs
                .get("f0_hz")
                .ok_or_else(|| anyhow!("FCPE output 'f0_hz' not found"))?;
            let (shape, _) = value.try_extract_tensor::<f32>()?;
            validate_fcpe_output_shape(shape, audio_samples)?;
            Ok(shape.to_vec())
        }
        #[cfg(not(feature = "ort"))]
        bail!("FCPE warmup requires the `ort` feature; native TensorRT reports its own shape")
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(&mut self, output_shape: &[i64]) -> Result<()> {
        if !provider_uses_fixed_shape(self.provider) || self.native.is_some() {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("FCPE IoBinding requires a fixed-shape profile"))?;
        let input_shape = profile.fixed_input_dims("audio")?;
        let output_shape = i64_shape_to_usize(output_shape, "FCPE output")?;
        let audio_samples = input_shape.get(1).copied().ok_or_else(|| {
            anyhow!(
                "FCPE TensorRT profile audio shape must be [1, samples, 1], got {}",
                format_usize_shape(input_shape)
            )
        })?;
        validate_fcpe_output_shape_usize(&output_shape, audio_samples)?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                HubertTensorRtBinding::Pinned(HubertTensorRtPinnedBinding::new(
                    &self.session,
                    "audio",
                    input_shape,
                    "f0_hz",
                    &output_shape,
                    profile.gpu_device_id,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let mut binding = HubertTensorRtGraphBinding::new(
                    &self.session,
                    "audio",
                    input_shape,
                    "f0_hz",
                    &output_shape,
                    None,
                    profile.gpu_device_id,
                )?;
                binding.warmup_capture(
                    &mut self.session,
                    "f0_hz",
                    ModelRole::Fcpe,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                HubertTensorRtBinding::CudaGraph(binding)
            }
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} input=audio input_shape={} output=f0_hz output_shape={}",
            self.provider.label(),
            ModelRole::Fcpe.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            format_usize_shape(input_shape),
            format_usize_shape(&output_shape)
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    /// Extract natural FCPE F0 into a reused buffer. NaN/non-positive values
    /// are normalized to the shared unvoiced representation before fusion.
    pub(super) fn extract(&mut self, audio_16k: &[f32]) -> Result<&[f32]> {
        let input_shape = [1usize, audio_16k.len(), 1usize];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            "audio",
            &input_shape,
        )?;
        #[cfg(feature = "ort")]
        if self.tensor_rt_binding.is_some() {
            return self.extract_with_binding(audio_16k);
        }
        if let Some(native) = self.native.as_mut() {
            self.output_scratch.clear();
            native.extract_into(audio_16k, &mut self.output_scratch)?;
            validate_fcpe_frame_count(self.output_scratch.len(), audio_16k.len())?;
            normalize_f0_in_place(&mut self.output_scratch);
            return Ok(self.output_scratch.as_slice());
        }
        #[cfg(feature = "ort")]
        {
            self.extract_with_session_run(audio_16k, &input_shape)
        }
        #[cfg(not(feature = "ort"))]
        bail!("FCPE session inference requires the `ort` feature; this build supports native TensorRT only")
    }

    #[cfg(feature = "ort")]
    fn extract_with_session_run(
        &mut self,
        audio_16k: &[f32],
        input_shape: &[usize; 3],
    ) -> Result<&[f32]> {
        let input = TensorRef::from_array_view((*input_shape, audio_16k))?;
        let run_start = Instant::now();
        let outputs = self.session.run(ort::inputs!["audio" => input])?;
        debug!(
            "fcpe session.run backend={} input=audio shape={} elapsed_us={}",
            self.provider.label(),
            format_usize_shape(input_shape),
            run_start.elapsed().as_micros()
        );
        let value = outputs
            .get("f0_hz")
            .ok_or_else(|| anyhow!("FCPE output 'f0_hz' not found"))?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        validate_fcpe_output_shape(shape, audio_16k.len())?;
        self.output_scratch.clear();
        self.output_scratch.extend_from_slice(data);
        normalize_f0_in_place(&mut self.output_scratch);
        Ok(self.output_scratch.as_slice())
    }

    #[cfg(feature = "ort")]
    fn extract_with_binding(&mut self, audio_16k: &[f32]) -> Result<&[f32]> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT FCPE IoBinding is not initialized"))?;
        match binding {
            HubertTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.audio, audio_16k, "audio")?;
                binding
                    .binding
                    .bind_input("audio", &binding.audio)
                    .context("failed to bind TensorRT FCPE input 'audio'")?;
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT FCPE bound output")?;
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                validate_fcpe_output_shape(shape, audio_16k.len())?;
                debug!(
                    "fcpe session.run_binding backend={} cuda_graph=false input_shape={} output_shape={} elapsed_us={}",
                    self.provider.label(),
                    format_usize_shape(&binding.input_shape),
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                self.output_scratch.clear();
                self.output_scratch.extend_from_slice(data);
                normalize_f0_in_place(&mut self.output_scratch);
                Ok(self.output_scratch.as_slice())
            }
            HubertTensorRtBinding::CudaGraph(binding) => {
                let h2d_us = binding
                    .copy_audio_to_device_if_owned(audio_16k, "audio")?
                    .unwrap_or(0);
                let run_start = Instant::now();
                let _outputs = self.session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                copy_f32_tensor_to_host(&binding.device_output, &mut binding.host_output, "f0_hz")?;
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                validate_fcpe_output_shape(shape, audio_16k.len())?;
                debug!(
                    "fcpe session.run_binding(device_io=true) backend={} cuda_graph={} input_shape={} output_shape={} h2d_us={} run_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    format_usize_shape(&binding.input_shape),
                    format_usize_shape(&binding.output_shape),
                    h2d_us,
                    run_us
                );
                self.output_scratch.clear();
                self.output_scratch.extend_from_slice(data);
                normalize_f0_in_place(&mut self.output_scratch);
                Ok(self.output_scratch.as_slice())
            }
        }
    }
}

fn normalize_f0_in_place(values: &mut [f32]) {
    for value in values {
        if !value.is_finite() || *value <= 0.0 {
            *value = 0.0;
        }
    }
}

fn validate_fcpe_frame_count(frames: usize, audio_samples: usize) -> Result<()> {
    let expected_frames = audio_samples
        .checked_div(160)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| anyhow!("FCPE input sample count is too large: {audio_samples}"))?;
    if frames != expected_frames {
        bail!(
            "FCPE output frame count {frames} does not match {audio_samples} input samples; the 160-sample contract requires {expected_frames}"
        );
    }
    Ok(())
}

fn validate_fcpe_output_shape(shape: &[i64], audio_samples: usize) -> Result<()> {
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 {
        bail!("FCPE output must have shape [1, frames, 1], got {shape:?}");
    }
    let frames = usize::try_from(shape[1]).with_context(|| {
        format!(
            "FCPE output frame dimension is negative or too large: {}",
            shape[1]
        )
    })?;
    validate_fcpe_frame_count(frames, audio_samples)
}

fn validate_fcpe_output_shape_usize(shape: &[usize], audio_samples: usize) -> Result<()> {
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 {
        bail!(
            "FCPE output must have shape [1, frames, 1], got {}",
            format_usize_shape(shape)
        );
    }
    validate_fcpe_frame_count(shape[1], audio_samples)
}

// Per-session latent-noise state for RVC exports that take the VITS
// reparameterization noise `z` (`rnd`) as a generator input. Present only when
// the model exposes that input. The scratch buffer is reused across chunks, and
// the counter-based generator keys every value to the shared absolute timeline.
#[cfg(feature = "ort")]
struct RvcRndState {
    name: String,
    channels: usize,
    generator: GaussianNoise,
    scratch: Vec<f32>,
}

#[cfg(feature = "ort")]
impl RvcRndState {
    fn from_io_names(io_names: &RvcIoNames) -> Result<Option<Self>> {
        let Some(rnd) = io_names.rnd.as_ref() else {
            return Ok(None);
        };
        let channels = usize::try_from(rnd.channels)
            .ok()
            .filter(|channels| *channels > 0)
            .ok_or_else(|| anyhow!("RVC '{}' input has non-positive channel count", rnd.name))?;
        Ok(Some(Self {
            name: rnd.name.clone(),
            channels,
            generator: GaussianNoise::new(RVC_RND_SEED),
            scratch: Vec::new(),
        }))
    }

    /// Refresh the reused scratch with timeline-stable `N(0, 1)` noise shaped
    /// `[1, channels, frame_len]` and return that shape.
    fn refresh(&mut self, frame_len: usize, window_start_frame: i64) -> Result<[usize; 3]> {
        let len = self
            .channels
            .checked_mul(frame_len)
            .context("RVC rnd input length overflow")?;
        self.scratch.resize(len, 0.0);
        self.generator.fill_window(
            &mut self.scratch,
            self.channels,
            frame_len,
            window_start_frame,
        );
        Ok([1, self.channels, frame_len])
    }
}

// CPU output binding is deliberately output-only: inputs still borrow the
// worker-owned buffers for each synchronous run, while the RVC "audio" tensor
// keeps stable preallocated storage across chunks with the same shapes.
#[cfg(feature = "ort")]
struct RvcCpuOutputBinding {
    binding: IoBinding,
    output: Tensor<f32>,
    output_shape: Vec<usize>,
    feats_shape: Vec<usize>,
    pitch_shape: Vec<usize>,
}

#[cfg(feature = "ort")]
impl RvcCpuOutputBinding {
    fn new(
        session: &Session,
        feats_shape: &[usize],
        pitch_shape: &[usize],
        output_shape: &[usize],
        audio_name: &str,
    ) -> Result<Self> {
        let allocator = Allocator::default();
        let mut output = Tensor::<f32>::new(&allocator, output_shape.to_vec())
            .context("failed to allocate CPU RVC output 'audio'")?;
        let mut binding = session
            .create_binding()
            .context("failed to create CPU RVC output IoBinding")?;
        bind_output_tensor(&mut binding, audio_name, &mut output)
            .context("failed to bind CPU RVC output 'audio'")?;
        Ok(Self {
            binding,
            output,
            output_shape: output_shape.to_vec(),
            feats_shape: feats_shape.to_vec(),
            pitch_shape: pitch_shape.to_vec(),
        })
    }

    fn matches_input(&self, feats_shape: &[usize], pitch_shape: &[usize]) -> bool {
        self.feats_shape == feats_shape && self.pitch_shape == pitch_shape
    }
}

pub(super) struct RvcModelSession {
    #[cfg(feature = "ort")]
    pub(super) session: Option<Session>,
    pub(super) provider: Provider,
    pub(super) tensor_rt_profile: Option<TensorRtSessionProfile>,
    pub(super) tensor_rt_run_mode: TensorRtRunMode,
    #[cfg(feature = "ort")]
    pub(super) tensor_rt_binding: Option<RvcTensorRtBinding>,
    native_rvc: Option<NativeRvcEngine>,
    #[cfg(feature = "ort")]
    cpu_output_binding: Option<RvcCpuOutputBinding>,
    pub(super) expected_feat_channels: i64,
    /// Generator I/O names for this model's export convention; every ORT/TensorRT
    /// bind site uses these instead of the canonical vcclient literals so RVC
    /// WebUI / converter exports (`phone`/`nsff0`/`ds`/`rnd`/...) bind correctly.
    io_names: RvcIoNames,
    /// Latent-noise generator for ORT-backed sessions whose export takes the
    /// `rnd` input. `None` when the model samples noise internally. The native
    /// TensorRT path keeps its own generator inside `NativeRvcEngine`, so this
    /// stays `None` there.
    #[cfg(feature = "ort")]
    rnd: Option<RvcRndState>,
}

impl RvcModelSession {
    pub(super) fn load(
        path: &Path,
        provider: Provider,
        tensor_rt_profile: Option<TensorRtSessionProfile>,
        expected_feat_channels_override: Option<i64>,
        tensor_rt_run_mode: TensorRtRunMode,
        tensor_rt_session_purpose: TensorRtSessionPurpose,
        io_names: RvcIoNames,
    ) -> Result<Self> {
        if provider.is_tensorrt() {
            let profile = tensor_rt_profile.as_ref().ok_or_else(|| {
                anyhow!("native TensorRT RVC requires a fixed-shape TensorRT profile")
            })?;
            let feats_shape = profile.fixed_input_dims(&io_names.feats)?;
            let channels = feats_shape
                .get(2)
                .copied()
                .ok_or_else(|| anyhow!("native TensorRT RVC feats profile must be rank-3"))?;
            let expected_feat_channels = expected_feat_channels_override
                .unwrap_or_else(|| i64::try_from(channels).unwrap_or(i64::MAX));
            let native_rvc = NativeRvcEngine::load(path, profile, channels, &io_names)?;
            info!(
                "loaded native TensorRT RVC model={} frames={} channels={} session_purpose={}",
                path.display(),
                native_rvc.frames(),
                native_rvc.channels(),
                tensor_rt_session_purpose.label()
            );
            return Ok(Self {
                #[cfg(feature = "ort")]
                session: None,
                provider,
                tensor_rt_profile,
                tensor_rt_run_mode,
                #[cfg(feature = "ort")]
                tensor_rt_binding: None,
                native_rvc: Some(native_rvc),
                #[cfg(feature = "ort")]
                cpu_output_binding: None,
                expected_feat_channels,
                io_names,
                // Native TensorRT generates rnd noise inside NativeRvcEngine.
                #[cfg(feature = "ort")]
                rnd: None,
            });
        }
        // CPU/CUDA only: validate via the provider-neutral reader, then load the
        // ORT session for inference. Unreachable in the TensorRT-only build.
        #[cfg(feature = "ort")]
        {
            let io = read_model_io(path)?;
            let mut required_inputs = vec![
                io_names.feats.as_str(),
                io_names.p_len.as_str(),
                io_names.pitch.as_str(),
                io_names.pitchf.as_str(),
                io_names.sid.as_str(),
            ];
            if let Some(rnd) = io_names.rnd.as_ref() {
                required_inputs.push(rnd.name.as_str());
            }
            io.require_inputs(&required_inputs)?;
            io.require_output(io_names.audio.as_str())?;
            let expected_feat_channels = match expected_feat_channels_override {
                Some(channels) => channels,
                None => io.feat_channels(&io_names.feats)?,
            };
            io.validate_rvc_metadata()?;
            let session = load_session(
                path,
                provider,
                ModelRole::Rvc,
                tensor_rt_profile.as_ref(),
                tensor_rt_run_mode,
                tensor_rt_session_purpose,
            )?;
            info!("loaded RVC model: {}", path.display());
            let rnd = RvcRndState::from_io_names(&io_names)?;
            Ok(Self {
                session: Some(session),
                provider,
                tensor_rt_profile,
                tensor_rt_run_mode,
                tensor_rt_binding: None,
                native_rvc: None,
                cpu_output_binding: None,
                expected_feat_channels,
                io_names,
                rnd,
            })
        }
        #[cfg(not(feature = "ort"))]
        bail!(
            "provider {} requires the `ort` feature; this build supports native TensorRT only",
            provider.label()
        )
    }

    pub(super) fn warmup_output_shape(
        &mut self,
        feature_len: usize,
        feature_channels: i64,
        speaker_id: i64,
    ) -> Result<Vec<i64>> {
        if let Some(native) = self.native_rvc.as_ref() {
            if native.frames() != feature_len {
                bail!(
                    "native TensorRT RVC engine frame count {} does not match runtime feature_len {}",
                    native.frames(),
                    feature_len
                );
            }
            if native.channels()
                != usize::try_from(feature_channels).context("invalid RVC channel count")?
            {
                bail!(
                    "native TensorRT RVC engine channel count {} does not match model channel count {}",
                    native.channels(),
                    feature_channels
                );
            }
            // The engine self-reports its fixed `audio` output length after
            // deserialize, so no warmup inference is needed to learn the shape.
            // `speaker_id` is consumed only by the ORT branch below.
            return Ok(vec![
                i64::try_from(native.output_len()).context("native RVC output length overflow")?
            ]);
        }
        #[cfg(not(feature = "ort"))]
        {
            let _ = (feature_len, feature_channels, speaker_id);
            bail!("RVC warmup requires the `ort` feature; native TensorRT reports its own shape")
        }
        #[cfg(feature = "ort")]
        {
            let feats_shape = vec![1i64, feature_len as i64, feature_channels];
            let feats_shape_usize = i64_shape_to_usize(&feats_shape, "feats")?;
            validate_tensorrt_input_shape(
                self.provider,
                self.tensor_rt_profile.as_ref(),
                self.io_names.feats.as_str(),
                &feats_shape_usize,
            )?;
            let pitch_shape = [1usize, feature_len];
            validate_tensorrt_input_shape(
                self.provider,
                self.tensor_rt_profile.as_ref(),
                self.io_names.pitch.as_str(),
                &pitch_shape,
            )?;
            validate_tensorrt_input_shape(
                self.provider,
                self.tensor_rt_profile.as_ref(),
                self.io_names.pitchf.as_str(),
                &pitch_shape,
            )?;
            let feats_len = feature_len
                .checked_mul(
                    usize::try_from(feature_channels).context("invalid RVC channel count")?,
                )
                .context("RVC warmup feats input length overflow")?;
            let feats = Tensor::from_array((feats_shape.clone(), vec![0.0f32; feats_len]))?;
            let p_len = Tensor::from_array(([1usize], vec![feature_len as i64]))?;
            let pitch = Tensor::from_array((pitch_shape, vec![1i64; feature_len]))?;
            let pitchf = Tensor::from_array((pitch_shape, vec![0.0f32; feature_len]))?;
            let sid = Tensor::from_array(([1usize], vec![speaker_id]))?;
            // Latent noise (when this export takes it): warmup only learns the
            // output shape, so zeros suffice — the per-chunk path feeds real noise.
            let rnd = match self.io_names.rnd.as_ref() {
                Some(rnd) => {
                    let channels =
                        usize::try_from(rnd.channels).context("invalid RVC rnd channel count")?;
                    let len = channels
                        .checked_mul(feature_len)
                        .context("RVC rnd warmup length overflow")?;
                    Some((
                        rnd.name.as_str(),
                        Tensor::from_array(([1usize, channels, feature_len], vec![0.0f32; len]))?,
                    ))
                }
                None => None,
            };
            let run_start = Instant::now();
            let names = &self.io_names;
            let session = self
                .session
                .as_mut()
                .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
            let outputs = if let Some((rnd_name, rnd)) = rnd {
                session.run(ort::inputs![
                    names.feats.as_str() => feats,
                    names.p_len.as_str() => p_len,
                    names.pitch.as_str() => pitch,
                    names.pitchf.as_str() => pitchf,
                    names.sid.as_str() => sid,
                    rnd_name => rnd,
                ])?
            } else {
                session.run(ort::inputs![
                    names.feats.as_str() => feats,
                    names.p_len.as_str() => p_len,
                    names.pitch.as_str() => pitch,
                    names.pitchf.as_str() => pitchf,
                    names.sid.as_str() => sid,
                ])?
            };
            debug!(
                "rvc warmup session.run backend={} feats_shape={} pitch_shape={} elapsed_us={}",
                self.provider.label(),
                format_usize_shape(&feats_shape_usize),
                format_usize_shape(&pitch_shape),
                run_start.elapsed().as_micros()
            );
            let value = outputs
                .get(self.io_names.audio.as_str())
                .ok_or_else(|| anyhow!("RVC output 'audio' not found"))?;
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            validate_rvc_audio_shape(shape, Some(data.len()))?;
            validate_rvc_audio_data(data)?;
            Ok(shape.to_vec())
        }
    }

    #[cfg(feature = "ort")]
    pub(super) fn enable_tensorrt_binding(
        &mut self,
        output_shape: &[i64],
        speaker_id: i64,
    ) -> Result<()> {
        if self.native_rvc.is_some() {
            return Ok(());
        }
        if !provider_uses_fixed_shape(self.provider) {
            return Ok(());
        }
        let profile = self
            .tensor_rt_profile
            .as_ref()
            .ok_or_else(|| anyhow!("RVC IoBinding requires a fixed-shape profile"))?;
        let feats_shape = profile.fixed_input_dims(&self.io_names.feats)?;
        let pitch_shape = profile.fixed_input_dims(&self.io_names.pitch)?;
        let frame_len = pitch_shape
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("TensorRT RVC pitch profile must be rank-2"))?;
        let output_shape = i64_shape_to_usize(output_shape, "rvc output")?;
        let binding = match self.tensor_rt_run_mode {
            TensorRtRunMode::PinnedCpu => {
                RvcTensorRtBinding::Pinned(RvcTensorRtPinnedBinding::new(
                    self.session
                        .as_ref()
                        .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?,
                    feats_shape,
                    pitch_shape,
                    &output_shape,
                    frame_len as i64,
                    speaker_id,
                    profile.gpu_device_id,
                    &self.io_names,
                )?)
            }
            TensorRtRunMode::DeviceIo | TensorRtRunMode::CudaGraph => {
                let session = self
                    .session
                    .as_mut()
                    .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
                let mut binding = RvcTensorRtGraphBinding::new(
                    session,
                    feats_shape,
                    pitch_shape,
                    &output_shape,
                    frame_len as i64,
                    speaker_id,
                    profile.gpu_device_id,
                    &self.io_names,
                )?;
                binding.warmup_capture(
                    session,
                    self.provider,
                    self.tensor_rt_run_mode.cuda_graph(),
                )?;
                RvcTensorRtBinding::CudaGraph(binding)
            }
        };
        info!(
            "GPU IoBinding enabled backend={} model_role={} mode={} cuda_graph={} device_io={} inputs=feats:{},pitch:{},pitchf:{},p_len:1,sid:1 output=audio output_shape={} host_input_memory=CUDA_PINNED/CPUInput host_output_memory=CUDA_PINNED/CPUOutput bound_input_memory={} bound_output_memory={}",
            self.provider.label(),
            ModelRole::Rvc.label(),
            self.tensor_rt_run_mode.label(),
            self.tensor_rt_run_mode.cuda_graph(),
            self.tensor_rt_run_mode.device_io(),
            format_usize_shape(feats_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(&output_shape),
            self.tensor_rt_run_mode.bound_input_memory(),
            self.tensor_rt_run_mode.bound_output_memory()
        );
        self.tensor_rt_binding = Some(binding);
        Ok(())
    }

    #[cfg(feature = "ort")]
    fn enable_cpu_output_binding(
        &mut self,
        feats_shape: &[usize],
        pitch_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<()> {
        if self.provider != Provider::Cpu {
            return Ok(());
        }
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let binding = RvcCpuOutputBinding::new(
            session,
            feats_shape,
            pitch_shape,
            output_shape,
            self.io_names.audio.as_str(),
        )?;
        info!(
            "CPU output IoBinding enabled model_role={} inputs=feats:{},pitch:{},pitchf:{} output=audio output_shape={}",
            ModelRole::Rvc.label(),
            format_usize_shape(feats_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(pitch_shape),
            format_usize_shape(output_shape)
        );
        self.cpu_output_binding = Some(binding);
        Ok(())
    }

    // Inputs mirror the ONNX RVC contract (feats/p_len/pitch/pitchf/sid) plus the
    // reused output buffer; an ad-hoc struct would only obscure that contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn infer(
        &mut self,
        feats: &[f32],
        feats_shape: &[i64],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        window_start_frame: i64,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let feats_shape_usize = i64_shape_to_usize(feats_shape, "feats")?;
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            self.io_names.feats.as_str(),
            &feats_shape_usize,
        )?;
        let pitch_shape = [1usize, frame_len];
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            self.io_names.pitch.as_str(),
            &pitch_shape,
        )?;
        validate_tensorrt_input_shape(
            self.provider,
            self.tensor_rt_profile.as_ref(),
            self.io_names.pitchf.as_str(),
            &pitch_shape,
        )?;
        if let Some(native) = self.native_rvc.as_mut() {
            let expected_output_len = native.output_len();
            native.infer_into(feats, pitch, pitchf, speaker_id, window_start_frame, out)?;
            if out.len() != expected_output_len {
                bail!(
                    "native TensorRT RVC audio output has {} values; expected {}",
                    out.len(),
                    expected_output_len
                );
            }
            if out.iter().any(|sample| !sample.is_finite()) {
                bail!("native TensorRT RVC audio output contains non-finite values");
            }
            return Ok(());
        }
        #[cfg(feature = "ort")]
        {
            if self.tensor_rt_binding.is_some() {
                return self.infer_with_binding(
                    feats,
                    frame_len,
                    pitch,
                    pitchf,
                    speaker_id,
                    window_start_frame,
                    out,
                );
            }
            if self.provider == Provider::Cpu
                && self
                    .cpu_output_binding
                    .as_ref()
                    .is_some_and(|binding| binding.matches_input(&feats_shape_usize, &pitch_shape))
            {
                return self.infer_with_cpu_output_binding(
                    feats,
                    feats_shape,
                    &feats_shape_usize,
                    frame_len,
                    pitch,
                    pitchf,
                    speaker_id,
                    window_start_frame,
                    &pitch_shape,
                    out,
                );
            }
            self.infer_with_session_run(
                feats,
                feats_shape,
                &feats_shape_usize,
                frame_len,
                pitch,
                pitchf,
                speaker_id,
                window_start_frame,
                &pitch_shape,
                out,
            )
        }
        #[cfg(not(feature = "ort"))]
        {
            let _ = out;
            bail!("RVC session inference requires the `ort` feature; this build supports native TensorRT only")
        }
    }

    // Keep the RVC tensor inputs explicit here: collapsing them into an ad-hoc
    // struct would obscure the ONNX input contract this function validates.
    #[cfg(feature = "ort")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn infer_with_session_run(
        &mut self,
        feats: &[f32],
        feats_shape: &[i64],
        feats_shape_usize: &[usize],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        window_start_frame: i64,
        pitch_shape: &[usize; 2],
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let p_len_value = [frame_len as i64];
        let sid_value = [speaker_id];
        // Refresh latent noise first (a self-disjoint mutable borrow of `rnd`),
        // then borrow the scratch immutably alongside the session below.
        let rnd_shape = match self.rnd.as_mut() {
            Some(state) => Some(state.refresh(frame_len, window_start_frame)?),
            None => None,
        };
        let feats = TensorRef::from_array_view((feats_shape, feats))?;
        let p_len = TensorRef::from_array_view(([1usize], p_len_value.as_slice()))?;
        let pitch = TensorRef::from_array_view((*pitch_shape, pitch))?;
        let pitchf = TensorRef::from_array_view((*pitch_shape, pitchf))?;
        let sid = TensorRef::from_array_view(([1usize], sid_value.as_slice()))?;
        let names = &self.io_names;
        let rnd_state = self.rnd.as_ref();
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let rnd = match (rnd_state, rnd_shape) {
            (Some(state), Some(shape)) => Some((
                state.name.as_str(),
                TensorRef::from_array_view((shape, state.scratch.as_slice()))?,
            )),
            _ => None,
        };
        let output_shape = {
            let run_start = Instant::now();
            let outputs = if let Some((rnd_name, rnd)) = rnd {
                session.run(ort::inputs![
                    names.feats.as_str() => feats,
                    names.p_len.as_str() => p_len,
                    names.pitch.as_str() => pitch,
                    names.pitchf.as_str() => pitchf,
                    names.sid.as_str() => sid,
                    rnd_name => rnd,
                ])?
            } else {
                session.run(ort::inputs![
                    names.feats.as_str() => feats,
                    names.p_len.as_str() => p_len,
                    names.pitch.as_str() => pitch,
                    names.pitchf.as_str() => pitchf,
                    names.sid.as_str() => sid,
                ])?
            };
            debug!(
                "rvc session.run backend={} feats_shape={} pitch_shape={} elapsed_us={}",
                self.provider.label(),
                format_usize_shape(feats_shape_usize),
                format_usize_shape(pitch_shape),
                run_start.elapsed().as_micros()
            );
            let value = outputs
                .get(names.audio.as_str())
                .ok_or_else(|| anyhow!("RVC output 'audio' not found"))?;
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            validate_rvc_audio_shape(shape, Some(data.len()))?;
            validate_rvc_audio_data(data)?;
            let output_shape = i64_shape_to_usize(shape, "rvc output")?;
            out.clear();
            out.extend_from_slice(data);
            output_shape
        };
        self.enable_cpu_output_binding(feats_shape_usize, pitch_shape, &output_shape)?;
        Ok(())
    }

    // Keep the RVC tensor inputs explicit here: collapsing them into an ad-hoc
    // struct would obscure the ONNX input contract this function validates.
    #[cfg(feature = "ort")]
    #[allow(clippy::too_many_arguments)]
    fn infer_with_cpu_output_binding(
        &mut self,
        feats: &[f32],
        feats_shape: &[i64],
        feats_shape_usize: &[usize],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        window_start_frame: i64,
        pitch_shape: &[usize; 2],
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let p_len_value = [frame_len as i64];
        let sid_value = [speaker_id];
        // Refresh latent noise first (self-disjoint mutable borrow of `rnd`),
        // then bind the scratch alongside the other inputs below.
        let rnd_shape = match self.rnd.as_mut() {
            Some(state) => Some(state.refresh(frame_len, window_start_frame)?),
            None => None,
        };
        let feats = TensorRef::from_array_view((feats_shape, feats))?;
        let p_len = TensorRef::from_array_view(([1usize], p_len_value.as_slice()))?;
        let pitch = TensorRef::from_array_view((*pitch_shape, pitch))?;
        let pitchf = TensorRef::from_array_view((*pitch_shape, pitchf))?;
        let sid = TensorRef::from_array_view(([1usize], sid_value.as_slice()))?;
        let provider = self.provider;
        let names = &self.io_names;
        let rnd_state = self.rnd.as_ref();
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let binding = self
            .cpu_output_binding
            .as_mut()
            .ok_or_else(|| anyhow!("CPU RVC output IoBinding is not initialized"))?;
        let rnd = match (rnd_state, rnd_shape) {
            (Some(state), Some(shape)) => Some((
                state.name.as_str(),
                TensorRef::from_array_view((shape, state.scratch.as_slice()))?,
            )),
            _ => None,
        };
        let run_start = Instant::now();
        // IoBinding retains bound input OrtValues after the run. These TensorRefs
        // borrow worker buffers, so clear inputs before returning on both success
        // and error paths; only the preallocated CPU output stays bound.
        let run_result: Result<()> = (|| {
            binding
                .binding
                .bind_input(names.feats.as_str(), &feats)
                .context("failed to bind CPU RVC input 'feats'")?;
            binding
                .binding
                .bind_input(names.p_len.as_str(), &p_len)
                .context("failed to bind CPU RVC input 'p_len'")?;
            binding
                .binding
                .bind_input(names.pitch.as_str(), &pitch)
                .context("failed to bind CPU RVC input 'pitch'")?;
            binding
                .binding
                .bind_input(names.pitchf.as_str(), &pitchf)
                .context("failed to bind CPU RVC input 'pitchf'")?;
            binding
                .binding
                .bind_input(names.sid.as_str(), &sid)
                .context("failed to bind CPU RVC input 'sid'")?;
            if let Some((rnd_name, rnd)) = rnd.as_ref() {
                binding
                    .binding
                    .bind_input(*rnd_name, rnd)
                    .context("failed to bind CPU RVC input 'rnd'")?;
            }
            let _outputs = session.run_binding(&binding.binding)?;
            binding
                .binding
                .synchronize_outputs()
                .context("failed to synchronize CPU RVC bound output")?;
            Ok(())
        })();
        binding.binding.clear_inputs();
        run_result?;
        debug!(
            "rvc session.run_binding backend={} cpu_output_binding=true feats_shape={} pitch_shape={} output_shape={} elapsed_us={}",
            provider.label(),
            format_usize_shape(feats_shape_usize),
            format_usize_shape(pitch_shape),
            format_usize_shape(&binding.output_shape),
            run_start.elapsed().as_micros()
        );
        let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
        validate_rvc_audio_shape(shape, Some(data.len()))?;
        validate_rvc_audio_data(data)?;
        let actual_shape = i64_shape_to_usize(shape, "rvc output")?;
        if actual_shape != binding.output_shape {
            bail!(
                "CPU RVC bound output shape changed from {} to {}",
                format_usize_shape(&binding.output_shape),
                format_usize_shape(&actual_shape)
            );
        }
        out.clear();
        out.extend_from_slice(data);
        Ok(())
    }

    #[cfg(feature = "ort")]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn infer_with_binding(
        &mut self,
        feats: &[f32],
        frame_len: usize,
        pitch: &[i64],
        pitchf: &[f32],
        speaker_id: i64,
        window_start_frame: i64,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let binding = self
            .tensor_rt_binding
            .as_mut()
            .ok_or_else(|| anyhow!("TensorRT RVC IoBinding is not initialized"))?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("RVC ORT session is not initialized"))?;
        let mut rnd_state = self.rnd.as_mut();
        match binding {
            RvcTensorRtBinding::Pinned(binding) => {
                copy_f32_tensor(&mut binding.feats, feats, "feats")?;
                copy_i64_tensor(&mut binding.pitch, pitch, "pitch")?;
                copy_f32_tensor(&mut binding.pitchf, pitchf, "pitchf")?;
                binding.bind_fixed_scalars_if_changed(frame_len as i64, speaker_id)?;
                binding
                    .binding
                    .bind_input(binding.names.feats.as_str(), &binding.feats)
                    .context("failed to bind TensorRT RVC input 'feats'")?;
                binding
                    .binding
                    .bind_input(binding.names.pitch.as_str(), &binding.pitch)
                    .context("failed to bind TensorRT RVC input 'pitch'")?;
                binding
                    .binding
                    .bind_input(binding.names.pitchf.as_str(), &binding.pitchf)
                    .context("failed to bind TensorRT RVC input 'pitchf'")?;
                // Stage absolute-timeline noise into the pinned buffer and
                // re-bind it (the pinned path copies inputs at bind time).
                if let Some(rnd_tensor) = binding.rnd.as_mut() {
                    let rnd_name = binding
                        .names
                        .rnd
                        .as_ref()
                        .ok_or_else(|| anyhow!("RVC rnd tensor without a resolved name"))?
                        .name
                        .as_str();
                    let state = rnd_state
                        .take()
                        .ok_or_else(|| anyhow!("RVC rnd noise generator is not initialized"))?;
                    state.refresh(frame_len, window_start_frame)?;
                    copy_f32_tensor(rnd_tensor, &state.scratch, "rnd")?;
                    binding
                        .binding
                        .bind_input(rnd_name, rnd_tensor)
                        .context("failed to bind TensorRT RVC input 'rnd'")?;
                }
                let run_start = Instant::now();
                let _outputs = session.run_binding(&binding.binding)?;
                binding
                    .binding
                    .synchronize_outputs()
                    .context("failed to synchronize TensorRT RVC bound output")?;
                debug!(
                    "rvc session.run_binding backend={} cuda_graph=false device_io=false feats_shape={} pitch_shape={} output_shape={} elapsed_us={}",
                    self.provider.label(),
                    format_usize_shape(&binding.feats_shape),
                    format_usize_shape(&binding.pitch_shape),
                    format_usize_shape(&binding.output_shape),
                    run_start.elapsed().as_micros()
                );
                let (shape, data) = binding.output.try_extract_tensor::<f32>()?;
                validate_rvc_audio_shape(shape, Some(data.len()))?;
                validate_rvc_audio_data(data)?;
                let actual_shape = i64_shape_to_usize(shape, "rvc output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RVC bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                out.clear();
                out.extend_from_slice(data);
                Ok(())
            }
            RvcTensorRtBinding::CudaGraph(binding) => {
                let h2d_start = Instant::now();
                copy_f32_tensor(&mut binding.host_feats, feats, "feats")?;
                copy_i64_tensor(&mut binding.host_pitch, pitch, "pitch")?;
                copy_f32_tensor(&mut binding.host_pitchf, pitchf, "pitchf")?;
                copy_f32_tensor_to_device(&binding.host_feats, &mut binding.device_feats, "feats")?;
                copy_i64_tensor_to_device(&binding.host_pitch, &mut binding.device_pitch, "pitch")?;
                copy_f32_tensor_to_device(
                    &binding.host_pitchf,
                    &mut binding.device_pitchf,
                    "pitchf",
                )?;
                // Stage absolute-timeline noise into host_rnd, then copy into the
                // already-bound device_rnd. Its address stays stable for the
                // captured graph; only its contents change.
                if let (Some(host_rnd), Some(device_rnd)) =
                    (binding.host_rnd.as_mut(), binding.device_rnd.as_mut())
                {
                    let state = rnd_state
                        .take()
                        .ok_or_else(|| anyhow!("RVC rnd noise generator is not initialized"))?;
                    state.refresh(frame_len, window_start_frame)?;
                    copy_f32_tensor(host_rnd, &state.scratch, "rnd")?;
                    copy_f32_tensor_to_device(host_rnd, device_rnd, "rnd")?;
                }
                binding.copy_fixed_scalars_if_changed(frame_len as i64, speaker_id)?;
                let h2d_us = h2d_start.elapsed().as_micros();
                let run_start = Instant::now();
                let _outputs = session.run_binding(&binding.binding)?;
                let run_us = run_start.elapsed().as_micros();
                let d2h_start = Instant::now();
                copy_f32_tensor_to_host(&binding.device_output, &mut binding.host_output, "audio")?;
                let d2h_us = d2h_start.elapsed().as_micros();
                debug!(
                    "rvc session.run_binding(device_io=true) backend={} cuda_graph={} feats_shape={} pitch_shape={} output_shape={} h2d_us={} run_us={} d2h_us={} elapsed_us={}",
                    self.provider.label(),
                    self.tensor_rt_run_mode.cuda_graph(),
                    format_usize_shape(&binding.feats_shape),
                    format_usize_shape(&binding.pitch_shape),
                    format_usize_shape(&binding.output_shape),
                    h2d_us,
                    run_us,
                    d2h_us,
                    h2d_us + run_us + d2h_us
                );
                let (shape, data) = binding.host_output.try_extract_tensor::<f32>()?;
                validate_rvc_audio_shape(shape, Some(data.len()))?;
                validate_rvc_audio_data(data)?;
                let actual_shape = i64_shape_to_usize(shape, "rvc output")?;
                if actual_shape != binding.output_shape {
                    bail!(
                        "TensorRT RVC bound output shape changed from {} to {}",
                        format_usize_shape(&binding.output_shape),
                        format_usize_shape(&actual_shape)
                    );
                }
                out.clear();
                out.extend_from_slice(data);
                Ok(())
            }
        }
    }
}

#[cfg(feature = "ort")]
#[cfg(all(windows, feature = "windowsml"))]
fn windows_ml_catalog_ep_for_provider(
    provider: Provider,
) -> Option<crate::windows_ml::CatalogExecutionProvider> {
    match provider {
        Provider::WindowsMlNvTensorRtRtx => {
            Some(crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx)
        }
        Provider::WindowsMlQnn => Some(crate::windows_ml::CatalogExecutionProvider::Qnn),
        Provider::WindowsMlOpenVino => Some(crate::windows_ml::CatalogExecutionProvider::OpenVino),
        Provider::WindowsMlMiGraphX => Some(crate::windows_ml::CatalogExecutionProvider::MiGraphX),
        Provider::WindowsMlVitisAi => Some(crate::windows_ml::CatalogExecutionProvider::VitisAi),
        _ => None,
    }
}

#[cfg(feature = "ort")]
#[cfg(all(windows, feature = "windowsml"))]
fn with_windows_ml_catalog_ep(
    builder: ort::session::builder::SessionBuilder,
    catalog_ep: crate::windows_ml::CatalogExecutionProvider,
    path: &Path,
    tensor_rt_profile: Option<&TensorRtSessionProfile>,
) -> Result<ort::session::builder::SessionBuilder> {
    let env = ort::environment::Environment::current()?;
    let devices = env
        .devices()
        .filter(|device| {
            device
                .ep()
                .ok()
                .and_then(crate::windows_ml::CatalogExecutionProvider::from_catalog_name)
                == Some(catalog_ep)
        })
        .collect::<Vec<_>>();
    if devices.is_empty() {
        bail!(
            "Windows ML catalog EP {} was registered, but ONNX Runtime did not expose a matching EP device for {}",
            catalog_ep.label(),
            path.display()
        );
    }
    let ep_name = devices[0].ep()?.to_string();
    let mut options = Vec::<(String, String)>::new();
    if catalog_ep == crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx {
        let profile = tensor_rt_profile.ok_or_else(|| {
            anyhow!(
                "Windows ML NvTensorRtRtx requires a fixed-shape profile for {}",
                path.display()
            )
        })?;
        // TensorRT RTX falls back to a fully dynamic profile when these are
        // omitted, which can generate invalid min/opt/max shapes for RVC models.
        // Use the same fixed profile machinery as the native TensorRT backend.
        for key in [
            "nv_profile_min_shapes",
            "nv_profile_opt_shapes",
            "nv_profile_max_shapes",
        ] {
            options.push((format!("{ep_name}.{key}"), profile.profile_shapes.clone()));
        }
        if let Ok(cache_root) = tensor_rt_cache_root() {
            if let Ok(cache_dir) = profile.cache_dir_from_root(&cache_root) {
                std::fs::create_dir_all(&cache_dir).with_context(|| {
                    format!(
                        "failed to create Windows ML NvTensorRtRtx runtime cache dir {}",
                        cache_dir.display()
                    )
                })?;
                options.push((
                    format!("{ep_name}.nv_runtime_cache_path"),
                    cache_dir.display().to_string(),
                ));
            }
        }
    }
    info!(
        "using Windows ML catalog EP {} via ORT EP device API for {} profile={} runtime_cache={}",
        catalog_ep.label(),
        path.display(),
        tensor_rt_profile
            .map(|profile| profile.profile_shapes.as_str())
            .unwrap_or("-"),
        options
            .iter()
            .find(|(key, _)| key.ends_with(".nv_runtime_cache_path"))
            .map(|(_, value)| value.as_str())
            .unwrap_or("-")
    );
    let options = (!options.is_empty()).then_some(options);
    builder
        .with_devices(devices, options.as_deref())
        .map_err(|err| {
            anyhow!(
                "failed to append Windows ML catalog EP {}: {err}",
                catalog_ep.label()
            )
        })
}

#[cfg(feature = "ort")]
pub(super) fn load_session(
    path: &Path,
    provider: Provider,
    role: ModelRole,
    tensor_rt_profile: Option<&TensorRtSessionProfile>,
    tensor_rt_run_mode: TensorRtRunMode,
    tensor_rt_session_purpose: TensorRtSessionPurpose,
) -> Result<Session> {
    // CUDA consumes the selected device ID from the fixed-shape profile.
    // Windows ML consumes the same profile only for TensorRT-RTX shape options;
    // its adapter selection remains owned by Windows ML.
    #[cfg(not(any(feature = "cuda", all(windows, feature = "windowsml"))))]
    let _ = tensor_rt_profile;
    #[cfg(feature = "cuda")]
    let gpu_device_id = tensor_rt_profile.map_or(0, |profile| profile.gpu_device_id);
    #[cfg(feature = "cuda")]
    let gpu_device_id_i32 = i32::try_from(gpu_device_id)
        .map_err(|_| anyhow!("GPU device ID {gpu_device_id} exceeds the supported i32 range"))?;

    #[cfg(not(all(windows, feature = "windowsml")))]
    if provider.is_windows_ml() {
        bail!(
            "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
            provider.label(),
            path.display()
        );
    }

    // In the Windows ML build ORT is loaded dynamically from the Windows App SDK
    // Runtime, so bootstrap it for *every* provider — not just windowsml* — and
    // bind ORT to that runtime's onnxruntime.dll. Otherwise the plain `cpu`
    // provider has no initialized dylib and fails to locate onnxruntime.dll on
    // the default search path (the Windows ML package does not bundle it).
    // `ensure_initialized` is idempotent (guarded by a OnceLock).
    #[cfg(all(windows, feature = "windowsml"))]
    crate::windows_ml::ensure_initialized()?;

    let mut builder = Session::builder()?
        .with_intra_threads(1)
        .map_err(|err| anyhow!(err.to_string()))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|err| anyhow!(err.to_string()))?;
    match provider {
        Provider::Cuda => {
            #[cfg(not(feature = "cuda"))]
            {
                // The ONNX Runtime CUDA EP is compiled out of this build. Keep
                // `tensor_rt_run_mode` referenced so the no-cuda build matches
                // the cuda build's signature without an unused-variable warning.
                let _ = tensor_rt_run_mode;
                bail!(
                    "Provider::Cuda is unavailable in this build (compiled without the `cuda` feature); rebuild with `--features cuda` or select a CPU/TensorRT provider for {}",
                    path.display()
                );
            }
            #[cfg(feature = "cuda")]
            {
                info!(
                    "requesting ONNX Runtime CUDA execution provider device_id={} cuda_graph={} device_io={} run_mode={}",
                    gpu_device_id,
                    tensor_rt_run_mode.cuda_graph(),
                    tensor_rt_run_mode.device_io(),
                    tensor_rt_run_mode.label()
                );
                if tensor_rt_run_mode.cuda_graph() {
                    builder = builder.with_disable_cpu_fallback().map_err(|err| {
                        anyhow!("failed to disable CPU fallback for CUDA backend: {err}")
                    })?;
                }
                builder = builder
                    .with_execution_providers([ep::CUDA::default()
                        .with_device_id(gpu_device_id_i32)
                        .with_cuda_graph(tensor_rt_run_mode.cuda_graph())
                        .build()
                        .error_on_failure()])
                    .map_err(|err| anyhow!("failed to register CUDA execution provider: {err}"))?;
            }
        }
        Provider::WindowsMl => {
            #[cfg(not(all(windows, feature = "windowsml")))]
            {
                bail!(
                    "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                    provider.label(),
                    path.display()
                );
            }
            #[cfg(all(windows, feature = "windowsml"))]
            {
                // Auto Windows ML optimizes for "works with the platform runtime":
                // catalog EP if present/preparable, then DirectML, then ORT's CPU fallback.
                // Explicit windowsml-* providers below intentionally fail
                // instead of silently changing the requested accelerator.
                match crate::windows_ml::try_register_best_catalog_ep()? {
                    Some(catalog_ep) => {
                        // The NvTensorRtRtx (TensorRT-RTX) EP cannot be combined
                        // with the DirectML EP in one session — ORT rejects it with
                        // "DML EP can only be used with CPU EPs". With its
                        // fixed-shape profile it covers the whole graph on its own,
                        // so skip the DirectML fallback for it. Other catalog EPs
                        // keep DirectML for ops they do not implement.
                        let with_directml_fallback = catalog_ep
                            != crate::windows_ml::CatalogExecutionProvider::NvTensorRtRtx;
                        info!(
                            "using Windows ML catalog EP {} ({}) for {}",
                            catalog_ep.label(),
                            if with_directml_fallback {
                                "with DirectML/CPU fallback"
                            } else {
                                "no DirectML fallback; TensorRT-RTX covers the full graph"
                            },
                            path.display()
                        );
                        builder = with_windows_ml_catalog_ep(
                            builder,
                            catalog_ep,
                            path,
                            tensor_rt_profile,
                        )?;
                        if with_directml_fallback {
                            builder = builder
                                .with_execution_providers([ep::DirectML::default().build()])
                                .map_err(|err| {
                                    anyhow!(
                                        "failed to configure Windows ML DirectML fallback EP: {err}"
                                    )
                                })?;
                        }
                    }
                    None => {
                        info!(
                            "no usable Windows ML catalog EP found; using DirectML/CPU fallback for {}",
                            path.display()
                        );
                        builder = builder
                            .with_execution_providers([ep::DirectML::default().build()])
                            .map_err(|err| {
                                anyhow!(
                                    "failed to configure Windows ML DirectML/CPU fallback EP: {err}"
                                )
                            })?;
                    }
                }
            }
        }
        Provider::WindowsMlNvTensorRtRtx
        | Provider::WindowsMlOpenVino
        | Provider::WindowsMlQnn
        | Provider::WindowsMlMiGraphX
        | Provider::WindowsMlVitisAi => {
            #[cfg(not(all(windows, feature = "windowsml")))]
            {
                bail!(
                    "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                    provider.label(),
                    path.display()
                );
            }
            #[cfg(all(windows, feature = "windowsml"))]
            {
                let catalog_ep = windows_ml_catalog_ep_for_provider(provider).ok_or_else(|| {
                    anyhow!(
                        "provider {} has no Windows ML catalog EP mapping for {}",
                        provider.label(),
                        path.display()
                    )
                })?;
                if !crate::windows_ml::try_register_catalog_ep(catalog_ep)? {
                    bail!(
                        "Windows ML catalog EP {} requested by provider {} is not present or not ready for {}; install/enable that EP with Windows ML tooling, or use provider windowsml for DirectML/CPU fallback",
                        catalog_ep.label(),
                        provider.label(),
                        path.display()
                    );
                }
                builder = with_windows_ml_catalog_ep(builder, catalog_ep, path, tensor_rt_profile)?;
            }
        }
        Provider::WindowsMlDirectMl => {
            #[cfg(not(all(windows, feature = "windowsml")))]
            {
                bail!(
                    "provider {} is unavailable in this build; rebuild on Windows with the `windowsml` feature for {}",
                    provider.label(),
                    path.display()
                );
            }
            #[cfg(all(windows, feature = "windowsml"))]
            {
                info!(
                    "using Windows ML DirectML execution provider via Windows App SDK Runtime for {}",
                    path.display()
                );
                builder = builder
                    .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
                    .map_err(|err| {
                        anyhow!("failed to register Windows ML DirectML execution provider: {err}")
                    })?;
            }
        }
        Provider::TensorRt => {
            bail!(
                "Provider::TensorRt is native-only; load a CPU inspection session or a native TensorRT engine for {}",
                path.display()
            );
        }
        Provider::Cpu | Provider::WindowsMlCpu => {
            info!(
                "using {} execution provider intra_threads={} inter_threads={} arena=true mem_pattern=true flush_to_zero=true",
                if provider == Provider::WindowsMlCpu {
                    "Windows ML CPU"
                } else {
                    "ONNX Runtime CPU"
                },
                CPU_ONNX_INTRA_THREADS,
                CPU_ONNX_INTER_THREADS
            );
            // CPU inference still feeds a latency-sensitive pipeline. Keep
            // these as load-time session options; per-chunk tuning here would
            // add allocation/logging pressure near the realtime path.
            builder = builder
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|err| anyhow!("failed to enable CPU graph optimizations: {err}"))?
                .with_intra_threads(CPU_ONNX_INTRA_THREADS)
                .map_err(|err| anyhow!("failed to set CPU intra-op threads: {err}"))?
                .with_parallel_execution(true)
                .map_err(|err| anyhow!("failed to enable CPU parallel execution: {err}"))?
                .with_inter_threads(CPU_ONNX_INTER_THREADS)
                .map_err(|err| anyhow!("failed to set CPU inter-op threads: {err}"))?
                // Repeated realtime chunks are shape-stable after stream
                // padding, so memory pattern plus the CPU arena avoids churn
                // in ORT's internal allocators. Revisit if CPU runs become
                // truly variable-shape.
                .with_memory_pattern(true)
                .map_err(|err| anyhow!("failed to enable CPU memory pattern: {err}"))?
                .with_prepacking(true)
                .map_err(|err| anyhow!("failed to enable CPU prepacking: {err}"))?
                .with_flush_to_zero()
                .map_err(|err| anyhow!("failed to enable CPU flush-to-zero: {err}"))?
                .with_intra_op_spinning(true)
                .map_err(|err| anyhow!("failed to enable CPU intra-op spinning: {err}"))?
                .with_inter_op_spinning(true)
                .map_err(|err| anyhow!("failed to enable CPU inter-op spinning: {err}"))?
                .with_execution_providers([ep::CPU::default()
                    .with_arena_allocator(true)
                    .build()
                    .error_on_failure()])
                .map_err(|err| anyhow!("failed to register CPU execution provider: {err}"))?;
        }
    }
    if provider.is_tensorrt() || provider.is_cuda() {
        info!(
            "starting {} session commit for {} session_purpose={} cuda_graph={}",
            provider.label(),
            role.label(),
            tensor_rt_session_purpose.label(),
            tensor_rt_run_mode.cuda_graph()
        );
    }
    let session = builder
        .commit_from_file(path)
        .with_context(|| format!("failed to load ONNX model {}", path.display()))?;
    info!(
        "created ONNX Runtime session backend={} model_role={} session_purpose={} cuda_graph={} model={}",
        provider.label(),
        role.label(),
        tensor_rt_session_purpose.label(),
        provider_uses_fixed_shape(provider) && tensor_rt_run_mode.cuda_graph(),
        path.display()
    );
    Ok(session)
}

/// Format an ORT output value type for the CLI `inspect` command. The pipeline's
/// own structural checks live on `onnx_meta::ModelIo` and need no ORT.
#[cfg(feature = "ort")]
pub(super) fn describe_value_type(value_type: &ValueType) -> String {
    match value_type {
        ValueType::Tensor { ty, shape, .. } => format!("{ty:?} {shape}"),
        other => format!("{other:?}"),
    }
}

// This is an opt-in integration check because FCPE weights are intentionally
// not shipped with vc-rs.  Run it with
// `VC_RS_FCPE_MODEL=<path> cargo test -p vc-core --features ort fcpe_ort_session`
// to exercise the real Rust session (rather than only the provider-neutral
// protobuf contract tests) at several dynamic input lengths.  Keeping the
// model external also prevents a developer-local path or a 43 MB weight file
// from entering a distributable crate.
#[cfg(all(test, feature = "ort"))]
mod fcpe_ort_session_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn fcpe_ort_session_runs_multiple_audio_lengths_when_model_is_provided() {
        let Some(path) = std::env::var_os("VC_RS_FCPE_MODEL").map(PathBuf::from) else {
            eprintln!("skip FCPE ORT integration test: VC_RS_FCPE_MODEL is unset");
            return;
        };
        assert!(
            path.is_file(),
            "FCPE model does not exist: {}",
            path.display()
        );

        let mut session = FcpePitchSession::load(
            &path,
            Provider::Cpu,
            None,
            TensorRtRunMode::PinnedCpu,
            TensorRtSessionPurpose::Main,
        )
        .expect("FCPE ORT session should load");

        for samples in [1_600usize, 4_960, 10_080, 15_200, 32_000] {
            let audio: Vec<f32> = (0..samples)
                .map(|index| (std::f32::consts::TAU * 220.0 * index as f32 / 16_000.0).sin() * 0.15)
                .collect();
            let f0 = session
                .extract(&audio)
                .expect("FCPE ORT inference should accept a dynamic length");
            assert_eq!(f0.len(), samples / 160 + 1, "samples={samples}");
            assert!(
                f0.iter().all(|value| value.is_finite() && *value >= 0.0),
                "FCPE output contains an invalid value for samples={samples}"
            );
        }
    }
}

// Native TensorRT is likewise opt-in: the test builds/loads one fixed engine
// per requested FCPE window through the same Rust wrapper used by the runtime,
// then runs inference.  It intentionally does not share an engine between
// lengths; fixed buffers and CUDA Graph capture are part of the contract.
#[cfg(all(test, native_tensorrt, windows))]
mod fcpe_native_tensorrt_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn native_fcpe_session_runs_each_requested_fixed_profile() {
        let Some(path) = std::env::var_os("VC_RS_FCPE_MODEL").map(PathBuf::from) else {
            eprintln!("skip native FCPE integration test: VC_RS_FCPE_MODEL is unset");
            return;
        };
        assert!(
            path.is_file(),
            "FCPE model does not exist: {}",
            path.display()
        );

        let samples = std::env::var("VC_RS_FCPE_SAMPLES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|part| {
                        part.trim()
                            .parse::<usize>()
                            .expect("valid FCPE sample length")
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![4_960, 10_080]);

        for audio_samples in samples {
            let model_cache_key =
                tensor_rt_model_cache_key(&path).expect("FCPE model cache key should be readable");
            let profile = TensorRtSessionProfile::new(
                ModelRole::Fcpe,
                vec![TensorRtInputShape {
                    name: "audio".to_string(),
                    dims: vec![1, audio_samples, 1],
                }],
            )
            .with_optional_model_cache_key(Some(model_cache_key));
            let mut session = FcpePitchSession::load(
                &path,
                Provider::TensorRt,
                Some(profile),
                TensorRtRunMode::PinnedCpu,
                TensorRtSessionPurpose::Main,
            )
            .expect("native FCPE session should load");
            let audio: Vec<f32> = (0..audio_samples)
                .map(|index| (std::f32::consts::TAU * 220.0 * index as f32 / 16_000.0).sin() * 0.15)
                .collect();
            let f0 = session
                .extract(&audio)
                .expect("native FCPE inference should succeed");
            assert_eq!(f0.len(), audio_samples / 160 + 1, "samples={audio_samples}");
            assert!(f0.iter().all(|value| value.is_finite() && *value >= 0.0));
        }
    }
}
