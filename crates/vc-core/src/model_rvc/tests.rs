use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::Provider;

use super::onnx_meta::RvcIoNames;
use super::pitch::{
    align_pitchf_to_features, center_crop_pitchf_to_features, pitchf_tail_for_output,
};
use super::shape::{
    aligned_rvc_input_len, extra_convert_samples_from_ms, keep_tail_in_place,
    onnx_silence_front_feature_frames, output_len_from_convert_size,
    resolve_rvc_context_samples_16k, rmvpe_model_input_samples_16k,
    rmvpe_model_input_samples_for_context_16k, rvc_context_samples_16k_for_frames,
    rvc_frames_for_context_samples_16k, tensor_rt_model_input_samples_16k, EMBEDDER_SAMPLE_RATE,
    RVC_SAMPLE_RATE,
};
use super::stream::{RvcStreamState, StreamInputTiming, VOLUME_DECAY};
use super::tensorrt::{
    format_usize_shape, i64_shape_to_usize, tensor_rt_benchmark_profile, tensor_rt_cache_key,
    tensor_rt_cache_root_from_override, tensor_rt_model_cache_key, tensor_rt_model_file_hash,
    tensor_rt_sanitize_cache_component, validate_tensorrt_input_shape, ModelRole, TensorRtRunMode,
    TensorRtSessionProfile,
};

fn tensor_rt_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("vc-rs-{name}-{}-{nanos}", std::process::id()))
}

#[test]
fn parses_cuda_graph_env_as_opt_in() {
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(None),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("")),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("0")),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("false")),
        TensorRtRunMode::DeviceIo
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("1")),
        TensorRtRunMode::CudaGraph
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("true")),
        TensorRtRunMode::CudaGraph
    );
    assert_eq!(
        TensorRtRunMode::parse_cuda_env(Some("on")),
        TensorRtRunMode::CudaGraph
    );
}

#[test]
fn tensorrt_run_mode_controls_graph_device_io() {
    assert!(TensorRtRunMode::CudaGraph.cuda_graph());
    assert!(TensorRtRunMode::CudaGraph.device_io());
    assert!(!TensorRtRunMode::DeviceIo.cuda_graph());
    assert!(TensorRtRunMode::DeviceIo.device_io());
    assert!(!TensorRtRunMode::PinnedCpu.cuda_graph());
    assert!(!TensorRtRunMode::PinnedCpu.device_io());
    assert!(Provider::TensorRt.is_tensorrt());
    assert!(Provider::Cuda.is_cuda());
    assert!(Provider::WindowsMl.is_windows_ml());
    assert!(Provider::WindowsMlDirectMl.is_windows_ml_directml());
    assert!(Provider::WindowsMlNvTensorRtRtx.is_windows_ml());
    assert!(Provider::WindowsMlOpenVino.is_windows_ml());
    assert!(Provider::WindowsMlQnn.is_windows_ml());
    assert!(Provider::WindowsMlMiGraphX.is_windows_ml());
    assert!(Provider::WindowsMlVitisAi.is_windows_ml());
    assert!(!Provider::WindowsMl.is_cuda());
    assert!(!Provider::WindowsMl.is_tensorrt());
    assert!(!Provider::WindowsMlNvTensorRtRtx.is_tensorrt());
    assert!(!Provider::Cpu.is_tensorrt());
}

#[test]
fn tensorrt_profiles_match_validated_shapes() {
    let contentvec = tensor_rt_benchmark_profile(ModelRole::ContentVec)
        .unwrap()
        .with_model_cache_key("content_vec_500_0123456789abcdef");
    let rmvpe = tensor_rt_benchmark_profile(ModelRole::Rmvpe).unwrap();
    let rvc = tensor_rt_benchmark_profile(ModelRole::Rvc).unwrap();

    assert_eq!(contentvec.profile_shapes, "audio:1x24000");
    assert_eq!(rmvpe.profile_shapes, "waveform:1x24000");
    assert_eq!(rvc.profile_shapes, "feats:1x75x768,pitch:1x75,pitchf:1x75");
    assert_eq!(
        contentvec
            .cache_dir_from_root(Path::new("cache-root"))
            .unwrap(),
        Path::new("cache-root")
            .join("device-0")
            .join("contentvec")
            .join("content_vec_500_0123456789abcdef")
            .join("audio_1x24000")
    );
}

#[test]
fn derives_tensorrt_contentvec_profile_from_default_realtime_chunking() {
    assert_eq!(
        tensor_rt_model_input_samples_16k(960, 48_000, 107, 48_000, 48_000),
        18_240
    );
    assert_eq!(rmvpe_model_input_samples_16k(960, 48_000), 4_960);
    let contentvec = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 18_240);
    let rmvpe = TensorRtSessionProfile::single_input(ModelRole::Rmvpe, "waveform", 4_960);
    let rvc = TensorRtSessionProfile::rvc(114, 768, &RvcIoNames::canonical());

    assert_eq!(contentvec.profile_shapes, "audio:1x18240");
    assert_eq!(rmvpe.profile_shapes, "waveform:1x4960");
    assert_eq!(
        rvc.profile_shapes,
        "feats:1x114x768,pitch:1x114,pitchf:1x114"
    );
    assert_eq!(
        tensor_rt_cache_key("feats:1x114x768,pitch:1x114,pitchf:1x114"),
        "feats_1x114x768_pitch_1x114_pitchf_1x114"
    );
}

#[test]
fn custom_rvc_frames_map_to_a_fixed_contentvec_context() {
    let extra_convert_samples = extra_convert_samples_from_ms(100, RVC_SAMPLE_RATE);
    let automatic_context = tensor_rt_model_input_samples_16k(
        24_000,
        RVC_SAMPLE_RATE,
        107,
        extra_convert_samples,
        RVC_SAMPLE_RATE,
    );
    assert_eq!(automatic_context, 11_520);
    assert_eq!(
        rvc_frames_for_context_samples_16k(
            automatic_context,
            extra_convert_samples,
            RVC_SAMPLE_RATE,
        )
        .unwrap(),
        62
    );
    assert_eq!(
        rvc_context_samples_16k_for_frames(200, extra_convert_samples, RVC_SAMPLE_RATE).unwrap(),
        33_600
    );
    assert_eq!(
        resolve_rvc_context_samples_16k(
            automatic_context,
            Some(200),
            None,
            extra_convert_samples,
            RVC_SAMPLE_RATE,
        )
        .unwrap(),
        33_600
    );
}

#[test]
fn custom_rvc_frames_enforce_model_and_timing_bounds() {
    let extra_convert_samples = extra_convert_samples_from_ms(100, RVC_SAMPLE_RATE);
    let automatic_context = 11_520;
    let static_mismatch = resolve_rvc_context_samples_16k(
        automatic_context,
        Some(200),
        Some(62),
        extra_convert_samples,
        RVC_SAMPLE_RATE,
    )
    .unwrap_err();
    assert!(static_mismatch.to_string().contains("supports only T=62"));
    assert!(resolve_rvc_context_samples_16k(
        automatic_context,
        Some(60),
        None,
        extra_convert_samples,
        RVC_SAMPLE_RATE,
    )
    .is_err());
    assert!(
        rvc_context_samples_16k_for_frames(201, extra_convert_samples, RVC_SAMPLE_RATE).is_err()
    );
}

#[test]
fn static_rvc_frames_are_checked_without_an_explicit_request() {
    let extra_convert_samples = extra_convert_samples_from_ms(100, RVC_SAMPLE_RATE);
    let automatic_context = 11_520; // The default 500 ms/85 ms test timing derives T=62.

    assert_eq!(
        resolve_rvc_context_samples_16k(
            automatic_context,
            None,
            Some(62),
            extra_convert_samples,
            RVC_SAMPLE_RATE,
        )
        .unwrap(),
        automatic_context
    );

    let mismatch = resolve_rvc_context_samples_16k(
        automatic_context,
        None,
        Some(200),
        extra_convert_samples,
        RVC_SAMPLE_RATE,
    )
    .unwrap_err();
    let message = mismatch.to_string();
    assert!(message.contains("static T=200"));
    assert!(message.contains("requires T=62"));
    assert!(message.contains("export-pth --frames 62"));
}

#[test]
fn rmvpe_input_uses_upstream_rvc_bucket_boundaries() {
    assert_eq!(rmvpe_model_input_samples_16k(12_960, 48_000), 4_960);
    assert_eq!(rmvpe_model_input_samples_16k(13_440, 48_000), 10_080);
    assert_eq!(rmvpe_model_input_samples_16k(28_320, 48_000), 10_080);
    assert_eq!(rmvpe_model_input_samples_16k(28_800, 48_000), 15_200);
    assert_eq!(rmvpe_model_input_samples_16k(43_680, 48_000), 15_200);
    assert_eq!(rmvpe_model_input_samples_16k(44_160, 48_000), 20_320);
}

#[test]
fn rmvpe_input_is_capped_to_available_context_without_padding() {
    assert_eq!(
        rmvpe_model_input_samples_for_context_16k(13_440, 48_000, 4_480),
        4_480
    );
    assert_eq!(
        rmvpe_model_input_samples_for_context_16k(13_440, 48_000, 12_000),
        10_080
    );
}

#[test]
fn contentvec_fixed_profile_allows_non_default_input_name() {
    let contentvec =
        TensorRtSessionProfile::single_input(ModelRole::ContentVec, "input_values", 18_240)
            .with_model_cache_key("content_vec_500_0123456789abcdef");

    assert_eq!(contentvec.profile_shapes, "input_values:1x18240");
    assert_eq!(
        contentvec.fixed_input_dims("input_values").unwrap(),
        &[1, 18_240]
    );
    assert!(contentvec.fixed_input_dims("audio").is_err());
    assert_eq!(
        contentvec
            .cache_dir_from_root(Path::new("cache-root"))
            .unwrap(),
        Path::new("cache-root")
            .join("device-0")
            .join("contentvec")
            .join("content_vec_500_0123456789abcdef")
            .join("input_values_1x18240")
    );
}

#[test]
fn tensor_rt_cache_root_override_wins() {
    assert_eq!(
        tensor_rt_cache_root_from_override(Some(OsStr::new("override-cache"))).unwrap(),
        PathBuf::from("override-cache")
    );
}

#[test]
fn tensor_rt_model_cache_key_hashes_file_contents() {
    let dir = tensor_rt_temp_dir("model-cache-key");
    fs::create_dir_all(&dir).unwrap();
    let model_a = dir.join("voice opt.onnx");
    let model_a_copy = dir.join("voice copy.onnx");
    let model_b = dir.join("voice changed.onnx");
    fs::write(&model_a, b"same model bytes").unwrap();
    fs::write(&model_a_copy, b"same model bytes").unwrap();
    fs::write(&model_b, b"different model bytes").unwrap();

    let hash_a = tensor_rt_model_file_hash(&model_a).unwrap();
    let hash_a_copy = tensor_rt_model_file_hash(&model_a_copy).unwrap();
    let hash_b = tensor_rt_model_file_hash(&model_b).unwrap();
    assert_eq!(hash_a, hash_a_copy);
    assert_ne!(hash_a, hash_b);

    let key = tensor_rt_model_cache_key(&model_a).unwrap();
    assert!(key.starts_with("voice_opt_"));
    assert!(key.ends_with(&format!("{hash_a:016x}")));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tensor_rt_sanitizes_model_cache_components() {
    assert_eq!(
        tensor_rt_sanitize_cache_component("voice opt+v1"),
        "voice_opt_v1"
    );
    assert_eq!(
        tensor_rt_sanitize_cache_component("abc_DEF-01"),
        "abc_DEF_01"
    );
    assert_eq!(tensor_rt_sanitize_cache_component(""), "model");
}

#[test]
fn validates_tensorrt_profile_input_shapes() {
    let contentvec = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 24_000);
    let rmvpe = TensorRtSessionProfile::single_input(ModelRole::Rmvpe, "waveform", 24_000);
    let rvc = TensorRtSessionProfile::rvc(75, 768, &RvcIoNames::canonical());

    validate_tensorrt_input_shape(Provider::TensorRt, Some(&contentvec), "audio", &[1, 24_000])
        .unwrap();
    validate_tensorrt_input_shape(Provider::TensorRt, Some(&rmvpe), "waveform", &[1, 24_000])
        .unwrap();
    validate_tensorrt_input_shape(Provider::TensorRt, Some(&rvc), "feats", &[1, 75, 768]).unwrap();
    validate_tensorrt_input_shape(Provider::TensorRt, Some(&rvc), "pitch", &[1, 75]).unwrap();
    validate_tensorrt_input_shape(Provider::Cpu, None, "pitch", &[1, 74]).unwrap();

    let err = validate_tensorrt_input_shape(Provider::TensorRt, Some(&rvc), "pitch", &[1, 74])
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("requires input 'pitch' shape 1x75"));
    let err =
        validate_tensorrt_input_shape(Provider::Cuda, Some(&rvc), "pitch", &[1, 74]).unwrap_err();
    assert!(err
        .to_string()
        .contains("requires input 'pitch' shape 1x75"));
    let err = validate_tensorrt_input_shape(
        Provider::WindowsMlNvTensorRtRtx,
        Some(&rvc),
        "pitch",
        &[1, 74],
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("requires input 'pitch' shape 1x75"));
    assert_eq!(format_usize_shape(&[1, 75, 768]), "1x75x768");
}

#[test]
fn looks_up_tensorrt_fixed_input_dims_explicitly() {
    let rvc = TensorRtSessionProfile::rvc(113, 768, &RvcIoNames::canonical());

    assert_eq!(rvc.fixed_input_dims("feats").unwrap(), &[1, 113, 768]);
    assert_eq!(rvc.fixed_input_dims("pitchf").unwrap(), &[1, 113]);
    let err = rvc.fixed_input_dims("threshold").unwrap_err();

    assert!(err
        .to_string()
        .contains("does not include input 'threshold'"));
}

#[test]
fn rejects_negative_runtime_output_shape_dims() {
    let err = i64_shape_to_usize(&[1, -1, 768], "contentvec output").unwrap_err();

    assert!(err.to_string().contains("negative or too-large dim -1"));
}

#[test]
fn aligns_pitchf_by_taking_tail_frames() {
    assert_eq!(
        align_pitchf_to_features(&[1.0, 2.0, 3.0, 4.0], 2),
        vec![3.0, 4.0]
    );
}

#[test]
fn keeps_only_requested_output_tail() {
    let mut audio = vec![1, 2, 3, 4, 5];
    keep_tail_in_place(&mut audio, 3);
    assert_eq!(audio, vec![3, 4, 5]);
}

// The RMS-mix reference now reads the 16 kHz rolling buffer (the signal
// ContentVec/F0 see), not the device-rate `audio_buffer`. These unit tests keep
// the matched-rate (16 kHz in == 16 kHz out) tail/pad math by populating
// `audio_16k_buffer` and passing `EMBEDDER_SAMPLE_RATE` — the assertion values
// are unchanged; only the buffer the reference is drawn from moved.
#[test]
fn output_reference_audio_uses_tail_matching_trimmed_output() {
    let mut state = RvcStreamState::new(48_000);
    state.audio_16k_buffer = (0..8).map(|value| value as f32).collect();
    let mut scratch = Vec::new();

    let reference = state
        .output_reference_audio(EMBEDDER_SAMPLE_RATE, EMBEDDER_SAMPLE_RATE, 5, &mut scratch)
        .unwrap();

    assert_eq!(reference, &[3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
fn output_reference_audio_left_pads_when_history_is_short() {
    let mut state = RvcStreamState::new(48_000);
    state.audio_16k_buffer = vec![1.0, 2.0];
    let mut scratch = Vec::new();

    let reference = state
        .output_reference_audio(EMBEDDER_SAMPLE_RATE, EMBEDDER_SAMPLE_RATE, 4, &mut scratch)
        .unwrap();

    assert_eq!(reference, &[0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn aligns_pitchf_by_left_padding_short_inputs() {
    assert_eq!(
        align_pitchf_to_features(&[3.0, 4.0], 4),
        vec![0.0, 0.0, 3.0, 4.0]
    );
}

#[test]
fn aligns_realtime_rvc_input_to_16k_hop_samples() {
    assert_eq!(aligned_rvc_input_len(4800, 48_000, 5632), 10560);
}

#[test]
fn derives_output_len_like_reference_pipeline() {
    assert_eq!(
        output_len_from_convert_size(3520, 48_000, 4096, 48_000),
        6465
    );
}

#[test]
fn stream_state_aligns_convert_size_to_16k_hop_samples() {
    let mut state = RvcStreamState::new(48_000);
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();
    assert_eq!(out.convert_size, 29_760);
}

#[test]
fn stream_state_derives_out_size_from_extra_convert_size() {
    let mut state = RvcStreamState::new(48_000);
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();
    assert_eq!(out.out_size, 25_665);
}

#[test]
fn stream_state_custom_context_keeps_the_output_cadence() {
    let mut state = RvcStreamState::new(48_000);
    state.set_contentvec_context_samples_16k(33_600);
    let input = vec![0.0; 24_000];
    let out = state
        .generate_input(&input, 48_000, 1_536, 1_536, 4_096)
        .unwrap();

    assert_eq!(out.convert_size, 100_800);
    // The extra context is left history only; the chunk output still follows
    // the normal timing window so it does not add output latency.
    assert_eq!(out.out_size, 25_665);
    assert_eq!(state.audio_16k_buffer.len(), 33_600);
}

#[test]
fn stream_state_zero_pads_initial_buffer() {
    let mut state = RvcStreamState::new(48_000);
    let out = state
        .generate_input(&[1.0, 2.0, 3.0, 4.0], 48_000, 0, 0, 4_096)
        .unwrap();
    assert_eq!(state.audio_buffer.len(), out.convert_size);
    assert!(state.audio_buffer[..state.audio_buffer.len() - 4]
        .iter()
        .all(|x| *x == 0.0));
    assert_eq!(
        &state.audio_buffer[state.audio_buffer.len() - 4..],
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn stream_state_keeps_16k_history_for_embedder() {
    let mut state = RvcStreamState::new(48_000);
    let input = vec![0.25; 4_800];

    state.generate_input(&input, 48_000, 0, 0, 0).unwrap();

    assert_eq!(state.audio_buffer.len(), 4_800);
    assert_eq!(state.audio_16k_buffer.len(), 1_600);
    assert_eq!(state.audio_16k_buffer, state.pitch_16k_buffer);
    assert!(
        state.pitch_resampler_16k.is_none(),
        "the no-denoiser path must not run a second 16 kHz resampler"
    );
    assert!(state
        .audio_16k_buffer
        .iter()
        .any(|sample| sample.abs() > 1e-4));
}

#[test]
fn stream_state_keeps_content_and_pitch_branches_aligned() {
    let mut state = RvcStreamState::new(48_000);
    let raw = vec![0.8; 4_800];
    let denoised = vec![0.2; 4_800];

    state
        .generate_input_with_pitch(
            &raw,
            &denoised,
            StreamInputTiming {
                sample_rate: 48_000,
                crossfade_and_search_samples: 0,
                volume_excluded_samples: 0,
                extra_convert_samples: 0,
                denoiser_content_mix: 0.25,
                denoiser_rmvpe_mix: 1.0,
            },
        )
        .unwrap();

    assert_eq!(state.audio_16k_buffer.len(), state.pitch_16k_buffer.len());
    assert_eq!(state.audio_16k_buffer.len(), 1_600);
    // Both independent streaming resamplers have the same filter startup
    // transient. Verify the resulting linear relation instead of assuming a
    // constant value before that transient settles: 0.75 * 0.8 + 0.25 * 0.2
    // equals 3.25 times the pitch branch's 0.2 source.
    assert!(state
        .audio_16k_buffer
        .iter()
        .zip(&state.pitch_16k_buffer)
        .all(|(content, pitch)| (*content - *pitch * 3.25).abs() < 1e-4));
    assert!(state
        .pitch_16k_buffer
        .iter()
        .any(|sample| sample.abs() > 1e-4));
}

#[test]
fn stream_state_rmvpe_mix_interpolates_aligned_raw_and_denoised_inputs() {
    let raw = vec![0.8; 4_800];
    let denoised = vec![0.2; 4_800];
    let run = |denoiser_rmvpe_mix| {
        let mut state = RvcStreamState::new(48_000);
        state
            .generate_input_with_pitch(
                &raw,
                &denoised,
                StreamInputTiming {
                    sample_rate: 48_000,
                    crossfade_and_search_samples: 0,
                    volume_excluded_samples: 0,
                    extra_convert_samples: 0,
                    denoiser_content_mix: 0.0,
                    denoiser_rmvpe_mix,
                },
            )
            .unwrap();
        state.pitch_16k_buffer.clone()
    };

    let raw_pitch = run(0.0);
    let mixed_pitch = run(0.5);
    let denoised_pitch = run(1.0);
    assert_eq!(raw_pitch.len(), denoised_pitch.len());
    assert!(raw_pitch
        .iter()
        .zip(&denoised_pitch)
        .zip(&mixed_pitch)
        .all(|((raw, denoised), mixed)| { (*mixed - (*raw + *denoised) * 0.5).abs() < 1e-5 }));
    assert!(raw_pitch
        .iter()
        .zip(&denoised_pitch)
        .any(|(raw, denoised)| (raw - denoised).abs() > 1e-4));
}

#[test]
fn stream_state_path_change_restarts_both_histories() {
    let mut state = RvcStreamState::new(48_000);
    let first = vec![1.0; 160];
    state.generate_input(&first, 16_000, 0, 0, 480).unwrap();

    let raw = vec![0.8; 160];
    let denoised = vec![0.2; 160];
    state
        .generate_input_with_pitch(
            &raw,
            &denoised,
            StreamInputTiming {
                sample_rate: 16_000,
                crossfade_and_search_samples: 0,
                volume_excluded_samples: 0,
                extra_convert_samples: 480,
                denoiser_content_mix: 0.25,
                denoiser_rmvpe_mix: 1.0,
            },
        )
        .unwrap();

    assert!(state.pitch_resampler_16k.is_some());
    assert_eq!(state.audio_16k_buffer.len(), 320);
    assert!(state.audio_16k_buffer[..160]
        .iter()
        .all(|sample| *sample == 0.0));
    assert!(state.pitch_16k_buffer[..160]
        .iter()
        .all(|sample| *sample == 0.0));
    assert!(state.audio_16k_buffer[160..]
        .iter()
        .all(|sample| (*sample - 0.65).abs() < 1e-6));
    assert!(state.pitch_16k_buffer[160..]
        .iter()
        .all(|sample| (*sample - 0.2).abs() < 1e-6));
}

#[test]
fn stream_state_volume_excludes_crossfade_not_sola_search() {
    let mut state = RvcStreamState::new(48_000);
    let mut input = vec![1.0; 80];
    input.extend(std::iter::repeat_n(0.0, 80));

    let out = state.generate_input(&input, 16_000, 480, 240, 0).unwrap();

    assert!((out.volume - 0.5f32.sqrt()).abs() < 1e-6);
}

#[test]
fn stream_state_volume_keeps_decay_from_previous_chunk() {
    let mut state = RvcStreamState::new(48_000);
    let loud = vec![1.0; 160];
    let quiet = vec![0.0; 160];

    let first = state.generate_input(&loud, 16_000, 0, 0, 0).unwrap();
    let second = state.generate_input(&quiet, 16_000, 0, 0, 0).unwrap();

    assert!((first.volume - 1.0).abs() < 1e-6);
    assert!((second.volume - VOLUME_DECAY).abs() < 1e-6);
}

#[test]
fn align_pitchf_to_features_uses_tail_for_feature_length() {
    let pitchf = vec![0.0, 10.0, 20.0, 30.0, 40.0];
    assert_eq!(align_pitchf_to_features(&pitchf, 3), vec![20.0, 30.0, 40.0]);
}

#[test]
fn center_crops_pitchf_to_feature_grid() {
    let pitchf: Vec<f32> = (0..183).map(|frame| frame as f32).collect();

    let aligned = center_crop_pitchf_to_features(&pitchf, 180);

    assert_eq!(aligned.len(), 180);
    assert_eq!(aligned[0], 1.0);
    assert_eq!(aligned[179], 180.0);
}

#[test]
fn pitchf_tail_for_output_matches_10ms_output_frames() {
    let pitchf = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    assert_eq!(
        pitchf_tail_for_output(&pitchf, 1_440, RVC_SAMPLE_RATE),
        vec![30.0, 40.0, 50.0]
    );
}

#[test]
fn stream_state_pitch_update_places_estimator_tail_window_at_absolute_frame() {
    let mut state = RvcStreamState::new(48_000);
    state.pitchf_buffer = (0..34).map(|frame| frame as f32).collect();

    state.update_pitchf_from_estimator_window(&[100.0, 101.0, 102.0, 103.0], 480);

    assert_eq!(state.pitchf_buffer[2], 2.0);
    assert_eq!(&state.pitchf_buffer[3..7], &[100.0, 101.0, 102.0, 103.0]);
}

#[test]
fn stream_state_pitch_update_drops_fcpe_center_padded_tail_frame() {
    let mut state = RvcStreamState::new(48_000);
    state.pitchf_buffer = vec![0.0, 1.0, 2.0];

    state.update_pitchf_from_estimator_window(&[10.0, 20.0, 30.0, 40.0], 0);

    assert_eq!(state.pitchf_buffer, vec![10.0, 20.0, 30.0]);
}

#[test]
fn derives_vcclient_onnx_silence_front_feature_offset() {
    assert_eq!(onnx_silence_front_feature_frames(4096, 48_000), 6);
}

#[test]
fn extra_convert_samples_scale_with_model_sample_rate() {
    // The same convert-context duration is fewer samples at a lower model rate.
    assert_eq!(extra_convert_samples_from_ms(100, 48_000), 4_800);
    assert_eq!(extra_convert_samples_from_ms(100, 40_000), 4_000);
    assert_eq!(extra_convert_samples_from_ms(100, 32_000), 3_200);
}

#[test]
fn out_size_tracks_model_sample_rate() {
    // Run the same device-rate input through models of different native rates.
    // The fix is rate-generic (reads `samplingRate`), not special-cased per rate,
    // so 32 kHz and 40 kHz are both handled like 48 kHz.
    let chunk = vec![0.1f32; 4_800]; // 100 ms at the 48 kHz device rate
    let device_rate = 48_000;

    let run = |rvc_rate: u32| {
        RvcStreamState::new(rvc_rate)
            .generate_input(
                &chunk,
                device_rate,
                0,
                0,
                extra_convert_samples_from_ms(100, rvc_rate),
            )
            .unwrap()
    };

    let base = run(48_000);
    for rvc_rate in [32_000u32, 40_000] {
        let out = run(rvc_rate);
        // The ContentVec/F0 window lives in the 16 kHz domain (independent of the
        // model output rate), so `convert_size` matches the 48 kHz baseline.
        assert_eq!(
            base.convert_size, out.convert_size,
            "device-rate convert window must not depend on model rate ({rvc_rate} Hz)"
        );
        // `out_size` is in the model's output-rate domain, so it scales by
        // rvc_rate/48000 relative to the 48 kHz baseline.
        assert!(
            out.out_size < base.out_size,
            "{rvc_rate} Hz output window must be shorter than 48 kHz: {} vs {}",
            out.out_size,
            base.out_size
        );
        let expected = base.out_size * rvc_rate as usize / 48_000;
        assert!(
            (out.out_size as i64 - expected as i64).abs() <= 1,
            "{rvc_rate} Hz out_size {} not ~{rvc_rate}/48000 of 48 kHz {} (expected {})",
            out.out_size,
            base.out_size,
            expected
        );
    }
}
#[test]
fn gpu_priority_defaults_to_high() {
    assert_eq!(super::GpuPriority::default(), super::GpuPriority::High);
}

#[test]
fn tensor_rt_cache_is_separated_by_gpu_device_id() {
    let profile = TensorRtSessionProfile::single_input(ModelRole::ContentVec, "audio", 24_000)
        .with_model_cache_key("model")
        .with_gpu_device_id(2);
    assert_eq!(
        profile
            .cache_dir_from_root(Path::new("cache-root"))
            .unwrap(),
        Path::new("cache-root")
            .join("device-2")
            .join("contentvec")
            .join("model")
            .join("audio_1x24000")
    );
}
