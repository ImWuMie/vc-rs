use anyhow::{bail, Context, Result};

// ContentVec emits frames on a 20 ms stride at 16 kHz. RVC's 10 ms grid is
// created later by repeating feature frames, so keep this alignment scoped to
// the shared ContentVec/RMVPE waveform context and not pitch/output sizing.
pub const EMBEDDER_SAMPLE_RATE: u32 = 16_000;
pub(super) const RVC_SAMPLE_RATE: u32 = 48_000;
pub(super) const CONTENTVEC_CONTEXT_ALIGN_SAMPLES: usize = 320;
pub(super) const RMVPE_FRAME_SAMPLES_16K: usize = 160;
pub(super) const RMVPE_BUCKET_FRAMES: usize = 32;
pub(super) const RMVPE_GUARD_FRAMES: usize = 5;
/// Upper bound for a manually selected dynamic RVC window. It prevents a typo
/// from requesting an impractically large fixed TensorRT engine or rolling
/// audio buffer. Dynamic ONNX models do not encode a numeric `frames` maximum.
pub(super) const MAX_CUSTOM_RVC_FRAMES: usize = 2_048;

pub(super) fn ms_to_samples(sample_rate: u32, ms: u32) -> usize {
    ((sample_rate as u64 * ms as u64) / 1000) as usize
}

pub(super) fn extra_convert_samples_from_ms(ms: u32, rvc_sample_rate: u32) -> usize {
    ms_to_samples(rvc_sample_rate, ms)
}

pub(super) fn feature_len_for_samples(samples: usize, sample_rate: u32) -> usize {
    (samples as u64 * 100 / sample_rate as u64) as usize
}

pub(super) enum Rounding {
    Floor,
    Ceil,
}

pub(super) fn samples_between_rates(
    samples: usize,
    from_sample_rate: u32,
    to_sample_rate: u32,
    rounding: Rounding,
) -> usize {
    let numerator = samples as u64 * to_sample_rate as u64;
    let denominator = from_sample_rate as u64;
    match rounding {
        Rounding::Floor => (numerator / denominator) as usize,
        Rounding::Ceil => numerator.div_ceil(denominator) as usize,
    }
}

pub(super) fn onnx_silence_front_feature_frames(
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> usize {
    let extra_16k_samples = (extra_convert_samples as u64 * EMBEDDER_SAMPLE_RATE as u64
        / rvc_sample_rate as u64) as usize;
    (extra_16k_samples / 360) * 2
}

pub(super) fn keep_tail_in_place<T>(values: &mut Vec<T>, len: usize) {
    if values.len() > len {
        values.drain(..values.len() - len);
    }
}

#[cfg(test)]
pub(super) fn aligned_rvc_input_len(
    chunk_len: usize,
    sample_rate: u32,
    extra_48k_samples: usize,
) -> usize {
    let chunk_16k = samples_between_rates(
        chunk_len,
        sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let extra_16k = samples_between_rates(
        extra_48k_samples,
        RVC_SAMPLE_RATE,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let convert_16k = align_up(chunk_16k + extra_16k, CONTENTVEC_CONTEXT_ALIGN_SAMPLES);
    samples_between_rates(
        convert_16k,
        EMBEDDER_SAMPLE_RATE,
        sample_rate,
        Rounding::Ceil,
    )
}

#[cfg(test)]
pub(super) fn output_len_from_convert_size(
    convert_len_16k: usize,
    _input_sample_rate: u32,
    extra_48k_samples: usize,
    output_sample_rate: u32,
) -> usize {
    let extra_16k = samples_between_rates(
        extra_48k_samples,
        RVC_SAMPLE_RATE,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    samples_between_rates(
        convert_len_16k.saturating_sub(extra_16k),
        EMBEDDER_SAMPLE_RATE,
        output_sample_rate,
        Rounding::Floor,
    )
    .max(1)
}

pub(super) fn align_up(value: usize, align: usize) -> usize {
    if align == 0 || value.is_multiple_of(align) {
        value
    } else {
        value + (align - value % align)
    }
}

pub(super) fn tensor_rt_model_input_samples_16k(
    chunk_samples: usize,
    sample_rate: u32,
    output_extra_ms: u32,
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> usize {
    tensor_rt_convert_size_16k(
        chunk_samples,
        sample_rate,
        ms_to_samples(rvc_sample_rate, output_extra_ms),
        extra_convert_samples,
        rvc_sample_rate,
    )
}

/// Returns the RVC generator frame count produced by the standard ContentVec
/// frontend for a fixed 16 kHz waveform window. ContentVec's convolutional
/// frontend has a 320-sample hop and emits one fewer frame than the number of
/// complete hops. Keep this paired with `rvc_context_samples_16k_for_frames`:
/// fixed TensorRT profiles and the rolling stream window must agree exactly.
pub(super) fn rvc_frames_for_context_samples_16k(
    context_samples_16k: usize,
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> Result<usize> {
    if !context_samples_16k.is_multiple_of(CONTENTVEC_CONTEXT_ALIGN_SAMPLES) {
        bail!(
            "ContentVec context {context_samples_16k} is not aligned to its {}-sample hop",
            CONTENTVEC_CONTEXT_ALIGN_SAMPLES
        );
    }
    let contentvec_frames = context_samples_16k
        .checked_div(CONTENTVEC_CONTEXT_ALIGN_SAMPLES)
        .and_then(|frames| frames.checked_sub(1))
        .context("ContentVec context is too short to emit a frame")?;
    let repeated_frames = contentvec_frames
        .checked_mul(2)
        .context("RVC frame count overflow")?;
    let silence_front_frames =
        onnx_silence_front_feature_frames(extra_convert_samples, rvc_sample_rate);
    let frames = if silence_front_frames > 0 && silence_front_frames < repeated_frames {
        repeated_frames - silence_front_frames
    } else {
        repeated_frames
    };
    if frames == 0 {
        bail!("derived zero RVC frames from ContentVec context")
    }
    Ok(frames)
}

/// Inverts the ContentVec/RVC time-grid mapping for a fixed generator profile.
/// A requested `T` changes only load-time context/history: the audio chunk size
/// and its output cadence remain unchanged. The realtime path must never turn a
/// custom T into a per-callback shape change or engine rebuild.
pub(super) fn rvc_context_samples_16k_for_frames(
    frames: usize,
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> Result<usize> {
    if frames == 0 {
        bail!("RVC frames must be greater than zero; use 0/auto to derive it from timing")
    }
    if frames > MAX_CUSTOM_RVC_FRAMES {
        bail!(
            "RVC frames {frames} exceed the supported custom limit {MAX_CUSTOM_RVC_FRAMES}; use 0/auto or a smaller value"
        )
    }
    let silence_front_frames =
        onnx_silence_front_feature_frames(extra_convert_samples, rvc_sample_rate);
    let repeated_frames = frames
        .checked_add(silence_front_frames)
        .context("RVC frame count overflow")?;
    if !repeated_frames.is_multiple_of(2) {
        bail!(
            "RVC frames {frames} cannot be represented by this ContentVec model; choose an even frame count"
        )
    }
    let contentvec_frames = repeated_frames / 2;
    let context_samples_16k = contentvec_frames
        .checked_add(1)
        .and_then(|frames| frames.checked_mul(CONTENTVEC_CONTEXT_ALIGN_SAMPLES))
        .context("ContentVec context size overflow")?;
    let actual = rvc_frames_for_context_samples_16k(
        context_samples_16k,
        extra_convert_samples,
        rvc_sample_rate,
    )?;
    if actual != frames {
        bail!(
            "RVC frame mapping produced {actual} frames for requested {frames}; choose a compatible frame count"
        )
    }
    Ok(context_samples_16k)
}

/// Resolves a requested fixed RVC frame count against the model contract and
/// current audio timing. A dynamic ONNX's symbolic `frames` axis has no numeric
/// range, so its lower bound is the current automatically derived context; a
/// static ONNX has one exact valid value.
pub(super) fn resolve_rvc_context_samples_16k(
    automatic_context_samples_16k: usize,
    requested_frames: Option<usize>,
    static_model_frames: Option<usize>,
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> Result<usize> {
    let automatic_frames = rvc_frames_for_context_samples_16k(
        automatic_context_samples_16k,
        extra_convert_samples,
        rvc_sample_rate,
    )?;
    let Some(requested_frames) = requested_frames else {
        // A static ONNX has exactly one legal generator frame count.  Do this
        // check before loading any backend session so CPU/DirectML fail at
        // startup just like TensorRT; the realtime worker must never discover
        // a profile mismatch on its first callback.  Dynamic models have no
        // numeric contract here and keep the automatically derived context.
        if let Some(static_model_frames) = static_model_frames {
            if static_model_frames != automatic_frames {
                bail!(
                    "RVC ONNX has static T={static_model_frames}, but the current runtime requires T={automatic_frames}; re-export with `export-pth --frames {automatic_frames}` or use a matching runtime configuration"
                );
            }
        }
        return Ok(automatic_context_samples_16k);
    };
    if let Some(static_model_frames) = static_model_frames {
        if requested_frames != static_model_frames {
            bail!(
                "RVC ONNX supports only T={static_model_frames}, but --rvc-frames requested T={requested_frames}; use T={static_model_frames} or load a dynamic ONNX"
            )
        }
    }
    if requested_frames < automatic_frames {
        bail!(
            "RVC T={requested_frames} is smaller than the current timing requires (minimum T={automatic_frames}); reduce chunk/context timing or use auto/a larger even T"
        )
    }
    rvc_context_samples_16k_for_frames(requested_frames, extra_convert_samples, rvc_sample_rate)
}

pub(super) fn rmvpe_model_input_samples_16k(chunk_samples: usize, sample_rate: u32) -> usize {
    // Match upstream RVC's RMVPE framing: 10 ms hop at 16 kHz, then mel2hidden
    // pads hidden frames to 32-frame buckets. The guard frames preserve a small
    // amount of F0 context without coupling RMVPE to ContentVec's larger window.
    let chunk_frames = (chunk_samples as u64 * 100).div_ceil(sample_rate as u64) as usize;
    let required_frames = chunk_frames.saturating_add(RMVPE_GUARD_FRAMES).max(1);
    let bucket_frames = align_up(required_frames, RMVPE_BUCKET_FRAMES);
    bucket_frames.saturating_sub(1) * RMVPE_FRAME_SAMPLES_16K
}

pub(super) fn rmvpe_model_input_samples_for_context_16k(
    chunk_samples: usize,
    sample_rate: u32,
    max_context_samples_16k: usize,
) -> usize {
    rmvpe_model_input_samples_16k(chunk_samples, sample_rate).min(max_context_samples_16k)
}

pub(super) fn tensor_rt_convert_size_16k(
    new_audio_samples: usize,
    sample_rate: u32,
    output_extra_samples: usize,
    extra_convert_samples: usize,
    rvc_sample_rate: u32,
) -> usize {
    let new_audio_16k_samples = samples_between_rates(
        new_audio_samples,
        sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let output_extra_16k_samples = samples_between_rates(
        output_extra_samples,
        rvc_sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    let extra_16k_samples = samples_between_rates(
        extra_convert_samples,
        rvc_sample_rate,
        EMBEDDER_SAMPLE_RATE,
        Rounding::Floor,
    );
    align_up(
        new_audio_16k_samples + output_extra_16k_samples + extra_16k_samples,
        CONTENTVEC_CONTEXT_ALIGN_SAMPLES,
    )
}
