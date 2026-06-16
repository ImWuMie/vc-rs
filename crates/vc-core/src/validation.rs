use anyhow::{bail, Result};

pub const CONVERSION_TIMING_LIMITS: ConversionTimingLimits = ConversionTimingLimits {
    min_chunk_ms: 20,
    max_chunk_ms: 2000,
    max_crossfade_ms: 1000,
    max_sola_search_ms: 1000,
    max_tail_discard_ms: 1000,
    max_output_extra_ms: 3000,
    min_extra_convert_ms: 20,
    max_extra_convert_ms: 3000,
};

#[derive(Clone, Copy, Debug)]
pub struct ConversionTiming {
    pub chunk_ms: u32,
    pub crossfade_ms: u32,
    pub sola_search_ms: u32,
    pub tail_discard_ms: u32,
    pub extra_convert_ms: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ConversionTimingLimits {
    pub min_chunk_ms: u32,
    pub max_chunk_ms: u32,
    pub max_crossfade_ms: u32,
    pub max_sola_search_ms: u32,
    pub max_tail_discard_ms: u32,
    pub max_output_extra_ms: u32,
    pub min_extra_convert_ms: u32,
    pub max_extra_convert_ms: u32,
}

pub fn validate_conversion_timing(
    timing: ConversionTiming,
    limits: ConversionTimingLimits,
) -> Result<()> {
    validate_u32_range(
        "chunk_ms",
        timing.chunk_ms,
        limits.min_chunk_ms,
        limits.max_chunk_ms,
    )?;
    validate_u32_range(
        "crossfade_ms",
        timing.crossfade_ms,
        0,
        limits.max_crossfade_ms,
    )?;
    validate_u32_range(
        "sola_search_ms",
        timing.sola_search_ms,
        0,
        limits.max_sola_search_ms,
    )?;
    validate_u32_range(
        "rvc_output_tail_discard_ms",
        timing.tail_discard_ms,
        0,
        limits.max_tail_discard_ms,
    )?;
    validate_u32_range(
        "extra_convert_ms",
        timing.extra_convert_ms,
        limits.min_extra_convert_ms,
        limits.max_extra_convert_ms,
    )?;

    // The three output-context knobs are added before pipeline construction and
    // then converted to sample counts. Keep this bound shared so front-ends
    // cannot accidentally request a huge TensorRT profile or ring buffer.
    let output_extra_ms = timing
        .crossfade_ms
        .checked_add(timing.sola_search_ms)
        .and_then(|v| v.checked_add(timing.tail_discard_ms))
        .ok_or_else(|| anyhow::anyhow!("output context milliseconds overflow u32"))?;
    validate_u32_range(
        "crossfade_ms + sola_search_ms + rvc_output_tail_discard_ms",
        output_extra_ms,
        0,
        limits.max_output_extra_ms,
    )?;

    Ok(())
}

pub fn validate_unit_interval(name: &str, value: f32) -> Result<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        bail!("{name} must be a finite value in 0.0..=1.0")
    }
}

pub fn validate_non_negative_f32(name: &str, value: f32) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        bail!("{name} must be a finite, non-negative value")
    }
}

pub fn validate_finite_f32(name: &str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        bail!("{name} must be finite")
    }
}

fn validate_u32_range(name: &str, value: u32, min: u32, max: u32) -> Result<()> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        bail!("{name} must be in {min}..={max} ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_timing() -> ConversionTiming {
        ConversionTiming {
            chunk_ms: 500,
            crossfade_ms: 85,
            sola_search_ms: 12,
            tail_discard_ms: 10,
            extra_convert_ms: 100,
        }
    }

    #[test]
    fn accepts_default_realtime_timing() {
        validate_conversion_timing(valid_timing(), CONVERSION_TIMING_LIMITS).unwrap();
    }

    #[test]
    fn rejects_chunk_values_outside_profile() {
        assert!(validate_conversion_timing(
            ConversionTiming {
                chunk_ms: CONVERSION_TIMING_LIMITS.min_chunk_ms - 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
        assert!(validate_conversion_timing(
            ConversionTiming {
                chunk_ms: CONVERSION_TIMING_LIMITS.max_chunk_ms + 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
    }

    #[test]
    fn rejects_extra_convert_values_outside_profile() {
        assert!(validate_conversion_timing(
            ConversionTiming {
                extra_convert_ms: CONVERSION_TIMING_LIMITS.min_extra_convert_ms - 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
        assert!(validate_conversion_timing(
            ConversionTiming {
                extra_convert_ms: CONVERSION_TIMING_LIMITS.max_extra_convert_ms + 1,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
    }

    #[test]
    fn rejects_excessive_output_context_sum() {
        assert!(validate_conversion_timing(
            ConversionTiming {
                crossfade_ms: 1000,
                sola_search_ms: 1000,
                tail_discard_ms: 1001,
                ..valid_timing()
            },
            CONVERSION_TIMING_LIMITS,
        )
        .is_err());
    }
}
