//! Read-only RVC feature retrieval over the standard FAISS IVF-Flat index.
//!
//! RVC training writes `added_IVF*_Flat_*.index` through
//! `faiss.index_factory(dim, "IVF<nlist>,Flat")`. Pulling FAISS or Python into
//! every vc-rs package would make deployment considerably heavier, so this file
//! implements the deliberately small, documented subset that those indexes use:
//! `IndexIVFFlat` with an `IndexFlatL2` coarse quantizer and
//! `ArrayInvertedLists`. Other FAISS index families fail at load time instead of
//! producing plausible but wrong features.
//!
//! Loading and decoding happen only while a pipeline is being constructed. The
//! worker-side query path reuses its scratch capacity after the first feature
//! window shape is seen and never opens files or takes locks, which is important
//! because it runs once per conversion chunk in realtime sessions.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

const FAISS_L2_METRIC: i32 = 1;
const NEIGHBORS: usize = 8;
// Stock RVC writers persist nprobe=1. Supporting a small number of probes is
// useful for compatible user-modified indexes, while bounding worker CPU time
// and scratch size for a realtime feature path.
const MAX_NPROBE: usize = 8;
const MAX_NLIST: usize = 1_000_000;
const MAX_INDEX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const EXACT_DISTANCE_EPSILON: f32 = 1.0e-12;
// Adaptive retrieval is deliberately conservative. A weak match still keeps a
// small target-voice share, while unvoiced/boundary frames lose substantially
// more index influence so consonants remain anchored to the source ContentVec.
const ADAPTIVE_INDEX_MIN_DISTANCE_FACTOR: f32 = 0.35;
const ADAPTIVE_INDEX_MIN_VOICING_FACTOR: f32 = 0.40;
const ADAPTIVE_INDEX_BOUNDARY_REDUCTION: f32 = 0.30;
const ADAPTIVE_PROTECT_MIN_FACTOR: f32 = 0.20;
const DISTANCE_CONFIDENCE_FULL_RATIO: f32 = 0.45;
const DISTANCE_CONFIDENCE_NONE_RATIO: f32 = 0.95;
// RMVPE's supported model contract exposes thresholded pitchf, not posterior
// probabilities. These controls therefore use a conservative temporal
// reliability proxy from raw F0 support and pitch continuity.
const F0_RELIABILITY_MIN_FACTOR: f32 = 0.55;
const F0_PAIR_STABLE_SEMITONES: f32 = 1.5;
const F0_PAIR_UNSTABLE_SEMITONES: f32 = 7.0;
const F0_NEIGHBOR_AGREEMENT_SEMITONES: f32 = 2.5;
const F0_ISOLATED_JUMP_START_SEMITONES: f32 = 3.5;
const F0_ISOLATED_JUMP_FULL_SEMITONES: f32 = 10.0;
const F0_MIN_VOICED_RELIABILITY: f32 = 0.35;
// A Schmitt-style boundary envelope reacts immediately to consonant/onset
// evidence, holds through threshold chatter, then releases over a few 20 ms
// ContentVec frames. Final Index/Protect controls use the same fast-drop,
// slow-recovery policy so target timbre returns without a hard feature step.
const F0_BOUNDARY_WEIGHT: f32 = 0.85;
const BOUNDARY_ENTER_THRESHOLD: f32 = 0.58;
const BOUNDARY_EXIT_THRESHOLD: f32 = 0.22;
const BOUNDARY_ATTACK_ALPHA: f32 = 0.85;
const BOUNDARY_RELEASE_ALPHA: f32 = 0.45;
const INDEX_DROP_ALPHA: f32 = 0.80;
const INDEX_RECOVERY_ALPHA: f32 = 0.38;
const PROTECT_DROP_ALPHA: f32 = 0.82;
const PROTECT_RECOVERY_ALPHA: f32 = 0.34;

#[derive(Clone, Copy, Debug)]
pub(super) struct FeatureIndexSummary {
    pub(super) dimensions: usize,
    pub(super) vectors: usize,
    pub(super) lists: usize,
    pub(super) probes: usize,
}

#[derive(Debug)]
struct InvertedList {
    /// Consecutive float32 vectors, each `dimensions` wide. FAISS stores the
    /// corresponding IDs after these codes, but RVC only needs the retrieved
    /// vector values, not their original training-row number.
    vectors: Vec<f32>,
}

/// A native representation of an RVC `added_IVF*_Flat_*.index` file.
#[derive(Debug)]
pub(super) struct FeatureIndex {
    dimensions: usize,
    nprobe: usize,
    centroids: Vec<f32>,
    lists: Vec<InvertedList>,
    vector_count: usize,
    // All fields below are allocated during load and reused by `blend_*`.
    centroid_distances: Vec<f32>,
    centroid_ids: Vec<usize>,
    mixed_frame: Vec<f32>,
    // Per-window adaptive controls. They are retained so FeatureTensor can
    // apply the same distance/boundary evidence to unvoiced Protect frames
    // after the generator's 2x frame expansion and silence-front trim.
    adaptive_protect_scales: Vec<f32>,
    boundary_strengths: Vec<f32>,
}

impl FeatureIndex {
    pub(super) fn load(path: &Path, expected_dimensions: usize) -> Result<Self> {
        let metadata = fs::metadata(path)
            .with_context(|| format!("failed to inspect RVC feature index {}", path.display()))?;
        if metadata.len() == 0 {
            bail!("RVC feature index {} is empty", path.display());
        }
        if metadata.len() > MAX_INDEX_BYTES {
            bail!(
                "RVC feature index {} is {} bytes; vc-rs limits a single index to {} bytes",
                path.display(),
                metadata.len(),
                MAX_INDEX_BYTES
            );
        }
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read RVC feature index {}", path.display()))?;
        let mut reader = FaissReader::new(&bytes);

        reader.expect_fourcc(*b"IwFl", "FAISS IndexIVFFlat")?;
        let root = reader.read_index_header("FAISS IndexIVFFlat")?;
        if !root.is_trained {
            bail!(
                "RVC feature index {} is not trained; use the training output named added_IVF*_Flat_*.index",
                path.display()
            );
        }
        if root.metric_type != FAISS_L2_METRIC {
            bail!(
                "RVC feature index {} uses FAISS metric {} instead of L2",
                path.display(),
                root.metric_type
            );
        }
        if root.vector_count == 0 {
            bail!(
                "RVC feature index {} contains no added vectors; choose added_IVF*_Flat_*.index, not trained_IVF*.index",
                path.display()
            );
        }
        let dimensions =
            usize::try_from(root.dimensions).context("FAISS index dimensions do not fit usize")?;
        if dimensions == 0 {
            bail!("RVC feature index {} has zero dimensions", path.display());
        }
        if dimensions != expected_dimensions {
            bail!(
                "RVC feature index {} has {} dimensions, but the RVC model expects {}. Use the matching v1 (256) or v2 (768) index.",
                path.display(),
                dimensions,
                expected_dimensions
            );
        }

        let nlist = reader.read_usize("FAISS IVF list count")?;
        if nlist == 0 || nlist > MAX_NLIST {
            bail!(
                "RVC feature index {} has unsupported IVF list count {}",
                path.display(),
                nlist
            );
        }
        let nprobe = reader.read_usize("FAISS IVF probe count")?;
        if nprobe == 0 || nprobe > MAX_NPROBE {
            bail!(
                "RVC feature index {} requests nprobe={}; vc-rs supports 1..={} for realtime-safe retrieval",
                path.display(),
                nprobe,
                MAX_NPROBE
            );
        }
        if nprobe > nlist {
            bail!(
                "RVC feature index {} requests {} probes for only {} IVF lists",
                path.display(),
                nprobe,
                nlist
            );
        }

        let centroids = reader.read_flat_l2_quantizer(dimensions, nlist)?;
        reader.skip_direct_map()?;
        let (lists, vector_count) = reader.read_array_inverted_lists(nlist, dimensions)?;

        let expected_vectors = usize::try_from(root.vector_count)
            .context("FAISS index vector count does not fit usize")?;
        if expected_vectors == 0 || vector_count == 0 {
            bail!(
                "RVC feature index {} contains no added vectors; choose added_IVF*_Flat_*.index, not trained_IVF*.index",
                path.display()
            );
        }
        if vector_count != expected_vectors {
            bail!(
                "RVC feature index {} is malformed: header says {} vectors but its inverted lists contain {}",
                path.display(),
                expected_vectors,
                vector_count
            );
        }
        if !reader.is_empty() {
            bail!(
                "RVC feature index {} has {} unexpected trailing bytes; only standard FAISS IVF-Flat indexes are supported",
                path.display(),
                reader.remaining()
            );
        }

        Ok(Self {
            dimensions,
            nprobe,
            centroids,
            lists,
            vector_count,
            centroid_distances: vec![f32::INFINITY; nprobe],
            centroid_ids: vec![usize::MAX; nprobe],
            mixed_frame: vec![0.0; dimensions],
            adaptive_protect_scales: Vec::new(),
            boundary_strengths: Vec::new(),
        })
    }

    pub(super) fn summary(&self) -> FeatureIndexSummary {
        FeatureIndexSummary {
            dimensions: self.dimensions,
            vectors: self.vector_count,
            lists: self.lists.len(),
            probes: self.nprobe,
        }
    }

    /// Blend every ContentVec frame with the inverse-squared-distance weighted
    /// eight nearest RVC index entries. This is the same retrieval equation used
    /// by the upstream RVC Python pipeline, before it doubles feature frames for
    /// the generator's 10 ms grid.
    #[allow(dead_code)]
    pub(super) fn blend_frames_in_place(
        &mut self,
        data: &mut [f32],
        frames: usize,
        dimensions: usize,
        index_rate: f32,
    ) -> Result<()> {
        if index_rate <= 0.0 {
            return Ok(());
        }
        if !index_rate.is_finite() {
            bail!("RVC index rate must be finite");
        }
        let index_rate = self.validate_blend_inputs(data, frames, dimensions, index_rate)?;
        for frame in data.chunks_exact_mut(dimensions) {
            self.blend_frame(frame, index_rate);
        }
        Ok(())
    }

    /// Blend retrieval with frame-local quality controls. `natural_pitchf_10ms`
    /// is the raw, unshifted F0 aligned to the untrimmed 10 ms generator grid;
    /// each ContentVec frame owns two adjacent values. The method performs the
    /// same nearest-neighbor query as standard RVC exactly once per frame, then
    /// lowers its effective Index rate for weak matches, unvoiced material, and
    /// sharp ContentVec transitions.
    pub(super) fn blend_frames_adaptive_in_place(
        &mut self,
        data: &mut [f32],
        frames: usize,
        dimensions: usize,
        index_rate: f32,
        natural_pitchf_10ms: &[f32],
    ) -> Result<()> {
        if index_rate <= 0.0 {
            self.adaptive_protect_scales.clear();
            self.boundary_strengths.clear();
            return Ok(());
        }
        let index_rate = self.validate_blend_inputs(data, frames, dimensions, index_rate)?;
        let expected_pitch_len = frames
            .checked_mul(2)
            .context("adaptive retrieval F0 frame length overflow")?;
        if natural_pitchf_10ms.len() != expected_pitch_len {
            bail!(
                "adaptive retrieval F0 frame count {} does not match {} ContentVec frames x 2",
                natural_pitchf_10ms.len(),
                frames
            );
        }

        self.boundary_strengths.resize(frames, 0.0);
        self.adaptive_protect_scales.resize(frames, 1.0);
        // Calculate boundaries before mutating any feature frame. Comparing a
        // frame against an already-retrieved neighbor would make the result
        // depend on loop order and create a subtle one-frame shimmer.
        for frame_index in 0..frames {
            let start = frame_index * dimensions;
            let frame = &data[start..start + dimensions];
            let previous_delta = if frame_index > 0 {
                let previous_start = start - dimensions;
                normalized_frame_delta(frame, &data[previous_start..previous_start + dimensions])
            } else {
                0.0
            };
            let next_delta = if frame_index + 1 < frames {
                let next_start = start + dimensions;
                normalized_frame_delta(frame, &data[next_start..next_start + dimensions])
            } else {
                0.0
            };
            let content_boundary = smoothstep(previous_delta.max(next_delta), 0.20, 0.75);
            let f0_boundary =
                F0_BOUNDARY_WEIGHT * f0_transition_strength(natural_pitchf_10ms, frame_index);
            self.boundary_strengths[frame_index] = content_boundary.max(f0_boundary);
        }

        // This state is intentionally window-local. Realtime calls replay a
        // rolling historical prefix; carrying the prior call's terminal EMA
        // into the next window's oldest frame would run time backwards and make
        // identical windows drift. Reconstructing from retained context is both
        // deterministic and continuous at the emitted tail.
        stabilize_boundary_strengths_in_place(&mut self.boundary_strengths);
        let mut previous_index_scale = None;
        let mut previous_protect_scale = None;

        for (frame_index, frame) in data.chunks_exact_mut(dimensions).enumerate() {
            let Some(distance_confidence) = self.retrieve_frame(frame) else {
                // No usable vector leaves the source untouched and must not
                // reduce Protect for this frame either. Reset the local
                // smoother so an empty IVF bucket cannot leak stale evidence
                // into the next usable bucket.
                self.adaptive_protect_scales[frame_index] = 1.0;
                previous_index_scale = None;
                previous_protect_scale = None;
                continue;
            };
            let voicing_factor = voicing_factor(natural_pitchf_10ms, frame_index);
            let f0_reliability = f0_temporal_reliability(natural_pitchf_10ms, frame_index);
            let f0_reliability_factor =
                F0_RELIABILITY_MIN_FACTOR + (1.0 - F0_RELIABILITY_MIN_FACTOR) * f0_reliability;
            let boundary_strength = self.boundary_strengths[frame_index];
            let distance_factor = ADAPTIVE_INDEX_MIN_DISTANCE_FACTOR
                + (1.0 - ADAPTIVE_INDEX_MIN_DISTANCE_FACTOR) * distance_confidence;
            let boundary_factor = 1.0 - ADAPTIVE_INDEX_BOUNDARY_REDUCTION * boundary_strength;
            let target_index_scale =
                (distance_factor * voicing_factor * f0_reliability_factor * boundary_factor)
                    .clamp(0.0, 1.0);
            let index_scale = smooth_adaptive_control(
                previous_index_scale,
                target_index_scale,
                INDEX_DROP_ALPHA,
                INDEX_RECOVERY_ALPHA,
            );
            previous_index_scale = Some(index_scale);
            let effective_rate = (index_rate * index_scale).clamp(0.0, 1.0);
            self.apply_retrieved_frame(frame, effective_rate);

            // Protect is only applied again on unvoiced frames. Distance and
            // boundary evidence already shaped the voiced Index rate above; a
            // second confidence multiplication there would over-attenuate
            // sustained vowels and make the target voice disappear.
            let target_protect_scale = ((ADAPTIVE_PROTECT_MIN_FACTOR
                + (1.0 - ADAPTIVE_PROTECT_MIN_FACTOR) * distance_confidence)
                * (1.0 - 0.65 * boundary_strength)
                * f0_reliability_factor)
                .clamp(0.0, 1.0);
            let protect_scale = smooth_adaptive_control(
                previous_protect_scale,
                target_protect_scale,
                PROTECT_DROP_ALPHA,
                PROTECT_RECOVERY_ALPHA,
            );
            previous_protect_scale = Some(protect_scale);
            self.adaptive_protect_scales[frame_index] = protect_scale;
        }
        Ok(())
    }

    /// Scales for the raw ContentVec frames from the most recent adaptive
    /// retrieval pass. The slice is worker-owned and remains valid until the
    /// next query; it is read after the pipeline expands/trims feature frames.
    pub(super) fn adaptive_protect_scales(&self) -> &[f32] {
        &self.adaptive_protect_scales
    }

    fn validate_blend_inputs(
        &self,
        data: &[f32],
        frames: usize,
        dimensions: usize,
        index_rate: f32,
    ) -> Result<f32> {
        if !index_rate.is_finite() {
            bail!("RVC index rate must be finite");
        }
        if dimensions != self.dimensions {
            bail!(
                "RVC feature index expects {} channels but received {}",
                self.dimensions,
                dimensions
            );
        }
        let expected_len = frames
            .checked_mul(dimensions)
            .context("RVC feature frame length overflow")?;
        if data.len() != expected_len {
            bail!(
                "RVC feature data has {} values, expected {} frames x {} channels",
                data.len(),
                frames,
                dimensions
            );
        }
        Ok(index_rate.clamp(0.0, 1.0))
    }

    #[allow(dead_code)]
    fn blend_frame(&mut self, frame: &mut [f32], index_rate: f32) {
        if self.retrieve_frame(frame).is_some() {
            self.apply_retrieved_frame(frame, index_rate);
        }
    }

    /// Fill `mixed_frame` with the weighted nearest index vector and return a
    /// scale-free match confidence. `None` means the selected IVF buckets had
    /// no usable vectors; `Some(0.0)` is still a valid but weak match and must
    /// retain the standard blend behavior. Comparing nearest distance with its
    /// IVF centroid distance makes confidence portable across v1/v2 feature
    /// dimensions and across indexes with different feature magnitudes.
    fn retrieve_frame(&mut self, frame: &[f32]) -> Option<f32> {
        self.select_nearest_centroids(frame);

        let mut neighbor_distances = [f32::INFINITY; NEIGHBORS];
        let mut neighbor_lists = [usize::MAX; NEIGHBORS];
        let mut neighbor_offsets = [usize::MAX; NEIGHBORS];
        let mut neighbor_count = 0usize;

        for centroid_slot in 0..self.nprobe {
            let list_index = self.centroid_ids[centroid_slot];
            if list_index == usize::MAX {
                continue;
            }
            let list = &self.lists[list_index];
            for (vector_index, candidate) in list.vectors.chunks_exact(self.dimensions).enumerate()
            {
                let distance = squared_l2(frame, candidate);
                insert_neighbor(
                    distance,
                    list_index,
                    vector_index,
                    &mut neighbor_distances,
                    &mut neighbor_lists,
                    &mut neighbor_offsets,
                    &mut neighbor_count,
                );
            }
        }

        if neighbor_count == 0 {
            // An empty nearest IVF bucket is unusual for a normal added RVC
            // index. Keeping the source ContentVec frame is safer than inventing
            // a substitute from an unrelated list.
            return None;
        }

        let lists = &self.lists;
        let mixed_frame = &mut self.mixed_frame;
        if neighbor_distances[0] <= EXACT_DISTANCE_EPSILON {
            let exact = &lists[neighbor_lists[0]].vectors[neighbor_offsets[0] * self.dimensions
                ..(neighbor_offsets[0] + 1) * self.dimensions];
            mixed_frame.copy_from_slice(exact);
        } else {
            mixed_frame.fill(0.0);
            let mut total_weight = 0.0f32;
            for neighbor in 0..neighbor_count {
                // FAISS returns squared L2 distance. Upstream RVC computes
                // `square(1 / score)`, i.e. 1 / squared_distance^2.
                let weight = 1.0 / (neighbor_distances[neighbor] * neighbor_distances[neighbor]);
                if !weight.is_finite() || weight <= 0.0 {
                    continue;
                }
                total_weight += weight;
                let candidate = &lists[neighbor_lists[neighbor]].vectors[neighbor_offsets[neighbor]
                    * self.dimensions
                    ..(neighbor_offsets[neighbor] + 1) * self.dimensions];
                for (mixed, value) in mixed_frame.iter_mut().zip(candidate) {
                    *mixed += value * weight;
                }
            }
            if !total_weight.is_finite() || total_weight <= 0.0 {
                return None;
            }
            for mixed in mixed_frame.iter_mut() {
                *mixed /= total_weight;
            }
        }

        Some(distance_confidence(
            neighbor_distances[0],
            self.centroid_distances[0],
        ))
    }

    fn apply_retrieved_frame(&mut self, frame: &mut [f32], index_rate: f32) {
        let source_weight = 1.0 - index_rate;
        for (source, retrieved) in frame.iter_mut().zip(self.mixed_frame.iter()) {
            *source = *source * source_weight + *retrieved * index_rate;
        }
    }

    fn select_nearest_centroids(&mut self, frame: &[f32]) {
        self.centroid_distances.fill(f32::INFINITY);
        self.centroid_ids.fill(usize::MAX);
        for (centroid_index, centroid) in self.centroids.chunks_exact(self.dimensions).enumerate() {
            let distance = squared_l2(frame, centroid);
            for slot in 0..self.nprobe {
                if distance < self.centroid_distances[slot] {
                    for shift in (slot + 1..self.nprobe).rev() {
                        self.centroid_distances[shift] = self.centroid_distances[shift - 1];
                        self.centroid_ids[shift] = self.centroid_ids[shift - 1];
                    }
                    self.centroid_distances[slot] = distance;
                    self.centroid_ids[slot] = centroid_index;
                    break;
                }
            }
        }
    }
}

fn distance_confidence(nearest_distance: f32, centroid_distance: f32) -> f32 {
    if nearest_distance <= EXACT_DISTANCE_EPSILON {
        return 1.0;
    }
    if !nearest_distance.is_finite()
        || !centroid_distance.is_finite()
        || centroid_distance <= EXACT_DISTANCE_EPSILON
    {
        return 0.0;
    }
    let ratio = (nearest_distance / centroid_distance).max(0.0);
    1.0 - smoothstep(
        ratio,
        DISTANCE_CONFIDENCE_FULL_RATIO,
        DISTANCE_CONFIDENCE_NONE_RATIO,
    )
}

fn smoothstep(value: f32, low: f32, high: f32) -> f32 {
    if value <= low {
        return 0.0;
    }
    if value >= high {
        return 1.0;
    }
    let t = (value - low) / (high - low);
    t * t * (3.0 - 2.0 * t)
}

fn normalized_frame_delta(left: &[f32], right: &[f32]) -> f32 {
    let mut distance = 0.0f32;
    let mut energy = 0.0f32;
    for (&left, &right) in left.iter().zip(right) {
        let delta = left - right;
        distance += delta * delta;
        energy += left * left + right * right;
    }
    if !distance.is_finite() || !energy.is_finite() {
        // Treat malformed/overflowing feature energy as a hard boundary. This
        // keeps the adaptive rate finite and favors the source frame instead of
        // allowing a NaN to propagate into the generator tensor.
        return 1.0;
    }
    (distance / energy.max(1.0e-6)).sqrt().clamp(0.0, 1.5)
}

fn voicing_factor(natural_pitchf_10ms: &[f32], frame_index: usize) -> f32 {
    let start = frame_index * 2;
    let voiced_in_frame = natural_pitchf_10ms[start..start + 2]
        .iter()
        .filter(|&&pitchf| valid_raw_f0(pitchf).is_some())
        .count();
    let current_voiced = voiced_in_frame > 0;
    let base = match voiced_in_frame {
        2 => 1.0,
        1 => 0.68,
        _ => ADAPTIVE_INDEX_MIN_VOICING_FACTOR,
    };
    let previous_voiced = frame_index > 0
        && natural_pitchf_10ms[start - 2..start]
            .iter()
            .any(|&pitchf| valid_raw_f0(pitchf).is_some());
    let next_voiced = start + 2 < natural_pitchf_10ms.len()
        && natural_pitchf_10ms[start + 2..start + 4]
            .iter()
            .any(|&pitchf| valid_raw_f0(pitchf).is_some());
    if (frame_index > 0 && previous_voiced != current_voiced)
        || (frame_index + 1 < natural_pitchf_10ms.len() / 2 && next_voiced != current_voiced)
    {
        base * 0.85
    } else {
        base
    }
}

fn f0_transition_strength(natural_pitchf_10ms: &[f32], frame_index: usize) -> f32 {
    let frame_count = natural_pitchf_10ms.len() / 2;
    if frame_index >= frame_count {
        return 0.0;
    }

    let current = frame_voiced_ratio(natural_pitchf_10ms, frame_index);
    // Compare complete ContentVec-sized (20 ms) voicing ratios. A partial
    // 10 ms pair by itself is not enough to call a sustained alternating
    // pattern a boundary; its F0 reliability is handled separately below.
    let mut max_delta: f32 = 0.0;
    if frame_index > 0 {
        max_delta = max_delta
            .max((current - frame_voiced_ratio(natural_pitchf_10ms, frame_index - 1)).abs());
    }
    if frame_index + 1 < frame_count {
        max_delta = max_delta
            .max((current - frame_voiced_ratio(natural_pitchf_10ms, frame_index + 1)).abs());
    }
    smoothstep(max_delta, 0.25, 0.75)
}

fn stabilize_boundary_strengths_in_place(strengths: &mut [f32]) {
    let mut latched = false;
    let mut envelope = 0.0f32;
    for strength in strengths {
        let raw = if strength.is_finite() {
            strength.clamp(0.0, 1.0)
        } else {
            // Malformed feature/F0 evidence must bias toward retaining source
            // articulation, never inject NaN into the adaptive controls.
            1.0
        };

        if latched {
            if raw <= BOUNDARY_EXIT_THRESHOLD {
                latched = false;
            }
        } else if raw >= BOUNDARY_ENTER_THRESHOLD {
            latched = true;
        }

        let target = if latched {
            raw.max(BOUNDARY_ENTER_THRESHOLD)
        } else {
            raw
        };
        let alpha = if target >= envelope {
            BOUNDARY_ATTACK_ALPHA
        } else {
            BOUNDARY_RELEASE_ALPHA
        };
        envelope += alpha * (target - envelope);
        envelope = envelope.clamp(0.0, 1.0);
        *strength = envelope;
    }
}

fn f0_temporal_reliability(natural_pitchf_10ms: &[f32], frame_index: usize) -> f32 {
    let frame_count = natural_pitchf_10ms.len() / 2;
    if frame_index >= frame_count {
        return 0.0;
    }

    let start = frame_index * 2;
    let left = valid_raw_f0(natural_pitchf_10ms[start]);
    let right = valid_raw_f0(natural_pitchf_10ms[start + 1]);
    let voiced = usize::from(left.is_some()) + usize::from(right.is_some());
    let mut reliability = match (left, right) {
        (Some(left), Some(right)) => {
            let pair_jump = semitone_distance(left, right);
            let pair_stability = 1.0
                - smoothstep(
                    pair_jump,
                    F0_PAIR_STABLE_SEMITONES,
                    F0_PAIR_UNSTABLE_SEMITONES,
                );
            F0_MIN_VOICED_RELIABILITY + (1.0 - F0_MIN_VOICED_RELIABILITY) * pair_stability
        }
        (Some(_), None) | (None, Some(_)) => 0.55,
        (None, None) => 0.45,
    };

    let has_previous = frame_index > 0;
    let has_next = frame_index + 1 < frame_count;
    let previous_voiced =
        has_previous && frame_voiced_count(natural_pitchf_10ms, frame_index - 1) > 0;
    let next_voiced = has_next && frame_voiced_count(natural_pitchf_10ms, frame_index + 1) > 0;

    // Only call a frame an island/dropout when both sides are visible. Rolling
    // window edges have unknown context and must not be penalized by a guess.
    if has_previous && has_next {
        if voiced == 0 && previous_voiced && next_voiced {
            reliability = reliability.min(0.38);
        } else if voiced > 0 && !previous_voiced && !next_voiced {
            reliability = reliability.min(0.32);
        }
    }

    // Suppress an isolated octave/large jump only when the two surrounding
    // ContentVec frames agree. A sustained real pitch transition has one
    // neighbor on the new contour and therefore does not meet this condition.
    if let (Some(previous), Some(current), Some(next)) = (
        frame_log2_pitch(natural_pitchf_10ms, frame_index.checked_sub(1)),
        frame_log2_pitch(natural_pitchf_10ms, Some(frame_index)),
        frame_log2_pitch(
            natural_pitchf_10ms,
            (frame_index + 1 < frame_count).then_some(frame_index + 1),
        ),
    ) {
        let neighbor_delta = 12.0 * (previous - next).abs();
        if neighbor_delta <= F0_NEIGHBOR_AGREEMENT_SEMITONES {
            let isolated_delta = 12.0 * (current - previous).abs().min((current - next).abs());
            let jump_strength = smoothstep(
                isolated_delta,
                F0_ISOLATED_JUMP_START_SEMITONES,
                F0_ISOLATED_JUMP_FULL_SEMITONES,
            );
            let jump_cap = 1.0 - (1.0 - F0_MIN_VOICED_RELIABILITY) * jump_strength;
            reliability = reliability.min(jump_cap);
        }
    }

    reliability.clamp(0.0, 1.0)
}

fn smooth_adaptive_control(
    previous: Option<f32>,
    target: f32,
    drop_alpha: f32,
    recovery_alpha: f32,
) -> f32 {
    let target = if target.is_finite() {
        target.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let Some(previous) = previous.filter(|value| value.is_finite()) else {
        return target;
    };
    let previous = previous.clamp(0.0, 1.0);
    let alpha = if target < previous {
        drop_alpha
    } else {
        recovery_alpha
    };
    let alpha = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    };
    (previous + alpha * (target - previous)).clamp(0.0, 1.0)
}

fn valid_raw_f0(pitchf: f32) -> Option<f32> {
    (pitchf.is_finite() && pitchf > 0.0).then_some(pitchf)
}

fn frame_voiced_count(natural_pitchf_10ms: &[f32], frame_index: usize) -> usize {
    let start = frame_index.saturating_mul(2);
    natural_pitchf_10ms
        .get(start..start.saturating_add(2))
        .unwrap_or_default()
        .iter()
        .filter(|&&pitchf| valid_raw_f0(pitchf).is_some())
        .count()
}

fn frame_voiced_ratio(natural_pitchf_10ms: &[f32], frame_index: usize) -> f32 {
    frame_voiced_count(natural_pitchf_10ms, frame_index) as f32 * 0.5
}

fn frame_log2_pitch(natural_pitchf_10ms: &[f32], frame_index: Option<usize>) -> Option<f32> {
    let start = frame_index?.checked_mul(2)?;
    let frame = natural_pitchf_10ms.get(start..start.checked_add(2)?)?;
    let mut log_sum = 0.0f32;
    let mut voiced = 0usize;
    for &pitchf in frame {
        if let Some(pitchf) = valid_raw_f0(pitchf) {
            log_sum += pitchf.log2();
            voiced += 1;
        }
    }
    (voiced > 0).then_some(log_sum / voiced as f32)
}

fn semitone_distance(left: f32, right: f32) -> f32 {
    (12.0 * (left.log2() - right.log2()).abs()).clamp(0.0, 120.0)
}

fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    // Keep this loop simple: release LLVM auto-vectorizes the contiguous f32
    // arithmetic on supported CPUs. More importantly, no temporary vector or
    // per-frame allocation is introduced into the worker path.
    left.iter().zip(right).fold(0.0f32, |sum, (left, right)| {
        let delta = left - right;
        sum + delta * delta
    })
}

fn insert_neighbor(
    distance: f32,
    list_index: usize,
    vector_index: usize,
    distances: &mut [f32; NEIGHBORS],
    lists: &mut [usize; NEIGHBORS],
    offsets: &mut [usize; NEIGHBORS],
    count: &mut usize,
) {
    if !distance.is_finite() {
        return;
    }
    let Some(slot) = distances.iter().position(|current| distance < *current) else {
        return;
    };
    for shift in (slot + 1..NEIGHBORS).rev() {
        distances[shift] = distances[shift - 1];
        lists[shift] = lists[shift - 1];
        offsets[shift] = offsets[shift - 1];
    }
    distances[slot] = distance;
    lists[slot] = list_index;
    offsets[slot] = vector_index;
    *count = (*count + 1).min(NEIGHBORS);
}

#[derive(Clone, Copy)]
struct IndexHeader {
    dimensions: i32,
    vector_count: i64,
    is_trained: bool,
    metric_type: i32,
}

struct FaissReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> FaissReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn expect_fourcc(&mut self, expected: [u8; 4], context: &str) -> Result<()> {
        let actual = self.read_exact(4, context)?;
        if actual != expected {
            bail!(
                "{context} is not supported (found fourcc '{}', expected '{}')",
                fourcc_text(actual),
                fourcc_text(&expected)
            );
        }
        Ok(())
    }

    fn read_index_header(&mut self, context: &str) -> Result<IndexHeader> {
        // `Index` serializes d, ntotal, two legacy dummy idx_t values, a bool,
        // and MetricType. This is FAISS's stable v1.7 x86_64 layout used by RVC
        // packages; the parser deliberately does not claim compatibility with
        // arbitrary ABI-dependent FAISS files.
        let dimensions = self.read_i32(context)?;
        let vector_count = self.read_i64(context)?;
        self.read_i64(context)?;
        self.read_i64(context)?;
        let is_trained = self.read_u8(context)? != 0;
        let metric_type = self.read_i32(context)?;
        if metric_type > 1 {
            self.read_f32(context)?;
        }
        Ok(IndexHeader {
            dimensions,
            vector_count,
            is_trained,
            metric_type,
        })
    }

    fn read_flat_l2_quantizer(
        &mut self,
        expected_dimensions: usize,
        expected_vectors: usize,
    ) -> Result<Vec<f32>> {
        let kind = self.read_exact(4, "FAISS coarse quantizer")?;
        if kind != b"IxF2" && kind != b"IxFl" {
            bail!(
                "RVC feature index uses unsupported coarse quantizer '{}'; expected IndexFlatL2",
                fourcc_text(kind)
            );
        }
        let header = self.read_index_header("FAISS coarse quantizer")?;
        let dimensions = usize::try_from(header.dimensions)
            .context("FAISS coarse quantizer dimensions do not fit usize")?;
        if dimensions != expected_dimensions || header.metric_type != FAISS_L2_METRIC {
            bail!("RVC feature index coarse quantizer does not match its L2 IVF dimensions");
        }
        let vectors = usize::try_from(header.vector_count)
            .context("FAISS coarse quantizer vector count does not fit usize")?;
        if !header.is_trained || vectors != expected_vectors {
            bail!("RVC feature index has an invalid IVF coarse quantizer");
        }
        let count = self.read_usize("FAISS coarse quantizer codes")?;
        let expected_count = dimensions
            .checked_mul(vectors)
            .context("FAISS coarse quantizer size overflow")?;
        if count != expected_count {
            bail!(
                "RVC feature index coarse quantizer has {} values, expected {}",
                count,
                expected_count
            );
        }
        self.read_f32_vec(count, "FAISS coarse quantizer codes")
    }

    fn skip_direct_map(&mut self) -> Result<()> {
        let map_kind = self.read_u8("FAISS IVF direct map")?;
        let array_len = self.read_usize("FAISS IVF direct map array")?;
        self.skip_items(array_len, 8, "FAISS IVF direct map array")?;
        match map_kind {
            0 | 1 => Ok(()),
            2 => {
                let pairs = self.read_usize("FAISS IVF direct map hash table")?;
                self.skip_items(pairs, 16, "FAISS IVF direct map hash table")
            }
            other => bail!("RVC feature index has unsupported FAISS direct-map type {other}"),
        }
    }

    fn read_array_inverted_lists(
        &mut self,
        expected_lists: usize,
        dimensions: usize,
    ) -> Result<(Vec<InvertedList>, usize)> {
        self.expect_fourcc(*b"ilar", "FAISS ArrayInvertedLists")?;
        let list_count = self.read_usize("FAISS ArrayInvertedLists list count")?;
        let code_size = self.read_usize("FAISS ArrayInvertedLists code size")?;
        let expected_code_size = dimensions
            .checked_mul(std::mem::size_of::<f32>())
            .context("RVC feature index code size overflow")?;
        if list_count != expected_lists || code_size != expected_code_size {
            bail!(
                "RVC feature index inverted-list layout does not match {}-D IVF-Flat vectors",
                dimensions
            );
        }
        self.expect_fourcc(*b"full", "FAISS ArrayInvertedLists sizes")?;
        let size_count = self.read_usize("FAISS ArrayInvertedLists size count")?;
        if size_count != expected_lists {
            bail!(
                "RVC feature index contains {} inverted-list sizes, expected {}",
                size_count,
                expected_lists
            );
        }
        let mut sizes = Vec::with_capacity(size_count);
        for _ in 0..size_count {
            sizes.push(self.read_usize("FAISS inverted-list size")?);
        }

        let mut lists = Vec::with_capacity(expected_lists);
        let mut vector_count = 0usize;
        for size in sizes {
            let values = size
                .checked_mul(dimensions)
                .context("RVC feature index list size overflow")?;
            let vectors = self.read_f32_vec(values, "FAISS IVF-Flat vector codes")?;
            // IndexIVFFlat serializes one int64 id after every vector code. The
            // retrieval mix needs the code values directly, so skip IDs rather
            // than relying on their usual sequential RVC-training numbering.
            self.skip_items(size, 8, "FAISS IVF-Flat vector ids")?;
            vector_count = vector_count
                .checked_add(size)
                .context("RVC feature index vector count overflow")?;
            lists.push(InvertedList { vectors });
        }
        Ok((lists, vector_count))
    }

    fn read_f32_vec(&mut self, count: usize, context: &str) -> Result<Vec<f32>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow!("{context} length overflow"))?;
        self.ensure(bytes, context)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_f32(context)?);
        }
        Ok(values)
    }

    fn read_usize(&mut self, context: &str) -> Result<usize> {
        let value = self.read_u64(context)?;
        usize::try_from(value).with_context(|| format!("{context} does not fit usize"))
    }

    fn read_u8(&mut self, context: &str) -> Result<u8> {
        Ok(self.read_exact(1, context)?[0])
    }

    fn read_i32(&mut self, context: &str) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_array(context)?))
    }

    fn read_i64(&mut self, context: &str) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_array(context)?))
    }

    fn read_u64(&mut self, context: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array(context)?))
    }

    fn read_f32(&mut self, context: &str) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_array(context)?))
    }

    fn read_array<const N: usize>(&mut self, context: &str) -> Result<[u8; N]> {
        let bytes = self.read_exact(N, context)?;
        bytes
            .try_into()
            .map_err(|_| anyhow!("internal FAISS reader length error for {context}"))
    }

    fn skip_items(&mut self, count: usize, item_size: usize, context: &str) -> Result<()> {
        let bytes = count
            .checked_mul(item_size)
            .ok_or_else(|| anyhow!("{context} length overflow"))?;
        self.skip(bytes, context)
    }

    fn skip(&mut self, count: usize, context: &str) -> Result<()> {
        self.ensure(count, context)?;
        self.offset += count;
        Ok(())
    }

    fn read_exact(&mut self, count: usize, context: &str) -> Result<&'a [u8]> {
        self.ensure(count, context)?;
        let start = self.offset;
        self.offset += count;
        Ok(&self.bytes[start..self.offset])
    }

    fn ensure(&self, count: usize, context: &str) -> Result<()> {
        if count > self.remaining() {
            bail!(
                "RVC feature index ended while reading {context} (need {count} bytes, {} remain)",
                self.remaining()
            );
        }
        Ok(())
    }
}

fn fourcc_text(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| {
            if byte.is_ascii_graphic() {
                char::from(*byte)
            } else {
                '?'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_standard_ivf_flat_and_matches_exact_vector() {
        let bytes = standard_index_bytes();
        let path = temp_index_path("valid");
        fs::write(&path, bytes).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        let summary = index.summary();
        assert_eq!(summary.dimensions, 2);
        assert_eq!(summary.vectors, 4);
        assert_eq!(summary.lists, 2);
        assert_eq!(summary.probes, 1);

        let mut frames = vec![0.0, 0.0, 2.0, 2.0];
        index.blend_frames_in_place(&mut frames, 2, 2, 1.0).unwrap();
        assert_eq!(frames, vec![0.0, 0.0, 2.0, 2.0]);
    }

    #[test]
    fn retrieval_rate_blends_source_and_retrieved_features() {
        let path = temp_index_path("blend");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        let mut frames = vec![0.5, 0.0];
        index.blend_frames_in_place(&mut frames, 1, 2, 0.5).unwrap();
        // Both entries in the selected IVF list participate. RVC weights each
        // by 1 / squared-L2-distance^2 before applying the 0.5 blend rate.
        let retrieved =
            0.2 * (1.0 / 0.09_f32.powi(2)) / ((1.0 / 0.09_f32.powi(2)) + (1.0 / 0.25_f32.powi(2)));
        assert!((frames[0] - (0.5 * 0.5 + retrieved * 0.5)).abs() < 1.0e-6);
        assert_eq!(frames[1], 0.0);
    }

    #[test]
    fn standard_blend_does_not_skip_a_valid_zero_confidence_match() {
        let path = temp_index_path("weak-standard");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        // The query and its nearest centroid are equally close, so the
        // adaptive confidence is zero even though the IVF bucket has a valid
        // vector. Standard RVC blending must still use that retrieved vector.
        let mut frames = vec![1.9, 1.9];
        index.blend_frames_in_place(&mut frames, 1, 2, 1.0).unwrap();
        assert!(frames[0] > 1.9);
        assert!(frames[1] > 1.9);
    }

    #[test]
    fn zero_index_rate_leaves_features_exactly_unchanged() {
        let path = temp_index_path("zero-rate");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        let mut frames = vec![0.5, -0.25];
        index.blend_frames_in_place(&mut frames, 1, 2, 0.0).unwrap();
        assert_eq!(frames, vec![0.5, -0.25]);
    }

    #[test]
    fn distance_confidence_is_bounded_and_decreases_for_weaker_matches() {
        let exact = distance_confidence(0.0, 1.0);
        let close = distance_confidence(0.4, 1.0);
        let weak = distance_confidence(0.8, 1.0);
        let outside = distance_confidence(1.2, 1.0);
        assert_eq!(exact, 1.0);
        assert!(close > weak);
        assert!(weak > outside);
        assert!((0.0..=1.0).contains(&close));
        assert!((0.0..=1.0).contains(&weak));
        assert_eq!(outside, 0.0);
    }

    #[test]
    fn voicing_factor_favors_fully_voiced_frames() {
        let fully_voiced = voicing_factor(&[120.0, 121.0, 120.0, 121.0], 0);
        let half_voiced = voicing_factor(&[120.0, 0.0, 120.0, 0.0], 0);
        let unvoiced = voicing_factor(&[0.0, 0.0, 0.0, 0.0], 0);
        assert_eq!(fully_voiced, 1.0);
        assert!(half_voiced < fully_voiced);
        assert_eq!(unvoiced, ADAPTIVE_INDEX_MIN_VOICING_FACTOR);
    }

    #[test]
    fn raw_f0_reliability_distinguishes_stable_and_unreliable_patterns() {
        let stable = f0_temporal_reliability(&[120.0, 121.0, 120.0, 121.0, 120.0, 121.0], 1);
        let partial = f0_temporal_reliability(&[120.0, 121.0, 120.0, 0.0, 120.0, 121.0], 1);
        let stable_unvoiced = f0_temporal_reliability(&[0.0; 6], 1);
        let dropout = f0_temporal_reliability(&[120.0, 121.0, 0.0, 0.0, 120.0, 121.0], 1);
        let isolated = f0_temporal_reliability(&[0.0, 0.0, 120.0, 121.0, 0.0, 0.0], 1);
        let octave_outlier =
            f0_temporal_reliability(&[120.0, 121.0, 240.0, 242.0, 119.0, 121.0], 1);
        let within_pair_jump = f0_temporal_reliability(&[120.0, 240.0], 0);

        assert!(stable > partial);
        assert!(partial > stable_unvoiced);
        assert!(stable_unvoiced > dropout);
        assert!(isolated < stable_unvoiced);
        assert!(octave_outlier <= F0_MIN_VOICED_RELIABILITY + 1.0e-6);
        assert!(within_pair_jump <= F0_MIN_VOICED_RELIABILITY + 1.0e-6);

        // A real transition persists into the right neighbor, so it must not
        // be classified as an isolated octave error.
        let sustained_octave =
            f0_temporal_reliability(&[120.0, 121.0, 240.0, 242.0, 241.0, 243.0], 1);
        assert!(sustained_octave > 0.95);
    }

    #[test]
    fn raw_f0_reliability_treats_invalid_values_as_unvoiced_and_stays_finite() {
        for pitchf in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -120.0, 0.0] {
            let reliability = f0_temporal_reliability(&[pitchf, pitchf], 0);
            assert!(reliability.is_finite());
            assert_eq!(reliability, 0.45);
        }
    }

    #[test]
    fn f0_transition_strength_detects_voicing_edges() {
        let pitchf = [120.0, 121.0, 120.0, 121.0, 0.0, 0.0];
        assert_eq!(
            f0_transition_strength(&[120.0, 121.0, 120.0, 121.0], 0),
            0.0
        );
        assert_eq!(f0_transition_strength(&pitchf, 1), 1.0);
        assert_eq!(f0_transition_strength(&[120.0, 0.0], 0), 0.0);
    }

    #[test]
    fn boundary_hysteresis_holds_through_threshold_chatter() {
        let mut strengths = [0.70, 0.50, 0.40, 0.25, 0.21, 0.50, f32::NAN];
        stabilize_boundary_strengths_in_place(&mut strengths);

        assert!(strengths[..4]
            .iter()
            .all(|strength| *strength >= BOUNDARY_ENTER_THRESHOLD));
        assert!(strengths[4] < BOUNDARY_ENTER_THRESHOLD);
        // After exiting, a value between the thresholds must not relatch.
        assert!(strengths[5] < BOUNDARY_ENTER_THRESHOLD);
        assert!(strengths[6].is_finite());
        assert!(strengths[6] > strengths[5]);
    }

    #[test]
    fn adaptive_control_drops_faster_than_it_recovers() {
        let dropped = smooth_adaptive_control(Some(1.0), 0.0, 0.80, 0.38);
        let recovered = smooth_adaptive_control(Some(0.0), 1.0, 0.80, 0.38);
        assert!(1.0 - dropped > recovered);
        assert_eq!(smooth_adaptive_control(None, 0.7, 0.80, 0.38), 0.7);
        assert!(smooth_adaptive_control(Some(f32::NAN), f32::NAN, 0.80, 0.38).is_finite());
    }

    #[test]
    fn adaptive_window_replay_is_deterministic_and_reuses_scratch() {
        let path = temp_index_path("adaptive-replay");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        let original = vec![0.5, 0.0, 1.0, 1.0, 2.0, 2.0];
        let pitchf = [120.0, 121.0, 120.0, 0.0, 0.0, 0.0];
        let mut first = original.clone();
        index
            .blend_frames_adaptive_in_place(&mut first, 3, 2, 1.0, &pitchf)
            .unwrap();
        let first_scales = index.adaptive_protect_scales().to_vec();
        let boundary_capacity = index.boundary_strengths.capacity();
        let protect_capacity = index.adaptive_protect_scales.capacity();

        let mut replay = original.clone();
        index
            .blend_frames_adaptive_in_place(&mut replay, 3, 2, 1.0, &pitchf)
            .unwrap();
        assert_eq!(replay, first);
        assert_eq!(index.adaptive_protect_scales(), first_scales);
        assert_eq!(index.boundary_strengths.capacity(), boundary_capacity);
        assert_eq!(index.adaptive_protect_scales.capacity(), protect_capacity);

        let mut smaller = original[..4].to_vec();
        index
            .blend_frames_adaptive_in_place(&mut smaller, 2, 2, 1.0, &pitchf[..4])
            .unwrap();
        assert_eq!(index.boundary_strengths.capacity(), boundary_capacity);
        assert_eq!(index.adaptive_protect_scales.capacity(), protect_capacity);

        let mut disabled = original.clone();
        index
            .blend_frames_adaptive_in_place(&mut disabled, 3, 2, 0.0, &[])
            .unwrap();
        assert_eq!(disabled, original);
        assert!(index.adaptive_protect_scales().is_empty());

        let mut reenabled = original.clone();
        index
            .blend_frames_adaptive_in_place(&mut reenabled, 3, 2, 1.0, &pitchf)
            .unwrap();
        assert_eq!(reenabled, first);
        assert_eq!(index.adaptive_protect_scales(), first_scales);
    }

    #[test]
    fn invalid_adaptive_f0_length_does_not_mutate_input_or_scratch() {
        let path = temp_index_path("adaptive-invalid-f0");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        let mut warmup = vec![0.5, 0.0];
        index
            .blend_frames_adaptive_in_place(&mut warmup, 1, 2, 1.0, &[120.0, 121.0])
            .unwrap();
        let old_scales = index.adaptive_protect_scales.clone();
        let old_boundaries = index.boundary_strengths.clone();

        let mut frames = vec![0.5, 0.0, 1.0, 1.0];
        let unchanged = frames.clone();
        let error = index
            .blend_frames_adaptive_in_place(&mut frames, 2, 2, 1.0, &[120.0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("F0 frame count"));
        assert_eq!(frames, unchanged);
        assert_eq!(index.adaptive_protect_scales, old_scales);
        assert_eq!(index.boundary_strengths, old_boundaries);
    }

    #[test]
    fn adaptive_retrieval_scales_weak_and_unvoiced_frames() {
        let path = temp_index_path("adaptive");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);

        // The first frame is an exact, voiced match. The second is farther from
        // both the IVF centroid and its stored vectors, so it receives less
        // target-voice influence and a smaller Protect scale.
        let mut frames = vec![0.5, 0.0, 1.0, 1.0];
        index
            .blend_frames_adaptive_in_place(&mut frames, 2, 2, 1.0, &[120.0, 120.0, 120.0, 120.0])
            .unwrap();
        let scales = index.adaptive_protect_scales();
        assert_eq!(scales.len(), 2);
        assert!(scales[0] > scales[1]);

        let path = temp_index_path("adaptive-voicing");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut voiced_index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);
        let mut voiced = vec![0.5, 0.0];
        voiced_index
            .blend_frames_adaptive_in_place(&mut voiced, 1, 2, 1.0, &[120.0, 120.0])
            .unwrap();

        let path = temp_index_path("adaptive-unvoiced");
        fs::write(&path, standard_index_bytes()).unwrap();
        let mut unvoiced_index = FeatureIndex::load(&path, 2).unwrap();
        let _ = fs::remove_file(&path);
        let mut unvoiced = vec![0.5, 0.0];
        unvoiced_index
            .blend_frames_adaptive_in_place(&mut unvoiced, 1, 2, 1.0, &[0.0, 0.0])
            .unwrap();

        let voiced_delta = (voiced[0] - 0.5).abs();
        let unvoiced_delta = (unvoiced[0] - 0.5).abs();
        assert!(voiced_delta > unvoiced_delta);
    }

    #[test]
    fn loads_opt_in_real_faiss_index() {
        // Developers can point this at a local RVC `added_IVF*_Flat_*.index`
        // without making a machine-specific model path part of the repository.
        // The synthetic fixture above remains the portable regression test.
        let Some(path) = std::env::var_os("VC_RS_TEST_RVC_INDEX") else {
            return;
        };
        let path = std::path::PathBuf::from(path);
        let dimensions = std::env::var("VC_RS_TEST_RVC_INDEX_DIMENSIONS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(768);
        let index = FeatureIndex::load(&path, dimensions).unwrap();
        assert_eq!(index.summary().dimensions, dimensions);
        assert!(index.summary().vectors > 0);
    }

    #[test]
    fn rejects_unadded_or_wrong_dimension_indexes() {
        let mut bytes = standard_index_bytes();
        // Root `ntotal` begins at byte 8. A trained-but-not-added index has no
        // vectors, and must never silently behave as an enabled retrieval path.
        bytes[8..16].fill(0);
        let path = temp_index_path("empty");
        fs::write(&path, bytes).unwrap();
        let error = FeatureIndex::load(&path, 2).unwrap_err().to_string();
        let _ = fs::remove_file(&path);
        assert!(error.contains("no added vectors"));

        let path = temp_index_path("dimension");
        fs::write(&path, standard_index_bytes()).unwrap();
        let error = FeatureIndex::load(&path, 3).unwrap_err().to_string();
        let _ = fs::remove_file(&path);
        assert!(error.contains("expects 3"));
    }

    fn standard_index_bytes() -> Vec<u8> {
        // A minimal FAISS 1.7 x86_64 `IndexIVFFlat` serialization: two L2
        // centroids, four 2D vectors, nprobe=1, and ArrayInvertedLists/full.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"IwFl");
        push_index_header(&mut bytes, 2, 4, true, FAISS_L2_METRIC);
        push_usize(&mut bytes, 2);
        push_usize(&mut bytes, 1);

        bytes.extend_from_slice(b"IxF2");
        push_index_header(&mut bytes, 2, 2, true, FAISS_L2_METRIC);
        push_usize(&mut bytes, 4);
        for value in [0.0, 0.0, 2.0, 2.0] {
            push_f32(&mut bytes, value);
        }

        bytes.push(0); // DirectMap::NoMap.
        push_usize(&mut bytes, 0);

        bytes.extend_from_slice(b"ilar");
        push_usize(&mut bytes, 2);
        push_usize(&mut bytes, 2 * std::mem::size_of::<f32>());
        bytes.extend_from_slice(b"full");
        push_usize(&mut bytes, 2);
        push_usize(&mut bytes, 2);
        push_usize(&mut bytes, 2);

        for value in [0.0, 0.0, 0.2, 0.0] {
            push_f32(&mut bytes, value);
        }
        for id in [0_i64, 1] {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        for value in [2.0, 2.0, 2.2, 2.0] {
            push_f32(&mut bytes, value);
        }
        for id in [2_i64, 3] {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        bytes
    }

    fn push_index_header(
        bytes: &mut Vec<u8>,
        dimensions: i32,
        vectors: i64,
        trained: bool,
        metric: i32,
    ) {
        bytes.extend_from_slice(&dimensions.to_le_bytes());
        bytes.extend_from_slice(&vectors.to_le_bytes());
        bytes.extend_from_slice(&0_i64.to_le_bytes());
        bytes.extend_from_slice(&0_i64.to_le_bytes());
        bytes.push(u8::from(trained));
        bytes.extend_from_slice(&metric.to_le_bytes());
    }

    fn push_usize(bytes: &mut Vec<u8>, value: usize) {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }

    fn push_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn temp_index_path(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_TEST_INDEX: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "vc-rs-feature-index-{}-{}-{}.index",
            label,
            std::process::id(),
            NEXT_TEST_INDEX.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
