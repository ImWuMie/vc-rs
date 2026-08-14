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
//! worker-side query path reuses all scratch buffers and never opens files,
//! locks, or allocates, which is important because it runs once per conversion
//! chunk in realtime sessions.

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
        let index_rate = index_rate.clamp(0.0, 1.0);
        for frame in data.chunks_exact_mut(dimensions) {
            self.blend_frame(frame, index_rate);
        }
        Ok(())
    }

    fn blend_frame(&mut self, frame: &mut [f32], index_rate: f32) {
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
            return;
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
                return;
            }
            for mixed in mixed_frame.iter_mut() {
                *mixed /= total_weight;
            }
        }

        let source_weight = 1.0 - index_rate;
        for (source, retrieved) in frame.iter_mut().zip(mixed_frame.iter()) {
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
