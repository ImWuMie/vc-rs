use std::path::Path;

use anyhow::{bail, Result};
use tracing::info;

use super::onnx_meta::{read_model_io, RvcIoNames};

#[cfg(feature = "ort")]
use super::sessions::{describe_value_type, load_session};
#[cfg(feature = "ort")]
use super::tensorrt::{ModelRole, TensorRtRunMode, TensorRtSessionPurpose};
#[cfg(feature = "ort")]
use crate::Provider;

/// CLI `inspect` command: prints a model's full I/O and metadata via ONNX
/// Runtime. The TensorRT-only build (no `ort`) falls back to the provider-neutral
/// `onnx_meta` reader below, which reports the same I/O and metadata without a
/// session.
#[cfg(feature = "ort")]
pub fn inspect_model(path: &Path) -> Result<()> {
    reject_pth_checkpoint(path)?;
    let structural_io = read_model_io(path)?;
    // Inspect is a structural ONNX query, so keep it CPU-only and provider-neutral.
    // CUDA/TensorRT load validation belongs to `run`/`wav`, where chunk-derived
    // fixed-shape profiles are available.
    let session = load_session(
        path,
        Provider::Cpu,
        ModelRole::Inspect,
        None,
        TensorRtRunMode::PinnedCpu,
        TensorRtSessionPurpose::Main,
    )?;
    println!("Model: {}", path.display());
    println!("Inputs:");
    for input in session.inputs() {
        println!("  {}: {}", input.name(), describe_value_type(input.dtype()));
    }
    println!("Outputs:");
    for output in session.outputs() {
        println!(
            "  {}: {}",
            output.name(),
            describe_value_type(output.dtype())
        );
    }
    println!("Opset version: {}", session.opset_for_domain("")?);
    if let Ok(metadata) = session.metadata() {
        println!("Metadata:");
        if let Some(name) = metadata.name() {
            println!("  name: {name}");
        }
        if let Some(producer) = metadata.producer() {
            println!("  producer: {producer}");
        }
        if let Some(description) = metadata.description() {
            println!("  description: {description}");
        }
        if let Some(domain) = metadata.domain() {
            println!("  domain: {domain}");
        }
        if let Some(graph_description) = metadata.graph_description() {
            println!("  graph_description: {graph_description}");
        }
        if let Some(version) = metadata.version() {
            println!("  version: {version}");
        }
        for key in metadata.custom_keys()? {
            if let Some(value) = metadata.custom(&key) {
                println!("  {key}: {value}");
            }
        }
    }
    if structural_io.resolve_rvc_io_names().is_ok() {
        print_rvc_speaker_count(&structural_io);
    }
    Ok(())
}

/// `ort`-free fallback for the TensorRT-only build: read the structural I/O and
/// `metadata_props` directly from the ONNX protobuf (`onnx_meta`) instead of
/// opening an ORT session. Shapes and metadata are reported; ORT-only extras
/// (opset version, the producer/domain header fields) are not available here.
#[cfg(not(feature = "ort"))]
pub fn inspect_model(path: &Path) -> Result<()> {
    reject_pth_checkpoint(path)?;
    let io = read_model_io(path)?;
    println!("Model: {}", path.display());
    println!("Inputs:");
    for input in &io.inputs {
        println!("  {}: {}", input.name, describe_tensor(input));
    }
    println!("Outputs:");
    for output in &io.outputs {
        println!("  {}: {}", output.name, describe_tensor(output));
    }
    if !io.metadata.is_empty() {
        println!("Metadata:");
        for (key, value) in &io.metadata {
            println!("  {key}: {value}");
        }
    }
    if io.resolve_rvc_io_names().is_ok() {
        print_rvc_speaker_count(&io);
    }
    Ok(())
}

/// Validate that an ONNX file exposes the RVC generator contract used by every
/// front end. This deliberately performs no inference: `export-pth` uses it at
/// the offline import boundary, and `RvcPipeline` repeats the same inspection
/// when the model is actually loaded.
pub fn validate_rvc_model(path: &Path) -> Result<()> {
    inspect_rvc_model(path).map(|_| ())
}

/// Validate an ONNX emitted by `export-pth`. A fixed-frame request must produce
/// a truly static public tensor contract at exactly that frame count; generic
/// exports retain the normal dynamic-or-static loader validation.
pub fn validate_exported_rvc_model(path: &Path, expected_frames: Option<usize>) -> Result<()> {
    let info = inspect_rvc_model(path)?;
    if let Some(expected_frames) = expected_frames {
        match info.static_feature_frames {
            Some(actual_frames) if actual_frames == expected_frames => {}
            Some(actual_frames) => bail!(
                "fixed-frame exporter produced {actual_frames} ONNX frames, expected {expected_frames}"
            ),
            None => bail!(
                "fixed-frame exporter produced dynamic ONNX time axes, expected {expected_frames} static frames"
            ),
        }
    }
    Ok(())
}

fn reject_pth_checkpoint(path: &Path) -> Result<()> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pth"))
    {
        bail!(
            "RVC .pth checkpoints cannot run directly. Export a trained generator checkpoint \
             to ONNX first (for vc-rs CLI: export-pth), then load the resulting .onnx file."
        );
    }
    Ok(())
}

fn print_rvc_speaker_count(io: &super::onnx_meta::ModelIo) {
    match io.rvc_speaker_count() {
        Some(count) => println!("RVC speaker embeddings: {count} (IDs 0..{})", count - 1),
        None => println!("RVC speaker embeddings: unknown (emb_g.weight not found)"),
    }
}

/// Human-readable element type + shape for the `ort`-free inspect fallback.
/// `dim_value` 0 marks a symbolic axis (`onnx_meta` collapses `dim_param` to 0).
#[cfg(not(feature = "ort"))]
fn describe_tensor(tensor: &super::onnx_meta::TensorInfo) -> String {
    // ONNX TensorProto.DataType: only the types RVC models use are named.
    let elem = match tensor.elem_type {
        1 => "float32".to_string(),
        7 => "int64".to_string(),
        9 => "bool".to_string(),
        10 => "float16".to_string(),
        11 => "float64".to_string(),
        other => format!("elem_type={other}"),
    };
    let dims: Vec<String> = tensor
        .dims
        .iter()
        .map(|dim| {
            if *dim > 0 {
                dim.to_string()
            } else {
                "?".to_string()
            }
        })
        .collect();
    format!("{elem}[{}]", dims.join(", "))
}

pub(super) struct RvcModelInfo {
    pub(super) expected_feat_channels: i64,
    /// Common static time dimension across feats/pitch/pitchf/rnd. `None` means
    /// all model time axes are dynamic.
    pub(super) static_feature_frames: Option<usize>,
    /// Generator I/O names resolved to this model's export convention; threaded
    /// to every binding site so vcclient, RVC WebUI, and third-party converter
    /// exports all load.
    pub(super) io_names: RvcIoNames,
    /// The model's native audio sample rate from metadata `samplingRate`, when
    /// recorded. `None` means unknown; the pipeline falls back to the default
    /// `RVC_SAMPLE_RATE`. Threaded so the convert/output windows are sized at the
    /// model's real rate (e.g. 32 kHz) instead of the hardcoded 48 kHz.
    pub(super) rvc_sample_rate: Option<u32>,
    /// Speaker IDs accepted by the generator, derived from `emb_g.weight`.
    /// `None` covers unusual exports that inline or rename the embedding table.
    pub(super) speaker_count: Option<usize>,
}

pub(super) fn inspect_contentvec_input_name(
    path: &Path,
    expected_channels: i64,
    requested_output: Option<&str>,
) -> Result<String> {
    let io = read_model_io(path)?;
    let input_name = io.single_input_name()?.to_string();
    let output_name = io.select_embedder_output(expected_channels, requested_output)?;
    info!(
        "inspected ContentVec model for fixed profile: {} input={} output={}",
        path.display(),
        input_name,
        output_name
    );
    Ok(input_name)
}

pub(super) fn inspect_rvc_model(path: &Path) -> Result<RvcModelInfo> {
    reject_pth_checkpoint(path)?;
    let io = read_model_io(path)?;
    let io_names = io.resolve_rvc_io_names()?;
    let expected_feat_channels = io.feat_channels(&io_names.feats)?;
    let static_feature_frames = io.validate_rvc_input_contract(&io_names)?;
    io.validate_rvc_metadata()?;
    let rvc_sample_rate = io.rvc_sample_rate();
    let speaker_count = io.rvc_speaker_count();
    let rnd_desc = io_names
        .rnd
        .as_ref()
        .map(|rnd| format!("{}[1,{},frames]", rnd.name, rnd.channels))
        .unwrap_or_else(|| "none".to_string());
    info!(
        "inspected RVC model: {} inputs=[{},{},{},{},{}] rnd={} output={} feat_channels={} frames={} sample_rate={} speakers={}",
        path.display(),
        io_names.feats,
        io_names.p_len,
        io_names.pitch,
        io_names.pitchf,
        io_names.sid,
        rnd_desc,
        io_names.audio,
        expected_feat_channels,
        static_feature_frames
            .map(|frames| frames.to_string())
            .unwrap_or_else(|| "dynamic".to_string()),
        rvc_sample_rate
            .map(|rate| rate.to_string())
            .unwrap_or_else(|| "default".to_string()),
        speaker_count
            .map(|count| count.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    Ok(RvcModelInfo {
        expected_feat_channels,
        static_feature_frames,
        io_names,
        rvc_sample_rate,
        speaker_count,
    })
}
