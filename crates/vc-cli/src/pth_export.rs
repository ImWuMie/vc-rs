//! Offline import of compatible RVC checkpoints.
//!
//! The Rust application intentionally never links PyTorch: it would enlarge
//! every package and, more importantly, has no place on the real-time path.
//! This command starts a user-selected local RVC Python environment once,
//! validates its ONNX output through the same structural contract that the
//! shared `RvcPipeline` loads, and then exits.

use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};

use crate::cli::ExportPthArgs;

const EXPORTER_SCRIPT: &str = include_str!("pth_exporter.py");
const EXPORTER_RELATIVE_PATH: &str = "infer/lib/infer_pack/models_onnx.py";

pub fn run(mut args: ExportPthArgs) -> Result<()> {
    if !args.trust_rvc_root {
        bail!(
            "--trust-rvc-root is required because exporting imports Python code from --rvc-root; \
             only use an RVC installation you trust"
        );
    }
    args.model = canonical_regular_file(&args.model, "PTH model")?;
    if !args
        .model
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pth"))
    {
        bail!(
            "--model must name a .pth checkpoint: {}",
            args.model.display()
        );
    }
    args.python = canonical_regular_file(&args.python, "Python executable")?;
    args.rvc_root = canonical_directory(&args.rvc_root, "RVC root")?;
    args.output = absolute_path(&args.output)?;

    let exporter = args.rvc_root.join(EXPORTER_RELATIVE_PATH);
    canonical_regular_file(&exporter, "RVC ONNX exporter definitions")?;
    if args.output.exists() {
        bail!(
            "output already exists (refusing to overwrite): {}; choose a new --output path",
            args.output.display()
        );
    }
    let output_parent = args.output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_parent).with_context(|| {
        format!(
            "failed to create ONNX output directory {}",
            output_parent.display()
        )
    })?;

    let script_path = create_exporter_script()?;
    let temporary_output = unique_output_path(&args.output)?;
    let export_result = invoke_exporter(&args, &script_path, &temporary_output);
    let _ = fs::remove_file(&script_path);

    if let Err(error) = export_result {
        let _ = fs::remove_file(&temporary_output);
        return Err(error);
    }

    if let Err(error) = vc_core::model_rvc::validate_rvc_model(&temporary_output) {
        let _ = fs::remove_file(&temporary_output);
        return Err(error).with_context(|| {
            format!(
                "exporter produced an ONNX file that vc-rs cannot use: {}",
                temporary_output.display()
            )
        });
    }

    // Recheck immediately before the rename: never replace a model another
    // process may have created while this potentially long export was running.
    if args.output.exists() {
        let _ = fs::remove_file(&temporary_output);
        bail!(
            "output appeared while exporting (refusing to overwrite): {}",
            args.output.display()
        );
    }
    fs::rename(&temporary_output, &args.output).with_context(|| {
        format!(
            "failed to move verified ONNX export to {}",
            args.output.display()
        )
    })?;

    println!(
        "Exported and validated RVC ONNX model: {}",
        args.output.display()
    );
    Ok(())
}

fn canonical_regular_file(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_file() {
        bail!("{label} was not found or is not a file: {}", path.display());
    }
    fs::canonicalize(path)
        .with_context(|| format!("failed to resolve {label} path {}", path.display()))
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_dir() {
        bail!(
            "{label} was not found or is not a directory: {}",
            path.display()
        );
    }
    fs::canonicalize(path)
        .with_context(|| format!("failed to resolve {label} path {}", path.display()))
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .context("failed to determine current directory for --output")
        .map(|directory| directory.join(path))
}

fn create_exporter_script() -> Result<PathBuf> {
    let base = std::env::temp_dir();
    for attempt in 0..128 {
        let path = base.join(unique_name("vc-rs-rvc-pth-export", "py", attempt));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create temporary exporter {}", path.display())
                })
            }
        };
        if let Err(error) = file.write_all(EXPORTER_SCRIPT.as_bytes()) {
            let _ = fs::remove_file(&path);
            return Err(error)
                .with_context(|| format!("failed to write temporary exporter {}", path.display()));
        }
        return Ok(path);
    }
    bail!("could not allocate a unique temporary exporter script")
}

fn unique_output_path(output: &Path) -> Result<PathBuf> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let filename = output
        .file_name()
        .ok_or_else(|| anyhow!("--output has no filename: {}", output.display()))?;
    let mut name = OsString::from(".");
    name.push(filename);
    for attempt in 0..128 {
        let mut candidate = name.clone();
        candidate.push(format!(".{}.exporting.onnx", unique_name("", "", attempt)));
        let candidate = parent.join(candidate);
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not allocate a temporary ONNX export path beside {}",
        output.display()
    )
}

fn unique_name(prefix: &str, extension: &str, attempt: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dot_extension = (!extension.is_empty()).then_some(".").unwrap_or_default();
    format!(
        "{prefix}-{}-{nanos}-{attempt}{dot_extension}{extension}",
        std::process::id()
    )
}

fn invoke_exporter(args: &ExportPthArgs, script: &Path, temporary_output: &Path) -> Result<()> {
    let output = Command::new(&args.python)
        .current_dir(&args.rvc_root)
        .arg(script)
        .arg("--model")
        .arg(&args.model)
        .arg("--output")
        .arg(temporary_output)
        .output()
        .with_context(|| format!("failed to start Python exporter {}", args.python.display()))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    bail!(
        "PTH export failed ({}).\n{}{}",
        output.status,
        tail(&stderr),
        if stdout.trim().is_empty() {
            String::new()
        } else {
            format!("\nstdout:\n{}", tail(&stdout))
        }
    )
}

fn tail(text: &str) -> &str {
    const MAX_ERROR_CHARS: usize = 16 * 1024;
    if text.len() <= MAX_ERROR_CHARS {
        text
    } else {
        let start = text
            .char_indices()
            .find_map(|(index, _)| (index >= text.len() - MAX_ERROR_CHARS).then_some(index))
            .unwrap_or(0);
        &text[start..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_temp_names_include_the_requested_extension() {
        assert!(unique_name("test", "py", 1).ends_with(".py"));
    }

    #[test]
    fn output_temp_path_stays_beside_final_model() {
        let output = Path::new("models/voice.onnx");
        let temporary = unique_output_path(output).unwrap();
        assert_eq!(temporary.parent(), Some(Path::new("models")));
        assert!(temporary
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("voice.onnx"));
    }

    #[test]
    fn relative_output_becomes_absolute_in_the_calling_directory() {
        let output = absolute_path(Path::new("voice.onnx")).unwrap();
        assert!(output.is_absolute());
        assert_eq!(output.file_name().unwrap(), "voice.onnx");
    }
}
