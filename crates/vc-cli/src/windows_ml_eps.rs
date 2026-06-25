use std::io::{self, Write};

use anyhow::{bail, Context, Result};
use vc_core::windows_ml::{self, CatalogExecutionProvider, CatalogProviderInfo, CatalogReadyState};

use crate::cli::{
    WindowsMlEpProvider, WindowsMlEpsArgs, WindowsMlEpsCommand, WindowsMlEpsInstallArgs,
};

pub fn run(args: WindowsMlEpsArgs) -> Result<()> {
    match args.command {
        WindowsMlEpsCommand::List => list(),
        WindowsMlEpsCommand::Install(args) => install(args),
    }
}

fn list() -> Result<()> {
    let providers = windows_ml::list_catalog_providers()?;
    if providers.is_empty() {
        println!("No Windows ML catalog execution providers are visible on this device.");
        return Ok(());
    }

    println!("Windows ML catalog execution providers:");
    for provider in &providers {
        print_provider(provider);
    }

    match windows_ml::select_best_catalog_provider_info(&providers) {
        Some(provider) => {
            let vc_provider = provider.vc_provider.with_context(|| {
                format!(
                    "Windows ML catalog EP {} matched vc-rs priority but has no vc provider mapping",
                    provider.name
                )
            })?;
            println!(
                "\nvc-rs priority would select: {} ({}) via {} ready-state={}",
                vc_provider.label(),
                vc_provider.vc_provider_name(),
                provider.name,
                provider.ready_state.label()
            );
        }
        None => {
            println!("\nNo vc-rs-supported Windows ML catalog EP is compatible on this device.");
            println!("The Windows ML provider can still use DirectML/CPU fallback.");
        }
    }
    Ok(())
}

fn install(args: WindowsMlEpsInstallArgs) -> Result<()> {
    let providers = windows_ml::list_catalog_providers()?;
    let selected_info = match args.provider {
        Some(provider) => {
            let selected = provider.into_catalog_provider();
            windows_ml::select_catalog_provider_info(&providers, selected)
        }
        None => Some(windows_ml::select_best_catalog_provider_info(&providers).with_context(
            || {
                "no vc-rs-supported Windows ML catalog EP is compatible on this device; use provider windowsml for DirectML/CPU fallback"
            },
        )?),
    };
    let selected = selected_info
        .and_then(|provider| provider.vc_provider)
        .or_else(|| {
            args.provider
                .map(WindowsMlEpProvider::into_catalog_provider)
        })
        .with_context(|| {
            "selected Windows ML catalog EP matched vc-rs priority but has no vc provider mapping"
        })?;

    println!(
        "Selected Windows ML catalog EP: {} ({})",
        selected.label(),
        selected.vc_provider_name()
    );
    if let Some(provider) = selected_info {
        println!("Catalog provider: {}", provider.name);
        println!("Current state: {}", provider.ready_state.label());
        if !provider.version.is_empty() {
            println!("Version: {}", provider.version);
        }
        if !provider.library_path.is_empty() {
            println!("Library: {}", provider.library_path);
        }
        if provider.ready_state == CatalogReadyState::Ready {
            println!("Already ready. No install action needed.");
            return Ok(());
        }
    } else {
        println!("Current state: not listed as compatible by Windows ML catalog.");
    }

    // Download/install can take minutes and may use Windows Update/Store-backed
    // services, so keep the explicit install command behind confirmation.
    if !args.yes {
        let action = match selected_info.map(|provider| provider.ready_state) {
            Some(CatalogReadyState::NotPresent) => "download and install",
            Some(CatalogReadyState::NotReady) => "prepare",
            Some(CatalogReadyState::Unknown(_)) | None => "attempt to prepare",
            Some(CatalogReadyState::Ready) => unreachable!("ready returned above"),
        };
        if !confirm(&format!(
            "Proceed to {action} {}? This may take several minutes.",
            selected.label()
        ))? {
            bail!("cancelled");
        }
    }

    let installed = windows_ml::ensure_catalog_provider_ready(selected)?;
    println!("Result state: {}", installed.ready_state.label());
    if !installed.library_path.is_empty() {
        println!("Library: {}", installed.library_path);
    }
    println!("Use with: --provider {}", selected.vc_provider_name());
    Ok(())
}

fn print_provider(provider: &CatalogProviderInfo) {
    let vc_provider = provider
        .vc_provider
        .map(CatalogExecutionProvider::vc_provider_name)
        .unwrap_or("-");
    let availability = match provider.ready_state {
        CatalogReadyState::Ready | CatalogReadyState::NotReady => "present",
        CatalogReadyState::NotPresent => "not-present",
        CatalogReadyState::Unknown(_) => "unknown",
    };
    println!(
        "  {} ready-state={} availability={} vc-provider={}",
        provider.name,
        provider.ready_state.label(),
        availability,
        vc_provider
    );
    if !provider.version.is_empty() {
        println!("    version={}", provider.version);
    }
    if !provider.package_family_name.is_empty() {
        println!("    package={}", provider.package_family_name);
    }
    if !provider.library_path.is_empty() {
        println!("    library={}", provider.library_path);
    }
}

fn confirm(prompt: &str) -> Result<bool> {
    print!("{prompt} [y/N]: ");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

impl WindowsMlEpProvider {
    fn into_catalog_provider(self) -> CatalogExecutionProvider {
        match self {
            Self::Nvtrtx => CatalogExecutionProvider::NvTensorRtRtx,
            Self::Qnn => CatalogExecutionProvider::Qnn,
            Self::Openvino => CatalogExecutionProvider::OpenVino,
            Self::Migraphx => CatalogExecutionProvider::MiGraphX,
            Self::Vitisai => CatalogExecutionProvider::VitisAi,
        }
    }
}
