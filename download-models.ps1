# Optional helper script.
# This script downloads third-party reference ONNX models.
#
# ContentVec and RMVPE come from https://huggingface.co/wok000/weights_gpl and
# are marked GPL-3.0. The downloaded model files are NOT part of vc-rs and are
# NOT covered by this repository's MIT license. Review and comply with the
# upstream license before using, modifying, or redistributing them.
#
# The optional GTCRN streaming denoiser model comes from the MIT-licensed
# upstream repo github.com/Xiaobin-Rong/gtcrn (see each item's License /
# Provenance below). It is fetched only when you pass -Gtcrn.
#
# The optional DeepFilterNet3 archives come from the MIT/Apache-2.0 dual
# licensed upstream repo github.com/Rikorose/DeepFilterNet. The URL is pinned
# to the exact commit used by libDF in vc-core so a moving branch cannot silently
# change the model bytes. They are fetched only when you pass -DeepFilterNet3
# (or the separate -DeepFilterNet3LowLatency switch).
#
# vc-rs does not redistribute pretrained model files. This script only downloads
# them directly from the upstream hosts at the user's request.

param(
    # Also fetch the optional GTCRN input-denoiser model into assets\gtcrn\.
    [switch]$Gtcrn,

    # Fetch the standard official DeepFilterNet3 ONNX archive into
    # assets\deepfilternet3\.
    [switch]$DeepFilterNet3,

    # Fetch the smaller/low-latency official DeepFilterNet3 ONNX archive too.
    [switch]$DeepFilterNet3LowLatency
)

$ErrorActionPreference = "Stop"

$assetsDir = Join-Path $PSScriptRoot "assets"
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

$downloads = @(
    @{
        Name = "ContentVec ONNX"
        Url  = "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx"
        Path = Join-Path $assetsDir "content_vec_500.onnx"
        License = "GPL-3.0 (wok000/weights_gpl)"
    },
    @{
        Name = "RMVPE ONNX"
        Url  = "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx"
        Path = Join-Path $assetsDir "rmvpe.onnx"
        License = "GPL-3.0 (wok000/weights_gpl)"
    }
)

if ($Gtcrn) {
    # The GTCRN model dir holds gtcrn_stream.onnx; pass it via --gtcrn-model
    # assets\gtcrn (CLI) or the GUI's GTCRN model dir picker.
    $downloads += @{
        Name = "GTCRN streaming ONNX"
        Url  = "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/stream/onnx_models/gtcrn.onnx"
        Path = Join-Path $assetsDir "gtcrn\gtcrn_stream.onnx"
        License = "MIT (Xiaobin-Rong/gtcrn)"
        Provenance = "Upstream prebuilt streaming export from github.com/Xiaobin-Rong/gtcrn (stream/onnx_models/gtcrn.onnx), MIT-licensed; saved here as gtcrn_stream.onnx."
    }
}

$deepFilterCommit = "d375b2d8309e0935d165700c91da9de862a99c31"
if ($DeepFilterNet3) {
    $downloads += @{
        Name = "DeepFilterNet3 ONNX"
        Url  = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/$deepFilterCommit/models/DeepFilterNet3_onnx.tar.gz"
        Path = Join-Path $assetsDir "deepfilternet3\DeepFilterNet3_onnx.tar.gz"
        Sha256 = "C94D91F70911001C946E0FABB4AA9ADC37045F45A03B56008CB0C8244CB63616"
        License = "MIT OR Apache-2.0 (Rikorose/DeepFilterNet)"
        Provenance = "Official DeepFilterNet3 ONNX archive at commit $deepFilterCommit."
    }
}
if ($DeepFilterNet3LowLatency) {
    $downloads += @{
        Name = "DeepFilterNet3 low-latency ONNX"
        Url  = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/$deepFilterCommit/models/DeepFilterNet3_ll_onnx.tar.gz"
        Path = Join-Path $assetsDir "deepfilternet3\DeepFilterNet3_ll_onnx.tar.gz"
        Sha256 = "5998E58E8BA0E09BB76986EF97B84AFA065A571EF282D4A1222F341E3251CF3A"
        License = "MIT OR Apache-2.0 (Rikorose/DeepFilterNet)"
        Provenance = "Official DeepFilterNet3 low-latency ONNX archive at commit $deepFilterCommit."
    }
}

foreach ($item in $downloads) {
    if (Test-Path $item.Path) {
        if ($item.Sha256) {
            $existingHash = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash
            if ($existingHash -ieq $item.Sha256) {
                Write-Host "[skip] $($item.Name) already exists and matches SHA-256: $($item.Path)"
                continue
            }
            Write-Host "[redownload] $($item.Name) exists but its SHA-256 does not match"
        } else {
            Write-Host "[skip] $($item.Name) already exists: $($item.Path)"
            continue
        }
    }

    Write-Host "[download] $($item.Name)"
    Write-Host "  from:    $($item.Url)"
    Write-Host "  to:      $($item.Path)"
    Write-Host "  license: $($item.License)"
    if ($item.Provenance) {
        Write-Host "  provenance: $($item.Provenance)"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $item.Path) | Out-Null
    $tmpPath = "$($item.Path).download"

    try {
        Invoke-WebRequest `
            -Uri $item.Url `
            -OutFile $tmpPath `
            -UseBasicParsing

        Move-Item -Force $tmpPath $item.Path

        if ($item.Sha256) {
            $actualHash = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash
            if ($actualHash -ine $item.Sha256) {
                Remove-Item -Force -LiteralPath $item.Path
                throw "SHA-256 mismatch for $($item.Name): expected $($item.Sha256), got $actualHash"
            }
            Write-Host "  sha256:  $actualHash"
        }

        $sizeMB = [math]::Round((Get-Item $item.Path).Length / 1MB, 2)
        Write-Host "[done] $($item.Name) ($sizeMB MB)"
    }
    catch {
        if (Test-Path $tmpPath) {
            Remove-Item -Force $tmpPath
        }
        throw
    }
}

Write-Host ""
Write-Host "All requested models are ready."
Write-Host "ContentVec: $assetsDir\content_vec_500.onnx"
Write-Host "RMVPE:      $assetsDir\rmvpe.onnx"
if ($Gtcrn) {
    Write-Host "GTCRN dir:  $assetsDir\gtcrn  (use --gtcrn-model `"$assetsDir\gtcrn`")"
}
if ($DeepFilterNet3) {
    Write-Host "DFN3:       $assetsDir\deepfilternet3\DeepFilterNet3_onnx.tar.gz"
}
if ($DeepFilterNet3LowLatency) {
    Write-Host "DFN3 LL:    $assetsDir\deepfilternet3\DeepFilterNet3_ll_onnx.tar.gz"
}
