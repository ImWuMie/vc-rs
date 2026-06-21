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
# vc-rs does not redistribute pretrained model files. This script only downloads
# them directly from the upstream hosts at the user's request.

param(
    # Also fetch the optional GTCRN input-denoiser model into assets\gtcrn\.
    [switch]$Gtcrn
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

foreach ($item in $downloads) {
    if (Test-Path $item.Path) {
        Write-Host "[skip] $($item.Name) already exists: $($item.Path)"
        continue
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
