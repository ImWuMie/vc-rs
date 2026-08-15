# Backend package builds overwrite the same Cargo output paths. Keep an identity
# stamp beside those ignored outputs so -SkipBuild can prove both the requested
# feature set and the exact bytes being repackaged. Paths are deliberately not
# serialized, which keeps machine-specific state out of the stamp.

function Get-PackageArtifactRecords {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.IDictionary]$Artifacts
    )

    $records = @()
    foreach ($name in @($Artifacts.Keys | ForEach-Object { [string]$_ } | Sort-Object)) {
        $artifactPath = [string]$Artifacts[$name]
        if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) {
            throw "Package artifact '$name' is missing: $artifactPath"
        }
        $records += [ordered]@{
            name = $name
            sha256 = (Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    return $records
}

function Write-PackageArtifactStamp {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$PackageKind,
        [Parameter(Mandatory = $true)] [string]$Variant,
        [Parameter(Mandatory = $true)] [string]$Features,
        [Parameter(Mandatory = $true)] [bool]$Asio,
        [Parameter(Mandatory = $true)] [bool]$DeepFilterNet3,
        [Parameter(Mandatory = $true)] [bool]$RuntimeOnly,
        [Parameter(Mandatory = $true)] [System.Collections.IDictionary]$Artifacts
    )

    $stamp = [ordered]@{
        schemaVersion = 1
        packageKind = $PackageKind
        variant = $Variant
        features = $Features
        noDefaultFeatures = $true
        asio = $Asio
        deepFilterNet3 = $DeepFilterNet3
        runtimeOnly = $RuntimeOnly
        artifacts = @(Get-PackageArtifactRecords -Artifacts $Artifacts)
    }

    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    # Replace only after every artifact was hashed and JSON serialization
    # completed; an interrupted build must not leave a valid-looking stamp.
    $temporaryPath = "$Path.tmp"
    try {
        $stamp | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
        Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

function Assert-PackageArtifactStamp {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$PackageKind,
        [Parameter(Mandatory = $true)] [string]$Variant,
        [Parameter(Mandatory = $true)] [string]$Features,
        [Parameter(Mandatory = $true)] [bool]$Asio,
        [Parameter(Mandatory = $true)] [bool]$DeepFilterNet3,
        [Parameter(Mandatory = $true)] [bool]$RuntimeOnly,
        [Parameter(Mandatory = $true)] [System.Collections.IDictionary]$Artifacts
    )

    $remediation = "Run this package command once without -SkipBuild for the requested configuration."
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Cannot use -SkipBuild because the package artifact identity stamp is missing: $Path. $remediation"
    }

    try {
        $stamp = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    }
    catch {
        throw "Cannot use -SkipBuild because the package artifact identity stamp is invalid: $Path. $remediation"
    }

    $expectedFields = [ordered]@{
        schemaVersion = 1
        packageKind = $PackageKind
        variant = $Variant
        features = $Features
        noDefaultFeatures = $true
        asio = $Asio
        deepFilterNet3 = $DeepFilterNet3
        runtimeOnly = $RuntimeOnly
    }
    foreach ($field in $expectedFields.Keys) {
        $actual = $stamp.$field
        $expected = $expectedFields[$field]
        if ([string]$actual -cne [string]$expected) {
            throw "Cannot use -SkipBuild because stamp field '$field' is '$actual', expected '$expected'. $remediation"
        }
    }

    $actualArtifacts = @($stamp.artifacts)
    if ($actualArtifacts.Count -ne $Artifacts.Count) {
        throw "Cannot use -SkipBuild because the stamp lists $($actualArtifacts.Count) artifacts, expected $($Artifacts.Count). $remediation"
    }

    foreach ($expectedArtifact in @(Get-PackageArtifactRecords -Artifacts $Artifacts)) {
        $matches = @($actualArtifacts | Where-Object { $_.name -ceq $expectedArtifact.name })
        if ($matches.Count -ne 1) {
            throw "Cannot use -SkipBuild because artifact '$($expectedArtifact.name)' is missing or duplicated in the stamp. $remediation"
        }
        if ([string]$matches[0].sha256 -cne [string]$expectedArtifact.sha256) {
            throw "Cannot use -SkipBuild because artifact '$($expectedArtifact.name)' no longer matches its recorded SHA-256. $remediation"
        }
    }
}
