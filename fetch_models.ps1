$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$assets = @(
    @{ Name = 'baseline.zip'; Directory = 'checkpoints'; Sha256 = '881F9FFC71174D3EB8AB0FE6D41C87A384E7E6F8B63B8FD38822A8A656F9EF9A' },
    @{ Name = 'improved.zip'; Directory = 'checkpoints'; Sha256 = '81EB923929868B2F8951E88FF9BD716694115AAA71728773ED0B3B48B1173B5E' },
    @{ Name = 'model.safetensors'; Directory = 'models/pythia-70m'; Sha256 = 'EBFA4E2F18696EBD83716A0D39FE2C025F2FF8483F72A83CA59C475692FC9D15' }
)

foreach ($asset in $assets) {
    $directory = Join-Path $root $asset.Directory
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $target = Join-Path $directory $asset.Name
    if (-not (Test-Path -LiteralPath $target) -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash -ne $asset.Sha256) {
        gh release download models-v1 --repo ARPurohit07/RocketRL --pattern $asset.Name --dir $directory --clobber
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($asset.Name)" }
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash -ne $asset.Sha256) {
        throw "Checksum mismatch: $target"
    }
    Write-Host "Verified $($asset.Name)"
}
