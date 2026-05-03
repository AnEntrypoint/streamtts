#Requires -Version 5.1
param(
    [string]$CcsniffFrom = "ccsniff-fresh.ndjson",
    [long]$Steps = 100000,
    [string]$CheckpointDir = "ckpt-full-history",
    [long]$CheckpointEvery = 1000,
    [string]$ModelRepo = "RWKV/RWKV7-Goose-World3-1.5B-HF",
    [string]$Binary = ".\bin\streamtts.exe"
)

$pidFile = ".\train.pid"
$logFile = ".\train.log"
$errFile = ".\train.err"

if (Test-Path $pidFile) {
    $existingPid = Get-Content $pidFile -Raw
    $proc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Training already running (PID $existingPid). Use train-stop.ps1 to stop it."
        exit 1
    }
    Remove-Item $pidFile -Force
}

if (-not (Test-Path $Binary)) {
    Write-Host "Binary not found: $Binary"
    Write-Host "Download from GitHub Actions artifacts or run: cargo build --release -p sttx-cli"
    exit 1
}

$args = @(
    "train",
    "--ccsniff-from", $CcsniffFrom,
    "--steps", $Steps,
    "--checkpoint-dir", $CheckpointDir,
    "--checkpoint-every", $CheckpointEvery,
    "--model-repo", $ModelRepo
)

$proc = Start-Process -FilePath $Binary `
    -ArgumentList $args `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -NoNewWindow `
    -PassThru

$proc.Id | Set-Content $pidFile
Write-Host "Training started (PID $($proc.Id))"
Write-Host "Log: $logFile"
Write-Host "Status: .\train-status.ps1"
Write-Host "Stop:   .\train-stop.ps1"
