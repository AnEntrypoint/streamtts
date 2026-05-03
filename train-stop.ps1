#Requires -Version 5.1
$pidFile = ".\train.pid"

if (-not (Test-Path $pidFile)) {
    Write-Host "No train.pid found — training not running."
    exit 0
}

$trainPid = Get-Content $pidFile -Raw
$proc = Get-Process -Id $trainPid -ErrorAction SilentlyContinue
if ($proc) {
    Stop-Process -Id $trainPid -Force
    Write-Host "Stopped training (PID $trainPid)"
} else {
    Write-Host "Process $trainPid already exited."
}
Remove-Item $pidFile -Force
