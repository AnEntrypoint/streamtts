#Requires -Version 5.1
param([int]$Lines = 40)

$pidFile = ".\train.pid"
$logFile = ".\train.log"

if (Test-Path $pidFile) {
    $trainPid = Get-Content $pidFile -Raw
    $proc = Get-Process -Id $trainPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Status: RUNNING (PID $trainPid, CPU $($proc.CPU)s, RAM $([math]::Round($proc.WorkingSet64/1MB))MB)"
    } else {
        Write-Host "Status: STOPPED (PID $trainPid no longer running)"
    }
} else {
    Write-Host "Status: NOT STARTED (no train.pid)"
}

if (Test-Path $logFile) {
    Write-Host ""
    Write-Host "--- Last $Lines lines of $logFile ---"
    Get-Content $logFile -Tail $Lines
}
