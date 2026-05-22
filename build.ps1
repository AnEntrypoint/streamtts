# Build helper that initializes the MSVC environment before invoking cargo.
#
# Why this exists: on this host, both `link.exe` and `rustc.exe` are shadowed
# by stale chocolatey/Git installations:
#   - C:\Program Files\Git\usr\bin\link.exe is coreutils `link`, not MSVC's.
#     It fails every build script with "link: unknown option -- dynamicbase".
#   - C:\ProgramData\chocolatey\bin\rustc.exe is 1.85.1 (GNU target), shadowing
#     rustup's 1.95.0-msvc. The icu_* crates require >= 1.86.
#
# This script ignores PATH order entirely: it directly invokes rustup's cargo
# under a vcvars64-initialized environment, so link.exe resolves to MSVC's
# Hostx64\x64\link.exe and the include/lib paths are correct.
#
# Usage:
#   .\build.ps1                                       # cargo build --release -p sttx-cli
#   .\build.ps1 check                                 # cargo check -p sttx-cli
#   .\build.ps1 train --ccsniff-from foo.ndjson       # cargo run --release -p sttx-cli -- train ...
#   .\build.ps1 raw -- check --workspace              # cargo check --workspace (bypass defaults)

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'

$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $VcVars)) {
    throw "vcvars64.bat not found at $VcVars -- install Visual Studio Build Tools or edit this path."
}

# Resolve rustup's cargo directly, ignoring chocolatey shadow.
$RustupBin   = "$env:USERPROFILE\.cargo\bin"
$RustupCargo = "$RustupBin\cargo.exe"
if (-not (Test-Path $RustupCargo)) {
    throw "rustup cargo not found at $RustupCargo -- run `rustup default 1.95.0-x86_64-pc-windows-msvc`."
}

# Pull vcvars64 env into this PowerShell session, once. (cmd.exe is at
# C:\Windows\System32\cmd.exe; that dir is always on the base PATH.)
if (-not $env:STREAMTTS_VCVARS_LOADED) {
    Write-Host "[build.ps1] loading MSVC env from vcvars64.bat..." -ForegroundColor DarkGray
    & cmd.exe /c "`"$VcVars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
        }
    }
    $env:STREAMTTS_VCVARS_LOADED = '1'
}

# Surgical PATH rewrite AFTER vcvars set it up:
#   - Prepend rustup's bin so cargo's spawned rustc.exe is rustup's, not the
#     chocolatey shadow (1.85.1 gnu-target).
#   - Drop chocolatey\bin AND Git\usr\bin entries so chocolatey's rustc /
#     Git's coreutils `link.exe` can never resolve ahead of MSVC link.exe.
#     The MSVC linker path was added to the front by vcvars64.
$pathParts = $env:PATH -split ';' | Where-Object {
    $_ -and ($_ -notmatch '\\chocolatey\\bin\\?$') -and ($_ -notmatch '\\Git\\usr\\bin\\?$')
}
$env:PATH = ((,$RustupBin) + $pathParts) -join ';'

# Belt-and-braces: pin cargo's choice of rustc and target.
$env:RUSTC = "$RustupBin\rustc.exe"
$env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-msvc'

# Decide which cargo invocation to run.
if ($Args.Count -eq 0) {
    $cargoArgs = @('build', '--release', '-p', 'sttx-cli')
}
else {
    $rest = if ($Args.Count -gt 1) { $Args[1..($Args.Count - 1)] } else { @() }
    if ($Args[0] -eq 'raw') {
        $cargoArgs = $rest
    }
    elseif ($Args[0] -eq 'check') {
        $cargoArgs = @('check', '-p', 'sttx-cli') + $rest
    }
    elseif ($Args[0] -in @('train', 'serve', 'inspect', 'merge-stats', 'validate-data', 'quality-assert')) {
        $cargoArgs = @('run', '--release', '-p', 'sttx-cli', '--') + $Args
    }
    else {
        # Pass-through: e.g. .\build.ps1 test, .\build.ps1 clippy
        $cargoArgs = $Args
    }
}

Write-Host "[build.ps1] cargo $($cargoArgs -join ' ')" -ForegroundColor Cyan
& $RustupCargo @cargoArgs
exit $LASTEXITCODE
