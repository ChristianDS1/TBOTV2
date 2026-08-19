# Auto-restart wrapper for 72h+ runs. Launcher only — no app code changes.
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Simulate,
    [int]$RestartDelaySec = 30
)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$logsDir = Join-Path $Root "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$logDate = Get-Date -Format "yyyy-MM-dd"
$logPath = Join-Path $logsDir "bot_$logDate.log"

function Write-LoopLine {
    param([string]$Line)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "[$ts] [loop] $Line"
    Write-Host $msg
    Add-Content -Path $logPath -Value $msg -Encoding utf8
}

$restart = 0
Write-LoopLine "=== T-BOT loop started (delay ${RestartDelaySec}s) ==="
Write-LoopLine "Monitor: http://${BindHost}:$Port/ | Ctrl+C to stop (no hot restart)"

try {
    while ($true) {
        $restart++
        Write-LoopLine "=== run #$restart ==="

        $psArgs = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $PSScriptRoot "start_tbot.ps1"),
            "-BindHost", $BindHost,
            "-Port", "$Port"
        )
        if ($Simulate) { $psArgs += "-Simulate" }
        # Browser only on first run; restarts stay quiet
        if ($restart -gt 1) { $psArgs += "-NoBrowser" }

        & powershell @psArgs
        $code = $LASTEXITCODE
        Write-LoopLine "Process exited code=$code — restart in ${RestartDelaySec}s (Ctrl+C to abort)"
        Start-Sleep -Seconds $RestartDelaySec
    }
} finally {
    Write-LoopLine "=== loop stopped ==="
}
