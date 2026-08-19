# Start T-BOT monitor + engine (Windows). Does NOT modify app code - launcher only.
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Simulate
)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

$logsDir = Join-Path $Root "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$logDate = Get-Date -Format "yyyy-MM-dd"
$logPath = Join-Path $logsDir "bot_$logDate.log"
$monitorUrl = "http://${BindHost}:$Port/"

function Get-GitShortHead {
    try {
        $h = git -C $Root rev-parse --short HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and $h) { return $h.Trim() }
    } catch { }
    return "unknown"
}

function Write-LogLine {
    param([string]$Line)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "[$ts] $Line"
    Write-Host $msg
    Add-Content -Path $logPath -Value $msg -Encoding utf8
}

function Show-StandbyWarning {
    try {
        $out = powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2>$null
        if ($out -match "Current AC Power Setting Index:\s*0x([0-9a-f]+)") {
            $hex = $Matches[1]
            $sec = [Convert]::ToInt32($hex, 16)
            if ($sec -gt 0 -and $sec -lt 3600) {
                Write-LogLine "WARN: AC standby timeout ~${sec}s - set Sleep to Never for 72h+ runs (see docs/RUN_72H.md)"
            }
        }
    } catch { }
}

function Open-MonitorBrowser {
    if ($NoBrowser) { return }
    Start-Job -ScriptBlock {
        param($Url)
        Start-Sleep -Seconds 2
        Start-Process $Url
    } -ArgumentList $monitorUrl | Out-Null
}

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-LogLine "ERROR: venv not found at $py - run: python -m venv .venv; pip install -r requirements.txt"
    exit 1
}

Write-LogLine "=== T-BOT start ==="
Write-LogLine "Root: $Root"
Write-LogLine "Git: $(Get-GitShortHead)"
Write-LogLine "Monitor: $monitorUrl"
Write-LogLine "Reports: reports/daily/"
Write-LogLine "Log: $logPath"
Write-LogLine "Host:Port: ${BindHost}:$Port Simulate=$Simulate"
Show-StandbyWarning
Open-MonitorBrowser

$runArgs = @(
    "-u", "-m", "trading_system", "run",
    "--host", $BindHost,
    "--port", "$Port"
)
if ($Simulate) {
    $runArgs += "--simulate"
}

Write-LogLine "Command: $py $($runArgs -join ' ')"

& $py @runArgs 2>&1 | Tee-Object -FilePath $logPath -Append
exit $LASTEXITCODE
