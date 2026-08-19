# Running T-BOT 72h+ on Windows

These scripts **only start the existing app** (`python -m trading_system run`). They do not change trading, learning, or risk logic.

## Quick start

| Script | Use |
|--------|-----|
| `scripts\start_tbot.bat` | Normal run: console + daily log + opens monitor once |
| `scripts\start_tbot_loop.bat` | 72h+: auto-restart ~30s after crash/exit |

Double-click either file, or from repo root:

```powershell
.\scripts\start_tbot.ps1
.\scripts\start_tbot_loop.ps1
```

Optional flags (PowerShell):

```powershell
.\scripts\start_tbot.ps1 -NoBrowser
.\scripts\start_tbot.ps1 -Simulate
.\scripts\start_tbot.ps1 -BindHost 127.0.0.1 -Port 8000
```

## Logs

- Folder: `logs/`
- File: `logs/bot_YYYY-MM-DD.log` (one file per calendar day)
- Console and file via `Tee-Object`; Python runs with `-u` and `PYTHONUNBUFFERED=1` to reduce buffering lag
- `*.log` is gitignored

If encoding looks wrong in the log, use the monitor + `reports/daily/`; the bot behavior is unchanged.

## Monitor (primary UI)

- URL: **http://127.0.0.1:8000/**
- Refreshes ~every 5s — you do **not** need to keep the browser open for 72h
- Health checklist:
  - **Kill switch** OFF (if ON with `api_failure:crypto`, reset in UI or restart the process)
  - **Crypto** `entries_enabled: true` on weekends
  - **Equity / trades** updating over time
  - Rejects not 100% dominated by `confirmed_loss_pattern:*` (learning lockup)

## Daily reports

- Auto: `reports/daily/daily_YYYY-MM-DD.md` at UTC day rollover
- Manual: `python -m trading_system report`
- Sections: aprendizaje, errores, oportunidades, cambios, progreso

## Database

- `data/trading.db` — trade history (do not commit)

## Prevent sleep (manual — required for 72h+)

1. **Settings → System → Power** → Plugged in: **Sleep = Never**
2. Disable hibernation if the PC uses it
3. Optional (admin CMD): `powercfg /change standby-timeout-ac 0`

The start script may warn if AC standby timeout looks low; it does **not** change power settings automatically.

## Auto-restart loop

`start_tbot_loop.bat`:

- Restarts the bot ~30s after exit or crash
- Logs `=== run #N ===` to the daily log
- Opens browser **only on the first** run
- **Ctrl+C** in the window stops the loop (no immediate hot restart)

Useful after transient `api_failure` kill-switch trips (restart clears in-memory kill state).

## Smoke test

1. Run `scripts\start_tbot.bat` → browser opens, monitor returns 200
2. Confirm `logs\bot_*.log` grows with INFO / uvicorn lines
3. Run `scripts\start_tbot_loop.bat`, kill `python.exe` once → new run within ~30s

## What these scripts do NOT do

- No Task Scheduler / Windows service
- No changes to `trading_system/` code or config
- No auto-reset of kill switch while the same process stays alive (use UI reset or restart)
