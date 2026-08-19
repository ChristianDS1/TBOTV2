# Adaptive Quantitative Trading System

Paper-first dual-venue bot: **crypto 24/7** + **forex session hours only** (no weekend OTC).

## Quick start

```bash
cd "d:\T-BOT V2"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# Simulated data (no exchange needed)
python -m trading_system run --simulate

# Live public OHLCV + internal paper fills
python -m trading_system run
```

Open **http://127.0.0.1:8000/** — monitor refreshes every 5s.

## Windows 72h run

Launcher scripts only (no change to bot logic):

- **Normal:** double-click `scripts\start_tbot.bat` — console, daily log under `logs/`, opens monitor once
- **Long run:** `scripts\start_tbot_loop.bat` — auto-restart ~30s after crash/exit

See [docs/RUN_72H.md](docs/RUN_72H.md) for logs, reports, sleep settings, and health checks.

```bash
python -m trading_system once --simulate
python -m trading_system backtest --bars 500
pytest -q
```

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md). Config: `trading_system/config/default.yaml`.

## Mode

Default is **paper**. Live is blocked until explicitly enabled later with hard risk gates.
