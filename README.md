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

```bash
python -m trading_system once --simulate
python -m trading_system backtest --bars 500
pytest -q
```

## Layout

See [ARCHITECTURE.md](ARCHITECTURE.md). Config: `trading_system/config/default.yaml`.

## Mode

Default is **paper**. Live is blocked until explicitly enabled later with hard risk gates.
