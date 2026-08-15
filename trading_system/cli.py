"""CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import uvicorn

from trading_system.config import load_config
from trading_system.dashboard import create_app
from trading_system.engine import TradingEngine


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trading_system", description="Adaptive Trading Bot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Start engine + monitor dashboard")
    run_p.add_argument("--simulate", action="store_true", help="Use simulated crypto data (no network)")
    run_p.add_argument("--host", default=None)
    run_p.add_argument("--port", type=int, default=None)

    sub.add_parser("once", help="Run a single tick and print snapshot")
    once_p = sub.choices["once"]
    once_p.add_argument("--simulate", action="store_true")

    bt = sub.add_parser("backtest", help="Run synthetic backtest")
    bt.add_argument("--bars", type=int, default=500)

    rp = sub.add_parser("report", help="Generate daily learning report now")
    rp.add_argument("--day", default=None, help="UTC day YYYY-MM-DD (default today)")
    rp.add_argument("--simulate", action="store_true")

    sub.add_parser(
        "rebuild-patterns",
        help="Backfill gross/cost_erosion on closed trades and rebuild pattern evidence",
    )

    purge = sub.add_parser(
        "purge-trades",
        help="Delete trades at/after a cutoff (default 2026-08-12T16:15 UTC) and rebuild patterns",
    )
    purge.add_argument(
        "--after",
        default="2026-08-12T16:15:00+00:00",
        help="UTC cutoff ISO datetime",
    )
    purge.add_argument("--db", default=None, help="Optional path to trading.db")
    purge.add_argument("--dry-run", action="store_true")
    purge.add_argument("--no-rebuild", action="store_true")
    purge.add_argument("--yes", "-y", action="store_true")

    reset = sub.add_parser(
        "reset-loss-learning",
        help="Wipe loss/cost_erosion patterns and delete time_stop trades (keep other wins)",
    )
    reset.add_argument("--db", default=None, help="Optional path to trading.db")
    reset.add_argument("--dry-run", action="store_true")
    reset.add_argument("--yes", "-y", action="store_true")

    hist = sub.add_parser(
        "historical-ml-run",
        help="Causal N-day historical walk with EXIT FIX labels + incremental ML (no 24/7 bot)",
    )
    hist.add_argument("--days", type=int, default=15)
    hist.add_argument("--out", default=None, help="Dataset dir (default data/ml/hist15_clean)")
    hist.add_argument("--model-dir", default=None, help="Model dir (default models/hist15_clean)")
    hist.add_argument("--simulate", action="store_true")
    hist.add_argument("--max-bars", type=int, default=None)
    hist.add_argument("--step", type=int, default=5, help="Evaluate every N bars (default 5)")
    hist.add_argument("--retrain-every", type=int, default=25)
    hist.add_argument(
        "--ml-min-p-win",
        type=float,
        default=0.0,
        help="Soft-gate after warmup (0=observe all; e.g. 0.35 to skip low p_win)",
    )

    val = sub.add_parser(
        "historical-pattern-validate",
        help="OOS walk on whitelisted hist15 patterns (no retrain, no live bridge)",
    )
    val.add_argument("--start", default="2026-07-20", help="UTC start YYYY-MM-DD")
    val.add_argument("--end", default="2026-07-22", help="UTC end YYYY-MM-DD (inclusive day)")
    val.add_argument(
        "--from-dataset",
        default=None,
        help="Hist15 dataset dir (default data/ml/hist15_clean)",
    )
    val.add_argument("--out", default=None, help="Output dir (default data/ml/hist15_validate_3d)")
    val.add_argument("--whitelist", default=None, help="Optional successful_patterns.json path")
    val.add_argument("--step", type=int, default=5)
    val.add_argument("--simulate", action="store_true")
    val.add_argument("--max-bars", type=int, default=None)
    val.add_argument("--min-wins", type=int, default=10)
    val.add_argument("--min-wr", type=float, default=0.12)
    val.add_argument(
        "--model-dir",
        default=None,
        help="Hist15 model dir (default models/hist15_clean)",
    )
    val.add_argument(
        "--ml-min-p-win",
        type=float,
        default=0.12,
        help="Soft-gate: skip if model p_win below this (0=score only, no skip)",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    cfg = load_config()

    if args.cmd == "run":
        simulate = bool(args.simulate)
        engine = TradingEngine(cfg, simulate=simulate)
        app = create_app(engine)
        host = args.host or cfg.dashboard.host
        port = args.port or cfg.dashboard.port
        print(f"Monitor: http://{host}:{port}/  (refresh 5s, mode={cfg.mode})")
        uvicorn.run(app, host=host, port=port, log_level="info")

    elif args.cmd == "once":
        engine = TradingEngine(cfg, simulate=bool(args.simulate))
        snap = engine.tick()
        print(snap.model_dump_json(indent=2))

    elif args.cmd == "backtest":
        from trading_system.backtesting import Backtester
        from trading_system.data.crypto import SimulatedCryptoAdapter

        adapter = SimulatedCryptoAdapter(seed=7)
        df = adapter.get_ohlcv("BTC/USDT", limit=args.bars)
        result = Backtester(cfg).run(df, "BTC/USDT")
        print(result.metrics)

    elif args.cmd == "report":
        engine = TradingEngine(cfg, simulate=True)
        path = engine.force_daily_report(args.day)
        print(path)

    elif args.cmd == "rebuild-patterns":
        from trading_system.database import Database
        from trading_system.learning.rebuild import rebuild_patterns

        db = Database(cfg.db_path())
        summary = rebuild_patterns(db, cfg.learning, quiet=True)
        print(summary)

    elif args.cmd == "purge-trades":
        import runpy

        from trading_system.config import ROOT

        script = ROOT / "scripts" / "purge_trades_after.py"
        argv = ["--after", args.after]
        if args.db:
            argv.extend(["--db", args.db])
        if args.dry_run:
            argv.append("--dry-run")
        if args.no_rebuild:
            argv.append("--no-rebuild")
        if args.yes:
            argv.append("--yes")
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")

    elif args.cmd == "reset-loss-learning":
        import runpy

        from trading_system.config import ROOT

        script = ROOT / "scripts" / "reset_loss_learning.py"
        argv = []
        if args.db:
            argv.extend(["--db", args.db])
        if args.dry_run:
            argv.append("--dry-run")
        if args.yes:
            argv.append("--yes")
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")

    elif args.cmd == "historical-ml-run":
        from trading_system.ml.hist_run import run_historical_ml

        manifest = run_historical_ml(
            cfg,
            days=int(args.days),
            out_dir=args.out,
            model_dir=args.model_dir,
            simulate=bool(args.simulate),
            max_bars=args.max_bars,
            step=int(args.step),
            retrain_every=int(args.retrain_every),
            ml_min_p_win=float(args.ml_min_p_win),
        )
        print(json.dumps(manifest, indent=2, default=str))

    elif args.cmd == "historical-pattern-validate":
        from trading_system.ml.validate_patterns import run_pattern_validation

        manifest = run_pattern_validation(
            cfg,
            start=str(args.start),
            end=str(args.end),
            from_dataset=args.from_dataset,
            out_dir=args.out,
            whitelist_path=args.whitelist,
            model_dir=args.model_dir,
            ml_min_p_win=float(args.ml_min_p_win),
            min_wins=int(args.min_wins),
            min_wr=float(args.min_wr),
            step=int(args.step),
            simulate=bool(args.simulate),
            max_bars=args.max_bars,
        )
        print(json.dumps(manifest, indent=2, default=str))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
