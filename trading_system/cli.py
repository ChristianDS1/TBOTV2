"""CLI."""

from __future__ import annotations

import argparse
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

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
