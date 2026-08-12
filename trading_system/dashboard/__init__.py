"""FastAPI dashboard — monitor refresh every 5s (client poll + WS)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from trading_system.engine import TradingEngine, get_engine

STATIC = Path(__file__).parent / "static"


def create_app(engine: TradingEngine | None = None) -> FastAPI:
    app = FastAPI(title="T-BOT Adaptive Monitor", version="0.1.0")
    eng = engine

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal eng
        if eng is None:
            eng = get_engine()
        eng.start_background()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if eng:
            eng.stop()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/monitor")
    async def monitor() -> dict:
        assert eng is not None
        return eng.get_monitor_payload()

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "mode": eng.cfg.mode if eng else None}

    @app.post("/api/kill")
    async def kill(reason: str = "manual") -> dict:
        assert eng is not None
        eng.risk.trip(reason)
        return {"kill_switch": True, "reason": reason}

    @app.post("/api/resume")
    async def resume() -> dict:
        assert eng is not None
        eng.risk.reset_kill()
        return {"kill_switch": False}

    @app.post("/api/report")
    async def report(day: str | None = None) -> dict:
        assert eng is not None
        path = eng.force_daily_report(day)
        return {"path": path, "day": day}

    @app.websocket("/ws/monitor")
    async def ws_monitor(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                assert eng is not None
                await ws.send_json(eng.get_monitor_payload())
                await asyncio.sleep(5)
        except WebSocketDisconnect:
            return

    return app
