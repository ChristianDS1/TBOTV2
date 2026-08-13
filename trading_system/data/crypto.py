"""CCXT crypto market adapter — live OHLCV, paper fills elsewhere."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd

from trading_system.data import MarketAdapter
from trading_system.types import Venue

logger = logging.getLogger(__name__)


class CryptoAdapter(MarketAdapter):
    venue = Venue.CRYPTO

    def __init__(self, exchange_id: str = "binance", sandbox: bool = False) -> None:
        exchange_cls = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_cls({"enableRateLimit": True})
        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)
        self._last_bar_time: dict[str, datetime] = {}
        self._last_error: str | None = None
        self._ok = True

    def get_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 200
    ) -> pd.DataFrame:
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            self._ok = True
            self._last_error = None
        except Exception as e:
            self._ok = False
            self._last_error = str(e)
            logger.exception("crypto OHLCV failed for %s", symbol)
            raise

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if not df.empty:
            self._last_bar_time[symbol] = df["timestamp"].iloc[-1].to_pydatetime()
        return df

    def get_last_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        price = float(ticker.get("last") or ticker.get("close") or 0)
        if price <= 0:
            raise ValueError(f"Invalid price for {symbol}")
        return price

    def is_tradable_now(self, symbol: str, ts: datetime | None = None) -> bool:
        return self._ok

    def health(self) -> dict[str, Any]:
        return {
            "venue": "crypto",
            "ok": self._ok,
            "error": self._last_error,
            "last_bars": {
                k: v.isoformat() for k, v in self._last_bar_time.items()
            },
        }


class SimulatedCryptoAdapter(MarketAdapter):
    """Deterministic OHLCV for offline tests / no-network runs."""

    venue = Venue.CRYPTO

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._prices: dict[str, float] = {"BTC/USDT": 65000.0, "ETH/USDT": 3500.0}

    def get_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 200
    ) -> pd.DataFrame:
        import numpy as np

        rng = np.random.default_rng(self.seed + hash(symbol) % 10_000)
        base = self._prices.get(symbol, 100.0)
        rets = rng.normal(0, 0.0008, size=limit)
        closes = base * np.cumprod(1 + rets)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.001, limit))
        lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.001, limit))
        vol = rng.uniform(10, 100, limit)
        freq_map = {
            "1s": "1s",
            "15s": "15s",
            "30s": "30s",
            "1m": "1min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "60m": "1h",
        }
        freq = freq_map.get(timeframe, "1min")
        now = pd.Timestamp.now(tz="UTC").floor("min")
        ts = pd.date_range(end=now, periods=limit, freq=freq, tz="UTC")
        self._prices[symbol] = float(closes[-1])
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": vol,
            }
        )

    def get_last_price(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, limit=5)
        return float(df["close"].iloc[-1])

    def is_tradable_now(self, symbol: str, ts: datetime | None = None) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {"venue": "crypto", "ok": True, "simulated": True}
