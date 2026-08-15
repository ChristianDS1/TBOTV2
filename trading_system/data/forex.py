"""Forex OHLCV adapter with session calendar gating."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trading_system.config import ForexSessionConfig
from trading_system.data import MarketAdapter, SessionCalendar
from trading_system.types import Venue

logger = logging.getLogger(__name__)

YF_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
}

YF_INTERVAL = {
    "1m": ("1m", "5d"),
    "5m": ("5m", "5d"),
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h": ("60m", "60d"),
    "60m": ("60m", "60d"),
}

_SYN_FREQ = {
    "1s": "1s",
    "15s": "15s",
    "30s": "30s",
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "60m": "1h",
}


class ForexAdapter(MarketAdapter):
    venue = Venue.FOREX

    def __init__(
        self,
        session_cfg: ForexSessionConfig,
        provider: str = "yfinance",
    ) -> None:
        self.calendar = SessionCalendar(session_cfg)
        self.provider = provider
        self._last_error: str | None = None
        self._ok = True
        self._cache: dict[str, pd.DataFrame] = {}
        self._last_fetch: dict[str, datetime] = {}

    def get_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 200
    ) -> pd.DataFrame:
        from trading_system.learning.sessions import is_weekend_utc

        tf = timeframe or "1m"
        cache_key = f"{symbol}:{tf}"
        try:
            if tf in ("15s", "30s", "1s"):
                raise ValueError(f"forex has no native {tf}")
            # Weekends: no live FX tape — avoid synthetic unless caller needs marks
            if is_weekend_utc() and self.provider == "yfinance" and tf == "1m":
                df = self._fetch_yfinance(symbol, tf, limit)
            elif self.provider == "yfinance":
                df = self._fetch_yfinance(symbol, tf, limit)
            else:
                df = self._synthetic(symbol, limit, tf)
            self._ok = True
            self._last_error = None
            self._cache[cache_key] = df
            if tf == "1m":
                self._cache[symbol] = df
            self._last_fetch[symbol] = datetime.now(timezone.utc)
            return df
        except Exception as e:
            self._ok = False
            self._last_error = str(e)
            if tf != "1m":
                logger.debug("forex OHLCV %s %s failed: %s", symbol, tf, e)
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            logger.warning("forex OHLCV failed for %s: %s — using synthetic", symbol, e)
            return self._synthetic(symbol, limit, "1m")

    def _fetch_yfinance(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        import yfinance as yf

        ticker = YF_MAP.get(symbol, symbol.replace("/", "") + "=X")
        interval, period = YF_INTERVAL.get(timeframe, ("1m", "5d"))
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            raise ValueError(f"No yfinance data for {ticker} {interval}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        ts_col = "Datetime" if "Datetime" in data.columns else "Date"
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data[ts_col], utc=True),
                "open": data["Open"].astype(float).values,
                "high": data["High"].astype(float).values,
                "low": data["Low"].astype(float).values,
                "close": data["Close"].astype(float).values,
                "volume": data["Volume"].fillna(0).astype(float).values
                if "Volume" in data.columns
                else 0.0,
            }
        )
        return df.tail(limit).reset_index(drop=True)

    def _synthetic(self, symbol: str, limit: int, timeframe: str = "1m") -> pd.DataFrame:
        bases = {
            "EUR/USD": 1.08,
            "GBP/USD": 1.27,
            "USD/JPY": 150.0,
            "AUD/USD": 0.66,
            "NZD/USD": 0.60,
            "USD/CAD": 1.36,
            "USD/CHF": 0.88,
            "EUR/GBP": 0.86,
            "EUR/JPY": 162.0,
            "GBP/JPY": 189.0,
        }
        base = bases.get(symbol, 1.0)
        rng = np.random.default_rng(abs(hash(symbol)) % 10_000)
        rets = rng.normal(0, 0.00015, size=limit)
        closes = base * np.cumprod(1 + rets)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.0003, limit))
        lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.0003, limit))
        freq = _SYN_FREQ.get(timeframe, "1min")
        now = pd.Timestamp.now(tz="UTC").floor("min")
        ts = pd.date_range(end=now, periods=limit, freq=freq, tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": rng.uniform(100, 1000, limit),
            }
        )

    def get_last_price(self, symbol: str) -> float:
        df = self._cache.get(symbol)
        if df is None or df.empty:
            df = self.get_ohlcv(symbol, limit=5)
        return float(df["close"].iloc[-1])

    def is_tradable_now(self, symbol: str, ts: datetime | None = None) -> bool:
        return self.calendar.is_open(ts) and self._ok

    def health(self) -> dict[str, Any]:
        sess = self.calendar.status()
        return {
            "venue": "forex",
            "ok": self._ok,
            "error": self._last_error,
            "provider": self.provider,
            **sess,
            "last_fetch": {k: v.isoformat() for k, v in self._last_fetch.items()},
        }
