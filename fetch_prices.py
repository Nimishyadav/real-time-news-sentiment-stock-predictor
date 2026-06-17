# -*- coding: utf-8 -*-
"""
fetch_prices.py
---------------
Fetches OHLCV (Open, High, Low, Close, Volume) stock price data from
Yahoo Finance via yfinance for NSE and BSE listed stocks.

Features:
  - NSE (.NS) and BSE (.BO) ticker support
  - Multiple intervals: 1d, 1h, 15m
  - SQLite caching - avoids re-downloading unchanged data
  - Derived columns: daily % return, 5-day moving average, volatility
  - Corporate actions: splits and dividends handled automatically by yfinance

Usage:
    python fetch_prices.py                      # all tickers, last 30 days, daily
    python fetch_prices.py --ticker RELIANCE     # single ticker
    python fetch_prices.py --days 90 --interval 1d
    python fetch_prices.py --ticker TCS --interval 1h --days 7
"""

import os
import logging
import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sentiment_cache.db")

WATCHLIST = [
    # Original 8
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "WIPRO.NS",
    "BAJFINANCE.NS",
    "SBIN.NS",
    # Added 12 - Nifty 50 large caps
    "HINDUNILVR.NS",
    "LT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "ITC.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "ADANIENT.NS",
    "ONGC.NS",
]

VALID_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"]

INTERVAL_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "1h": 730, "1d": None, "1wk": None, "1mo": None,
}

# Windows UTF-8 fix -- must run BEFORE any logging handler is created.
import sys as _sys, io as _io, logging as _log

def _make_utf8_handler():
    if hasattr(_sys.stderr, "buffer"):
        stream = _io.TextIOWrapper(
            _sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    else:
        stream = _sys.stderr
    h = _log.StreamHandler(stream)
    h.setFormatter(_log.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    return h

_root_logger = _log.getLogger()
_root_logger.handlers.clear()
_root_logger.addHandler(_make_utf8_handler())
_root_logger.setLevel(_log.INFO)

logging = _log
log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id         TEXT PRIMARY KEY,
            ticker     TEXT NOT NULL,
            datetime   TEXT NOT NULL,
            interval   TEXT NOT NULL,
            open       REAL,
            high       REAL,
            low        REAL,
            close      REAL,
            volume     INTEGER,
            adj_close  REAL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker   ON prices(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price_datetime ON prices(datetime)")
    conn.commit()
    log.info("Database ready: %s", db_path)
    return conn


def save_prices(conn: sqlite3.Connection, df: pd.DataFrame, ticker: str, interval: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for _, row in df.iterrows():
        dt_str = str(row["datetime"])
        pid = f"{ticker}::{dt_str}::{interval}"
        try:
            conn.execute(
                """INSERT OR REPLACE INTO prices
                   (id, ticker, datetime, interval, open, high, low, close, volume, adj_close, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pid, ticker, dt_str, interval,
                    round(float(row.get("open",  0) or 0), 4),
                    round(float(row.get("high",  0) or 0), 4),
                    round(float(row.get("low",   0) or 0), 4),
                    round(float(row.get("close", 0) or 0), 4),
                    int(row.get("volume", 0) or 0),
                    round(float(row.get("adj_close", row.get("close", 0)) or 0), 4),
                    now,
                ),
            )
            inserted += 1
        except sqlite3.Error as exc:
            log.warning("DB insert error for %s: %s", ticker, exc)
    conn.commit()
    return inserted


def load_prices_from_db(
    conn: sqlite3.Connection, ticker: str, interval: str, from_date: datetime
) -> pd.DataFrame:
    df = pd.read_sql_query(
        """SELECT datetime, open, high, low, close, volume, adj_close
           FROM prices
           WHERE ticker=? AND interval=? AND datetime>=?
           ORDER BY datetime ASC""",
        conn,
        params=(ticker, interval, from_date.isoformat()),
    )
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# ---------------------------------------------------------------------------
# yfinance fetch  (fixed for yfinance >= 0.2.x MultiIndex columns)
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance returns a MultiIndex like [('Close','RELIANCE.NS'), ...]
    when downloading a single ticker. This flattens it to plain lowercase
    column names: close, open, high, low, volume, adj close.
    Also renames the Date/Datetime index column -> 'datetime'.
    """
    df = df.copy()

    # Flatten MultiIndex columns - take only the first level (metric name)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower().strip() for col in df.columns]
    else:
        df.columns = [str(c).lower().strip() for c in df.columns]

    # After reset_index the date becomes a regular column named 'date' or 'datetime'
    # (yfinance uses 'Date' for daily, 'Datetime' for intraday)
    for candidate in ("date", "datetime", "index"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "datetime"})
            break

    # Normalise 'adj close' -> 'adj_close'
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})

    return df


def _clamp_days(days: int, interval: str) -> int:
    max_days = INTERVAL_MAX_DAYS.get(interval)
    if max_days and days > max_days:
        log.warning("interval='%s' max=%d days, clamping from %d", interval, max_days, days)
        return max_days
    return days


def fetch_prices_yfinance(
    ticker: str,
    days: int = 30,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance. Returns clean DataFrame or empty DF on failure."""
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}")

    days = _clamp_days(days, interval)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    log.info("[%s] Downloading %d days @ interval=%s ...", ticker, days, interval)

    try:
        raw = yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        log.error("[%s] yfinance download failed: %s", ticker, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        log.warning("[%s] No data returned - check ticker symbol (use RELIANCE.NS not RELIANCE)", ticker)
        return pd.DataFrame()

    # --- Flatten columns and reset index ---
    raw = raw.reset_index()
    raw = _flatten_columns(raw)

    # Confirm 'datetime' column now exists
    if "datetime" not in raw.columns:
        log.error("[%s] Could not find date column. Columns are: %s", ticker, raw.columns.tolist())
        return pd.DataFrame()

    # Ensure UTC timezone on datetime column
    if hasattr(raw["datetime"].dt, "tz") and raw["datetime"].dt.tz is None:
        raw["datetime"] = raw["datetime"].dt.tz_localize("UTC")
    else:
        try:
            raw["datetime"] = raw["datetime"].dt.tz_convert("UTC")
        except TypeError:
            raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)

    # Add adj_close if missing
    if "adj_close" not in raw.columns:
        raw["adj_close"] = raw["close"]

    # Select final columns
    keep = ["datetime", "open", "high", "low", "close", "volume", "adj_close"]
    existing_keep = [c for c in keep if c in raw.columns]
    df = raw[existing_keep].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_values("datetime").reset_index(drop=True)

    log.info("[%s] Got %d rows of OHLCV data", ticker, len(df))
    return df


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add pct_change, MA5, MA20, volatility, and direction labels."""
    df = df.copy()
    df["pct_change"]    = (df["close"].pct_change() * 100).round(4)
    df["ma5"]           = df["close"].rolling(window=5,  min_periods=1).mean().round(2)
    df["ma20"]          = df["close"].rolling(window=20, min_periods=1).mean().round(2)
    df["volatility_5d"] = (
        df["pct_change"].rolling(window=5, min_periods=2).std() * (252 ** 0.5)
    ).round(4)
    df["price_direction"] = df["pct_change"].apply(
        lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0)
    )
    df["next_day_direction"] = df["price_direction"].shift(-1)
    return df


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_prices(
    ticker: str,
    days: int = 30,
    interval: str = "1d",
    conn: Optional[sqlite3.Connection] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    from_date = datetime.now(timezone.utc) - timedelta(days=days)

    if use_cache and conn:
        cached = load_prices_from_db(conn, ticker, interval, from_date)
        if len(cached) >= max(5, days // 2):
            log.info("[%s] Using %d cached price rows", ticker, len(cached))
            return add_derived_columns(cached)

    df = fetch_prices_yfinance(ticker, days=days, interval=interval)
    if df.empty:
        return df

    if conn:
        n = save_prices(conn, df, ticker, interval)
        log.info("[%s] Saved %d price rows to DB", ticker, n)

    return add_derived_columns(df)


def fetch_all_prices(
    watchlist: list = WATCHLIST,
    days: int = 30,
    interval: str = "1d",
    db_path: str = DB_PATH,
) -> dict:
    conn = init_db(db_path)
    results = {}
    for ticker in watchlist:
        df = fetch_prices(ticker, days=days, interval=interval, conn=conn)
        if not df.empty:
            results[ticker] = df
        else:
            log.warning("[%s] Skipped - no data", ticker)
    conn.close()
    return results


def get_price_summary(df: pd.DataFrame, ticker: str) -> dict:
    if df.empty:
        return {}
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    return {
        "ticker":        ticker,
        "last_close":    round(float(latest["close"]), 2),
        "prev_close":    round(float(prev["close"]), 2),
        "pct_change":    round(float(latest["pct_change"]), 2),
        "volume":        int(latest["volume"]),
        "ma5":           round(float(latest["ma5"]), 2),
        "ma20":          round(float(latest["ma20"]), 2),
        "volatility_5d": round(float(latest.get("volatility_5d", 0) or 0), 2),
        "trend":         "up" if latest["pct_change"] > 0 else "down",
        "as_of":         str(latest["datetime"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch NSE/BSE stock price data")
    parser.add_argument("--ticker",   type=str, default=None,
                        help="Ticker without suffix e.g. RELIANCE  (adds .NS automatically)")
    parser.add_argument("--days",     type=int, default=30)
    parser.add_argument("--interval", type=str, default="1d", choices=VALID_INTERVALS)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.ticker:
        key = f"{args.ticker.upper()}.NS"
        conn = init_db()
        df = fetch_prices(key, days=args.days, interval=args.interval,
                          conn=conn, use_cache=not args.no_cache)
        conn.close()

        if df.empty:
            print(f"\nNo data for {key}.")
            print("Tips:")
            print("  - NSE tickers end with .NS  e.g. RELIANCE.NS")
            print("  - BSE tickers end with .BO  e.g. RELIANCE.BO")
            print("  - Run: python fetch_prices.py --ticker RELIANCE")
            return

        summary = get_price_summary(df, key)
        sign = "+" if summary["pct_change"] >= 0 else ""
        print(f"\n{'='*60}")
        print(f"  {key}  |  Last close: Rs.{summary['last_close']}  "
              f"({sign}{summary['pct_change']}%)")
        print(f"  MA5: Rs.{summary['ma5']}  |  MA20: Rs.{summary['ma20']}  "
              f"|  Volume: {summary['volume']:,}")
        print(f"{'='*60}\n")
        cols = ["datetime", "open", "high", "low", "close", "volume", "pct_change", "ma5"]
        print(df[cols].tail(10).to_string(index=False))
        print(f"\nData saved to: {DB_PATH}")

    else:
        data = fetch_all_prices(days=args.days, interval=args.interval)
        print(f"\n{'='*60}")
        print(f"  Fetched data for {len(data)} tickers")
        print(f"{'='*60}")
        for ticker, df in data.items():
            s = get_price_summary(df, ticker)
            sign = "+" if s.get("pct_change", 0) >= 0 else ""
            print(f"  {ticker:<18}  Rs.{s.get('last_close','?'):<10}"
                  f"  {sign}{s.get('pct_change', 0):.2f}%   {len(df)} rows")
        print(f"\nAll data saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
