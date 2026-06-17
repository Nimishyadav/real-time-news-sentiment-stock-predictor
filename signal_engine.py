# -*- coding: utf-8 -*-
"""
signal_engine.py
----------------
Correlates FinBERT sentiment scores (from sentiment_engine.py) with actual
NSE/BSE stock price movements to measure predictive power, run a backtest,
and rank all tickers by current signal strength.

Analysis performed:
  1. Lag correlation  - Pearson r between today's sentiment and price returns
                        at lag 0 (same day), lag 1 (next day), lag 2 (day after)
  2. Backtest         - Walk-forward accuracy: did BULLISH -> price up, BEARISH -> price down?
  3. Rolling accuracy - 20-day rolling window to see if the signal is improving
  4. Ticker ranking   - Score all tickers by weighted_compound + backtest accuracy
  5. CSV export       - Full results saved for the Streamlit dashboard

Output tables added to sentiment_cache.db:
  - correlation_results : lag-0/1/2 Pearson r and p-values per ticker
  - backtest_results    : daily signal vs actual direction per ticker

Usage:
    python signal_engine.py                    # analyse all tickers
    python signal_engine.py --ticker RELIANCE  # single ticker deep-dive
    python signal_engine.py --days 90          # use 90 days of history
    python signal_engine.py --export           # save CSV reports
    python signal_engine.py --min-rows 10      # lower threshold for sparse data
"""

import os
import logging
import argparse
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
# Configuration
# ---------------------------------------------------------------------------

DB_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sentiment_cache.db")
EXPORT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reports")
MIN_ROWS     = 5       # minimum overlapping days needed for any analysis
SIGNAL_UP    = 0.15    # compound score threshold for BULLISH
SIGNAL_DOWN  = -0.15   # compound score threshold for BEARISH
RETURN_UP    = 0.2     # % return threshold to call a day "up"  (+0.2%)
RETURN_DOWN  = -0.2    # % return threshold to call a day "down" (-0.2%)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Add signal analysis tables to the existing DB."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS correlation_results (
            id           TEXT PRIMARY KEY,   -- ticker::computed_date
            ticker       TEXT NOT NULL,
            computed_at  TEXT NOT NULL,
            n_days       INTEGER,            -- overlapping data points used
            lag0_r       REAL,               -- same-day correlation
            lag0_p       REAL,
            lag1_r       REAL,               -- next-day correlation (key metric)
            lag1_p       REAL,
            lag2_r       REAL,               -- day-after-next correlation
            lag2_p       REAL,
            best_lag     INTEGER,            -- lag with highest |r|
            best_r       REAL,
            is_significant INTEGER           -- 1 if best p-value < 0.05
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id               TEXT PRIMARY KEY,   -- ticker::date
            ticker           TEXT NOT NULL,
            date             TEXT NOT NULL,
            sentiment_signal TEXT,               -- BULLISH / BEARISH / NEUTRAL
            weighted_compound REAL,
            actual_return    REAL,               -- next-day % return
            price_direction  INTEGER,            -- +1 / -1 / 0
            signal_correct   INTEGER,            -- 1=correct, 0=wrong, NULL=neutral
            computed_at      TEXT NOT NULL
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_bt_ticker ON backtest_results(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bt_date   ON backtest_results(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cr_ticker ON correlation_results(ticker)")
    conn.commit()
    log.info("DB ready (signal tables): %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sentiment(
    conn: sqlite3.Connection,
    ticker: str,
    days: int,
) -> pd.DataFrame:
    """Load daily_sentiment rows for one ticker."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        """SELECT date, weighted_compound, avg_compound, signal,
                  avg_pos, avg_neg, headline_count
           FROM   daily_sentiment
           WHERE  ticker = ? AND date >= ?
           ORDER BY date ASC""",
        conn, params=(ticker, cutoff),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices(
    conn: sqlite3.Connection,
    ticker: str,
    days: int,
) -> pd.DataFrame:
    """
    Load prices and compute pct_change.
    pct_change is calculated here rather than stored, since fetch_prices.py
    stores raw OHLCV and derives metrics in Python.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days + 5)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        """SELECT datetime, close, volume
           FROM   prices
           WHERE  ticker = ? AND interval = '1d' AND datetime >= ?
           ORDER BY datetime ASC""",
        conn, params=(ticker, cutoff),
    )
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"]     = df["datetime"].dt.normalize()
    # One row per calendar date (take last close if duplicates)
    df = df.sort_values("datetime").drop_duplicates("date", keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    # Compute returns
    df["pct_change"]         = df["close"].pct_change() * 100
    df["next_day_pct"]       = df["pct_change"].shift(-1)   # tomorrow's return
    df["day_after_pct"]      = df["pct_change"].shift(-2)   # day after tomorrow

    df["price_direction"]    = df["pct_change"].apply(
        lambda x: 1 if x > RETURN_UP else (-1 if x < RETURN_DOWN else 0)
    )
    df["next_day_direction"] = df["price_direction"].shift(-1)

    return df


def merge_sentiment_prices(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join sentiment and prices on date.
    Returns combined DataFrame with both sentiment scores and price returns.
    """
    if sentiment_df.empty or price_df.empty:
        return pd.DataFrame()

    # Normalise date columns to UTC midnight for joining
    s = sentiment_df.copy()
    p = price_df.copy()

    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
    if p["date"].dt.tz is None:
        p["date"] = p["date"].dt.tz_localize("UTC")

    merged = pd.merge(s, p[["date", "close", "pct_change", "next_day_pct",
                              "day_after_pct", "price_direction", "next_day_direction"]],
                      on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Lag correlation analysis
# ---------------------------------------------------------------------------

def compute_lag_correlation(
    merged: pd.DataFrame,
    ticker: str,
) -> dict:
    """
    Compute Pearson correlation between weighted_compound and price returns
    at three lags:
      lag 0: sentiment[t] vs return[t]       (same day)
      lag 1: sentiment[t] vs return[t+1]     (next day - most predictive)
      lag 2: sentiment[t] vs return[t+2]     (day after next)

    Returns a dict with r, p-value, n for each lag, plus best_lag info.
    """
    result = {
        "ticker": ticker,
        "n_days": len(merged),
        "lag0_r": None, "lag0_p": None,
        "lag1_r": None, "lag1_p": None,
        "lag2_r": None, "lag2_p": None,
        "best_lag": None, "best_r": None, "is_significant": 0,
    }

    sentiment = merged["weighted_compound"].values

    lag_configs = [
        ("lag0", sentiment,       merged["pct_change"].values),
        ("lag1", sentiment[:-1],  merged["next_day_pct"].values[:-1]),
        ("lag2", sentiment[:-2],  merged["day_after_pct"].values[:-2]),
    ]

    best_abs_r = 0.0

    for lag_name, s_vals, r_vals in lag_configs:
        # Drop NaN pairs
        mask = ~(np.isnan(s_vals) | np.isnan(r_vals))
        s_clean = s_vals[mask]
        r_clean = r_vals[mask]

        if len(s_clean) < MIN_ROWS:
            log.debug("[%s] %s: not enough clean data (%d rows)", ticker, lag_name, len(s_clean))
            continue

        try:
            r_val, p_val = stats.pearsonr(s_clean, r_clean)
            result[f"{lag_name}_r"] = round(float(r_val), 6)
            result[f"{lag_name}_p"] = round(float(p_val), 6)

            if abs(r_val) > best_abs_r:
                best_abs_r = abs(r_val)
                lag_num = int(lag_name[-1])
                result["best_lag"] = lag_num
                result["best_r"]   = round(float(r_val), 6)
                result["is_significant"] = 1 if p_val < 0.05 else 0

        except Exception as exc:
            log.warning("[%s] Correlation failed for %s: %s", ticker, lag_name, exc)

    return result


def save_correlation(conn: sqlite3.Connection, result: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rid = f"{result['ticker']}::{now[:10]}"
    conn.execute(
        """INSERT OR REPLACE INTO correlation_results
           (id, ticker, computed_at, n_days,
            lag0_r, lag0_p, lag1_r, lag1_p, lag2_r, lag2_p,
            best_lag, best_r, is_significant)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rid, result["ticker"], now, result["n_days"],
            result["lag0_r"], result["lag0_p"],
            result["lag1_r"], result["lag1_p"],
            result["lag2_r"], result["lag2_p"],
            result["best_lag"], result["best_r"], result["is_significant"],
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(
    merged: pd.DataFrame,
    ticker: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Walk-forward backtest: for each day t where we have a sentiment signal,
    check whether the NEXT day's price movement matched the signal.

    Rules:
      BULLISH  (compound > +0.15) + next_day_return > +0.2%  -> CORRECT
      BULLISH  (compound > +0.15) + next_day_return < -0.2%  -> WRONG
      BEARISH  (compound < -0.15) + next_day_return < -0.2%  -> CORRECT
      BEARISH  (compound < -0.15) + next_day_return > +0.2%  -> WRONG
      NEUTRAL  -> skipped (not counted in accuracy)

    Saves results to backtest_results table.
    Returns DataFrame of backtest rows.
    """
    if merged.empty or len(merged) < MIN_ROWS:
        log.warning("[%s] Not enough data for backtest (%d rows)", ticker, len(merged))
        return pd.DataFrame()

    now_str = datetime.now(timezone.utc).isoformat()
    rows    = []

    for i, row in merged.iterrows():
        date_str   = str(row["date"])[:10]
        compound   = float(row["weighted_compound"])
        signal     = str(row["signal"])
        next_ret   = row.get("next_day_pct")
        next_dir   = row.get("next_day_direction")

        # Skip if no next-day data
        if pd.isna(next_ret) or pd.isna(next_dir):
            continue

        next_ret = float(next_ret)
        next_dir = int(next_dir)

        # Evaluate correctness
        if signal == "BULLISH":
            correct = 1 if next_dir == 1 else (0 if next_dir == -1 else None)
        elif signal == "BEARISH":
            correct = 1 if next_dir == -1 else (0 if next_dir == 1 else None)
        else:
            correct = None   # NEUTRAL - not counted

        row_id = f"{ticker}::{date_str}"
        rows.append({
            "id":               row_id,
            "ticker":           ticker,
            "date":             date_str,
            "sentiment_signal": signal,
            "weighted_compound": compound,
            "actual_return":    round(next_ret, 4),
            "price_direction":  next_dir,
            "signal_correct":   correct,
            "computed_at":      now_str,
        })

    if not rows:
        return pd.DataFrame()

    # Persist to DB
    for r in rows:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO backtest_results
                   (id, ticker, date, sentiment_signal, weighted_compound,
                    actual_return, price_direction, signal_correct, computed_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    r["id"], r["ticker"], r["date"],
                    r["sentiment_signal"], r["weighted_compound"],
                    r["actual_return"], r["price_direction"],
                    r["signal_correct"], r["computed_at"],
                ),
            )
        except sqlite3.Error as exc:
            log.warning("Backtest DB error for %s: %s", r["id"], exc)
    conn.commit()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def compute_accuracy_metrics(bt_df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute overall and per-signal accuracy metrics from a backtest DataFrame.

    Returns dict with:
      overall_accuracy   : % of non-neutral signals that were correct
      bullish_accuracy   : accuracy of BULLISH signals specifically
      bearish_accuracy   : accuracy of BEARISH signals specifically
      total_signals      : count of non-neutral signals
      correct_signals    : count of correct non-neutral signals
      precision_score    : harmonic mean of bullish and bearish accuracy
    """
    if bt_df.empty:
        return {"ticker": ticker, "overall_accuracy": None, "total_signals": 0}

    # Filter to only actionable (non-neutral) signals with a definitive outcome
    actionable = bt_df[
        (bt_df["sentiment_signal"] != "NEUTRAL") &
        (bt_df["signal_correct"].notna())
    ].copy()

    if actionable.empty:
        return {"ticker": ticker, "overall_accuracy": None, "total_signals": 0}

    total   = len(actionable)
    correct = int(actionable["signal_correct"].sum())
    overall = correct / total if total > 0 else 0.0

    # Per-signal breakdown
    bullish = actionable[actionable["sentiment_signal"] == "BULLISH"]
    bearish = actionable[actionable["sentiment_signal"] == "BEARISH"]

    bull_acc = float(bullish["signal_correct"].mean()) if len(bullish) > 0 else None
    bear_acc = float(bearish["signal_correct"].mean()) if len(bearish) > 0 else None

    # Precision score: harmonic mean (penalises if one direction is much worse)
    if bull_acc is not None and bear_acc is not None and (bull_acc + bear_acc) > 0:
        precision = 2 * bull_acc * bear_acc / (bull_acc + bear_acc)
    else:
        precision = overall

    return {
        "ticker":            ticker,
        "overall_accuracy":  round(overall, 4),
        "bullish_accuracy":  round(bull_acc, 4) if bull_acc is not None else None,
        "bearish_accuracy":  round(bear_acc, 4) if bear_acc is not None else None,
        "total_signals":     total,
        "correct_signals":   correct,
        "neutral_signals":   int((bt_df["sentiment_signal"] == "NEUTRAL").sum()),
        "precision_score":   round(precision, 4),
    }


def compute_rolling_accuracy(bt_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling N-day accuracy to see if the signal is improving over time.
    Returns DataFrame with columns: date, rolling_accuracy, signal_count.
    """
    if bt_df.empty:
        return pd.DataFrame()

    actionable = bt_df[
        (bt_df["sentiment_signal"] != "NEUTRAL") &
        (bt_df["signal_correct"].notna())
    ].copy()

    if len(actionable) < window:
        window = max(3, len(actionable) // 2)

    actionable = actionable.sort_values("date").reset_index(drop=True)
    actionable["rolling_accuracy"] = (
        actionable["signal_correct"]
        .rolling(window=window, min_periods=max(3, window // 2))
        .mean()
    )
    actionable["signal_count"] = (
        actionable["signal_correct"]
        .rolling(window=window, min_periods=1)
        .count()
        .astype(int)
    )
    return actionable[["date", "rolling_accuracy", "signal_count"]].dropna()


# ---------------------------------------------------------------------------
# Multi-ticker ranking
# ---------------------------------------------------------------------------

def rank_tickers(
    conn: sqlite3.Connection,
    accuracy_map: dict[str, dict],
    days: int = 7,
) -> pd.DataFrame:
    """
    Rank all tickers by a composite score combining:
      - Latest weighted_compound (current sentiment strength)
      - Backtest overall_accuracy (historical signal reliability)
      - Headline count (confidence in the sentiment reading)

    Composite = 0.5 * |compound| * sign(compound)   [direction + strength]
               + 0.3 * (accuracy - 0.5)              [above-random accuracy]
               + 0.2 * min(headline_count / 10, 1)   [data confidence, capped at 10]

    Returns ranked DataFrame sorted by composite score descending.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    latest = pd.read_sql_query(
        """SELECT ds.*
           FROM   daily_sentiment ds
           INNER JOIN (
               SELECT ticker, MAX(date) AS max_date
               FROM   daily_sentiment
               WHERE  date >= ?
               GROUP BY ticker
           ) latest ON ds.ticker = latest.ticker AND ds.date = latest.max_date
           ORDER BY ds.date DESC""",
        conn, params=(cutoff,),
    )

    if latest.empty:
        log.warning("No recent daily_sentiment data for ranking")
        return pd.DataFrame()

    rows = []
    for _, row in latest.iterrows():
        ticker   = row["ticker"]
        compound = float(row["weighted_compound"])
        n_heads  = int(row["headline_count"])
        signal   = str(row["signal"])
        acc_data = accuracy_map.get(ticker, {})
        accuracy = acc_data.get("overall_accuracy") or 0.5  # default: coin-flip

        # Composite score
        direction_strength = compound  # already in [-1, +1] with direction embedded
        accuracy_bonus     = accuracy - 0.5   # positive if better than random
        data_confidence    = min(n_heads / 10, 1.0)

        composite = (
            0.5 * direction_strength +
            0.3 * accuracy_bonus +
            0.2 * data_confidence
        )

        rows.append({
            "ticker":           ticker,
            "signal":           signal,
            "compound":         round(compound, 4),
            "accuracy":         round(accuracy, 4),
            "headline_count":   n_heads,
            "composite_score":  round(composite, 4),
            "as_of_date":       str(row["date"])[:10],
            "total_signals":    acc_data.get("total_signals", 0),
        })

    ranked = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_reports(
    correlation_map: dict,
    accuracy_map: dict,
    ranked_df: pd.DataFrame,
    backtest_map: dict[str, pd.DataFrame],
    export_dir: str = EXPORT_DIR,
) -> list[str]:
    """Save all analysis results to CSV files for the Streamlit dashboard."""
    os.makedirs(export_dir, exist_ok=True)
    today   = datetime.now(timezone.utc).strftime("%Y%m%d")
    files   = []

    # 1. Correlation results
    corr_rows = list(correlation_map.values())
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        path = os.path.join(export_dir, f"correlation_{today}.csv")
        corr_df.to_csv(path, index=False)
        files.append(path)
        log.info("Exported: %s", path)

    # 2. Accuracy summary
    acc_rows = list(accuracy_map.values())
    if acc_rows:
        acc_df = pd.DataFrame(acc_rows)
        path = os.path.join(export_dir, f"accuracy_{today}.csv")
        acc_df.to_csv(path, index=False)
        files.append(path)
        log.info("Exported: %s", path)

    # 3. Ticker ranking
    if not ranked_df.empty:
        path = os.path.join(export_dir, f"ranking_{today}.csv")
        ranked_df.to_csv(path, index=False)
        files.append(path)
        log.info("Exported: %s", path)

    # 4. Full backtest detail (all tickers combined)
    all_bt = pd.concat(
        [df for df in backtest_map.values() if not df.empty],
        ignore_index=True,
    ) if backtest_map else pd.DataFrame()

    if not all_bt.empty:
        path = os.path.join(export_dir, f"backtest_{today}.csv")
        all_bt.to_csv(path, index=False)
        files.append(path)
        log.info("Exported: %s", path)

    return files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_signal_pipeline(
    tickers: Optional[list[str]] = None,
    days: int = 90,
    export: bool = False,
    min_rows: int = MIN_ROWS,
    db_path: str = DB_PATH,
) -> dict:
    """
    Full signal analysis pipeline for all tickers (or a subset).

    Returns dict:
      {
        'correlations': {ticker: corr_dict},
        'accuracy':     {ticker: acc_dict},
        'backtests':    {ticker: bt_df},
        'ranking':      ranked_df,
      }
    """
    conn = init_db(db_path)

    # Discover tickers from DB if not specified
    if tickers is None:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM daily_sentiment ORDER BY ticker"
        ).fetchall()
        tickers = [r[0] for r in rows]

    if not tickers:
        log.warning("No tickers found in daily_sentiment. Run sentiment_engine.py first.")
        conn.close()
        return {}

    log.info("Running signal analysis for %d ticker(s): %s", len(tickers), tickers)

    correlation_map = {}
    accuracy_map    = {}
    backtest_map    = {}

    for ticker in tickers:
        log.info("[%s] Loading data...", ticker)

        sentiment_df = load_sentiment(conn, ticker, days)
        price_df     = load_prices(conn, ticker, days)

        if sentiment_df.empty:
            log.warning("[%s] No sentiment data - run sentiment_engine.py first", ticker)
            continue

        if price_df.empty:
            log.warning("[%s] No price data - run fetch_prices.py first", ticker)
            # Still compute signal metrics without price correlation
            continue

        merged = merge_sentiment_prices(sentiment_df, price_df)

        if len(merged) < min_rows:
            log.warning(
                "[%s] Only %d overlapping days (need %d) - skipping correlation",
                ticker, len(merged), min_rows,
            )
            continue

        log.info("[%s] %d overlapping sentiment+price days", ticker, len(merged))

        # --- Lag correlation ---
        corr = compute_lag_correlation(merged, ticker)
        correlation_map[ticker] = corr
        save_correlation(conn, corr)
        log.info(
            "[%s] Lag correlation: lag0=%.3f  lag1=%.3f  lag2=%.3f  best=lag%s (r=%.3f, sig=%s)",
            ticker,
            corr.get("lag0_r") or 0,
            corr.get("lag1_r") or 0,
            corr.get("lag2_r") or 0,
            corr.get("best_lag"),
            corr.get("best_r") or 0,
            "YES" if corr.get("is_significant") else "NO",
        )

        # --- Backtest ---
        bt_df = run_backtest(merged, ticker, conn)
        backtest_map[ticker] = bt_df

        # --- Accuracy metrics ---
        acc = compute_accuracy_metrics(bt_df, ticker)
        accuracy_map[ticker] = acc
        log.info(
            "[%s] Backtest: accuracy=%.1f%%  signals=%d  correct=%d  (bull=%.1f%%  bear=%.1f%%)",
            ticker,
            (acc.get("overall_accuracy") or 0) * 100,
            acc.get("total_signals", 0),
            acc.get("correct_signals", 0),
            (acc.get("bullish_accuracy") or 0) * 100,
            (acc.get("bearish_accuracy") or 0) * 100,
        )

    # --- Multi-ticker ranking ---
    ranked_df = rank_tickers(conn, accuracy_map, days=days)

    # --- Export ---
    if export:
        files = export_reports(correlation_map, accuracy_map, ranked_df, backtest_map)
        log.info("Exported %d report files", len(files))

    conn.close()

    return {
        "correlations": correlation_map,
        "accuracy":     accuracy_map,
        "backtests":    backtest_map,
        "ranking":      ranked_df,
    }


# ---------------------------------------------------------------------------
# Dashboard helpers (used by app.py)
# ---------------------------------------------------------------------------

def get_signal_ranking(conn: sqlite3.Connection, days: int = 7) -> pd.DataFrame:
    """Quick ranking read for Streamlit dashboard - no re-computation."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return pd.read_sql_query(
        """SELECT ds.ticker, ds.date, ds.weighted_compound, ds.signal,
                  ds.headline_count, ds.avg_pos, ds.avg_neg
           FROM   daily_sentiment ds
           INNER JOIN (
               SELECT ticker, MAX(date) AS max_date
               FROM   daily_sentiment WHERE date >= ?
               GROUP BY ticker
           ) latest ON ds.ticker=latest.ticker AND ds.date=latest.max_date
           ORDER BY ds.weighted_compound DESC""",
        conn, params=(cutoff,),
    )


def get_backtest_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Aggregate backtest accuracy per ticker for the dashboard."""
    return pd.read_sql_query(
        """SELECT ticker,
                  COUNT(*) AS total_rows,
                  SUM(CASE WHEN sentiment_signal != 'NEUTRAL' THEN 1 ELSE 0 END) AS total_signals,
                  SUM(CASE WHEN signal_correct = 1 THEN 1 ELSE 0 END) AS correct,
                  ROUND(
                      100.0 * SUM(CASE WHEN signal_correct=1 THEN 1.0 ELSE 0 END)
                      / NULLIF(SUM(CASE WHEN sentiment_signal!='NEUTRAL'
                                        AND signal_correct IS NOT NULL
                                   THEN 1 ELSE 0 END), 0),
                      1
                  ) AS accuracy_pct
           FROM   backtest_results
           GROUP BY ticker
           ORDER BY accuracy_pct DESC NULLS LAST""",
        conn,
    )


def get_correlation_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Latest correlation results per ticker for the dashboard."""
    return pd.read_sql_query(
        """SELECT cr.*
           FROM   correlation_results cr
           INNER JOIN (
               SELECT ticker, MAX(computed_at) AS max_at
               FROM   correlation_results GROUP BY ticker
           ) latest ON cr.ticker=latest.ticker AND cr.computed_at=latest.max_at
           ORDER BY ABS(cr.best_r) DESC NULLS LAST""",
        conn,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for correlation/accuracy display."""
    filled = int(abs(value) * width)
    bar    = "#" * filled + "." * (width - filled)
    sign   = "+" if value >= 0 else "-"
    return f"{sign}[{bar}]"


def main():
    parser = argparse.ArgumentParser(description="Signal analysis and backtest engine")
    parser.add_argument("--ticker",   type=str, default=None,
                        help="Single ticker e.g. RELIANCE (adds .NS)")
    parser.add_argument("--days",     type=int, default=90,
                        help="Days of history to analyse (default: 90)")
    parser.add_argument("--export",   action="store_true",
                        help="Export CSV reports to data/reports/")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS,
                        help=f"Min overlapping days for analysis (default: {MIN_ROWS})")
    args = parser.parse_args()

    tickers = [f"{args.ticker.upper()}.NS"] if args.ticker else None

    results = run_signal_pipeline(
        tickers=tickers,
        days=args.days,
        export=args.export,
        min_rows=args.min_rows,
    )

    if not results:
        print("\nNo results - make sure you have run:")
        print("  1. python fetch_news.py")
        print("  2. python fetch_prices.py")
        print("  3. python sentiment_engine.py")
        print("  4. python signal_engine.py")
        return

    W = 70

    # -- Lag Correlation ----------------------------------------------------
    print(f"\n{'='*W}")
    print("  LAG CORRELATION ANALYSIS")
    print(f"  (Pearson r: sentiment compound vs price % return)")
    print(f"{'='*W}")
    print(f"  {'Ticker':<18}  {'Lag-0':>8}  {'Lag-1':>8}  {'Lag-2':>8}  {'Best':>6}  {'Sig?':>5}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*5}")

    for ticker, corr in results.get("correlations", {}).items():
        lag0 = f"{corr.get('lag0_r') or 0:+.4f}"
        lag1 = f"{corr.get('lag1_r') or 0:+.4f}"
        lag2 = f"{corr.get('lag2_r') or 0:+.4f}"
        best = f"lag{corr.get('best_lag','?')}"
        sig  = "OK" if corr.get("is_significant") else "X"
        n    = corr.get("n_days", 0)
        print(f"  {ticker:<18}  {lag0:>8}  {lag1:>8}  {lag2:>8}  {best:>6}  {sig:>5}  (n={n})")

    if not results.get("correlations"):
        print("  No correlation data - need more overlapping sentiment+price days")
        print("  Tip: run fetch_news.py --days 30 to get more headlines")

    # -- Backtest Accuracy --------------------------------------------------
    print(f"\n{'='*W}")
    print("  BACKTEST ACCURACY")
    print(f"  (Did today's signal predict tomorrow's direction?)")
    print(f"{'='*W}")
    print(f"  {'Ticker':<18}  {'Overall':>8}  {'Bullish':>8}  {'Bearish':>8}  {'Signals':>8}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for ticker, acc in results.get("accuracy", {}).items():
        overall = f"{(acc.get('overall_accuracy') or 0)*100:.1f}%"
        bull    = f"{(acc.get('bullish_accuracy') or 0)*100:.1f}%" if acc.get('bullish_accuracy') is not None else "  N/A"
        bear    = f"{(acc.get('bearish_accuracy') or 0)*100:.1f}%" if acc.get('bearish_accuracy') is not None else "  N/A"
        nsig    = acc.get("total_signals", 0)
        ncorr   = acc.get("correct_signals", 0)
        print(f"  {ticker:<18}  {overall:>8}  {bull:>8}  {bear:>8}  {ncorr}/{nsig:>5}")

    if not results.get("accuracy"):
        print("  No backtest data available yet")

    # -- Ticker Ranking -----------------------------------------------------
    ranked = results.get("ranking", pd.DataFrame())
    if not ranked.empty:
        print(f"\n{'='*W}")
        print("  TICKER RANKING  (composite: sentiment strength + accuracy + confidence)")
        print(f"{'='*W}")
        print(f"  {'#':<3}  {'Ticker':<18}  {'Signal':<8}  {'Compound':>9}  {'Accuracy':>9}  {'Score':>7}")
        print(f"  {'-'*3}  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}")

        for _, row in ranked.iterrows():
            icon     = {"BULLISH": "[+]", "BEARISH": "[-]", "NEUTRAL": "[=]"}.get(row["signal"], "?")
            compound = f"{row['compound']:+.4f}"
            accuracy = f"{row['accuracy']*100:.1f}%" if row['accuracy'] else "   N/A"
            score    = f"{row['composite_score']:+.4f}"
            print(f"  {int(row['rank']):<3}  {row['ticker']:<18}  {icon} {row['signal']:<7}  {compound:>9}  {accuracy:>9}  {score:>7}")

    # -- Interpretation guide -----------------------------------------------
    print(f"\n{'='*W}")
    print("  HOW TO READ THESE RESULTS")
    print(f"{'='*W}")
    print("  Lag-1 r > +0.20 with p < 0.05  -> sentiment leads price (predictive)")
    print("  Lag-1 r near 0                  -> sentiment not correlated with returns")
    print("  Accuracy > 55%                  -> better than random (coin flip = 50%)")
    print("  Accuracy > 60%                  -> strong signal worth trading attention")
    print("  Composite score > +0.20         -> strong bullish candidate")
    print("  Composite score < -0.20         -> strong bearish candidate")
    print()
    print("  NOTE: This is for educational purposes only.")
    print("  Sentiment alone is one input - not a trading system.")
    print(f"{'='*W}")

    if args.export:
        print(f"\n  Reports saved to: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
