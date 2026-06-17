# -*- coding: utf-8 -*-
"""
sentiment_engine.py
--------------------
Runs FinBERT (ProsusAI/finbert) sentiment inference on financial headlines
fetched by fetch_news.py, then aggregates scores per ticker per day using
exponential time-decay weighting.

Pipeline:
  1. Load unscored headlines from SQLite (headlines table)
  2. Run FinBERT in batches -> positive / negative / neutral probabilities
  3. Compute a single compound score per headline  [-1.0 ... +1.0]
  4. Persist raw scores to `headline_sentiment` table (never re-scores)
  5. Aggregate into daily ticker-level scores -> `daily_sentiment` table
  6. Exponential decay: recent headlines weighted higher (half-life = 24 h)

Output tables added to sentiment_cache.db:
  - headline_sentiment  : one row per headline with pos/neg/neu scores
  - daily_sentiment     : one row per (ticker, date) with aggregated score

Usage:
    python sentiment_engine.py                    # score all pending headlines
    python sentiment_engine.py --ticker RELIANCE  # one ticker only
    python sentiment_engine.py --days 7           # re-aggregate last 7 days
    python sentiment_engine.py --device cpu       # force CPU (default: auto)
    python sentiment_engine.py --batch-size 8     # smaller batch for low RAM
"""

import os
import logging
import argparse
import sqlite3
import math
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

# Suppress HuggingFace progress bars in production
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ---------------------------------------------------------------------------
# Windows UTF-8 fix -- must run BEFORE any logging handler is created.
# Replaces default cp1252 stderr handler with explicit utf-8 handler.
# ---------------------------------------------------------------------------
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
# Constants
# ---------------------------------------------------------------------------

DB_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sentiment_cache.db")
FINBERT_MODEL = "ProsusAI/finbert"   # 3-class: positive / negative / neutral
MAX_TOKENS    = 512                   # FinBERT's context window
DEFAULT_BATCH = 16                    # headlines per inference call
DECAY_HALF_LIFE_HOURS = 24            # weight halves every 24 hours


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Extend the existing DB with sentiment tables.
    Safe to call on a DB that already has the headlines / prices tables.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # --- Per-headline scores (raw FinBERT output) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS headline_sentiment (
            headline_id  TEXT PRIMARY KEY,   -- FK -> headlines.id
            ticker       TEXT NOT NULL,
            title        TEXT NOT NULL,
            published    TEXT,
            pos_score    REAL NOT NULL,      -- FinBERT positive probability [0,1]
            neg_score    REAL NOT NULL,      -- FinBERT negative probability [0,1]
            neu_score    REAL NOT NULL,      -- FinBERT neutral  probability [0,1]
            compound     REAL NOT NULL,      -- pos_score - neg_score  in [-1, +1]
            label        TEXT NOT NULL,      -- dominant label string
            scored_at    TEXT NOT NULL       -- UTC ISO-8601
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hs_ticker ON headline_sentiment(ticker)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hs_pub ON headline_sentiment(published)"
    )

    # --- Daily aggregated sentiment per ticker ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id              TEXT PRIMARY KEY,  -- ticker::date
            ticker          TEXT NOT NULL,
            date            TEXT NOT NULL,     -- YYYY-MM-DD UTC
            avg_compound    REAL NOT NULL,     -- simple mean of compound scores
            weighted_compound REAL NOT NULL,   -- exponential-decay weighted mean
            avg_pos         REAL NOT NULL,
            avg_neg         REAL NOT NULL,
            avg_neu         REAL NOT NULL,
            headline_count  INTEGER NOT NULL,
            signal          TEXT NOT NULL,     -- BULLISH / BEARISH / NEUTRAL
            computed_at     TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ds_ticker ON daily_sentiment(ticker)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ds_date ON daily_sentiment(date)"
    )

    conn.commit()
    log.info("DB ready (sentiment tables): %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# FinBERT model loader
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}   # module-level cache so we load once per process

def load_finbert(device: str = "auto") -> object:
    """
    Load the FinBERT pipeline. Cached after first call.

    device: "auto" -> GPU if available, else CPU
            "cpu"  -> force CPU
            "cuda" -> force GPU (raises if unavailable)
    """
    import torch
    from transformers import pipeline as hf_pipeline

    global _pipeline_cache
    if device in _pipeline_cache:
        return _pipeline_cache[device]

    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device

    log.info("Loading FinBERT (%s) on device='%s' ...", FINBERT_MODEL, resolved)

    finbert = hf_pipeline(
        task="text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        top_k=None,              # replaces return_all_scores=True (removed in transformers 5.x)
        device=0 if resolved == "cuda" else -1,
        truncation=True,
        max_length=MAX_TOKENS,
    )

    log.info("FinBERT loaded successfully.")
    _pipeline_cache[device] = finbert
    return finbert


# ---------------------------------------------------------------------------
# Headline scoring
# ---------------------------------------------------------------------------

def _scores_to_dict(raw_scores) -> dict:
    """
    Convert FinBERT output for one headline to a standard score dict.

    With top_k=None, both transformers 4.x and 5.x return:
        [{'label': 'positive', 'score': 0.88},
         {'label': 'negative', 'score': 0.04},
         {'label': 'neutral',  'score': 0.08}]

    Returns: {'pos_score', 'neg_score', 'neu_score', 'compound', 'label'}
    """
    # Build {label: score} mapping - handle both list-of-dicts and plain dict
    if isinstance(raw_scores, dict):
        scores = {k: float(v) for k, v in raw_scores.items() if isinstance(v, (int, float))}
    else:
        scores = {item["label"]: float(item["score"]) for item in raw_scores}

    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    neu = scores.get("neutral",  0.0)

    # Compound: net sentiment direction normalised to [-1, +1]
    compound = round(pos - neg, 6)

    # Dominant label
    label = max(scores, key=scores.get)

    return {
        "pos_score": round(pos, 6),
        "neg_score": round(neg, 6),
        "neu_score": round(neu, 6),
        "compound":  compound,
        "label":     label,
    }


def score_headlines_batch(
    texts: list[str],
    finbert,
    batch_size: int = DEFAULT_BATCH,
) -> list[dict]:
    """
    Run FinBERT on a list of headline texts in batches.

    Returns a list of score dicts (same order as input texts).
    Handles errors per-batch gracefully - failed batches get neutral scores.
    """
    results = []
    total   = len(texts)
    n_batches = math.ceil(total / batch_size)

    for i in range(n_batches):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        # Truncate each text to avoid tokeniser warnings
        batch = [t[:2000] for t in batch]

        try:
            raw = finbert(batch)

            # Normalise output format:
            # transformers 4.x returns list-of-lists: [[{label,score},...], ...]
            # transformers 5.x returns flat list of dicts: [{'positive':0.8,...}, ...]
            # After normalisation, each element is what _scores_to_dict expects.
            if raw and isinstance(raw[0], dict):
                # 5.x format - already one dict per headline
                normalised = raw
            else:
                # 4.x format - one list-of-dicts per headline
                normalised = raw

            for item_scores in normalised:
                results.append(_scores_to_dict(item_scores))

        except Exception as exc:
            log.warning("Batch %d/%d failed (%s) - using neutral fallback", i + 1, n_batches, exc)
            for _ in batch:
                results.append({
                    "pos_score": 0.0, "neg_score": 0.0, "neu_score": 1.0,
                    "compound": 0.0, "label": "neutral",
                })

        pct = round((i + 1) / n_batches * 100)
        log.info("  Scored batch %d/%d  (%d%%)  -  %d headlines so far",
                 i + 1, n_batches, pct, len(results))

    return results


# ---------------------------------------------------------------------------
# Fetch unscored headlines from DB
# ---------------------------------------------------------------------------

def get_unscored_headlines(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """
    Return headlines that have no entry in headline_sentiment yet.
    Optionally filter by ticker and recency.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    ticker_clause = "AND h.ticker = ?" if ticker else ""
    params: list = [cutoff]
    if ticker:
        params.append(ticker)

    df = pd.read_sql_query(
        f"""
        SELECT h.id, h.ticker, h.title, h.description, h.published
        FROM   headlines h
        LEFT JOIN headline_sentiment hs ON h.id = hs.headline_id
        WHERE  hs.headline_id IS NULL
          AND  h.published >= ?
          {ticker_clause}
        ORDER BY h.published DESC
        """,
        conn,
        params=params,
    )
    return df


# ---------------------------------------------------------------------------
# Save scored headlines to DB
# ---------------------------------------------------------------------------

def save_headline_scores(
    conn: sqlite3.Connection,
    rows: list[dict],
) -> int:
    """
    Insert scored headline rows into headline_sentiment.
    Skips duplicates (headline_id is PRIMARY KEY).
    Returns count inserted.
    """
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for row in rows:
        try:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO headline_sentiment
                  (headline_id, ticker, title, published,
                   pos_score, neg_score, neu_score, compound, label, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["headline_id"],
                    row["ticker"],
                    row["title"],
                    row.get("published", ""),
                    row["pos_score"],
                    row["neg_score"],
                    row["neu_score"],
                    row["compound"],
                    row["label"],
                    now,
                ),
            )
            inserted += cursor.rowcount   # 1 if inserted, 0 if ignored (duplicate)
        except sqlite3.Error as exc:
            log.warning("Insert error for headline %s: %s", row.get("headline_id"), exc)
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Exponential decay weight
# ---------------------------------------------------------------------------

def exp_decay_weight(published_utc: str, half_life_hours: float = DECAY_HALF_LIFE_HOURS) -> float:
    """
    Return a weight in (0, 1] based on how old the headline is.

    weight = exp( -ln(2) * age_hours / half_life_hours )

    A headline published NOW     -> weight ~ 1.0
    A headline published 24 h ago -> weight ~ 0.5
    A headline published 48 h ago -> weight ~ 0.25
    A headline published 7 d ago  -> weight ~ 0.02

    If the date can't be parsed, returns 0.5 (neutral weight).
    """
    try:
        pub = pd.to_datetime(published_utc, utc=True)
        now = datetime.now(timezone.utc)
        age_hours = max((now - pub).total_seconds() / 3600, 0)
        return math.exp(-math.log(2) * age_hours / half_life_hours)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _infer_signal(weighted_compound: float) -> str:
    """
    Convert a weighted compound score to a trading signal label.
    Thresholds tuned for FinBERT's typical output distribution.

    > +0.15  -> BULLISH
    < -0.15  -> BEARISH
    else     -> NEUTRAL
    """
    if weighted_compound > 0.15:
        return "BULLISH"
    elif weighted_compound < -0.15:
        return "BEARISH"
    else:
        return "NEUTRAL"


def aggregate_daily_sentiment(
    conn: sqlite3.Connection,
    ticker: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """
    Aggregate headline_sentiment into daily scores per ticker.

    For each (ticker, date):
      - avg_compound         : simple arithmetic mean
      - weighted_compound    : exponential-decay weighted mean
      - avg_pos / neg / neu  : mean class probabilities
      - headline_count       : number of scored headlines
      - signal               : BULLISH / BEARISH / NEUTRAL

    Persists results to daily_sentiment table.
    Returns the aggregated DataFrame.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    ticker_clause = "AND ticker = ?" if ticker else ""
    params: list = [cutoff]
    if ticker:
        params.append(ticker)

    df = pd.read_sql_query(
        f"""
        SELECT headline_id, ticker, title, published,
               pos_score, neg_score, neu_score, compound, label
        FROM   headline_sentiment
        WHERE  published >= ?
               {ticker_clause}
        ORDER BY published ASC
        """,
        conn,
        params=params,
    )

    if df.empty:
        log.warning("No scored headlines found for aggregation.")
        return pd.DataFrame()

    # Parse published -> UTC date
    df["published_dt"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_dt"])
    df["date"] = df["published_dt"].dt.strftime("%Y-%m-%d")

    # Compute per-row decay weight
    df["weight"] = df["published"].apply(exp_decay_weight)

    # Group by ticker + date
    records = []
    now_str = datetime.now(timezone.utc).isoformat()

    for (tick, date), grp in df.groupby(["ticker", "date"]):
        n = len(grp)
        weights = grp["weight"].values
        compounds = grp["compound"].values

        avg_compound      = float(np.mean(compounds))
        w_sum             = float(np.sum(weights))
        weighted_compound = float(np.dot(weights, compounds) / w_sum) if w_sum > 0 else 0.0
        avg_pos           = float(np.mean(grp["pos_score"].values))
        avg_neg           = float(np.mean(grp["neg_score"].values))
        avg_neu           = float(np.mean(grp["neu_score"].values))
        signal            = _infer_signal(weighted_compound)
        row_id            = f"{tick}::{date}"

        records.append({
            "id":                 row_id,
            "ticker":             tick,
            "date":               date,
            "avg_compound":       round(avg_compound,      4),
            "weighted_compound":  round(weighted_compound, 4),
            "avg_pos":            round(avg_pos, 4),
            "avg_neg":            round(avg_neg, 4),
            "avg_neu":            round(avg_neu, 4),
            "headline_count":     n,
            "signal":             signal,
            "computed_at":        now_str,
        })

    # Persist to DB
    for rec in records:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_sentiment
                  (id, ticker, date, avg_compound, weighted_compound,
                   avg_pos, avg_neg, avg_neu, headline_count, signal, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["id"], rec["ticker"], rec["date"],
                    rec["avg_compound"], rec["weighted_compound"],
                    rec["avg_pos"], rec["avg_neg"], rec["avg_neu"],
                    rec["headline_count"], rec["signal"], rec["computed_at"],
                ),
            )
        except sqlite3.Error as exc:
            log.warning("Aggregation insert error for %s: %s", rec["id"], exc)

    conn.commit()
    log.info("Aggregated %d (ticker, date) pairs into daily_sentiment", len(records))

    result_df = pd.DataFrame(records)
    return result_df


# ---------------------------------------------------------------------------
# Dashboard helper: latest sentiment per ticker
# ---------------------------------------------------------------------------

def get_latest_sentiment(
    conn: sqlite3.Connection,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return the most recent daily_sentiment row per ticker.
    Used by the Streamlit dashboard for the summary metrics panel.
    """
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        df = pd.read_sql_query(
            f"""
            SELECT ds.*
            FROM   daily_sentiment ds
            INNER JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM   daily_sentiment
                WHERE  ticker IN ({placeholders})
                GROUP BY ticker
            ) latest ON ds.ticker = latest.ticker AND ds.date = latest.max_date
            ORDER BY ds.weighted_compound DESC
            """,
            conn,
            params=tickers,
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT ds.*
            FROM   daily_sentiment ds
            INNER JOIN (
                SELECT ticker, MAX(date) AS max_date
                FROM   daily_sentiment
                GROUP BY ticker
            ) latest ON ds.ticker = latest.ticker AND ds.date = latest.max_date
            ORDER BY ds.weighted_compound DESC
            """,
            conn,
        )
    return df


def get_sentiment_history(
    conn: sqlite3.Connection,
    ticker: str,
    days: int = 30,
) -> pd.DataFrame:
    """
    Return daily sentiment history for one ticker, ready for chart overlay.
    Joins with prices table when available.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(
        """
        SELECT date, weighted_compound, avg_pos, avg_neg, avg_neu,
               headline_count, signal
        FROM   daily_sentiment
        WHERE  ticker = ?
          AND  date   >= ?
        ORDER BY date ASC
        """,
        conn,
        params=(ticker, cutoff),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Main pipeline: score + aggregate
# ---------------------------------------------------------------------------

def run_pipeline(
    ticker: Optional[str] = None,
    days: int = 30,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH,
    db_path: str = DB_PATH,
    skip_scoring: bool = False,
) -> pd.DataFrame:
    """
    Full sentiment pipeline for one ticker or the entire watchlist.

    Steps:
      1. Fetch unscored headlines from DB
      2. Run FinBERT batch inference
      3. Save per-headline scores
      4. Aggregate into daily_sentiment
      5. Return aggregated DataFrame

    Parameters
    ----------
    ticker       : NSE ticker e.g. 'RELIANCE.NS' or None for all
    days         : How many days of headlines to process
    device       : 'auto' | 'cpu' | 'cuda'
    batch_size   : Headlines per FinBERT batch (reduce if OOM)
    db_path      : Path to SQLite DB
    skip_scoring : Skip FinBERT and only re-aggregate (for quick refresh)
    """
    conn = init_db(db_path)

    # ---- Step 1: Find unscored headlines ----
    if not skip_scoring:
        unscored = get_unscored_headlines(conn, ticker=ticker, days=days)
        log.info("Found %d unscored headlines to process", len(unscored))

        if not unscored.empty:
            # ---- Step 2: Load model ----
            finbert = load_finbert(device=device)

            # Combine title + description for richer context
            texts = []
            for _, row in unscored.iterrows():
                title = str(row.get("title", "") or "")
                desc  = str(row.get("description", "") or "")
                # Use title alone if description is empty or too similar
                if desc and desc.lower() != title.lower() and len(desc) > 20:
                    combined = f"{title}. {desc}"
                else:
                    combined = title
                texts.append(combined[:1000])   # cap at ~200 tokens

            # ---- Step 3: Run FinBERT ----
            log.info("Running FinBERT on %d headlines (batch_size=%d)...",
                     len(texts), batch_size)
            scores = score_headlines_batch(texts, finbert, batch_size=batch_size)

            # ---- Step 4: Save per-headline scores ----
            rows_to_save = []
            for (_, headline_row), score in zip(unscored.iterrows(), scores):
                rows_to_save.append({
                    "headline_id": headline_row["id"],
                    "ticker":      headline_row["ticker"],
                    "title":       headline_row["title"],
                    "published":   headline_row.get("published", ""),
                    **score,
                })
            n_saved = save_headline_scores(conn, rows_to_save)
            log.info("Saved %d new headline scores to DB", n_saved)

        else:
            log.info("No new headlines to score - all up to date.")

    # ---- Step 5: Aggregate daily sentiment ----
    log.info("Aggregating daily sentiment scores...")
    daily_df = aggregate_daily_sentiment(conn, ticker=ticker, days=days)
    conn.close()

    return daily_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run FinBERT sentiment pipeline on fetched headlines"
    )
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="NSE ticker without suffix e.g. RELIANCE (processes all if omitted)",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Days of headlines to score/aggregate (default: 30)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto -> GPU if available, else CPU)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"Headlines per batch (default: {DEFAULT_BATCH}). Reduce to 4-8 on low RAM.",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip FinBERT inference, only re-aggregate existing scores",
    )
    parser.add_argument(
        "--show-headlines", action="store_true",
        help="Print individual headline scores after running",
    )
    args = parser.parse_args()

    ticker_key = f"{args.ticker.upper()}.NS" if args.ticker else None

    # Run the full pipeline
    daily_df = run_pipeline(
        ticker=ticker_key,
        days=args.days,
        device=args.device,
        batch_size=args.batch_size,
        skip_scoring=args.skip_scoring,
    )

    # Print results
    if daily_df.empty:
        print("\nNo sentiment data generated.")
        print("Make sure you have run fetch_news.py first to populate headlines.")
        return

    print(f"\n{'='*70}")
    print(f"  DAILY SENTIMENT SUMMARY  ({len(daily_df)} ticker-days)")
    print(f"{'='*70}")

    display_cols = ["ticker", "date", "weighted_compound", "signal",
                    "headline_count", "avg_pos", "avg_neg"]
    print(daily_df[display_cols].to_string(index=False))

    # Signal breakdown
    print(f"\n{'='*70}")
    print("  SIGNAL BREAKDOWN")
    print(f"{'='*70}")
    for sig in ["BULLISH", "NEUTRAL", "BEARISH"]:
        subset = daily_df[daily_df["signal"] == sig]
        if not subset.empty:
            symbol = {"BULLISH": "[+]", "NEUTRAL": "[=]", "BEARISH": "[-]"}[sig]
            print(f"  {symbol} {sig:<10} {len(subset):>3} days  "
                  f"| avg compound: {subset['weighted_compound'].mean():.4f}")

    # Per-ticker summary
    print(f"\n{'='*70}")
    print("  LATEST SIGNAL PER TICKER")
    print(f"{'='*70}")
    conn = sqlite3.connect(DB_PATH)
    latest = get_latest_sentiment(conn)
    conn.close()

    if not latest.empty:
        for _, row in latest.iterrows():
            signal_icon = {"BULLISH": "[+]", "NEUTRAL": "[=]", "BEARISH": "[-]"}.get(
                row["signal"], "?"
            )
            print(
                f"  {row['ticker']:<18}  {signal_icon} {row['signal']:<10}"
                f"  compound={row['weighted_compound']:+.4f}"
                f"  ({row['headline_count']} headlines on {row['date']})"
            )

    # Optional: show individual headline scores
    if args.show_headlines:
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        ticker_clause = f"AND ticker = '{ticker_key}'" if ticker_key else ""
        hs = pd.read_sql_query(
            f"""SELECT ticker, published, compound, label, title
                FROM headline_sentiment
                WHERE published >= '{cutoff}' {ticker_clause}
                ORDER BY published DESC
                LIMIT 30""",
            conn,
        )
        conn.close()
        if not hs.empty:
            print(f"\n{'='*70}")
            print("  RECENT HEADLINE SCORES (last 30)")
            print(f"{'='*70}")
            for _, r in hs.iterrows():
                icon = {"positive": "+", "negative": "-", "neutral": "~"}.get(r["label"], "?")
                print(f"  [{icon}] {r['compound']:+.3f}  {r['ticker']:<14}"
                      f"  {str(r['published'])[:16]}  {r['title'][:65]}")

    print(f"\nAll scores saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
