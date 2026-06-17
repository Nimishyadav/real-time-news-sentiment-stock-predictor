# -*- coding: utf-8 -*-
"""
fetch_news.py
-------------
Fetches financial headlines for a list of stock tickers from:
  1. NewsAPI (https://newsapi.org - free tier: 100 req/day)
  2. RSS feeds: Economic Times Markets, Moneycontrol, Mint

Headlines are deduplicated, stored in SQLite, and returned as a
clean pandas DataFrame ready for FinBERT inference.

Usage:
    python fetch_news.py                     # fetch all tickers in WATCHLIST
    python fetch_news.py --ticker RELIANCE   # fetch single ticker
    python fetch_news.py --days 7            # fetch last 7 days (default: 3)
"""

import os
import hashlib
import logging
import argparse
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import pandas as pd
import requests
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # reads NEWSAPI_KEY from .env file

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "sentiment_cache.db")

# NSE/BSE tickers mapped to human-readable search terms for NewsAPI
# Format: { "TICKER.NS": ["search term 1", "search term 2"] }
WATCHLIST = {
    # Original 8
    "RELIANCE.NS":   ["Reliance Industries", "RIL"],
    "TCS.NS":        ["Tata Consultancy Services", "TCS"],
    "INFY.NS":       ["Infosys"],
    "HDFCBANK.NS":   ["HDFC Bank"],
    "ICICIBANK.NS":  ["ICICI Bank"],
    "WIPRO.NS":      ["Wipro"],
    "BAJFINANCE.NS": ["Bajaj Finance"],
    "SBIN.NS":       ["State Bank of India", "SBI"],
    # Added 12 - Nifty 50 large caps
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],
    "LT.NS":         ["Larsen Toubro", "L&T", "Larsen"],
    "AXISBANK.NS":   ["Axis Bank"],
    "KOTAKBANK.NS":  ["Kotak Mahindra Bank", "Kotak Bank"],
    "ITC.NS":        ["ITC Limited", "ITC"],
    "MARUTI.NS":     ["Maruti Suzuki", "Maruti"],
    "TITAN.NS":      ["Titan Company", "Titan"],
    "SUNPHARMA.NS":  ["Sun Pharmaceutical", "Sun Pharma"],
    "TATAMOTORS.NS": ["Tata Motors"],
    "TATASTEEL.NS":  ["Tata Steel"],
    "ADANIENT.NS":   ["Adani Enterprises", "Adani"],
    "ONGC.NS":       ["ONGC", "Oil and Natural Gas"],
}

# Indian financial RSS feeds - no API key needed
RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rss.cms",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.livemint.com/rss/markets",
]

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
REQUEST_DELAY = 0.5   # seconds between NewsAPI calls (rate limiting)
MAX_HEADLINES_PER_TICKER = 50

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
# Database setup
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create the headlines table if it doesn't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS headlines (
            id          TEXT PRIMARY KEY,   -- SHA-256 of (ticker + title)
            ticker      TEXT NOT NULL,
            title       TEXT NOT NULL,
            description TEXT,
            source      TEXT,
            url         TEXT,
            published   TEXT,              -- ISO-8601 UTC string
            fetched_at  TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON headlines(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_published ON headlines(published)")
    conn.commit()
    log.info("Database ready: %s", db_path)
    return conn


def make_headline_id(ticker: str, title: str) -> str:
    """Stable deduplication key - SHA-256 of ticker + normalised title."""
    raw = f"{ticker}::{title.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def insert_headlines(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert new headlines, skip duplicates. Returns count inserted."""
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for row in rows:
        hid = make_headline_id(row["ticker"], row["title"])
        try:
            conn.execute(
                """INSERT INTO headlines
                   (id, ticker, title, description, source, url, published, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    hid,
                    row["ticker"],
                    row["title"],
                    row.get("description", ""),
                    row.get("source", ""),
                    row.get("url", ""),
                    row.get("published", ""),
                    now,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate - already in DB
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# NewsAPI fetcher
# ---------------------------------------------------------------------------

def fetch_newsapi(
    ticker: str,
    search_terms: list[str],
    from_date: datetime,
    to_date: datetime,
) -> list[dict]:
    """
    Call NewsAPI /v2/everything for each search term, return combined results.
    Requires NEWSAPI_KEY in environment (free tier = 100 calls/day, 1 month history).
    """
    if not NEWSAPI_KEY:
        log.warning("NEWSAPI_KEY not set - skipping NewsAPI for %s", ticker)
        return []

    results = []
    for term in search_terms:
        params = {
            "q":        f'"{term}"',          # exact phrase match
            "from":     from_date.strftime("%Y-%m-%d"),
            "to":       to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 100,
            "apiKey":   NEWSAPI_KEY,
        }
        try:
            resp = requests.get(NEWSAPI_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            log.info("  NewsAPI '%s' -> %d articles", term, len(articles))

            for article in articles:
                title = (article.get("title") or "").strip()
                if not title or title == "[Removed]":
                    continue
                results.append({
                    "ticker":      ticker,
                    "title":       title,
                    "description": (article.get("description") or "").strip(),
                    "source":      article.get("source", {}).get("name", ""),
                    "url":         article.get("url", ""),
                    "published":   article.get("publishedAt", ""),
                })
        except requests.RequestException as exc:
            log.error("  NewsAPI error for '%s': %s", term, exc)

        time.sleep(REQUEST_DELAY)

    return results


# ---------------------------------------------------------------------------
# RSS fetcher
# ---------------------------------------------------------------------------

def _parse_rss_date(entry) -> str:
    """Extract published date from feedparser entry -> ISO-8601 UTC string."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(timezone.utc).isoformat()


def fetch_rss(
    ticker: str,
    search_terms: list[str],
    from_date: datetime,
) -> list[dict]:
    """
    Parse Indian financial RSS feeds and filter entries that mention any
    search term for this ticker. No API key required.
    """
    results = []
    terms_lower = [t.lower() for t in search_terms]

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            source_name = feed.feed.get("title", feed_url)
            matched = 0

            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                summary = (entry.get("summary") or "").strip()
                combined = f"{title} {summary}".lower()

                # keyword match
                if not any(term in combined for term in terms_lower):
                    continue

                # date filter - skip entries older than from_date
                pub_str = _parse_rss_date(entry)
                try:
                    pub_dt = datetime.fromisoformat(pub_str)
                    if pub_dt < from_date:
                        continue
                except ValueError:
                    pass

                results.append({
                    "ticker":      ticker,
                    "title":       title,
                    "description": summary[:500],
                    "source":      source_name,
                    "url":         entry.get("link", ""),
                    "published":   pub_str,
                })
                matched += 1

            if matched:
                log.info("  RSS '%s' -> %d matches for %s", source_name, matched, ticker)

        except Exception as exc:
            log.warning("  RSS error for %s: %s", feed_url, exc)

    return results


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_headlines(
    ticker: str,
    search_terms: list[str],
    days: int = 3,
    conn: Optional[sqlite3.Connection] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch headlines for one ticker from NewsAPI + RSS.
    If use_cache=True and recent data exists in DB, return from cache.

    Returns a DataFrame with columns:
        ticker, title, description, source, url, published
    """
    to_date   = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days)

    # --- Check cache first ---
    if use_cache and conn:
        cached = pd.read_sql_query(
            """SELECT ticker, title, description, source, url, published
               FROM headlines
               WHERE ticker = ?
                 AND published >= ?
               ORDER BY published DESC""",
            conn,
            params=(ticker, from_date.isoformat()),
        )
        if len(cached) >= 10:
            log.info("[%s] Using %d cached headlines", ticker, len(cached))
            return cached

    log.info("[%s] Fetching fresh headlines (%d days)...", ticker, days)

    # --- Fetch from sources ---
    rows: list[dict] = []
    rows.extend(fetch_newsapi(ticker, search_terms, from_date, to_date))
    rows.extend(fetch_rss(ticker, search_terms, from_date))

    # Deduplicate within this batch by title
    seen_titles: set[str] = set()
    unique_rows = []
    for row in rows:
        key = row["title"].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_rows.append(row)

    # Cap at MAX_HEADLINES_PER_TICKER (most recent first)
    unique_rows = sorted(
        unique_rows,
        key=lambda r: r.get("published", ""),
        reverse=True,
    )[:MAX_HEADLINES_PER_TICKER]

    log.info("[%s] %d unique headlines after dedup", ticker, len(unique_rows))

    # --- Persist to DB ---
    if conn and unique_rows:
        n = insert_headlines(conn, unique_rows)
        log.info("[%s] Inserted %d new rows into DB", ticker, n)

    return pd.DataFrame(unique_rows) if unique_rows else pd.DataFrame(
        columns=["ticker", "title", "description", "source", "url", "published"]
    )


def fetch_all_tickers(
    watchlist: dict[str, list[str]] = WATCHLIST,
    days: int = 3,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Fetch headlines for every ticker in the watchlist.
    Returns one combined DataFrame, stored in SQLite.
    """
    conn = init_db(db_path)
    frames = []

    for ticker, terms in watchlist.items():
        df = fetch_headlines(ticker, terms, days=days, conn=conn)
        frames.append(df)
        time.sleep(REQUEST_DELAY)  # be polite to APIs

    conn.close()
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["published"] = pd.to_datetime(combined["published"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["title"])
    combined = combined.sort_values("published", ascending=False).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch financial news headlines")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single NSE ticker e.g. RELIANCE (no .NS suffix)")
    parser.add_argument("--days", type=int, default=3,
                        help="Number of past days to fetch (default: 3)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass DB cache and force fresh fetch")
    args = parser.parse_args()

    if args.ticker:
        # Build ticker key
        key = f"{args.ticker.upper()}.NS"
        terms = WATCHLIST.get(key)
        if not terms:
            # Fallback: use raw ticker name as search term
            terms = [args.ticker.upper()]
            log.warning("'%s' not in WATCHLIST - searching by name only", key)

        conn = init_db()
        df = fetch_headlines(key, terms, days=args.days, conn=conn,
                             use_cache=not args.no_cache)
        conn.close()
    else:
        df = fetch_all_tickers(days=args.days)

    if df.empty:
        print("\nNo headlines found. Check your NEWSAPI_KEY in .env")
        return

    print(f"\n{'='*60}")
    print(f"  Fetched {len(df)} headlines across {df['ticker'].nunique()} ticker(s)")
    print(f"{'='*60}\n")
    print(df[["ticker", "published", "source", "title"]].to_string(index=False))
    print(f"\nData saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
