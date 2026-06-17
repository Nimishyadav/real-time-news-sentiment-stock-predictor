"""
app.py - Real-Time News Sentiment Stock Predictor
--------------------------------------------------
Streamlit dashboard with:
  - 20 NSE stocks (Nifty 50 large caps)
  - Real-time price refresh via yfinance (every N minutes)
  - Auto-running fetch_prices + sentiment_engine in background
  - Dual-axis chart: price line + sentiment bars
  - Headline feed with FinBERT scores
  - Live signal ranking for all tickers

Run:
    streamlit run app.py
"""

import os
import sys
import time
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# -- Page config --------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Signals · NSE",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Constants -----------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(PROJECT_DIR, "data", "sentiment_cache.db")
PYTHON      = sys.executable   # same Python that's running this app

# 20-stock expanded watchlist
WATCHLIST = {
    "RELIANCE.NS":   "Reliance Inds.",
    "TCS.NS":        "TCS",
    "INFY.NS":       "Infosys",
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "WIPRO.NS":      "Wipro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "SBIN.NS":       "SBI",
    "HINDUNILVR.NS": "HUL",
    "LT.NS":         "L&T",
    "AXISBANK.NS":   "Axis Bank",
    "KOTAKBANK.NS":  "Kotak Bank",
    "ITC.NS":        "ITC",
    "MARUTI.NS":     "Maruti Suzuki",
    "TITAN.NS":      "Titan",
    "SUNPHARMA.NS":  "Sun Pharma",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS":  "Tata Steel",
    "ADANIENT.NS":   "Adani Ent.",
    "ONGC.NS":       "ONGC",
}

SIGNAL_COLOR = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "NEUTRAL": "#94a3b8"}
SIGNAL_ICON  = {"BULLISH": "(B)",       "BEARISH": "(S)",       "NEUTRAL": "(N)"}

REFRESH_OPTIONS = {
    "Off":    0,
    "5 min":  5 * 60 * 1000,
    "10 min": 10 * 60 * 1000,
    "30 min": 30 * 60 * 1000,
}


# -- CSS -----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --txt-primary:   #1e293b;
    --txt-secondary: #475569;
    --txt-muted:     #94a3b8;
    --border-soft:   rgba(0,0,0,0.08);
    --card-bg:       rgba(0,0,0,0.03);
    --grid-color:    rgba(0,0,0,0.06);
}
[data-theme="dark"], .stApp[data-theme="dark"] {
    --txt-primary:   #f1f5f9;
    --txt-secondary: #94a3b8;
    --txt-muted:     #64748b;
    --border-soft:   rgba(255,255,255,0.08);
    --card-bg:       rgba(255,255,255,0.03);
    --grid-color:    rgba(255,255,255,0.05);
}

.header-bar {
    display:flex; align-items:baseline; gap:12px;
    padding:4px 0 18px;
    border-bottom:1px solid var(--border-soft); margin-bottom:18px;
}
.header-title { font-size:1.3rem; font-weight:600; color:var(--txt-primary); margin:0; }
.header-sub   { font-family:'IBM Plex Mono',monospace; font-size:0.71rem; color:var(--txt-muted); margin:0; }

.signal-badge {
    display:inline-flex; align-items:center; gap:7px;
    padding:10px 18px; border-radius:8px;
    font-family:'IBM Plex Mono',monospace;
    font-size:0.93rem; font-weight:500; letter-spacing:0.04em;
    width:100%; justify-content:center;
}
.signal-badge.bullish { background:rgba(34,197,94,0.12);  color:#22c55e; border:1px solid rgba(34,197,94,0.3);  }
.signal-badge.bearish { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.3);  }
.signal-badge.neutral { background:rgba(148,163,184,0.10);color:#64748b; border:1px solid rgba(148,163,184,0.2);}

.metric-card {
    background:var(--card-bg);
    border:1px solid var(--border-soft);
    border-radius:10px; padding:14px 18px;
}
.metric-label {
    font-family:'IBM Plex Mono',monospace; font-size:0.67rem;
    color:var(--txt-muted); text-transform:uppercase;
    letter-spacing:0.09em; margin-bottom:4px;
}
.metric-value { font-size:1.4rem; font-weight:600; color:var(--txt-primary); line-height:1; }
.metric-delta { font-family:'IBM Plex Mono',monospace; font-size:0.74rem; margin-top:5px; }
.up   { color:#22c55e; }
.down { color:#ef4444; }
.flat { color:var(--txt-muted); }

.section-label {
    font-family:'IBM Plex Mono',monospace; font-size:0.67rem;
    color:var(--txt-muted); text-transform:uppercase; letter-spacing:0.1em;
    padding:14px 0 8px; border-top:1px solid var(--border-soft); margin-top:4px;
}

.headline-card { padding:9px 0; border-bottom:1px solid var(--border-soft); }
.headline-title { font-size:0.84rem; color:var(--txt-primary); line-height:1.45; margin-bottom:4px; }
.headline-meta  {
    font-family:'IBM Plex Mono',monospace; font-size:0.67rem;
    color:var(--txt-muted); display:flex; gap:10px; align-items:center;
}
.hl-badge {
    display:inline-block; font-size:0.64rem;
    font-family:'IBM Plex Mono',monospace;
    padding:1px 7px; border-radius:4px; font-weight:500;
}
.hl-pos { background:rgba(34,197,94,0.15);  color:#22c55e; }
.hl-neg { background:rgba(239,68,68,0.15);  color:#ef4444; }
.hl-neu { background:rgba(148,163,184,0.12);color:#64748b; }

.rank-row {
    display:flex; align-items:center; padding:7px 0;
    border-bottom:1px solid var(--border-soft); gap:8px;
}
.rank-num  { font-family:'IBM Plex Mono',monospace; font-size:0.7rem;  color:var(--txt-muted); width:18px; }
.rank-name { font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:var(--txt-secondary); flex:1; }
.rank-n    { font-size:0.66rem; color:var(--txt-muted); }

.live-dot {
    display:inline-block; width:7px; height:7px; border-radius:50%;
    background:#22c55e; margin-right:6px; animation:blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.refresh-bar {
    font-family:'IBM Plex Mono',monospace; font-size:0.67rem;
    color:var(--txt-muted); text-align:right; padding-top:10px;
}

.no-data { text-align:center; padding:30px 10px; color:var(--txt-muted); font-size:0.84rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


# -- DB helpers ----------------------------------------------------------------

@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def get_available_tickers() -> list:
    # Show ALL tickers from prices table (not just ones with sentiment)
    # so stocks appear even before their first headline is scored
    df = q("SELECT DISTINCT ticker FROM prices ORDER BY ticker")
    if df.empty:
        df = q("SELECT DISTINCT ticker FROM daily_sentiment ORDER BY ticker")
    if df.empty:
        return list(WATCHLIST.keys())
    db_tickers = set(df["ticker"].tolist())
    # Order: known watchlist tickers first (preserves display name order)
    known   = list(WATCHLIST.keys())
    ordered = [t for t in known if t in db_tickers]
    extra   = [t for t in db_tickers if t not in known]
    return ordered + extra


def get_live_price(ticker: str) -> dict:
    """
    Pull the most recent close from the prices table.
    fetch_prices.py populates this - call it to refresh.
    """
    df = q(
        """SELECT datetime, close, open, high, low, volume
           FROM prices WHERE ticker=? AND interval='1d'
           ORDER BY datetime DESC LIMIT 2""",
        (ticker,),
    )
    if df.empty:
        return {}
    latest = df.iloc[0]
    prev   = df.iloc[1] if len(df) > 1 else latest
    pct    = ((float(latest["close"]) - float(prev["close"])) / float(prev["close"])) * 100
    return {
        "close":      float(latest["close"]),
        "open":       float(latest["open"]),
        "high":       float(latest["high"]),
        "low":        float(latest["low"]),
        "volume":     int(latest["volume"]),
        "pct_change": round(pct, 2),
        "datetime":   str(latest["datetime"])[:10],
    }


def get_prices_df(ticker: str, days: int) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days + 5)).strftime("%Y-%m-%d")
    df = q(
        """SELECT datetime, close, high, low, open, volume
           FROM prices WHERE ticker=? AND interval='1d' AND datetime>=?
           ORDER BY datetime ASC""",
        (ticker, cutoff),
    )
    if not df.empty:
        df["datetime"]   = pd.to_datetime(df["datetime"], utc=True)
        df["date"]       = df["datetime"].dt.normalize().dt.tz_localize(None)
        df               = df.drop_duplicates("date", keep="last").sort_values("date")
        df["pct_change"] = df["close"].pct_change() * 100
    return df


def get_sentiment_df(ticker: str, days: int) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    df = q(
        """SELECT date, weighted_compound, signal, avg_pos, avg_neg, headline_count
           FROM daily_sentiment WHERE ticker=? AND date>=?
           ORDER BY date ASC""",
        (ticker, cutoff),
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_latest_signal(ticker: str) -> dict:
    df = q(
        """SELECT date, weighted_compound, signal, headline_count, avg_pos, avg_neg
           FROM daily_sentiment WHERE ticker=?
           ORDER BY date DESC LIMIT 1""",
        (ticker,),
    )
    return df.iloc[0].to_dict() if not df.empty else {}


def get_all_signals() -> pd.DataFrame:
    # Left join prices -> daily_sentiment so ALL tickers with prices appear
    # Tickers without sentiment show as NEUTRAL with 0 compound
    return q(
        """SELECT
               p.ticker,
               COALESCE(ds.date, date('now')) AS date,
               COALESCE(ds.weighted_compound, 0.0) AS weighted_compound,
               COALESCE(ds.signal, 'NEUTRAL') AS signal,
               COALESCE(ds.headline_count, 0) AS headline_count
           FROM (SELECT DISTINCT ticker FROM prices) p
           LEFT JOIN (
               SELECT ds2.*
               FROM daily_sentiment ds2
               INNER JOIN (
                   SELECT ticker, MAX(date) AS mx FROM daily_sentiment GROUP BY ticker
               ) l ON ds2.ticker=l.ticker AND ds2.date=l.mx
           ) ds ON p.ticker = ds.ticker
           ORDER BY COALESCE(ds.weighted_compound, 0.0) DESC"""
    )


def get_headlines(ticker: str) -> pd.DataFrame:
    df = q(
        """SELECT hs.title, hs.published, hs.compound, hs.label,
                  h.source, h.url
           FROM headline_sentiment hs
           JOIN headlines h ON h.id=hs.headline_id
           WHERE hs.ticker=?
           ORDER BY hs.published DESC LIMIT 15""",
        (ticker,),
    )
    if df.empty:
        df = q(
            """SELECT title, published, NULL as compound, NULL as label,
                      source, url
               FROM headlines WHERE ticker=?
               ORDER BY published DESC LIMIT 15""",
            (ticker,),
        )
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    return df


def get_backtest_accuracy(ticker: str) -> Optional[float]:
    df = q(
        """SELECT ROUND(100.0 * SUM(CASE WHEN signal_correct=1 THEN 1.0 ELSE 0 END)
                  / NULLIF(SUM(CASE WHEN sentiment_signal!='NEUTRAL'
                                    AND signal_correct IS NOT NULL THEN 1 ELSE 0 END),0),1)
                  AS acc FROM backtest_results WHERE ticker=?""",
        (ticker,),
    )
    if df.empty or df["acc"].isna().all():
        return None
    return float(df["acc"].iloc[0])


# -- Background data refresh ---------------------------------------------------


def get_top_impact_headlines(limit: int = 5) -> pd.DataFrame:
    """Top headlines by absolute sentiment magnitude across all tickers today."""
    df = q(
        """SELECT hs.ticker, hs.title, hs.compound, hs.label,
                  hs.pos_score, hs.neg_score, hs.published, h.source, h.url
           FROM   headline_sentiment hs
           JOIN   headlines h ON h.id = hs.headline_id
           WHERE  hs.published >= date('now', '-3 days')
           ORDER BY ABS(hs.compound) DESC
           LIMIT ?""",
        (limit,),
    )
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    return df


def get_sentiment_trend(days: int = 14) -> pd.DataFrame:
    """Daily average sentiment across all tickers for sparkline."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return q(
        """SELECT date,
                  AVG(weighted_compound)  AS avg_compound,
                  SUM(headline_count)     AS total_articles,
                  COUNT(DISTINCT ticker)  AS n_tickers
           FROM   daily_sentiment
           WHERE  date >= ?
           GROUP BY date
           ORDER BY date ASC""",
        (cutoff,),
    )


def get_most_bullish_bearish() -> tuple:
    """Return single most bullish and most bearish headline from the last 3 days."""
    df = q(
        """SELECT hs.ticker, hs.title, hs.compound, hs.label,
                  hs.published, h.source, h.url
           FROM   headline_sentiment hs
           JOIN   headlines h ON h.id = hs.headline_id
           WHERE  hs.published >= date('now', '-3 days')
             AND  hs.label != 'neutral'""",
    )
    if df.empty:
        return None, None
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    bullish = df.nlargest(1, "compound").iloc[0].to_dict() if not df[df["compound"] > 0].empty else None
    bearish = df.nsmallest(1, "compound").iloc[0].to_dict() if not df[df["compound"] < 0].empty else None
    return bullish, bearish


def run_refresh(days: int = 3) -> tuple[bool, str]:
    """
    Run fetch_prices.py + sentiment_engine.py as subprocesses.
    Returns (success, message).
    This updates the SQLite DB with fresh prices and sentiment.
    """
    scripts = [
        [PYTHON, os.path.join(PROJECT_DIR, "fetch_prices.py"), "--days", str(days), "--no-cache"],
        [PYTHON, os.path.join(PROJECT_DIR, "sentiment_engine.py"), "--days", str(days)],
    ]
    errors = []
    for cmd in scripts:
        script_name = os.path.basename(cmd[1])
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"]       = "1"
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                cwd=PROJECT_DIR,
                env=env,
            )
            if result.returncode != 0:
                safe_err = result.stderr[-300:].encode("ascii", errors="replace").decode("ascii")
                errors.append(f"{script_name}: {safe_err}")
        except subprocess.TimeoutExpired:
            errors.append(f"{script_name}: timed out after 120s")
        except Exception as e:
            errors.append(f"{script_name}: {e}")

    if errors:
        return False, " | ".join(errors)
    return True, "Refreshed successfully"


# -- Chart ---------------------------------------------------------------------

def build_chart(price_df: pd.DataFrame, sent_df: pd.DataFrame, ticker: str) -> go.Figure:
    fig  = make_subplots(specs=[[{"secondary_y": True}]])
    has_p = not price_df.empty
    has_s = not sent_df.empty

    if has_p:
        # Price line
        fig.add_trace(
            go.Scatter(
                x=price_df["date"], y=price_df["close"].round(2),
                name="Price (₹)", mode="lines",
                line=dict(color="#3b82f6", width=2.5),
                hovertemplate="<b>%{x|%d %b}</b>  ₹%{y:,.2f}<extra></extra>",
            ),
            secondary_y=False,
        )

    if has_s:
        colors  = [SIGNAL_COLOR.get(s, "#94a3b8") for s in sent_df["signal"]]
        opacity = [0.85 if s != "NEUTRAL" else 0.35 for s in sent_df["signal"]]
        fig.add_trace(
            go.Bar(
                x=sent_df["date"], y=sent_df["weighted_compound"].round(4),
                name="Sentiment", marker_color=colors, marker_opacity=opacity,
                yaxis="y2",
                hovertemplate="<b>%{x|%d %b}</b>  Sentiment: %{y:+.3f}<extra></extra>",
            ),
        )
        # Threshold lines at ±0.15
        for y_val, col, dash in [
            (0,     "rgba(148,163,184,0.25)", "dot"),
            (0.15,  "rgba(34,197,94,0.3)",    "dash"),
            (-0.15, "rgba(239,68,68,0.3)",     "dash"),
        ]:
            fig.add_shape(
                type="line", xref="paper", x0=0, x1=1,
                yref="y2", y0=y_val, y1=y_val,
                line=dict(dash=dash, color=col, width=1),
            )

    short = ticker.replace(".NS", "").replace(".BO", "")
    fig.update_layout(
        title=dict(
            text=f"<b>{short}</b>  ·  Price & Sentiment",
            font=dict(size=13, color="#64748b"), x=0, pad=dict(l=2, b=6),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=380,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#64748b"), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, rangeslider=dict(visible=False),
            tickfont=dict(size=10, color="#64748b", family="IBM Plex Mono"),
            tickformat="%d %b",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(128,128,128,0.1)", zeroline=False,
            autorange=True, tickprefix="₹",
            tickfont=dict(size=10, color="#3b82f6", family="IBM Plex Mono"),
            title=dict(text="Price (₹)", font=dict(color="#3b82f6", size=11)),
        ),
        yaxis2=dict(
            showgrid=False, zeroline=False, range=[-1.15, 1.15],
            tickfont=dict(size=10, color="#64748b", family="IBM Plex Mono"),
            title=dict(text="Sentiment", font=dict(color="#64748b", size=11)),
            tickformat="+.1f",
        ),
        bargap=0.25,
    )

    if not has_p and not has_s:
        fig.add_annotation(
            text="No data - run fetch_prices.py and sentiment_engine.py",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="#64748b", size=13),
        )
    return fig


# -- UI helpers ----------------------------------------------------------------

def badge(signal: str, compound: float) -> None:
    cls  = signal.lower() if signal in SIGNAL_COLOR else "neutral"
    icon = SIGNAL_ICON.get(signal, "(N)")
    sign = "+" if compound >= 0 else ""
    st.markdown(
        f'<div class="signal-badge {cls}">{icon} {signal} &nbsp;·&nbsp; {sign}{compound:.4f}</div>',
        unsafe_allow_html=True,
    )


def metric(label: str, value: str, delta: str = "", dcls: str = "flat") -> None:
    d = f'<div class="metric-delta {dcls}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{d}</div>',
        unsafe_allow_html=True,
    )


def price_ticker_row(all_signals_df: pd.DataFrame) -> None:
    """Horizontal scrolling mini-ticker bar across the top."""
    if all_signals_df.empty:
        return
    parts = []
    for _, row in all_signals_df.iterrows():
        t    = str(row["ticker"]).replace(".NS","")
        sig  = str(row.get("signal","NEUTRAL"))
        comp = float(row.get("weighted_compound", 0))
        col  = SIGNAL_COLOR.get(sig, "#94a3b8")
        icon = SIGNAL_ICON.get(sig, "(N)")
        sign = "+" if comp >= 0 else ""
        parts.append(
            f'<span style="margin:0 14px;font-family:IBM Plex Mono,monospace;font-size:0.75rem;">'
            f'<b style="color:var(--txt-secondary)">{t}</b>'
            f'<span style="color:{col};margin-left:6px">{icon} {sign}{comp:.2f}</span>'
            f'</span>'
        )
    html = (
        '<div style="overflow-x:auto;white-space:nowrap;padding:8px 0 14px;'
        'border-bottom:1px solid var(--border-soft);margin-bottom:16px;">'
        + "".join(parts) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def headline_feed(df: pd.DataFrame) -> None:
    if df.empty:
        st.markdown(
            '<div class="no-data">No headlines yet.<br>'
            'Run <code>fetch_news.py</code> then <code>sentiment_engine.py</code></div>',
            unsafe_allow_html=True,
        )
        return
    for _, row in df.iterrows():
        label = str(row.get("label") or "")
        score = row.get("compound")
        bcls  = "hl-pos" if label == "positive" else "hl-neg" if label == "negative" else "hl-neu"
        icon  = "(B)" if label == "positive" else "(S)" if label == "negative" else "(N)"
        stxt  = f" {score:+.2f}" if score is not None else ""
        bdg   = f'<span class="hl-badge {bcls}">{icon} {label.upper() or "-"}{stxt}</span>'
        pub   = row.get("published")
        pstr  = pd.Timestamp(pub).strftime("%d %b %H:%M") if pd.notnull(pub) else ""
        src   = str(row.get("source") or "")
        url   = str(row.get("url") or "")
        title = str(row.get("title") or "")
        link  = (
            f'<a href="{url}" target="_blank" style="color:var(--txt-primary);text-decoration:none">{title}</a>'
            if url and url not in ("None","") else title
        )
        st.markdown(
            f'<div class="headline-card">'
            f'<div class="headline-title">{link}</div>'
            f'<div class="headline-meta">{bdg}<span>{pstr}</span>'
            f'{"<span>" + src + "</span>" if src and src!="None" else ""}'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def ranking_panel(df: pd.DataFrame, selected: str) -> None:
    if df.empty:
        st.markdown('<div class="no-data">No data yet.</div>', unsafe_allow_html=True)
        return
    for i, row in enumerate(df.itertuples(), 1):
        ticker = str(row.ticker)
        sig    = str(getattr(row, "signal", "NEUTRAL"))
        comp   = float(getattr(row, "weighted_compound", 0))
        n      = int(getattr(row, "headline_count", 0))
        col    = SIGNAL_COLOR.get(sig, "#94a3b8")
        icon   = SIGNAL_ICON.get(sig, "(N)")
        sign   = "+" if comp >= 0 else ""
        name   = WATCHLIST.get(ticker, ticker.replace(".NS",""))
        is_sel = ticker == selected
        bg     = "rgba(59,130,246,0.07)" if is_sel else "transparent"
        fw     = "600" if is_sel else "400"
        st.markdown(
            f'<div class="rank-row" style="background:{bg};border-radius:5px;padding-left:4px">'
            f'<span class="rank-num">{i}</span>'
            f'<span class="rank-name" style="font-weight:{fw}">{name}</span>'
            f'<span class="rank-n">{n}◆</span>'
            f'<span style="font-size:0.72rem;font-family:IBM Plex Mono,monospace;'
            f'width:84px;text-align:right;color:{col}">{icon} {sig}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;'
            f'width:60px;text-align:right;color:{col}">{sign}{comp:.3f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# -- Sidebar -------------------------------------------------------------------


def render_impact_analyzer(bullish: dict, bearish: dict,
                           top5: pd.DataFrame, trend_df: pd.DataFrame) -> None:
    """News Impact Analyzer panel with top headlines and sentiment sparkline."""
    st.markdown('<div class="section-label">NEWS IMPACT ANALYZER</div>', unsafe_allow_html=True)

    has_data = bullish is not None or bearish is not None or not top5.empty

    if not has_data:
        st.markdown(
            '<div class="no-data">No scored headlines yet.<br>'
            'Run <code>fetch_news.py</code> then <code>sentiment_engine.py</code></div>',
            unsafe_allow_html=True,
        )
        return

    # -- Most Bullish / Most Bearish cards
    cb, cs = st.columns(2)

    with cb:
        if bullish:
            comp = float(bullish["compound"])
            ticker = str(bullish.get("ticker","")).replace(".NS","")
            title  = str(bullish.get("title",""))[:90]
            src    = str(bullish.get("source",""))
            url    = str(bullish.get("url","") or "")
            pub    = bullish.get("published")
            pstr   = pd.Timestamp(pub).strftime("%d %b") if pd.notnull(pub) else ""
            link   = f'<a href="{url}" target="_blank" style="color:#22c55e;text-decoration:none;font-weight:500">{title}{"..." if len(str(bullish.get("title","")))>90 else ""}</a>' if url not in ("","None") else f'<span style="color:#22c55e;font-weight:500">{title}</span>'
            st.markdown(
                f'<div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.25);'
                f'border-radius:8px;padding:12px 14px;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#22c55e;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">'
                f'(B) Most Bullish Today</div>'
                f'<div style="font-size:0.82rem;line-height:1.4;margin-bottom:6px">{link}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.67rem;color:#475569;">'
                f'{ticker} &nbsp;|&nbsp; score: <span style="color:#22c55e">+{comp:.3f}</span>'
                f'&nbsp;|&nbsp; {src} &nbsp;|&nbsp; {pstr}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(34,197,94,0.04);border:1px solid rgba(34,197,94,0.1);'
                'border-radius:8px;padding:12px 14px;color:#475569;font-size:0.82rem">'
                'No bullish headlines in last 3 days</div>',
                unsafe_allow_html=True,
            )

    with cs:
        if bearish:
            comp   = float(bearish["compound"])
            ticker = str(bearish.get("ticker","")).replace(".NS","")
            title  = str(bearish.get("title",""))[:90]
            src    = str(bearish.get("source",""))
            url    = str(bearish.get("url","") or "")
            pub    = bearish.get("published")
            pstr   = pd.Timestamp(pub).strftime("%d %b") if pd.notnull(pub) else ""
            link   = f'<a href="{url}" target="_blank" style="color:#ef4444;text-decoration:none;font-weight:500">{title}{"..." if len(str(bearish.get("title","")))>90 else ""}</a>' if url not in ("","None") else f'<span style="color:#ef4444;font-weight:500">{title}</span>'
            st.markdown(
                f'<div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.25);'
                f'border-radius:8px;padding:12px 14px;">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#ef4444;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">'
                f'(S) Most Bearish Today</div>'
                f'<div style="font-size:0.82rem;line-height:1.4;margin-bottom:6px">{link}</div>'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.67rem;color:#475569;">'
                f'{ticker} &nbsp;|&nbsp; score: <span style="color:#ef4444">{comp:.3f}</span>'
                f'&nbsp;|&nbsp; {src} &nbsp;|&nbsp; {pstr}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.04);border:1px solid rgba(239,68,68,0.1);'
                'border-radius:8px;padding:12px 14px;color:#475569;font-size:0.82rem">'
                'No bearish headlines in last 3 days</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # -- Top 5 by magnitude
    if not top5.empty:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.67rem;'
            'color:var(--txt-muted);text-transform:uppercase;letter-spacing:0.08em;'
            'margin-bottom:6px">Top 5 by Sentiment Magnitude</div>',
            unsafe_allow_html=True,
        )
        for _, row in top5.iterrows():
            comp   = float(row.get("compound", 0))
            label  = str(row.get("label",""))
            ticker = str(row.get("ticker","")).replace(".NS","")
            title  = str(row.get("title",""))
            url    = str(row.get("url","") or "")
            col    = "#22c55e" if label == "positive" else "#ef4444" if label == "negative" else "#94a3b8"
            icon   = "(B)" if label == "positive" else "(S)" if label == "negative" else "(N)"
            bar_w  = int(abs(comp) * 60)
            link   = f'<a href="{url}" target="_blank" style="color:var(--txt-primary);text-decoration:none">{title[:80]}{"..." if len(title)>80 else ""}</a>' if url not in ("","None") else f'{title[:80]}'

            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;'
                f'border-bottom:1px solid var(--border-soft);">'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
                f'color:{col};width:38px;flex-shrink:0">{icon}<br>{comp:+.2f}</span>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.8rem;line-height:1.4;color:var(--txt-primary);margin-bottom:3px">{link}</div>'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="height:4px;width:{bar_w}px;background:{col};'
                f'border-radius:2px;opacity:0.7;flex-shrink:0"></div>'
                f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#475569">{ticker}</span>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

    # -- Sentiment trend sparkline
    if not trend_df.empty:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.67rem;'
            'color:var(--txt-muted);text-transform:uppercase;letter-spacing:0.08em;'
            'margin-top:12px;margin-bottom:4px">Market Sentiment Trend (14 days)</div>',
            unsafe_allow_html=True,
        )
        spark = build_sparkline(trend_df)
        st.plotly_chart(spark, width="stretch", config={"displayModeBar": False})


def sidebar() -> tuple:
    with st.sidebar:
        st.markdown("### 📡 Sentiment Signals")
        st.markdown(
            '<span style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
            'color:var(--txt-muted)">NSE · FinBERT · 20 stocks</span>',
            unsafe_allow_html=True,
        )
        st.divider()

        available = get_available_tickers()
        fmt       = lambda t: WATCHLIST.get(t, t.replace(".NS",""))
        ticker    = st.selectbox("Stock", available, index=0, format_func=fmt)
        days      = st.slider("History (days)", 7, 180, 30, 7)

        st.divider()
        st.markdown(
            '<span style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
            'color:var(--txt-muted)">AUTO-REFRESH</span>',
            unsafe_allow_html=True,
        )
        ref_label = st.radio(
            "interval", list(REFRESH_OPTIONS.keys()),
            index=0, horizontal=True, label_visibility="collapsed",
        )
        ref_ms = REFRESH_OPTIONS[ref_label]

        st.divider()

        # Manual refresh button
        if st.button("↺  Refresh data now", use_container_width=True):
            with st.spinner("Fetching prices & scoring headlines..."):
                ok, msg = run_refresh(days=7)
            if ok:
                st.success("Data updated!")
                st.cache_data.clear()
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"Refresh failed: {msg}")

        # Last refresh timestamp
        if "last_refresh" in st.session_state:
            lr = st.session_state["last_refresh"]
            ago = int((datetime.now() - lr).total_seconds() / 60)
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
                f'color:var(--txt-muted);margin-top:8px">'
                f'<span class="live-dot"></span>Last refresh {ago} min ago</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:var(--txt-muted)">'
            f'{now_ist.strftime("%d %b %Y · %H:%M")} IST</div>'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;'
            f'color:var(--txt-muted);margin-top:5px">⚠ Educational only. Not financial advice.</div>',
            unsafe_allow_html=True,
        )

    return ticker, days, ref_ms


# -- Main ----------------------------------------------------------------------

def main():
    ticker, days, ref_ms = sidebar()

    # -- Auto-refresh via streamlit-autorefresh -----------------------------
    if ref_ms > 0 and HAS_AUTOREFRESH:
        count = st_autorefresh(interval=ref_ms, key="autorefresh")
        # On each auto-refresh cycle, update data
        if count > 0:
            if "last_refresh" not in st.session_state or \
               (datetime.now() - st.session_state["last_refresh"]).total_seconds() > (ref_ms / 1000 - 30):
                run_refresh(days=3)
                st.session_state["last_refresh"] = datetime.now()
                st.cache_data.clear()

    # -- Load data ----------------------------------------------------------
    price_info   = get_live_price(ticker)
    price_df     = get_prices_df(ticker, days)
    sentiment_df = get_sentiment_df(ticker, days)
    latest       = get_latest_signal(ticker)
    headlines_df = get_headlines(ticker)
    accuracy     = get_backtest_accuracy(ticker)
    all_signals  = get_all_signals()
    bullish_h, bearish_h = get_most_bullish_bearish()
    top5_df      = get_top_impact_headlines(limit=5)
    trend_df     = get_sentiment_trend(days=14)

    signal   = latest.get("signal", "NEUTRAL")
    compound = float(latest.get("weighted_compound", 0) or 0)
    n_heads  = int(latest.get("headline_count", 0) or 0)
    name     = WATCHLIST.get(ticker, ticker.replace(".NS",""))

    # -- Scrolling ticker bar -----------------------------------------------
    price_ticker_row(all_signals)

    # -- Header -------------------------------------------------------------
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.markdown(
        f'<div class="header-bar">'
        f'<p class="header-title">📡 {name} · Sentiment Signal</p>'
        f'<p class="header-sub">FinBERT · NSE · {now_ist.strftime("%d %b %Y %H:%M")} IST</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # -- Top metrics --------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1])

    with c1:
        st.markdown('<div class="metric-label">CURRENT SIGNAL</div>', unsafe_allow_html=True)
        badge(signal, compound)
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.64rem;'
            f'color:var(--txt-muted);margin-top:4px;text-align:center">'
            f'{n_heads} article{"s" if n_heads!=1 else ""}  ·  '
            f'{str(latest.get("date",""))[:10]}</div>',
            unsafe_allow_html=True,
        )

    with c2:
        if price_info:
            lc  = price_info["close"]
            pct = price_info["pct_change"]
            dcl = "up" if pct > 0 else "down" if pct < 0 else "flat"
            metric("LAST CLOSE", f"₹{lc:,.2f}",
                   f"{'(B)' if pct>0 else '(S)' if pct<0 else '-'} {abs(pct):.2f}%", dcl)
        else:
            metric("LAST CLOSE", "-", "Run fetch_prices.py")

    with c3:
        if price_info:
            h, l = price_info.get("high",0), price_info.get("low",0)
            metric("DAY HIGH / LOW", f"₹{h:,.0f}", f"Low ₹{l:,.0f}", "flat")
        else:
            metric("DAY HIGH / LOW", "-")

    with c4:
        pos = float(latest.get("avg_pos", 0) or 0)
        neg = float(latest.get("avg_neg", 0) or 0)
        dcl = "up" if compound > 0.15 else "down" if compound < -0.15 else "flat"
        metric("SENTIMENT", f"{compound:+.4f}", f"pos {pos:.0%}  neg {neg:.0%}", dcl)

    with c5:
        if accuracy is not None:
            dcl = "up" if accuracy > 55 else "down" if accuracy < 45 else "flat"
            metric("SIGNAL ACC.", f"{accuracy:.1f}%",
                   "above random ✓" if accuracy > 55 else "near random", dcl)
        else:
            metric("SIGNAL ACC.", "-", "Need more data")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # -- Chart --------------------------------------------------------------
    fig = build_chart(price_df, sentiment_df, ticker)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # -- Bottom -------------------------------------------------------------
    col_feed, col_rank = st.columns([1.6, 1])

    with col_feed:
        st.markdown('<div class="section-label">HEADLINE FEED</div>', unsafe_allow_html=True)
        headline_feed(headlines_df)

    with col_rank:
        st.markdown(
            f'<div class="section-label">ALL {len(all_signals)} STOCKS TODAY</div>',
            unsafe_allow_html=True,
        )
        ranking_panel(all_signals, ticker)

    # -- Refresh status bar -------------------------------------------------
    if ref_ms > 0:
        interval_str = next(k for k, v in REFRESH_OPTIONS.items() if v == ref_ms)
        st.markdown(
            f'<div class="refresh-bar">'
            f'<span class="live-dot"></span>Auto-refreshing every {interval_str}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="refresh-bar">Auto-refresh off - use sidebar button or set interval</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

def build_sparkline(trend_df: pd.DataFrame) -> go.Figure:
    """Mini sentiment trend line chart for the Impact Analyzer."""
    fig = go.Figure()
    if trend_df.empty:
        return fig

    trend_df = trend_df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    y = trend_df["avg_compound"].astype(float)
    colors = ["#22c55e" if v > 0.05 else "#ef4444" if v < -0.05 else "#94a3b8" for v in y]

    fig.add_trace(go.Bar(
        x=trend_df["date"], y=y,
        marker_color=colors, marker_opacity=0.75,
        hovertemplate="<b>%{x|%d %b}</b><br>Avg sentiment: %{y:+.3f}<extra></extra>",
        name="Market Sentiment",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.4)", line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=110,
        showlegend=False,
        hovermode="x",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=True,
            tickformat="%d %b", tickfont=dict(size=9, color="#64748b", family="IBM Plex Mono"),
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-1.1, 1.1],
        ),
    )
    return fig


