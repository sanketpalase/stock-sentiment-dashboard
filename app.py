import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STOCK SENTIMENT DASHBOARD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── imports with graceful fallbacks ──────────────────────────────────────────
try:
    import praw
    PRAW_OK = True
except ImportError:
    PRAW_OK = False

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    from transformers import pipeline
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False

try:
    import requests
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

try:
    import ta
    TA_OK = True
except ImportError:
    TA_OK = False

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3154;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #8b92b2; margin-top: 4px; }
    .positive { color: #00d4a3; }
    .negative { color: #ff4c6a; }
    .neutral  { color: #f7b731; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c3c8e8;
        border-bottom: 1px solid #2d3154;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .news-item {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 3px solid #5b6bdc;
        font-size: 0.88rem;
    }
    .tag-positive { background:#00d4a333; color:#00d4a3; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
    .tag-negative { background:#ff4c6a33; color:#ff4c6a; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
    .tag-neutral  { background:#f7b73133; color:#f7b731; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
    div[data-testid="stTabs"] button { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_sentiment_model():
    if TRANSFORMERS_OK:
        try:
            return pipeline(
                "text-classification",
                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                top_k=None,
            )
        except Exception:
            pass
    return None


def simple_sentiment(text: str) -> dict:
    """Rule-based fallback sentiment."""
    pos_words = {"bullish","buy","growth","profit","surge","rally","strong","beat","gain","up","positive","good","great","excellent","high","rise","boom"}
    neg_words = {"bearish","sell","loss","crash","fall","weak","miss","drop","down","negative","bad","poor","low","decline","bust","risk","concern","debt"}
    tokens = set(text.lower().split())
    pos = len(tokens & pos_words)
    neg = len(tokens & neg_words)
    total = pos + neg or 1
    return {"positive": pos/total, "negative": neg/total, "neutral": 1 - (pos+neg)/max(total,1)}


def analyze_sentiment(texts: list, model) -> list:
    results = []
    for text in texts:
        if model:
            try:
                raw = model(text[:512])[0]
                scores = {r["label"].lower(): r["score"] for r in raw}
                label = max(scores, key=scores.get)
                results.append({"label": label, "scores": scores, "text": text})
                continue
            except Exception:
                pass
        scores = simple_sentiment(text)
        label = max(scores, key=scores.get)
        results.append({"label": label, "scores": scores, "text": text})
    return results


@st.cache_data(ttl=300)
def fetch_reddit_posts(keyword: str, reddit_cfg: dict, limit: int = 20) -> list:
    if PRAW_OK and reddit_cfg.get("client_id"):
        try:
            reddit = praw.Reddit(**reddit_cfg, user_agent="SentimentDash/1.0")
            posts = []
            for sub in reddit.subreddit("all").search(keyword, limit=limit):
                posts.append(sub.title + " " + (sub.selftext[:200] if sub.selftext else ""))
            return posts
        except Exception as e:
            st.warning(f"Reddit API error: {e}")
    # Mock fallback
    mock = [
        f"{keyword} stock looking bullish after strong quarterly earnings beat expectations",
        f"Should I buy {keyword}? The fundamentals look solid with growing revenue",
        f"{keyword} facing regulatory headwinds — risk of correction ahead",
        f"Analyst upgrades {keyword} to BUY with 20% upside target",
        f"{keyword} insider selling raises concerns among retail investors",
        f"Technical breakout in {keyword} — momentum traders piling in",
        f"{keyword} dividend cut disappoints long-term holders",
        f"Strong institutional buying detected in {keyword} this week",
    ]
    return mock


@st.cache_data(ttl=300)
def fetch_news(keyword: str) -> list:
    if BS4_OK:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = f"https://economictimes.indiatimes.com/searchresult.cms?query={keyword}"
            resp = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(resp.text, "html.parser")
            headlines = [h.get_text(strip=True) for h in soup.select("h3, h4")[:15]]
            return [h for h in headlines if len(h) > 20][:12]
        except Exception:
            pass
    # Mock fallback
    return [
        f"{keyword} Q4 results: Revenue up 18% YoY, beats street estimates",
        f"RBI policy impact: What it means for {keyword} going forward",
        f"{keyword} announces strategic partnership to expand into new markets",
        f"FII outflows put pressure on {keyword} amid global risk-off sentiment",
        f"{keyword} management guidance raised — positive outlook for FY26",
        f"Sector rotation favors {keyword}; mutual funds increase holdings",
    ]


@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    if YF_OK:
        try:
            # Try NSE suffix first, then BSE, then as-is
            for suffix in [".NS", ".BO", ""]:
                t = ticker + suffix if suffix else ticker
                df = yf.download(t, period=period, progress=False, auto_adjust=True)
                if not df.empty and len(df) > 10:
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    return df.reset_index()
        except Exception as e:
            st.warning(f"yfinance error: {e}")
    # Generate realistic mock OHLCV
    dates = pd.date_range(end=datetime.today(), periods=130, freq="B")
    np.random.seed(hash(ticker) % 2**31)
    close = 1000 * np.exp(np.cumsum(np.random.randn(130) * 0.012))
    high  = close * (1 + np.abs(np.random.randn(130)) * 0.01)
    low   = close * (1 - np.abs(np.random.randn(130)) * 0.01)
    open_ = close * (1 + np.random.randn(130) * 0.008)
    vol   = np.random.randint(500_000, 5_000_000, 130).astype(float)
    return pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    # Moving averages
    df["SMA20"]  = c.rolling(20).mean()
    df["SMA50"]  = c.rolling(50).mean()
    df["EMA20"]  = c.ewm(span=20).mean()

    # Bollinger Bands
    std20 = c.rolling(20).std()
    df["BB_upper"] = df["SMA20"] + 2 * std20
    df["BB_lower"] = df["SMA20"] - 2 * std20

    # RSI
    if TA_OK:
        df["RSI"] = ta.momentum.RSIIndicator(c).rsi()
        macd_obj  = ta.trend.MACD(c)
        df["MACD"]        = macd_obj.macd()
        df["MACD_signal"] = macd_obj.macd_signal()
        df["MACD_hist"]   = macd_obj.macd_diff()
        df["ATR"]         = ta.volatility.AverageTrueRange(df["High"].astype(float), df["Low"].astype(float), c).average_true_range()
        df["OBV"]         = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    else:
        # Manual RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - 100 / (1 + rs)
        # Manual MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
        # Manual ATR
        h = df["High"].astype(float); l = df["Low"].astype(float)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["OBV"] = (np.sign(c.diff()) * v).cumsum()

    # Returns & Volatility
    df["Returns"]    = c.pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100
    return df


def compute_quant_stats(df: pd.DataFrame) -> dict:
    c = df["Close"].astype(float)
    ret = c.pct_change().dropna()
    total_ret = (c.iloc[-1] / c.iloc[0] - 1) * 100
    vol_ann   = ret.std() * np.sqrt(252) * 100
    sharpe    = (ret.mean() * 252) / (ret.std() * np.sqrt(252) + 1e-9)
    roll_max  = c.cummax()
    drawdown  = ((c - roll_max) / roll_max * 100)
    max_dd    = drawdown.min()
    rsi_now   = df["RSI"].iloc[-1] if "RSI" in df else None
    return {
        "current_price": c.iloc[-1],
        "total_return":  total_ret,
        "volatility":    vol_ann,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "rsi":           rsi_now,
        "sma20":         df["SMA20"].iloc[-1],
        "sma50":         df["SMA50"].iloc[-1],
    }


def sentiment_vs_price_correlation(sentiment_scores: list, df: pd.DataFrame) -> float:
    """Approximate correlation: map daily sentiment score to last N trading days."""
    n = min(len(sentiment_scores), len(df) - 1)
    if n < 3:
        return 0.0
    prices = df["Close"].astype(float).pct_change().dropna().values[-n:]
    scores = np.array([s["scores"].get("positive", 0.5) - s["scores"].get("negative", 0.5) for s in sentiment_scores[-n:]])
    if len(prices) != len(scores):
        min_len = min(len(prices), len(scores))
        prices = prices[-min_len:]
        scores = scores[-min_len:]
    corr = np.corrcoef(scores, prices)[0, 1]
    return round(float(corr), 3) if not np.isnan(corr) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

DARK = dict(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#c3c8e8", size=12),
            xaxis=dict(gridcolor="#1e2130", showgrid=True),
            yaxis=dict(gridcolor="#1e2130", showgrid=True))


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00d4a3", decreasing_line_color="#ff4c6a",
        name="OHLC"), row=1, col=1)
    # Overlays
    for col, color, name in [("SMA20","#f7b731","SMA 20"), ("SMA50","#5b6bdc","SMA 50"), ("BB_upper","#aaa","BB Upper"), ("BB_lower","#aaa","BB Lower")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], line=dict(color=color, width=1, dash="dot" if "BB" in col else "solid"), name=name), row=1, col=1)
    # Volume
    colors = ["#00d4a3" if r >= 0 else "#ff4c6a" for r in df["Returns"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=colors, name="Volume", opacity=0.7), row=2, col=1)
    # RSI
    if "RSI" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#f7b731", width=1.5), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="#ff4c6a", dash="dash", width=1), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#00d4a3", dash="dash", width=1), row=3, col=1)

    fig.update_layout(title=f"{ticker} — Price, Volume & RSI", height=650,
                      showlegend=True, legend=dict(orientation="h", y=1.02),
                      xaxis_rangeslider_visible=False, **DARK)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    if "MACD" not in df:
        return go.Figure()
    fig = make_subplots(rows=1, cols=1)
    colors = ["#00d4a3" if v >= 0 else "#ff4c6a" for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_hist"], marker_color=colors, name="MACD Histogram"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], line=dict(color="#5b6bdc", width=1.5), name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"], line=dict(color="#f7b731", width=1.5), name="Signal"))
    fig.update_layout(title="MACD", height=280, **DARK)
    return fig


def volatility_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "Volatility" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Volatility"],
                                 fill="tozeroy", fillcolor="rgba(91,107,220,0.15)",
                                 line=dict(color="#5b6bdc", width=2), name="Annualised Vol %"))
    fig.update_layout(title="Rolling 20-Day Volatility (Annualised %)", height=280, **DARK)
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    c = df["Close"].astype(float)
    dd = ((c - c.cummax()) / c.cummax() * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=dd,
                             fill="tozeroy", fillcolor="rgba(255,76,106,0.15)",
                             line=dict(color="#ff4c6a", width=2), name="Drawdown %"))
    fig.update_layout(title="Drawdown from Peak (%)", height=280, **DARK)
    return fig


def sentiment_donut(pos: float, neg: float, neu: float) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[pos, neg, neu],
        hole=0.62,
        marker_colors=["#00d4a3", "#ff4c6a", "#f7b731"],
        textinfo="percent+label",
        textfont_size=13,
    ))
    fig.update_layout(height=320, showlegend=False,
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      font=dict(color="#c3c8e8"),
                      annotations=[dict(text=f"{int(pos*100)}%<br>Positive", x=0.5, y=0.5,
                                        font=dict(size=16, color="#00d4a3"), showarrow=False)])
    return fig


def sentiment_timeline(results: list, title: str = "Sentiment Timeline") -> go.Figure:
    pos_s = [r["scores"].get("positive", 0) for r in results]
    neg_s = [r["scores"].get("negative", 0) for r in results]
    neu_s = [r["scores"].get("neutral",  0) for r in results]
    idx   = list(range(1, len(results) + 1))
    fig   = go.Figure()
    fill_colors = {"#00d4a3": "rgba(0,212,163,0.08)", "#ff4c6a": "rgba(255,76,106,0.08)", "#f7b731": "rgba(247,183,49,0.08)"}
    for arr, color, name in [(pos_s,"#00d4a3","Positive"), (neg_s,"#ff4c6a","Negative"), (neu_s,"#f7b731","Neutral")]:
        fig.add_trace(go.Scatter(x=idx, y=arr, mode="lines+markers",
                                 line=dict(color=color, width=2), name=name,
                                 fill="tozeroy", fillcolor=fill_colors[color]))
    fig.update_layout(title=title, height=300, **DARK)
    return fig


def combined_sentiment_price_chart(sentiment_results: list, df: pd.DataFrame, ticker: str) -> go.Figure:
    n = min(len(sentiment_results), 30)
    sent_scores = [r["scores"].get("positive", 0.5) - r["scores"].get("negative", 0.5) for r in sentiment_results[-n:]]
    prices = df["Close"].astype(float).values[-n:]
    x = list(range(1, n + 1))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=prices, name=f"{ticker} Price",
                             line=dict(color="#5b6bdc", width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=x, y=sent_scores, name="Net Sentiment",
                         marker_color=["#00d4a3" if v >= 0 else "#ff4c6a" for v in sent_scores],
                         opacity=0.7), secondary_y=True)
    fig.update_layout(title=f"Price vs Sentiment Overlap — {ticker}", height=350, **DARK)
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Net Sentiment", secondary_y=True)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Dashboard Settings")
    ticker  = st.text_input("Stock Ticker / Keyword", value="RELIANCE", placeholder="e.g. RELIANCE, INFY, TCS").strip().upper()
    period  = st.selectbox("Price History", ["1mo","3mo","6mo","1y","2y"], index=2)
    n_posts = st.slider("Reddit Posts to Fetch", 5, 50, 20)

    st.markdown("---")
    st.markdown("#### Reddit API (optional)")
    r_id  = st.text_input("Client ID",     type="password", placeholder="Leave blank for mock data")
    r_sec = st.text_input("Client Secret", type="password")
    r_usr = st.text_input("Username")
    r_pwd = st.text_input("Password", type="password")

    reddit_cfg = dict(client_id=r_id, client_secret=r_sec,
                      username=r_usr, password=r_pwd) if r_id else {}

    st.markdown("---")
    run_btn = st.button("🔍 Analyse", use_container_width=True, type="primary")
    st.markdown(f"<small>Library status: {'✅' if YF_OK else '⚠️'} yfinance &nbsp; {'✅' if TRANSFORMERS_OK else '⚠️'} transformers &nbsp; {'✅' if TA_OK else '⚠️'} ta</small>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"# 📈 STOCK SENTIMENT DASHBOARD")
st.markdown(f"<small style='color:#8b92b2'>Analysing: <b style='color:#c3c8e8'>{ticker}</b> &nbsp;|&nbsp; {datetime.now().strftime('%d %b %Y %H:%M')}</small>", unsafe_allow_html=True)

if not run_btn and "last_ticker" not in st.session_state:
    st.info("👈 Enter a ticker or keyword in the sidebar and click **Analyse**.")
    st.stop()

if run_btn:
    st.session_state["last_ticker"] = ticker
    st.session_state["period"]      = period

ticker = st.session_state.get("last_ticker", ticker)
period = st.session_state.get("period", period)


# ── fetch everything ──────────────────────────────────────────────────────────
with st.spinner("Fetching data & running analysis…"):
    model         = load_sentiment_model()
    reddit_posts  = fetch_reddit_posts(ticker, reddit_cfg, n_posts)
    news_items    = fetch_news(ticker)
    all_texts     = reddit_posts + news_items
    all_results   = analyze_sentiment(all_texts, model)
    reddit_res    = all_results[:len(reddit_posts)]
    news_res      = all_results[len(reddit_posts):]

    price_df = fetch_price_data(ticker, period)
    price_df = add_technical_indicators(price_df)
    stats    = compute_quant_stats(price_df)
    corr     = sentiment_vs_price_correlation(all_results, price_df)

# Aggregate sentiment
pos_pct = sum(1 for r in all_results if r["label"] == "positive") / len(all_results)
neg_pct = sum(1 for r in all_results if r["label"] == "negative") / len(all_results)
neu_pct = 1 - pos_pct - neg_pct
overall = "BULLISH 🟢" if pos_pct > 0.5 else ("BEARISH 🔴" if neg_pct > 0.5 else "NEUTRAL 🟡")


# ── top KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    ("Current Price", f"₹{stats['current_price']:,.1f}", "neutral"),
    ("Return", f"{stats['total_return']:+.1f}%", "positive" if stats["total_return"] >= 0 else "negative"),
    ("Volatility", f"{stats['volatility']:.1f}%", "neutral"),
    ("Sharpe Ratio", f"{stats['sharpe']:.2f}", "positive" if stats["sharpe"] >= 1 else "negative"),
    ("Max Drawdown", f"{stats['max_drawdown']:.1f}%", "negative"),
    ("Overall Mood", overall, "positive" if "BULL" in overall else ("negative" if "BEAR" in overall else "neutral")),
]
for col, (label, val, cls) in zip([c1,c2,c3,c4,c5,c6], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {cls}">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 Sentiment Analysis", "📈 Stock Analysis", "🔗 Combined View", "📰 News & Posts"])


# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Sentiment Overview")
    col_d, col_r, col_n = st.columns(3)

    with col_d:
        st.plotly_chart(sentiment_donut(pos_pct, neg_pct, neu_pct), use_container_width=True)

    with col_r:
        st.markdown("**Reddit Sentiment**")
        r_pos = sum(1 for r in reddit_res if r["label"] == "positive") / max(len(reddit_res), 1)
        r_neg = sum(1 for r in reddit_res if r["label"] == "negative") / max(len(reddit_res), 1)
        r_neu = 1 - r_pos - r_neg
        st.plotly_chart(sentiment_donut(r_pos, r_neg, r_neu), use_container_width=True)

    with col_n:
        st.markdown("**News Sentiment**")
        n_pos = sum(1 for r in news_res if r["label"] == "positive") / max(len(news_res), 1)
        n_neg = sum(1 for r in news_res if r["label"] == "negative") / max(len(news_res), 1)
        n_neu = 1 - n_pos - n_neg
        st.plotly_chart(sentiment_donut(n_pos, n_neg, n_neu), use_container_width=True)

    st.plotly_chart(sentiment_timeline(all_results, "Sentiment Scores Across All Posts"), use_container_width=True)

    # Sentiment breakdown table
    df_sent = pd.DataFrame([{
        "Source": "Reddit" if i < len(reddit_res) else "News",
        "Text": r["text"][:90] + "…",
        "Label": r["label"].capitalize(),
        "Positive": f"{r['scores'].get('positive',0):.2f}",
        "Negative": f"{r['scores'].get('negative',0):.2f}",
        "Neutral":  f"{r['scores'].get('neutral', 0):.2f}",
    } for i, r in enumerate(all_results)])
    st.dataframe(df_sent, use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Price & Technical Indicators")
    st.plotly_chart(candlestick_chart(price_df, ticker), use_container_width=True)

    col_m, col_v = st.columns(2)
    with col_m:
        st.plotly_chart(macd_chart(price_df), use_container_width=True)
    with col_v:
        st.plotly_chart(volatility_chart(price_df), use_container_width=True)

    col_dd, col_obv = st.columns(2)
    with col_dd:
        st.plotly_chart(drawdown_chart(price_df), use_container_width=True)
    with col_obv:
        if "OBV" in price_df:
            fig_obv = go.Figure(go.Scatter(x=price_df["Date"], y=price_df["OBV"],
                                           line=dict(color="#5b6bdc", width=2), name="OBV"))
            fig_obv.update_layout(title="On-Balance Volume (OBV)", height=280, **DARK)
            st.plotly_chart(fig_obv, use_container_width=True)

    st.markdown("---")
    st.markdown("### Key Quantitative Metrics")
    k1, k2, k3, k4 = st.columns(4)
    rsi_val = stats["rsi"]
    rsi_signal = "Overbought ⚠️" if rsi_val > 70 else ("Oversold 🟢" if rsi_val < 30 else "Neutral") if rsi_val else "N/A"
    trend_signal = "Bullish ↑" if stats["sma20"] > stats["sma50"] else "Bearish ↓"
    with k1:
        st.metric("RSI (14)", f"{rsi_val:.1f}" if rsi_val else "N/A", rsi_signal)
    with k2:
        st.metric("SMA 20 vs 50", trend_signal, f"₹{stats['sma20']:,.1f} vs ₹{stats['sma50']:,.1f}")
    with k3:
        st.metric("Annualised Volatility", f"{stats['volatility']:.2f}%")
    with k4:
        st.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}", "Good ✅" if stats["sharpe"] > 1 else "Low ⚠️")

    # Raw OHLCV table
    with st.expander("📋 Raw OHLCV Data"):
        disp = price_df[["Date","Open","High","Low","Close","Volume"]].tail(30).copy()
        for col in ["Open","High","Low","Close"]:
            disp[col] = disp[col].map(lambda x: f"₹{x:,.2f}")
        disp["Volume"] = disp["Volume"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Sentiment × Price Correlation")
    corr_color = "#00d4a3" if corr > 0.2 else ("#ff4c6a" if corr < -0.2 else "#f7b731")
    st.markdown(f"""
    <div class="metric-card" style="max-width:320px;">
        <div class="metric-value" style="color:{corr_color}">{corr:+.3f}</div>
        <div class="metric-label">Pearson Correlation — Sentiment Score vs Daily Returns</div>
    </div><br>
    """, unsafe_allow_html=True)

    interp = ("Positive correlation: bullish sentiment tends to coincide with upward price moves." if corr > 0.2
              else "Negative correlation: bearish sentiment tends to coincide with downward price moves." if corr < -0.2
              else "Weak correlation: sentiment and price moves are largely independent in this window.")
    st.info(f"**Interpretation:** {interp}")

    st.plotly_chart(combined_sentiment_price_chart(all_results, price_df, ticker), use_container_width=True)

    st.markdown("### Signal Summary")
    sig_data = {
        "Signal": ["RSI", "MACD Crossover", "MA Trend", "Sentiment Bias", "Price Momentum"],
        "Value": [
            f"{rsi_val:.1f}" if rsi_val else "N/A",
            "Bullish" if price_df["MACD_hist"].iloc[-1] > 0 else "Bearish",
            "Bullish" if stats["sma20"] > stats["sma50"] else "Bearish",
            "Bullish" if pos_pct > neg_pct else ("Bearish" if neg_pct > pos_pct else "Neutral"),
            "Positive" if price_df["Returns"].tail(5).mean() > 0 else "Negative",
        ],
        "Interpretation": [
            rsi_signal,
            "MACD above signal" if price_df["MACD_hist"].iloc[-1] > 0 else "MACD below signal",
            "SMA20 above SMA50 (Golden cross zone)" if stats["sma20"] > stats["sma50"] else "SMA20 below SMA50 (Death cross zone)",
            f"{pos_pct*100:.0f}% positive posts",
            f"Avg 5-day return: {price_df['Returns'].tail(5).mean()*100:+.2f}%",
        ],
    }
    st.dataframe(pd.DataFrame(sig_data), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📰 News Headlines")
    for item, res in zip(news_items, news_res):
        tag_cls  = f"tag-{res['label']}"
        tag_text = res["label"].upper()
        st.markdown(f"""
        <div class="news-item">
            {item}&nbsp;&nbsp;<span class="{tag_cls}">{tag_text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 💬 Reddit Posts")
    for post, res in zip(reddit_posts, reddit_res):
        tag_cls  = f"tag-{res['label']}"
        tag_text = res["label"].upper()
        st.markdown(f"""
        <div class="news-item" style="border-left-color:#ff5700;">
            {post[:150]}…&nbsp;&nbsp;<span class="{tag_cls}">{tag_text}</span>
        </div>""", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<small style='color:#4a4f6a'>Data may include mock values where APIs are unavailable. For production use, configure Reddit API keys and ensure yfinance can reach NSE/BSE tickers.</small>", unsafe_allow_html=True)
