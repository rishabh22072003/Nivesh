# app.py ── Nivesh AI Dashboard v5
# Changes from v4:
#   • Dark/light toggle REMOVED (was unreliable — use .streamlit/config.toml)
#   • Unified "Nivesh AI Consensus" signal — reconciles conflicting Technical/Prediction
#   • Conflict explainer shown whenever Technical ≠ Prediction direction
#   • Technical tab now clearly labelled as "individual signals only"
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from main import FinancialBotInterface
from stock_agent import MasterAgent, calculate_investment
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Nivesh AI - Stock Analysis Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ── Session state defaults ─────────────────────────────────────────────────────
for k, v in [
    ("chat_history",     []),
    ("pending_question", None),
    ("current_ticker",   None),
    ("news_data",        None),
    ("news_analysis",    None),
    ("master_summary",   None),
    ("last_prediction",  None),
    ("user_name",        "Investor"),
    ("disclaimer_accepted", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Disclaimer / Terms Gate ───────────────────────────────────────────────────
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.markdown("""
<style>
.disc-outer{
  max-width:700px;margin:40px auto;
  background:linear-gradient(135deg,#ffffff,#f0f6ff);
  border:2px solid #0077b6;border-radius:20px;
  padding:36px 40px;box-shadow:0 8px 32px rgba(0,119,182,0.15);
}
.disc-title{
  font-size:26px;font-weight:800;color:#023e8a;
  text-align:center;margin-bottom:4px;
}
.disc-sub{
  font-size:13px;color:#0077b6;text-align:center;
  margin-bottom:24px;letter-spacing:0.5px;
}
.disc-section{
  background:#e8f4fd;border-left:4px solid #0077b6;
  border-radius:0 10px 10px 0;padding:12px 16px;
  margin:12px 0;
}
.disc-section-title{
  font-size:13px;font-weight:700;color:#023e8a;
  text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;
}
.disc-section p{font-size:13px;color:#1a1a2e;margin:0;line-height:1.7;}
.disc-warn{
  background:linear-gradient(135deg,#fff3cd,#ffe69c);
  border:1px solid #e6a817;border-radius:10px;
  padding:14px 18px;margin:16px 0;
  font-size:13px;color:#3d2b00;line-height:1.7;
}
.disc-footer{
  text-align:center;font-size:12px;color:#0077b6;
  margin-top:20px;line-height:1.8;
}
</style>
<div class="disc-outer">
  <div class="disc-title">📜 Nivesh AI — निवेशक सहमति पत्र</div>
  <div class="disc-sub">Investor Acknowledgement & Risk Disclosure | कृपया ध्यान से पढ़ें और समझें</div>

  <div class="disc-section">
    <div class="disc-section-title">⚠️ जोखिम की चेतावनी / Risk Warning</div>
    <p>
      <b>निवेश विद्या जोखिम के अधीन है।</b> शेयर बाज़ार में निवेश करने से आपकी मूल राशि का नुकसान हो सकता है।
      पिछला प्रदर्शन भविष्य के परिणामों की गारंटी नहीं देता।<br><br>
      <b>Investing is subject to market risks.</b> The value of your investments can go up or down.
      Past performance does not guarantee future results. You may lose some or all of your invested capital.
    </p>
  </div>

  <div class="disc-section">
    <div class="disc-section-title">🤖 AI की सीमाएं / AI Limitations</div>
    <p>
      Nivesh AI एक <b>शैक्षिक उपकरण</b> है। यह SEBI-पंजीकृत निवेश सलाहकार नहीं है।
      इसकी predictions statistical models पर आधारित हैं — ये 100% सटीक नहीं होतीं।<br><br>
      Nivesh AI is an <b>educational tool only</b>. It is NOT a SEBI-registered investment advisor.
      All predictions are based on statistical models and historical data — they are NOT guarantees.
    </p>
  </div>

  <div class="disc-section">
    <div class="disc-section-title">📋 आपकी ज़िम्मेदारी / Your Responsibility</div>
    <p>
      कोई भी निवेश निर्णय लेने से पहले <b>अपना खुद का शोध करें (DYOR — Do Your Own Research)</b>।
      ज़रूरत पड़े तो SEBI-पंजीकृत वित्तीय सलाहकार से परामर्श लें।<br><br>
      Before making any investment decision, <b>always do your own research</b>.
      Consult a SEBI-registered financial advisor if needed. Never invest money you cannot afford to lose.
    </p>
  </div>

  <div class="disc-warn">
    🔔 <b>याद रखें / Remember:</b><br>
    • शेयर बाज़ार में <b>कोई guaranteed return नहीं होता।</b><br>
    • Nivesh AI की signals <b>सुझाव हैं, आदेश नहीं।</b><br>
    • <b>पहले सीखो, फिर निवेश करो।</b> Gyan Kendra tab ज़रूर पढ़ें।<br>
    • <b>कभी भी उधार लेकर या EMI के पैसे से</b> शेयर न खरीदें।<br><br>
    No guaranteed returns exist in stock markets. Nivesh AI signals are suggestions, not commands.
    Never invest borrowed money or funds needed for daily expenses.
  </div>

  <div class="disc-footer">
    इस सॉफ्टवेयर का उपयोग करके आप उपरोक्त सभी शर्तों से सहमत होते हैं।<br>
    By clicking <b>"मैं सहमत हूँ / I Agree"</b> below, you confirm you have read,
    understood, and accepted all terms and risk disclosures above.
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        name_input = st.text_input(
            "अपना नाम लिखें / Enter your name",
            placeholder="e.g. Rahul Sharma",
            help="We use your name only to personalise your experience in this session.",
        )
        if st.button(
            "✅ मैं सहमत हूँ — I Have Read & Agree to All Terms",
            type="primary",
            use_container_width=True,
            disabled=(len(name_input.strip()) < 2),
        ):
            st.session_state.disclaimer_accepted = True
            st.session_state.user_name = name_input.strip()
            st.rerun()

        st.caption("⚠️ आगे बढ़ने के लिए पहले नाम लिखें। Name required to proceed.")
    st.stop()

# ── Personalised greeting after disclaimer ─────────────────────────────────────
user_name = st.session_state.get("user_name", "Investor")

# ── Custom component styles (only our own classes, never Streamlit internals) ──
st.markdown("""<style>
/* ── Prediction cards ── light theme */
.pred-card{
  background:linear-gradient(135deg,#e8f4fd,#d0e8f7,#b8dcf0);
  border:1px solid #0077b644;border-radius:14px;
  padding:18px 20px;margin:6px 0;color:#1a1a2e;min-height:110px;
  box-shadow:0 2px 8px rgba(0,119,182,0.10);}
.pred-card h3{margin:0 0 6px 0;font-size:11px;color:#0077b6;
  letter-spacing:1.5px;text-transform:uppercase;}
.pred-card .price{font-size:26px;font-weight:700;color:#023e8a;}
.pred-card .delta{font-size:12px;margin-top:4px;}

/* ── Buy banners ── light */
.buy-banner{border-radius:12px;padding:16px 20px;margin:10px 0;
  font-size:15px;font-weight:600;color:#1a1a2e;}
.buy-strong-buy {background:linear-gradient(90deg,#d4f7d4,#a8edbc);border:2px solid #1b9e4b;}
.buy-buy        {background:linear-gradient(90deg,#e0f7e9,#c3f0d5);border:2px solid #2dc653;}
.buy-hold       {background:linear-gradient(90deg,#fff9e6,#fff0b3);border:2px solid #e6a817;}
.buy-sell       {background:linear-gradient(90deg,#fde8e8,#facaca);border:2px solid #e03131;}
.buy-strong-sell{background:linear-gradient(90deg,#ffd0d0,#ffb3b3);border:2px solid #c0392b;}

/* ── Consensus box ── light */
.consensus-box{border-radius:14px;padding:20px 24px;margin:12px 0;
  font-size:15px;font-weight:600;color:#1a1a2e;border:2px solid #0077b6;
  box-shadow:0 2px 12px rgba(0,0,0,0.08);}
.consensus-buy     {background:linear-gradient(135deg,#d4f7d4,#b8f0c8,#a0e8b8);border-color:#1b9e4b;color:#0a3d1a;}
.consensus-hold    {background:linear-gradient(135deg,#fff9e6,#fff3cc,#ffe8a0);border-color:#e6a817;color:#3d2b00;}
.consensus-sell    {background:linear-gradient(135deg,#fde8e8,#facaca,#f5a0a0);border-color:#e03131;color:#3d0000;}
.consensus-conflict{background:linear-gradient(135deg,#ede9ff,#ddd5ff,#ccc0ff);border-color:#5c4fd6;color:#1a1040;}
.consensus-sub{font-size:13px;font-weight:400;opacity:0.90;margin-top:6px;line-height:1.6;}
.signal-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;}
.signal-pill{border-radius:20px;padding:4px 14px;font-size:12px;font-weight:600;border:1px solid;}
.pill-buy {background:#d4f7d466;border-color:#1b9e4b;color:#0a3d1a;}
.pill-sell{background:#fde8e866;border-color:#e03131;color:#3d0000;}
.pill-hold{background:#fff9e666;border-color:#e6a817;color:#3d2b00;}
.conflict-explain{background:#f0eeff;border:1px solid #5c4fd644;border-radius:10px;
  padding:14px 18px;margin:8px 0;color:#1a1040;font-size:13px;line-height:1.7;}

/* ── Calc card ── light */
.calc-card{background:linear-gradient(135deg,#eef4fb,#ddeaf7);
  border:1px solid #90c2e7;border-radius:12px;
  padding:20px;margin:8px 0;color:#1a1a2e;
  box-shadow:0 2px 8px rgba(0,0,0,0.07);}
.calc-profit{color:#1b9e4b;font-size:22px;font-weight:700;}
.calc-loss  {color:#e03131;font-size:22px;font-weight:700;}
.calc-label {font-size:11px;color:#0077b6;text-transform:uppercase;letter-spacing:1px;}

/* ── News cards ── light */
.news-card{background:linear-gradient(135deg,#f0f6ff,#e4eeff);
  border-left:3px solid #0077b6;border-radius:8px;
  padding:12px 16px;margin:5px 0;color:#1a1a2e;
  box-shadow:0 1px 4px rgba(0,0,0,0.07);}
.news-card-global{border-left-color:#e6a817;}
.news-title{font-size:14px;font-weight:600;color:#1a1a2e;}
.news-meta {font-size:11px;color:#0077b6;margin-top:3px;}
.news-snip {font-size:12px;color:#555;margin-top:4px;}

/* ── Summary box ── light */
.summary-box{background:linear-gradient(135deg,#eef4fb,#ddeaf7);
  border:1px solid #90c2e7;border-radius:12px;
  padding:20px;margin:10px 0;color:#1a1a2e;line-height:1.7;
  box-shadow:0 2px 8px rgba(0,0,0,0.07);}

/* ── Conflict card ── light */
.conflict-card{background:linear-gradient(135deg,#fff0f0,#ffe0e0);
  border-left:3px solid #e03131;border-radius:8px;
  padding:12px;margin:4px 0;color:#2c1010;font-size:13px;}

/* ── Sentiment spans ── light */
.sbull{color:#1b9e4b;font-weight:700;}
.sbear{color:#e03131;font-weight:700;}
.sneut{color:#e6a817;font-weight:700;}
.up  {color:#1b9e4b;}
.down{color:#e03131;}
.flat{color:#e6a817;}
</style>""", unsafe_allow_html=True)

# ── API key ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("⚠️ Nivesh AI not configured. Please set GEMINI_API_KEY.")
    st.stop()

bot_interface = FinancialBotInterface(API_KEY, model="gemini-2.5-flash")

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_indian_ticker(sym):
    sym = sym.upper().strip()
    return sym if (".NS" in sym or ".BO" in sym) else f"{sym}.NS"

def safe_fmt(v, fmt="currency", d=2):
    if v is None or v == "N/A": return "N/A"
    try:
        if isinstance(v, str): v = float(v.replace(",", ""))
        if fmt == "currency":   return f"₹{v:,.{d}f}"
        if fmt == "percentage": return f"{v*100:.{d}f}%"
        if fmt == "number":     return f"{v:,.{d}f}"
        return str(v)
    except: return "N/A"

def fmt_market_cap(v):
    if v is None: return "N/A"
    try:
        v = float(v)
        if v >= 1e12: return f"₹{v/1e12:.2f}T"
        if v >= 1e9:  return f"₹{v/1e9:.2f}B"
        if v >= 1e7:  return f"₹{v/1e7:.2f} Cr"
        if v >= 1e5:  return f"₹{v/1e5:.2f} Lakh"
        return f"₹{v:,.0f}"
    except: return "N/A"

def get_live_price_info(stock):
    try:
        info = stock.info
        cp   = info.get("currentPrice") or info.get("regularMarketPrice")
        prev = info.get("previousClose")
        if cp and prev:
            chg = cp - prev
            return cp, chg, chg / prev * 100
    except: pass
    return None, None, None

def delta_card(pred, base):
    if pred is None or base is None: return ""
    pct   = (pred - base) / base * 100
    cls   = "up" if pct > 0 else "down"
    arrow = "▲" if pct > 0 else "▼"
    return f'<span class="{cls}">{arrow} {pct:+.2f}% vs current</span>'

def sentiment_css(label):
    l = label.lower()
    if "bull" in l: return "sbull"
    if "bear" in l: return "sbear"
    return "sneut"

def buy_banner_css(signal):
    s = signal.lower()
    if "strong buy"  in s: return "buy-strong-buy"
    if "buy"         in s: return "buy-buy"
    if "strong sell" in s: return "buy-strong-sell"
    if "sell"        in s: return "buy-sell"
    return "buy-hold"

def pl_color(val):
    return "calc-profit" if val >= 0 else "calc-loss"

def render_list(items, header, icon):
    clean = [str(i) for i in (items or []) if i is not None and str(i).strip()]
    if not clean: return
    st.subheader(f"{icon} {header}")
    for item in clean:
        st.write(f"• {item}")

def _pill(label, verdict):
    v   = verdict.lower()
    cls = "pill-buy" if "buy" in v else "pill-sell" if "sell" in v else "pill-hold"
    return f'<span class="signal-pill {cls}">{label}: {verdict}</span>'

def build_consensus(tech_short, fund_long, pred_signal, pred_pct, sentiment_score):
    """Reconcile all signals into one honest verdict with explanation."""
    t = tech_short.lower()
    f = fund_long.lower()
    p = pred_signal.lower()

    buy_votes  = sum(1 for v in [t, f, p] if "buy"  in v)
    sell_votes = sum(1 for v in [t, f, p] if "sell" in v)

    # True conflict = technical says sell but model says buy (or vice versa)
    tech_sell    = "sell" in t
    model_buy    = "buy"  in p
    tech_buy     = "buy"  in t
    model_sell   = "sell" in p
    conflicting  = (tech_sell and model_buy) or (tech_buy and model_sell)

    conflict_note = ""

    if conflicting:
        css  = "consensus-conflict"
        head = "⚠️ Mixed Signals — Caution Advised"
        exp  = (
            f"Technical analysis says <b>{'Sell' if tech_sell else 'Buy'}</b> "
            f"(current price momentum is {'bearish' if tech_sell else 'bullish'} — "
            f"price is {'below' if tech_sell else 'above'} key moving averages). "
            f"However, the statistical model predicts a <b>{pred_pct:+.1f}%</b> move "
            f"{'upward' if pred_pct > 0 else 'downward'}. "
            f"<b>Nivesh AI recommendation: Do not buy yet. Wait for technical indicators "
            f"(RSI, MACD, SMA crossover) to confirm before entering.</b>"
        )
        conflict_note = (
            "<b>Why do Technical and Prediction disagree?</b> "
            "Technical analysis reads <i>current</i> price momentum — it is bearish right now. "
            "The statistical model uses historical patterns and may anticipate a future mean-reversion bounce. "
            "They are measuring different timeframes. When they conflict like this, <b>reduce risk: "
            "wait for the RSI to cross above 40 and MACD to turn positive before buying.</b>"
        )
    elif buy_votes >= 2:
        css  = "consensus-buy"
        head = "✅ Consensus: Buy / Accumulate"
        exp  = (
            f"{buy_votes}/3 signals align bullish. "
            f"Statistical model forecasts <b>{pred_pct:+.1f}%</b> upside. "
            f"{'Positive news sentiment supports this move.' if sentiment_score > 0.1 else 'Monitor news sentiment for additional confirmation.'}"
        )
    elif sell_votes >= 2:
        css  = "consensus-sell"
        head = "🔴 Consensus: Sell / Avoid"
        exp  = (
            f"{sell_votes}/3 signals align bearish. "
            f"Statistical model forecasts <b>{pred_pct:+.1f}%</b> change. "
            f"{'Negative news sentiment adds to downside risk.' if sentiment_score < -0.1 else 'Wait for conditions to improve before entering.'}"
        )
    else:
        css  = "consensus-hold"
        head = "🟡 Consensus: Hold / Watch"
        exp  = (
            f"Signals are mixed with no strong conviction. "
            f"Statistical model forecasts <b>{pred_pct:+.1f}%</b> change. "
            f"No clear edge in either direction — wait for a clearer setup."
        )

    return css, head, exp, conflict_note


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Stock Selection")
    stock_input = st.text_input("Enter Stock Symbol", "ITC", help="e.g. ITC, RELIANCE, TCS")
    exchange    = st.selectbox("Exchange", ["NSE", "BSE"])
    ticker = (get_indian_ticker(stock_input) if exchange == "NSE"
              else f"{stock_input.upper()}.BO")

    if st.session_state.current_ticker != ticker:
        for k in ["chat_history", "news_data", "news_analysis", "master_summary"]:
            st.session_state[k] = [] if k == "chat_history" else None
        st.session_state.pending_question = None
        if st.session_state.current_ticker is not None:
            st.success(f"📊 Switched to: **{ticker}**")
    st.session_state.current_ticker = ticker
    st.caption(f"Analyzing: **{ticker}**")

    st.subheader("Date Range")
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = st.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date,
    )
    st.divider()
    st.success("🤖 **Powered by Nivesh AI**")
    st.caption("📰 India + Global live news")
    st.caption("💰 Investment P&L calculator")
    st.caption("🎯 Unified consensus signal")
    st.caption("📊 Optimised for Indian markets")
    st.divider()
    if st.button("🚪 Logout / Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
user_name = st.session_state.get("user_name", "Investor")
st.title("🤖 Nivesh AI - Stock Analysis Dashboard")
st.markdown(
    f"Namaste **{user_name}** 🙏 — Comprehensive stock analysis for Indian markets. "
    "Live news · Sentiment predictions · Unified consensus signal."
)

if not ticker:
    st.info("👈 Enter a ticker in the sidebar to begin.")
    st.stop()

try:
    stock = yf.Ticker(ticker)
    current_price, price_change, change_pct = get_live_price_info(stock)

    if current_price:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.metric(
                f"Live Price — {ticker}",
                safe_fmt(current_price, "currency"),
                (f"{safe_fmt(price_change,'currency')} ({change_pct:.2f}%)"
                 if price_change else None),
            )
        with c2:
            info = stock.info
            st.metric("Market Cap", fmt_market_cap(info.get("marketCap")))
        with c3:
            vol = info.get("volume")
        vol_str = (f"{vol/1e6:.1f}M" if vol and vol>=1e6 else f"{vol/1e3:.0f}K" if vol and vol>=1e3 else str(vol) if vol else "N/A")
        st.metric("Volume", vol_str)

    st.divider()
    dr_start, dr_end = (date_range if len(date_range)==2 else (date_range[0], date_range[0]))
    hist = stock.history(start=dr_start, end=dr_end)
    if hist.empty:
        st.error(f"No data for {ticker}.")
        st.stop()

    try:
        master_agent    = MasterAgent(ticker, gemini_api_key=API_KEY)
        analysis_result = master_agent.get_final_verdict()
    except Exception as e:
        st.warning(f"Analysis partial: {e}")
        master_agent    = None
        analysis_result = {
            "verdict":  {"short_term": "Hold", "long_term": "Hold"},
            "rationale":{"technical": ["In progress…"], "fundamental": ["In progress…"]},
        }

    plotly_tpl = "plotly_white"
    plot_bg    = "#ffffff"

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Technical", "📑 Fundamental", "📰 News & Summary",
        "🔮 Price Prediction", "🤖 AI Advisor", "💬 Ask Questions",
        "📚 Gyan Kendra",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — TECHNICAL
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Price Chart with Technical Indicators")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_sma = st.checkbox("Show SMA", True)
            sma_p    = st.slider("SMA Period", 5, 200, 20)
        with c2:
            show_rsi = st.checkbox("Show RSI", True)
            rsi_p    = st.slider("RSI Period", 5, 30, 14)
        with c3:
            show_macd = st.checkbox("Show MACD", True)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=("Price Chart", "RSI", "MACD"),
        )
        fig.add_trace(
            go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"], name="Price",
            ), row=1, col=1,
        )
        if show_sma:
            sma = hist["Close"].rolling(sma_p).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma, name=f"SMA {sma_p}",
                           line=dict(color="#3498db", width=2)), row=1, col=1,
            )
        if show_rsi:
            d = hist["Close"].diff()
            g = d.where(d > 0, 0).rolling(rsi_p).mean()
            l = (-d.where(d < 0, 0)).rolling(rsi_p).mean()
            rsi_vals = 100 - (100 / (1 + g / l))
            fig.add_trace(
                go.Scatter(x=hist.index, y=rsi_vals, name="RSI",
                           line=dict(color="purple", width=2)), row=2, col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red",   row=2, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
        if show_macd:
            e1        = hist["Close"].ewm(span=12, adjust=False).mean()
            e2        = hist["Close"].ewm(span=26, adjust=False).mean()
            macd_line = e1 - e2
            sig       = macd_line.ewm(span=9, adjust=False).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=macd_line, name="MACD",
                           line=dict(color="#3498db", width=2)), row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist.index, y=sig, name="Signal",
                           line=dict(color="orange", width=2)), row=3, col=1,
            )

        fig.update_layout(
            height=800, template=plotly_tpl,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False, showlegend=True, hovermode="x unified",
        )
        fig.update_xaxes(title_text="Date",      row=3, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI",       row=2, col=1)
        fig.update_yaxes(title_text="MACD",      row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # ── Individual signals — clearly NOT the final verdict ──────────────
        st.subheader("📊 Individual Indicator Signals")
        st.caption(
            "ℹ️ These are raw signals from each indicator independently. "
            "They may conflict with each other — that's normal. "
            "Go to **🔮 Price Prediction** to see the unified **Nivesh AI Consensus** that resolves them."
        )
        c1, c2 = st.columns(2)
        with c1:
            vs = analysis_result["verdict"]["short_term"]
            (st.success if vs == "Buy" else st.error if vs == "Sell" else st.warning)(
                f"**Technical Short-term:** {'🟢' if vs=='Buy' else '🔴' if vs=='Sell' else '🟡'} {vs}"
            )
            st.write("**Technical reasoning:**")
            for p in analysis_result["rationale"]["technical"]:
                st.write(f"• {p}")
        with c2:
            vl = analysis_result["verdict"]["long_term"]
            (st.success if vl == "Buy" else st.error if vl == "Sell" else st.warning)(
                f"**Fundamental Long-term:** {'🟢' if vl=='Buy' else '🔴' if vl=='Sell' else '🟡'} {vl}"
            )
            st.write("**Fundamental reasoning:**")
            for p in analysis_result["rationale"]["fundamental"]:
                st.write(f"• {p}")

        st.info(
            "💡 **Tip:** These signals sometimes disagree with the prediction model — "
            "that's expected since they use different methods. "
            "Head to **🔮 Price Prediction** for the single Nivesh AI Consensus that tells you what to actually do."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — FUNDAMENTAL
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Fundamental Analysis")
        info = stock.info
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Company:** {info.get('longName','N/A')}")
            st.write(f"**Sector:** {info.get('sector','N/A')}")
            st.write(f"**Industry:** {info.get('industry','N/A')}")
        with c2:
            st.write(f"**Country:** {info.get('country','N/A')}")
            st.write(f"**Exchange:** {info.get('exchange','N/A')}")
            st.write(f"**Currency:** {info.get('currency','INR')}")
        st.divider()
        st.subheader("💰 Valuation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Price", safe_fmt(info.get("currentPrice"), "currency"))
            st.metric("Market Cap",    fmt_market_cap(info.get("marketCap")))
            st.metric("P/E Ratio",     safe_fmt(info.get("trailingPE"),   "number"))
        with c2:
            st.metric("52W High", safe_fmt(info.get("fiftyTwoWeekHigh"), "currency"))
            st.metric("52W Low",  safe_fmt(info.get("fiftyTwoWeekLow"),  "currency"))
            st.metric("Volume",   safe_fmt(info.get("volume"),           "number", 0))
        with c3:
            dy = info.get("dividendYield")
            st.metric("Dividend Yield", safe_fmt(dy, "percentage") if dy else "N/A")
            st.metric("Beta", safe_fmt(info.get("beta"),        "number"))
            st.metric("EPS",  safe_fmt(info.get("trailingEps"), "currency"))
        st.divider()
        st.subheader("📊 Financial Health")
        c1, c2 = st.columns(2)
        with c1:
            pm = info.get("profitMargins")
            st.metric("Profit Margin",  safe_fmt(pm,  "percentage") if pm  else "N/A")
            roe = info.get("returnOnEquity")
            st.metric("ROE",            safe_fmt(roe, "percentage") if roe else "N/A")
            rg = info.get("revenueGrowth")
            st.metric("Revenue Growth", safe_fmt(rg,  "percentage") if rg  else "N/A")
        with c2:
            cr = info.get("currentRatio")
            st.metric("Current Ratio", safe_fmt(cr, "number") if cr else "N/A")
            de = info.get("debtToEquity")
            st.metric("Debt/Equity",   safe_fmt(de, "number") if de else "N/A")
            eg = info.get("earningsQuarterlyGrowth")
            st.metric("Earnings Growth", safe_fmt(eg, "percentage") if eg else "N/A")
        st.divider()
        st.subheader("📝 About")
        st.write(info.get("longBusinessSummary", "No description available."))

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — NEWS & SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("📰 Live News Intelligence")
        st.markdown(
            "News from **Google News + Yahoo Finance** split into "
            "**🇮🇳 India** and **🌍 Global** feeds, analysed by **Nivesh AI**."
        )
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            fetch_btn = st.button("🔄 Fetch & Analyse News", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear & Refresh", use_container_width=True):
                st.session_state.news_data = st.session_state.news_analysis = st.session_state.master_summary = None
                st.rerun()

        if fetch_btn or st.session_state.news_analysis:
            if fetch_btn or not st.session_state.news_analysis:
                with st.spinner("📡 Fetching India + Global news… analysing with Nivesh AI (~20 s)"):
                    try:
                        if master_agent:
                            st.session_state.news_data      = master_agent.get_live_news()
                            st.session_state.news_analysis  = master_agent.get_news_analysis()
                            st.session_state.master_summary = master_agent.get_master_summary()
                    except Exception as e:
                        st.error(f"Error: {e}")

            na = st.session_state.news_analysis
            nd = st.session_state.news_data
            ms = st.session_state.master_summary

            if na and ms:
                st.markdown("---")
                overall = ms.get("overall_verdict", "Hold 🟡")
                (st.success if "Buy" in overall else st.error if "Sell" in overall else st.warning)(
                    f"## 🏆 Overall Verdict: {overall}"
                )
                vc1, vc2, vc3 = st.columns(3)
                for col, label, val in [
                    (vc1, "📊 Technical (Short)",  ms["verdicts"]["technical_short"]),
                    (vc2, "📑 Fundamental (Long)", ms["verdicts"]["fundamental_long"]),
                    (vc3, "📰 News Signal",         ms["verdicts"]["news_based"]),
                ]:
                    with col:
                        icon = "🟢" if "Buy" in val else "🔴" if "Sell" in val else "🟡"
                        st.metric(label, f"{icon} {val}")

                st.markdown("---")
                st.subheader("🧠 Nivesh AI Market Summary")
                sl    = na.get("sentiment_label", "Neutral")
                sc    = na.get("sentiment_score", 0)
                css_s = sentiment_css(sl)
                st.markdown(f"""<div class="summary-box">
<p>{na.get('summary','')}</p>
<hr style="border-color:#415a77;margin:10px 0">
<b>Sentiment:</b> <span class="{css_s}">{sl} ({sc:+.2f})</span>
&nbsp;|&nbsp; <b>Est. Price Impact:</b> {na.get('price_impact_estimate',0):+.1f}%
&nbsp;|&nbsp; <b>News Action:</b> {na.get('buy_signal','Hold')}
</div>""", unsafe_allow_html=True)

                ic1, ic2 = st.columns(2)
                with ic1:
                    st.subheader("🇮🇳 India Market Impact")
                    st.info(na.get("india_impact", "No India analysis available."))
                with ic2:
                    st.subheader("🌍 Global / War Impact")
                    st.warning(na.get("global_impact", "No global analysis available."))

                active_conflicts = [
                    str(c) for c in na.get("active_conflicts", [])
                    if c is not None and str(c).strip()
                ]
                if active_conflicts:
                    st.subheader("⚔️ Active Global Conflicts & Impact")
                    st.caption("Nivesh AI monitors current wars and geopolitical events affecting Indian markets")
                    conflict_cols = st.columns(min(len(active_conflicts), 3))
                    icons = ["🔴", "🟠", "🟡"]
                    for i, conflict in enumerate(active_conflicts[:3]):
                        with conflict_cols[i]:
                            st.markdown(
                                f'<div class="conflict-card">{icons[i%3]} {conflict}</div>',
                                unsafe_allow_html=True,
                            )

                tc1, tc2, tc3 = st.columns(3)
                with tc1: render_list(na.get("key_themes"),        "Key Themes", "🔑")
                with tc2: render_list(na.get("risk_factors"),       "Risks",      "⚠️")
                with tc3: render_list(na.get("positive_catalysts"), "Catalysts",  "🚀")

                st.markdown("---")
                if nd:
                    news_tab1, news_tab2, news_tab3 = st.tabs([
                        f"🇮🇳 India Stock ({len(nd.get('india_stock_news',[]))})",
                        f"🇮🇳 India Macro ({len(nd.get('india_macro_news',[]))})",
                        f"🌍 Global ({len(nd.get('global_news',[]))})",
                    ])
                    with news_tab1:
                        st.caption("Stock-specific news from Indian sources + Yahoo Finance")
                        for a in nd.get("india_stock_news", [])[:12]:
                            st.markdown(
                                f'<div class="news-card"><div class="news-title">📰 {a["title"]}</div>'
                                f'<div class="news-meta">🕒 {a.get("published","")[:25]}'
                                f' &nbsp;|&nbsp; 📡 {a.get("source","")}</div>'
                                f'{"<div class=news-snip>"+a["snippet"]+"</div>" if a.get("snippet") else ""}'
                                f'</div>', unsafe_allow_html=True,
                            )
                    with news_tab2:
                        st.caption("RBI, Sensex, Nifty, India economy, budget, FII/DII")
                        for a in nd.get("india_macro_news", [])[:15]:
                            st.markdown(
                                f'<div class="news-card"><div class="news-title">🇮🇳 {a["title"]}</div>'
                                f'<div class="news-meta">🕒 {a.get("published","")[:25]}'
                                f' &nbsp;|&nbsp; 📡 {a.get("source","")}</div></div>',
                                unsafe_allow_html=True,
                            )
                    with news_tab3:
                        st.caption("US Fed, Iran-Israel/US war, Russia-Ukraine, oil prices, China, dollar")
                        for a in nd.get("global_news", [])[:15]:
                            st.markdown(
                                f'<div class="news-card news-card-global"><div class="news-title">🌍 {a["title"]}</div>'
                                f'<div class="news-meta">🕒 {a.get("published","")[:25]}'
                                f' &nbsp;|&nbsp; 📡 {a.get("source","")}</div></div>',
                                unsafe_allow_html=True,
                            )
                st.caption(f"🕐 Fetched: {nd.get('fetched_at','') if nd else ''} &nbsp;|&nbsp; Powered by Nivesh AI")
        else:
            st.info("👆 Click **Fetch & Analyse News** to get India + Global news with Nivesh AI analysis.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — PRICE PREDICTION + CONSENSUS + INVESTMENT CALCULATOR
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🔮 Nivesh AI Price Prediction")
        days_ahead = st.slider("Forecast horizon (trading days)", 7, 60, 30, key="pred_days")
        show_band  = st.checkbox("Show ±2% confidence band", value=True)
        use_news   = st.checkbox("🗞️ Apply news sentiment adjustment", value=True)

        if st.button("🚀 Run Nivesh AI Prediction", type="primary", use_container_width=True):
            with st.spinner("⚙️ Running prediction models…"):
                try:
                    pa   = master_agent if master_agent else MasterAgent(ticker, API_KEY)
                    pred = pa.get_price_prediction(days_ahead, use_news_sentiment=use_news)
                    pred["_fut_dates_str"] = [d.strftime("%Y-%m-%d") for d in pred["future_dates"]]
                    st.session_state.last_prediction = pred
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

        pred = st.session_state.get("last_prediction")
        if pred:
            from datetime import datetime as _dt
            fut_dates   = [_dt.strptime(d, "%Y-%m-%d") for d in pred.get("_fut_dates_str", [])]
            cp          = pred["current_price"]
            lr_preds    = pred["lr_predictions"]
            arima_preds = pred["arima_predictions"]
            ensemble    = pred["ensemble"]
            pred_id     = pred.get("prediction_id", "")
            buy_now     = pred.get("buy_now", {})
            sentiment   = pred.get("sentiment_score", 0)
            end_price   = (ensemble or lr_preds or arima_preds or [cp])[-1]
            pct         = (end_price - cp) / cp * 100

            if pct > 1:    st.success(f"📈 {pred['summary']}")
            elif pct < -1: st.error(f"📉 {pred['summary']}")
            else:          st.info(f"➡️ {pred['summary']}")

            # ── NIVESH AI CONSENSUS ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("🎯 Nivesh AI Consensus — What Should You Do?")
            st.caption(
                "This box combines Technical + Fundamental + Statistical model signals into "
                "one honest recommendation. When signals conflict, it tells you exactly why."
            )

            tech_short  = analysis_result["verdict"]["short_term"]
            fund_long   = analysis_result["verdict"]["long_term"]
            pred_signal = buy_now.get("signal", "Hold 🟡")

            con_css, con_head, con_exp, con_conflict = build_consensus(
                tech_short, fund_long, pred_signal, pct, sentiment
            )

            sent_label = ("Bullish" if sentiment > 0.1
                          else "Bearish" if sentiment < -0.1 else "Neutral")
            pills_html = (
                f'<div class="signal-row">'
                f'{_pill("📊 Technical", tech_short)}'
                f'{_pill("📑 Fundamental", fund_long)}'
                f'{_pill("📈 Model", pred_signal.split(" ")[0])}'
                f'{_pill("📰 Sentiment", sent_label) if sentiment != 0 else ""}'
                f'</div>'
            )

            st.markdown(
                f'<div class="consensus-box {con_css}">'
                f'<div style="font-size:18px;font-weight:700">{con_head}</div>'
                f'<div class="consensus-sub">{con_exp}</div>'
                f'{pills_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if con_conflict:
                st.markdown(
                    f'<div class="conflict-explain">📖 {con_conflict}</div>',
                    unsafe_allow_html=True,
                )

            # ── Raw model signal (kept for transparency) ────────────────────
            st.markdown("---")
            with st.expander("📊 Raw Statistical Model Signal (detail)", expanded=False):
                st.caption("This is the raw output from the price prediction model alone, before consensus reconciliation.")
                signal  = buy_now.get("signal",     "Hold 🟡")
                reason  = buy_now.get("reason",     "")
                conf    = buy_now.get("confidence", 0)
                css_cls = buy_banner_css(signal)
                st.markdown(
                    f'<div class="buy-banner {css_cls}">'
                    f'📈 &nbsp;<b>Model Signal:</b> {signal}<br>'
                    f'<span style="font-size:13px;font-weight:400;opacity:0.9">{reason}</span><br>'
                    f'<span style="font-size:12px;opacity:0.7">Confidence: {conf:.0f}%'
                    f' &nbsp;|&nbsp; Predicted Change: {pct:+.2f}%'
                    f'{"&nbsp;|&nbsp; Sentiment: "+str(sentiment) if use_news else ""}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

            # ── Price cards ─────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📌 Price Targets at End of Forecast")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="pred-card"><h3>Current Price</h3>'
                    f'<div class="price">₹{cp:.2f}</div>'
                    f'<div class="delta flat">Live market price</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                lr_end = lr_preds[-1] if lr_preds else None
                st.markdown(
                    f'<div class="pred-card"><h3>Linear Regression</h3>'
                    f'<div class="price">{"₹"+f"{lr_end:.2f}" if lr_end else "N/A"}</div>'
                    f'<div class="delta">{delta_card(lr_end, cp)}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                ar_end = arima_preds[-1] if arima_preds else None
                st.markdown(
                    f'<div class="pred-card"><h3>ARIMA Model</h3>'
                    f'<div class="price">{"₹"+f"{ar_end:.2f}" if ar_end else "N/A"}</div>'
                    f'<div class="delta">{delta_card(ar_end, cp)}</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                ens_end = ensemble[-1] if ensemble else None
                st.markdown(
                    f'<div class="pred-card"><h3>🏆 Ensemble (Best)</h3>'
                    f'<div class="price">{"₹"+f"{ens_end:.2f}" if ens_end else "N/A"}</div>'
                    f'<div class="delta">{delta_card(ens_end, cp)}</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Forecast chart ──────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 Forecast Chart")
            fig_p  = go.Figure()
            hist60 = stock.history(
                start=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            ).tail(60)
            fig_p.add_trace(go.Scatter(
                x=hist60.index, y=hist60["Close"], name="Historical",
                mode="lines", line=dict(color="#0077b6", width=2),
            ))
            fig_p.add_trace(go.Scatter(
                x=[hist60.index[-1]], y=[cp], mode="markers",
                marker=dict(color="#023e8a", size=9), name="Today",
            ))
            if lr_preds:
                fig_p.add_trace(go.Scatter(
                    x=fut_dates[:len(lr_preds)], y=lr_preds, name="LR", mode="lines",
                    line=dict(color="#06d6a0", width=2, dash="dot"),
                ))
            if arima_preds:
                fig_p.add_trace(go.Scatter(
                    x=fut_dates[:len(arima_preds)], y=arima_preds, name="ARIMA", mode="lines",
                    line=dict(color="#ffd166", width=2, dash="dash"),
                ))
            if ensemble:
                fig_p.add_trace(go.Scatter(
                    x=fut_dates[:len(ensemble)], y=ensemble,
                    name="🏆 Ensemble", mode="lines", line=dict(color="#ff6b6b", width=3),
                ))
                if show_band:
                    upper = [p * 1.02 for p in ensemble]
                    lower = [p * 0.98 for p in ensemble]
                    fig_p.add_trace(go.Scatter(
                        x=fut_dates[:len(ensemble)] + fut_dates[:len(ensemble)][::-1],
                        y=upper + lower[::-1], fill="toself",
                        fillcolor="rgba(255,107,107,0.10)",
                        line=dict(color="rgba(0,0,0,0)"), name="±2% Band",
                    ))
            fig_p.add_hline(y=cp, line_dash="dot", line_color="#0077b6", opacity=0.5,
                            annotation_text=f"Current ₹{cp:.2f}", annotation_position="top left")
            fig_p.update_layout(
                height=480, template=plotly_tpl, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                yaxis=dict(title="Price (₹)", gridcolor="rgba(128,128,128,0.15)"),
                xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # ── Weekly table ────────────────────────────────────────────────
            if pred.get("weekly_table"):
                st.subheader("📅 Week-by-Week Price Targets")
                rows = []
                for row in pred["weekly_table"]:
                    r = {"Period": f"{row['week']} ({row['date_range']})"}
                    if "lr"       in row: r["Linear Regression"] = f"₹{row['lr']}"
                    if "arima"    in row: r["ARIMA"]             = f"₹{row['arima']}"
                    if "ensemble" in row: r["🏆 Ensemble"]       = f"₹{row['ensemble']}"
                    rows.append(r)
                st.dataframe(pd.DataFrame(rows).set_index("Period"), use_container_width=True)

            # ── Accuracy ────────────────────────────────────────────────────
            st.subheader("📐 Model Accuracy (Back-test)")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if pred["lr_mae"] and pred["lr_mape"]:
                    st.metric("Linear Regression", f"~{max(0,100-pred['lr_mape']):.1f}%",
                              delta=f"MAE ₹{pred['lr_mae']:.2f}")
            with ac2:
                if pred["arima_mae"] and pred["arima_mape"]:
                    st.metric("ARIMA", f"~{max(0,100-pred['arima_mape']):.1f}%",
                              delta=f"MAE ₹{pred['arima_mae']:.2f}")
            with ac3:
                vals = [x for x in [pred["lr_mape"] or 0, pred["arima_mape"] or 0] if x > 0]
                if vals:
                    st.metric("Ensemble", f"~{max(0,100-sum(vals)/len(vals)):.1f}%",
                              delta="55% LR + 45% ARIMA")

            # ── Investment Calculator ────────────────────────────────────────
            st.markdown("---")
            st.subheader("💰 Investment Calculator")
            st.markdown("Enter how much you want to invest — Nivesh AI shows your projected profit or loss.")

            inv_col1, inv_col2 = st.columns([1, 2])
            with inv_col1:
                invest_amt = st.number_input(
                    "Investment Amount (₹)",
                    min_value=100.0, max_value=10_000_000.0,
                    value=10000.0, step=500.0, format="%.0f",
                )

            calc = calculate_investment(invest_amt, cp, ensemble or lr_preds or [], fut_dates)
            if calc:
                with inv_col2:
                    st.caption(f"{calc['shares']:.4f} shares @ ₹{cp:.2f} each")
                pl_val  = calc["profit_loss"]
                pl_pct  = calc["pl_pct"]
                pl_cls  = pl_color(pl_val)
                pl_sign = "+" if pl_val >= 0 else ""
                pl_icon = "📈" if pl_val >= 0 else "📉"
                st.markdown(f"""<div class="calc-card">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px">
  <div>
    <div class="calc-label">You Invest</div>
    <div style="font-size:20px;font-weight:700">₹{calc['investment']:,.0f}</div>
    <div class="calc-label" style="margin-top:6px">Shares Bought</div>
    <div style="font-size:16px">{calc['shares']:.4f} shares</div>
  </div>
  <div style="text-align:center">
    <div class="calc-label">Projected Value in {days_ahead} Trading Days</div>
    <div class="{pl_cls}" style="font-size:28px">₹{calc['end_value']:,.2f}</div>
    <div class="{pl_cls}" style="font-size:16px">
      {pl_icon} {pl_sign}₹{abs(pl_val):,.2f} ({pl_sign}{pl_pct:.2f}%)
    </div>
  </div>
  <div>
    <div class="calc-label">Best Case 🚀</div>
    <div style="font-size:14px;font-weight:600">
      ₹{calc['best_case']['value']:,.2f} (+₹{calc['best_case']['profit']:,.2f})
    </div>
    <div class="calc-label" style="margin-top:6px">Worst Case ⚠️</div>
    <div style="font-size:14px;font-weight:600">
      ₹{calc['worst_case']['value']:,.2f} (-₹{abs(calc['worst_case']['loss']):,.2f})
    </div>
  </div>
</div>
</div>""", unsafe_allow_html=True)

                if calc.get("weekly"):
                    st.markdown("**📅 Week-by-Week Projection:**")
                    df_calc = pd.DataFrame([{
                        "Week":            w["week"],
                        "Date":            w["date"],
                        "Est. Price":      f"₹{w['price']:,.2f}",
                        "Portfolio Value": f"₹{w['value']:,.2f}",
                        "Profit/Loss":     f"{'+'if w['profit_loss']>=0 else ''}₹{w['profit_loss']:,.2f}",
                        "Return %":        f"{'+'if w['pl_pct']>=0 else ''}{w['pl_pct']:.2f}%",
                    } for w in calc["weekly"]])
                    st.dataframe(df_calc.set_index("Week"), use_container_width=True)
                st.caption("⚠️ Uses Nivesh AI ensemble forecast. Actual returns may differ. Not financial advice.")

            if pred_id:
                st.caption(f"📌 Prediction ID: `{pred_id}` — verify it in the Tracker below.")

        # ── Prediction tracker ───────────────────────────────────────────────
        st.markdown("---")
        with st.expander("✅ Prediction Tracker — Mark Correct / Incorrect", expanded=False):
            st.caption("One prediction saved per stock per day. Mark outcome when target date arrives.")
            if master_agent:
                stats = master_agent.get_accuracy_stats()
                if stats["total"] > 0:
                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    sc1.metric("Total",     stats["total"])
                    sc2.metric("✅ Correct", stats["correct"])
                    sc3.metric("⚠️ Partial", stats["partial"])
                    sc4.metric("❌ Wrong",   stats["incorrect"])
                    sc5.metric("Accuracy",  f"{stats['accuracy_pct']}%")
                    st.progress(stats["accuracy_pct"] / 100)
                col_clr, _ = st.columns([1, 3])
                with col_clr:
                    if st.button("🗑️ Clear All Predictions", type="secondary"):
                        master_agent.tracker.history = []
                        master_agent.tracker._save()
                        st.success("Cleared!")
                        st.rerun()
                pending = master_agent.get_pending_predictions()
                if pending:
                    st.markdown(f"**{len(pending)} pending verification(s):**")
                    for p in pending:
                        cols = st.columns([2, 1, 1, 1])
                        with cols[0]:
                            chg  = p["predicted_change_pct"]
                            icon = "📈" if chg >= 0 else "📉"
                            st.markdown(
                                f"{icon} **{p['ticker']}** — ₹{p['predicted_price']}"
                                f" ({chg:+.1f}%) | Made: {p['date_made'][:10]}"
                                f" | Verify by: {p['target_date']}"
                            )
                        with cols[1]:
                            actual = st.number_input(
                                "Actual ₹", min_value=0.0, key=f"a_{p['id']}",
                                step=0.5, label_visibility="collapsed", placeholder="Actual price",
                            )
                        with cols[2]:
                            outcome = st.selectbox(
                                "", ["auto", "correct", "partial", "incorrect"],
                                key=f"o_{p['id']}", label_visibility="collapsed",
                            )
                        with cols[3]:
                            if st.button("Save", key=f"s_{p['id']}", type="primary"):
                                if actual > 0:
                                    master_agent.mark_prediction(p["id"], actual, outcome, "")
                                    st.success("Saved!")
                                    st.rerun()
                                else:
                                    st.warning("Enter price first.")
                else:
                    st.info("No pending predictions. Run a prediction above to start tracking!")

                recent   = master_agent.get_prediction_history(10)
                verified = [r for r in recent if r["outcome"] is not None]
                if verified:
                    st.markdown("**Recent verified predictions:**")
                    df_h = pd.DataFrame([{
                        "Date":      r["date_made"][:10],
                        "Predicted": f"₹{r['predicted_price']}",
                        "Change":    f"{r['predicted_change_pct']:+.1f}%",
                        "Actual":    f"₹{r['actual_price']}" if r["actual_price"] else "—",
                        "Outcome":   {"correct":"✅","partial":"⚠️","incorrect":"❌"}.get(r["outcome"],"—"),
                    } for r in verified])
                    st.dataframe(df_h.set_index("Date"), use_container_width=True)

        st.warning("⚠️ Statistical models for educational reference only. Not financial advice.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — AI ADVISOR
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("🤖 Nivesh AI Investment Advisor")
        atype = st.radio("Analysis Type:", ["📋 Comprehensive", "⚡ Quick Summary"], horizontal=True)
        if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):
            with st.spinner("🧠 Nivesh AI analysing…"):
                try:
                    resp = bot_interface.analyze(
                        ticker,
                        "comprehensive" if "Comprehensive" in atype else "quick",
                    )
                    st.success("✅ Done!")
                    st.markdown("---")
                    st.markdown(resp)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": resp,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.session_state.chat_history:
            with st.expander("📜 Previous Analyses"):
                for msg in reversed(st.session_state.chat_history[-5:]):
                    if msg["role"] == "assistant":
                        st.markdown(f"**{msg.get('timestamp','')}:**")
                        st.markdown(
                            (msg["content"][:500] + "…")
                            if len(msg["content"]) > 500
                            else msg["content"]
                        )
                        st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — ASK QUESTIONS
    # ══════════════════════════════════════════════════════════════════════
    with tab6:
        st.subheader("💬 Ask Nivesh AI")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info(f"📊 Analysing: **{ticker}**")
        with c2:
            if st.button("🗑️ Clear"):
                st.session_state.chat_history = []
                st.rerun()

        for msg in st.session_state.chat_history:
            if msg.get("role") == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg.get("role") == "assistant" and "timestamp" not in msg:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        if st.session_state.pending_question:
            prompt = st.session_state.pending_question
            st.session_state.pending_question = None
            if not any(
                m.get("content") == prompt and m.get("role") == "user"
                for m in st.session_state.chat_history[-1:]
            ):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analysing…"):
                    try:
                        r = bot_interface.ask(ticker, prompt)
                        st.write(r)
                        st.session_state.chat_history.append({"role": "assistant", "content": r})
                    except Exception as e:
                        st.error(f"❌ {e}")

        prompt = st.chat_input(f"Ask about {ticker}…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analysing…"):
                    try:
                        r = bot_interface.ask(ticker, prompt)
                        st.write(r)
                        st.session_state.chat_history.append({"role": "assistant", "content": r})
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.divider()
        st.caption("💡 Suggested:")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Should I buy now?"):
                st.session_state.pending_question = "Should I buy this stock now?"
                st.rerun()
            if st.button("Main risks?"):
                st.session_state.pending_question = "What are the main risks?"
                st.rerun()
        with c2:
            if st.button("Good for long-term?"):
                st.session_state.pending_question = "Is this good for long-term investment?"
                st.rerun()
            if st.button("Compare competitors"):
                st.session_state.pending_question = "How does this compare with competitors?"
                st.rerun()

        st.divider()
        with st.expander("📊 Compare Multiple Stocks"):
            comp_input = st.text_input(
                "Competitor tickers (comma-separated)",
                placeholder="e.g. HINDUNILVR.NS, DABUR.NS",
            )
            if st.button("🔎 Compare", type="primary"):
                if comp_input:
                    comps = [t.strip().upper() for t in comp_input.split(",")]
                    if 1 <= len(comps) <= 4:
                        with st.spinner("Comparing…"):
                            try:
                                r = bot_interface.compare_stocks([ticker] + comps)
                                st.success("✅ Done!")
                                st.markdown(r)
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Enter 1–4 tickers.")
                else:
                    st.warning("Enter competitor tickers.")


    # ══════════════════════════════════════════════════════════════════════
    # TAB 7 — GYAN KENDRA — pure HTML/JS loaded from file, zero escaping
    # ══════════════════════════════════════════════════════════════════════
    with tab7:
        import streamlit.components.v1 as _c
        import os as _os
        # Try multiple path strategies — Streamlit __file__ can be unreliable
        _gk_candidates = [
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "gyan_kendra.html"),
            _os.path.join(_os.getcwd(), "gyan_kendra.html"),
            "gyan_kendra.html",
        ]
        _gk_path = next((p for p in _gk_candidates if _os.path.exists(p)), None)
        if _gk_path:
            with open(_gk_path, "r", encoding="utf-8") as _f:
                _gk_html = _f.read()
            _c.html(_gk_html, height=3400, scrolling=True)
        else:
            st.error("gyan_kendra.html not found. Please place it in the same folder as app.py.")
except Exception as e:
    st.error(f"❌ Error analysing {ticker}")
    st.exception(e)

st.divider()
st.caption("🤖 Nivesh AI v5 — India + Global News | Investment Calculator | Unified Consensus | 📚 Gyan Kendra | Data: Yahoo Finance + Google News")
st.caption("⚠️ Educational purposes only. Not financial advice. Always do your own research before investing.")