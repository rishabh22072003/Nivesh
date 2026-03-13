# stock_agent.py  ──  Nivesh AI Engine v3
# ─────────────────────────────────────────────────────────────────────────────
# v3 changes:
#   • All AI branding = "Nivesh AI" only (no external AI names shown)
#   • News split into INDIA news + GLOBAL/FOREIGN news
#   • India queries: NSE, BSE, RBI, Sensex, Nifty, Indian budget, FII/DII
#   • Global queries: US Fed, wars (Russia-Ukraine, Israel-Gaza), oil, China
#   • Investment calculator: ₹ amount → shares, projected value, P&L, weekly
#   • "Buy Now?" signal on every prediction
#   • Live price fix (always yf.Ticker.info, never stale historical close)
#   • Sentiment-adjusted ensemble forecast
#   • Prediction override tracker (mark correct/incorrect, track accuracy)
# ─────────────────────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
import numpy as np
from finta import TA
import json, os, re
import urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE NEWS AGENT  ── India + Global separated
# ══════════════════════════════════════════════════════════════════════════════

class LiveNewsAgent:
    """
    Fetches news in two buckets:
      INDIA  — stock-specific + Indian macro (RBI, Sensex, Nifty, FII, budget)
      GLOBAL — US Fed, wars, oil, China, dollar, world markets

    All AI-powered analysis is branded as "Nivesh AI" only.
    """

    INDIA_QUERIES = [
        "NSE BSE India stock market today",
        "RBI repo rate India monetary policy",
        "Sensex Nifty 50 India today",
        "India inflation CPI WPI",
        "India GDP economy budget",
        "FII DII investment India NSE",
        "India rupee forex today",
    ]

    # ── Global queries — updated for current 2026 conflicts ─────────────────
    # Always searches for CURRENT wars and geopolitical events dynamically.
    # Primary conflicts as of early 2026:
    #   • Iran-Israel/US war (Strait of Hormuz, oil tankers, missile strikes)
    #   • Russia-Ukraine war (ongoing since 2022)
    #   • Israel-Gaza conflict (ongoing)
    #   • Sudan / Sahel instability
    GLOBAL_QUERIES = [
        # Current primary conflict: Iran-Israel/US
        "Iran Israel US war Strait of Hormuz oil 2026",
        "Iran missile strike oil price Middle East war",
        "Strait of Hormuz closure crude oil impact",
        # Russia-Ukraine
        "Russia Ukraine war sanctions economy 2026",
        "Ukraine war energy Europe impact",
        # Israel-Gaza
        "Israel Gaza war ceasefire 2026",
        # Oil & energy (directly tied to India's import costs)
        "crude oil price OPEC Brent WTI today",
        "oil price India import inflation impact",
        # US macro
        "US Federal Reserve interest rate decision 2026",
        "US Dow Jones Nasdaq stock market today",
        "US dollar index DXY rupee",
        # China
        "China economy slowdown trade 2026",
        "US China trade war tariff",
        # Global macro
        "global recession inflation 2026",
        "IMF World Bank global growth forecast",
    ]

    def __init__(self, ticker: str, api_key: str = None):
        self.ticker       = ticker
        self.api_key      = api_key or os.getenv("GEMINI_API_KEY")
        self.company_name = self._get_company_name()

    def _get_company_name(self) -> str:
        try:
            info = yf.Ticker(self.ticker).info
            return info.get("longName",
                            self.ticker.replace(".NS","").replace(".BO",""))
        except Exception:
            return self.ticker.replace(".NS","").replace(".BO","")

    def _fetch_rss(self, query: str, max_items: int = 6, geo: str = "IN") -> list:
        articles = []
        try:
            url = (f"https://news.google.com/rss/search"
                   f"?q={urllib.parse.quote(query)}&hl=en-IN&gl={geo}&ceid={geo}:en")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                root  = ET.fromstring(resp.read())
            for item in root.findall(".//item")[:max_items]:
                title = item.findtext("title","").strip()
                pub   = item.findtext("pubDate","").strip()
                desc  = re.sub(r"<[^>]+>","", item.findtext("description",""))[:200]
                src   = item.find("source")
                if title:
                    articles.append({
                        "title":     title,
                        "published": pub,
                        "snippet":   desc.strip(),
                        "source":    src.text if src is not None else "Google News",
                    })
        except Exception:
            pass
        return articles

    def _fetch_yfinance_news(self, max_items: int = 8) -> list:
        articles = []
        try:
            for item in (yf.Ticker(self.ticker).news or [])[:max_items]:
                c = item.get("content", {})
                t = c.get("title") or item.get("title","")
                s = c.get("summary","")
                p = c.get("pubDate") or str(item.get("providerPublishTime",""))
                provider = (c.get("provider",{}) or {}).get("displayName","Yahoo Finance")
                if t:
                    articles.append({"title":t,"published":p[:30],"snippet":s[:200],"source":provider})
        except Exception:
            pass
        return articles

    @staticmethod
    def _dedup(lst: list) -> list:
        seen, out = set(), []
        for a in lst:
            k = a["title"][:55].lower()
            if k not in seen:
                seen.add(k)
                out.append(a)
        return out

    def fetch_all_news(self) -> dict:
        short = self.company_name.split()[0] if self.company_name else self.ticker

        # India stock-specific
        india_stock  = self._fetch_yfinance_news(8)
        india_stock += self._fetch_rss(f"{short} NSE stock India", 5, "IN")
        india_stock += self._fetch_rss(
            f"{self.ticker.replace('.NS','').replace('.BO','')} share price", 4, "IN")

        # India macro
        india_macro = []
        for q in self.INDIA_QUERIES:
            india_macro += self._fetch_rss(q, 4, "IN")

        # Global / foreign
        global_news = []
        for q in self.GLOBAL_QUERIES:
            global_news += self._fetch_rss(q, 3, "US")

        return {
            "india_stock_news": self._dedup(india_stock)[:15],
            "india_macro_news": self._dedup(india_macro)[:18],
            "global_news":      self._dedup(global_news)[:18],
            "fetched_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        }

    # ── Rule-based sentiment fallback ─────────────────────────────────────────
    @staticmethod
    def _rule_sentiment(news: dict) -> float:
        pos_w = ["surge","rally","gain","profit","growth","buy","upgrade","strong",
                 "record","bullish","rise","beat","dividend","expand","outperform",
                 "recover","boost","high","increase","investment","inflow"]
        neg_w = ["fall","drop","loss","decline","sell","downgrade","weak","bearish",
                 "crash","miss","war","sanction","recession","default","crisis","risk",
                 "cut","ban","inflation","rate hike","conflict","tension","outflow",
                 "rupee fall","FII selling","slowdown"]
        all_text = " ".join(
            a["title"]+" "+a.get("snippet","")
            for a in (news.get("india_stock_news",[]) +
                      news.get("india_macro_news",[]) +
                      news.get("global_news",[]))
        ).lower()
        pos = sum(all_text.count(w) for w in pos_w)
        neg = sum(all_text.count(w) for w in neg_w)
        total = pos + neg
        return 0.0 if total == 0 else round(max(-1.0, min(1.0, (pos-neg)/total)), 3)

    # ── Nivesh AI-powered analysis ────────────────────────────────────────────
    def analyze_news(self, news: dict, current_price: float) -> dict:
        """Analyze news with Nivesh AI. All branding stays as Nivesh AI."""
        if not _GENAI_OK or not self.api_key:
            return self._fallback(news)
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")

            india_hdl = "\n".join(
                f"• [{a['source']}] {a['title']}"
                for a in (news.get("india_stock_news",[])[:8] +
                           news.get("india_macro_news",[])[:8])
            )
            global_hdl = "\n".join(
                f"• [{a['source']}] {a['title']}"
                for a in news.get("global_news",[])[:10]
            )

            current_year = datetime.now().year
            current_month = datetime.now().strftime("%B %Y")
            prompt = f"""You are Nivesh AI, an expert Indian stock market analyst ({current_month}).

Analyze ALL news below for Indian stock {self.ticker} (₹{current_price:.2f}).

IMPORTANT — Always identify and analyze CURRENTLY ACTIVE global conflicts:
As of {current_month}, key active conflicts include:
• Iran-Israel/US war — Strait of Hormuz threatened, oil tanker attacks, missile strikes (HIGH impact on India's oil import costs)
• Russia-Ukraine war (ongoing since 2022) — energy/commodity prices
• Israel-Gaza conflict (ongoing) — regional Middle East instability
• Any new conflicts mentioned in the news below

For each active conflict, explain the SPECIFIC impact on this Indian stock/sector.

=== INDIA NEWS (Stock + Macro) ===
{india_hdl or 'No India news available.'}

=== GLOBAL / FOREIGN NEWS (Wars, Fed, Oil, etc.) ===
{global_hdl or 'No global news available.'}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "summary": "3-4 sentences covering: company-specific news, Indian market conditions (RBI/FII/Sensex), AND how current active wars/conflicts (especially Iran-Israel/US war and its oil impact) affect this stock",
  "india_impact": "Specific impact of Indian factors: RBI policy, rupee movement, FII/DII flows, inflation, budget on this stock",
  "global_impact": "Specific impact of EACH active war/conflict on this stock — Iran-Israel/US (oil/rupee), Russia-Ukraine (commodities), any new conflicts. Also US Fed and China effects.",
  "active_conflicts": ["conflict1 and its impact", "conflict2 and its impact"],
  "sentiment_score": <float -1.0 to 1.0>,
  "sentiment_label": "<Very Bearish|Bearish|Neutral|Bullish|Very Bullish>",
  "key_themes": ["theme1","theme2","theme3"],
  "risk_factors": ["risk1","risk2","risk3"],
  "positive_catalysts": ["catalyst1","catalyst2"],
  "price_impact_estimate": <estimated % price move from news alone, e.g. -1.5 or +2.0>,
  "buy_signal": "<Strong Buy|Buy|Hold|Sell|Strong Sell>"
}}"""

            raw    = re.sub(r"```json|```","", model.generate_content(prompt).text.strip()).strip()
            result = json.loads(raw)

            # Sanitize all list fields — remove None / empty entries
            for list_field in ["key_themes","risk_factors","positive_catalysts","active_conflicts"]:
                raw_list = result.get(list_field, [])
                if isinstance(raw_list, list):
                    result[list_field] = [str(i) for i in raw_list if i is not None and str(i).strip()]
                else:
                    result[list_field] = []

            result["powered_by"]    = "Nivesh AI"
            result["analysis_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return result
        except Exception:
            return self._fallback(news)

    def _fallback(self, news: dict) -> dict:
        score = self._rule_sentiment(news)
        label = self._label(score)
        india_cnt  = len(news.get("india_stock_news",[]))+len(news.get("india_macro_news",[]))
        global_cnt = len(news.get("global_news",[]))
        return {
            "summary":              (f"Nivesh AI analysis: {label} sentiment based on "
                                     f"{india_cnt} India news and {global_cnt} global news articles."),
            "india_impact":         "Indian market analysis based on available news.",
            "global_impact":        "Global market analysis: monitoring Iran-Israel/US war (oil/Hormuz), Russia-Ukraine, Israel-Gaza impacts.",
            "active_conflicts":     ["Iran-Israel/US war — oil price & rupee impact",
                                     "Russia-Ukraine war — commodity/energy prices",
                                     "Israel-Gaza conflict — regional instability"],
            "sentiment_score":      score,
            "sentiment_label":      label,
            "key_themes":           [],
            "risk_factors":         [],
            "positive_catalysts":   [],
            "price_impact_estimate":round(score * 2.0, 2),
            "buy_signal":           "Hold",
            "powered_by":           "Nivesh AI",
            "analysis_time":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.5:   return "Very Bullish"
        if score >= 0.15:  return "Bullish"
        if score <= -0.5:  return "Very Bearish"
        if score <= -0.15: return "Bearish"
        return "Neutral"


# ══════════════════════════════════════════════════════════════════════════════
#  INVESTMENT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_investment(investment_amount: float, current_price: float,
                          predictions: list, future_dates: list) -> dict:
    """
    Given ₹ investment amount, returns:
      - shares you can buy
      - projected portfolio value at end of forecast
      - profit / loss in ₹ and %
      - week-by-week P&L snapshots
      - best-case and worst-case within the forecast horizon
    """
    if not predictions or current_price <= 0 or investment_amount <= 0:
        return {}

    shares        = investment_amount / current_price
    end_price     = predictions[-1]
    end_value     = shares * end_price
    profit_loss   = end_value - investment_amount
    pl_pct        = profit_loss / investment_amount * 100

    weekly = []
    for w in range(min(4, len(predictions) // 5)):
        i     = min((w + 1) * 5 - 1, len(predictions) - 1)
        price = predictions[i]
        val   = shares * price
        pl    = val - investment_amount
        weekly.append({
            "week":        f"Week {w + 1}",
            "date":        future_dates[i].strftime("%b %d") if i < len(future_dates) else "",
            "price":       round(price, 2),
            "value":       round(val, 2),
            "profit_loss": round(pl, 2),
            "pl_pct":      round(pl / investment_amount * 100, 2),
        })

    max_p = max(predictions)
    min_p = min(predictions)

    return {
        "investment":      round(investment_amount, 2),
        "current_price":   round(current_price, 2),
        "shares":          round(shares, 4),
        "end_price":       round(end_price, 2),
        "end_value":       round(end_value, 2),
        "profit_loss":     round(profit_loss, 2),
        "pl_pct":          round(pl_pct, 2),
        "weekly":          weekly,
        "best_case":  {"price": round(max_p,2), "value": round(shares*max_p,2),
                       "profit": round(shares*max_p - investment_amount, 2)},
        "worst_case": {"price": round(min_p,2), "value": round(shares*min_p,2),
                       "loss":  round(shares*min_p - investment_amount, 2)},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION OVERRIDE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class PredictionOverrideTracker:
    def __init__(self, filepath: str = "prediction_history.json"):
        self.filepath = filepath
        self.history  = self._load()

    def _load(self) -> list:
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath) as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save(self):
        try:
            with open(self.filepath,"w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception:
            pass

    def save_prediction(self, ticker, current_price, ensemble_end,
                        days, sentiment_score=0.0, news_summary="") -> str:
        today = datetime.now().strftime("%Y-%m-%d")

        # Only keep ONE pending prediction per ticker per day — update it instead of appending
        for entry in self.history:
            if (entry["ticker"] == ticker
                    and entry["date_made"][:10] == today
                    and entry["outcome"] is None):
                # Update existing entry with latest values
                entry["predicted_price"]      = round(ensemble_end, 2)
                entry["predicted_change_pct"] = round((ensemble_end - current_price) / current_price * 100, 2)
                entry["price_at_prediction"]  = round(current_price, 2)
                entry["sentiment_score"]       = sentiment_score
                entry["news_summary"]          = news_summary[:300]
                entry["days_ahead"]            = days
                entry["target_date"]           = (datetime.now() + timedelta(days=int(days * 1.4))).strftime("%Y-%m-%d")
                self._save()
                return entry["id"]

        # No existing entry for today — create new
        pid = f"{ticker}_{today}"
        self.history.append({
            "id":                  pid,
            "ticker":              ticker,
            "date_made":           datetime.now().isoformat(),
            "target_date":         (datetime.now() + timedelta(days=int(days * 1.4))).strftime("%Y-%m-%d"),
            "price_at_prediction": round(current_price, 2),
            "predicted_price":     round(ensemble_end, 2),
            "predicted_change_pct":round((ensemble_end - current_price) / current_price * 100, 2),
            "days_ahead":          days,
            "sentiment_score":     sentiment_score,
            "news_summary":        news_summary[:300],
            "outcome":             None,
            "actual_price":        None,
            "override_note":       "",
            "verified_at":         None,
        })
        self._save()
        return pid

    def mark_outcome(self, pred_id, actual_price, outcome="auto", note="") -> dict:
        for e in self.history:
            if e["id"] == pred_id:
                e["actual_price"]  = actual_price
                e["verified_at"]   = datetime.now().isoformat()
                e["override_note"] = note
                if outcome == "auto":
                    pred_dir = "up" if e["predicted_price"] > e["price_at_prediction"] else "down"
                    act_dir  = "up" if actual_price > e["price_at_prediction"] else "down"
                    pct_err  = abs(actual_price - e["predicted_price"]) / e["price_at_prediction"] * 100
                    outcome  = ("correct" if pred_dir==act_dir and pct_err<=3
                                else "partial" if pred_dir==act_dir else "incorrect")
                e["outcome"] = outcome
                self._save()
                return e
        return {}

    def get_stats(self, ticker=None) -> dict:
        recs = [r for r in self.history
                if r["outcome"] and (ticker is None or r["ticker"]==ticker)]
        if not recs:
            return {"total":0,"correct":0,"partial":0,"incorrect":0,"accuracy_pct":0}
        c = sum(1 for r in recs if r["outcome"]=="correct")
        p = sum(1 for r in recs if r["outcome"]=="partial")
        i = sum(1 for r in recs if r["outcome"]=="incorrect")
        return {"total":len(recs),"correct":c,"partial":p,"incorrect":i,
                "accuracy_pct":round((c+p*0.5)/len(recs)*100,1),"ticker":ticker or "all"}

    def get_pending(self, ticker=None) -> list:
        return [r for r in self.history
                if r["outcome"] is None and (ticker is None or r["ticker"]==ticker)]

    def get_recent(self, ticker=None, n=10) -> list:
        recs = [r for r in self.history if (ticker is None or r["ticker"]==ticker)]
        return sorted(recs, key=lambda x: x["date_made"], reverse=True)[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalAnalysisAgent:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data   = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, period="6mo", interval="1d", progress=False)
        if df.empty: raise ValueError(f"No data for {self.ticker}")
        df.dropna(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = ['Open','High','Low','Close','Volume']
        return df

    def analyze(self) -> dict:
        df = self.data.copy()
        rationale, short_term, long_term = [], "Hold", "Hold"
        try:
            df['RSI']   = TA.RSI(df)
            df['MACD']  = TA.MACD(df)['MACD']
            df['SMA_20']= TA.SMA(df, 20)
            df['SMA_50']= TA.SMA(df, 50)
            df['EMA_20']= TA.EMA(df, 20)
            df['ADX']   = TA.ADX(df)
            df['MOM']   = TA.MOM(df)
            df['CCI']   = TA.CCI(df)
            df['OBV']   = TA.OBV(df)
            l = df.iloc[-1]

            if   l['RSI']<30 and l['MACD']>0 and l['MOM']>0:  short_term="Buy";  rationale.append("RSI<30, MACD>0, MOM>0 → bullish momentum.")
            elif l['RSI']>70 and l['MACD']<0 and l['MOM']<0:  short_term="Sell"; rationale.append("RSI>70, MACD<0, MOM<0 → bearish momentum.")
            elif l['CCI']>100:  short_term="Buy";  rationale.append("CCI>100 → strong upside potential.")
            elif l['CCI']<-100: short_term="Sell"; rationale.append("CCI<-100 → downside risk.")
            else: rationale.append(f"RSI at {l['RSI']:.2f} → neutral short-term.")

            if   l['SMA_20']>l['SMA_50'] and l['EMA_20']>l['SMA_50']: long_term="Buy";  rationale.append("SMA20 & EMA20 > SMA50 → long-term uptrend.")
            elif l['SMA_20']<l['SMA_50'] and l['EMA_20']<l['SMA_50']: long_term="Sell"; rationale.append("SMA20 & EMA20 < SMA50 → long-term downtrend.")
            else: rationale.append("Mixed SMA/EMA → long-term Hold.")
            rationale.append(f"ADX {l['ADX']:.2f} → {'strong' if l['ADX']>25 else 'weak/sideways'} trend.")

        except Exception as e:
            return {"agent":"technical","verdict":{"short_term":"Hold","long_term":"Hold"},"rationale":[f"Error: {e}"]}

        return {"agent":"technical","verdict":{"short_term":short_term,"long_term":long_term},"rationale":rationale}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTAL ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

class FundamentalAnalysisAgent:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data   = self._fetch_data()

    def _fetch_data(self) -> dict:
        try:    return yf.Ticker(self.ticker).info
        except: return {"error": "Failed to fetch fundamental data"}

    def _safe(self, key, default=None):
        v = self.data.get(key, default)
        return v if v not in [None, np.nan] else default

    def analyze(self) -> dict:
        if "error" in self.data:
            return {"agent":"fundamental","verdict":{"short_term":"Hold","long_term":"Hold"},"rationale":[self.data["error"]]}

        rationale, score = [], 0

        pe = self._safe("trailingPE")
        if pe is not None:
            if pe<15:   score+=1; rationale.append(f"PE {pe:.2f} → undervalued.")
            elif pe>30: score-=1; rationale.append(f"PE {pe:.2f} → overvalued.")
            else:       rationale.append(f"PE {pe:.2f} → fair range.")
        else: rationale.append("PE unavailable.")

        pm = self._safe("profitMargins")
        if pm is not None:
            if pm>0.15:   score+=1; rationale.append(f"Profit margin {pm*100:.1f}% → strong.")
            elif pm<0.05: score-=1; rationale.append(f"Profit margin {pm*100:.1f}% → weak.")
            else:         rationale.append(f"Profit margin {pm*100:.1f}% → average.")
        else: rationale.append("Profit margin unavailable.")

        roe = self._safe("returnOnEquity")
        if roe is not None:
            if roe>0.15:   score+=1; rationale.append(f"ROE {roe*100:.1f}% → efficient.")
            elif roe<0.07: score-=1; rationale.append(f"ROE {roe*100:.1f}% → inefficient.")
            else:          rationale.append(f"ROE {roe*100:.1f}% → moderate.")
        else: rationale.append("ROE unavailable.")

        cr = self._safe("currentRatio")
        if cr is not None:
            if cr>=1.5:  score+=1; rationale.append(f"Current ratio {cr:.2f} → healthy.")
            elif cr<1.0: score-=1; rationale.append(f"Current ratio {cr:.2f} → liquidity risk.")
            else:        rationale.append(f"Current ratio {cr:.2f} → acceptable.")
        else: rationale.append("Current ratio unavailable.")

        de = self._safe("debtToEquity")
        if de is not None:
            if de<1.0:  score+=1; rationale.append(f"D/E {de:.2f} → manageable.")
            elif de>2.0:score-=1; rationale.append(f"D/E {de:.2f} → high debt risk.")
            else:       rationale.append(f"D/E {de:.2f} → reasonable.")
        else: rationale.append("D/E unavailable.")

        fcf = self._safe("freeCashflow")
        rev = self._safe("totalRevenue")
        if fcf and rev:
            fcf_m = fcf/rev
            if fcf_m>0.1:   score+=1; rationale.append(f"FCF margin {fcf_m*100:.1f}% → strong.")
            elif fcf_m<0.02:score-=1; rationale.append(f"FCF margin {fcf_m*100:.1f}% → weak.")
            else:           rationale.append(f"FCF margin {fcf_m*100:.1f}% → moderate.")
        else: rationale.append("FCF unavailable.")

        eg = self._safe("earningsQuarterlyGrowth")
        if eg is not None:
            if eg>0.2:  score+=1; rationale.append(f"Earnings growth {eg*100:.1f}% → strong.")
            elif eg<0:  score-=1; rationale.append(f"Earnings growth {eg*100:.1f}% → negative.")
            else:       rationale.append(f"Earnings growth {eg*100:.1f}% → moderate.")
        else: rationale.append("Earnings growth unavailable.")

        long_term  = "Buy" if score>=4 else "Sell" if score<=-2 else "Hold"
        short_term = ("Buy"  if pm and pm>0.15 and eg and eg>0.15
                      else "Sell" if pm and pm<0.05 and eg and eg<0
                      else "Hold")
        return {"agent":"fundamental","verdict":{"short_term":short_term,"long_term":long_term},"rationale":rationale}


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE PREDICTION AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PricePredictionAgent:
    def __init__(self, ticker: str):
        self.ticker      = ticker
        self.df          = self._fetch_data()
        self._live_price = self._get_live_price()

    def _fetch_data(self) -> pd.DataFrame:
        df = yf.download(self.ticker, period="2y", interval="1d", progress=False)
        if df.empty: raise ValueError(f"No price data for {self.ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = ['Open','High','Low','Close','Volume']
        df.dropna(inplace=True)
        return df

    def _get_live_price(self) -> float:
        """Always use live price — never stale historical close."""
        try:
            info = yf.Ticker(self.ticker).info
            p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            if p: return float(p)
        except Exception: pass
        return float(self.df['Close'].iloc[-1])

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c = df['Close']
        f['returns']     = c.pct_change()
        f['log_returns'] = np.log(c / c.shift(1))
        for n in [5,10,20]: f[f'sma_{n}'] = c.rolling(n).mean()
        f['ema_12'] = c.ewm(span=12,adjust=False).mean()
        f['ema_26'] = c.ewm(span=26,adjust=False).mean()
        f['macd']   = f['ema_12'] - f['ema_26']
        std20 = c.rolling(20).std()
        upper, lower = f['sma_20']+2*std20, f['sma_20']-2*std20
        f['bb_pct'] = (c - lower) / (upper - lower).replace(0,np.nan)
        delta = c.diff()
        gain  = delta.where(delta>0,0.0).rolling(14).mean()
        loss  = (-delta.where(delta<0,0.0)).rolling(14).mean()
        f['rsi'] = 100 - (100 / (1 + gain/loss.replace(0,np.nan)))
        f['vol_chg']        = df['Volume'].pct_change()
        f['high_low_pct']   = (df['High']-df['Low']) / c.replace(0,np.nan)
        f['close_open_pct'] = (c - df['Open'].replace(0,np.nan)) / df['Open'].replace(0,np.nan)
        for lag in [1,2,3,5,10]: f[f'lag_{lag}'] = c.shift(lag)
        f['target'] = c.shift(-1)
        f.replace([np.inf,-np.inf],np.nan,inplace=True)
        f.dropna(inplace=True)
        return f

    def _lr_forecast(self, days: int):
        feat = self._build_features(self.df)
        if len(feat)<60: return None,None,None
        fcols = [c for c in feat.columns if c!='target']
        X = np.clip(feat[fcols].values.astype(np.float64),-1e9,1e9)
        y = np.clip(feat['target'].values.astype(np.float64),-1e9,1e9)
        sX,sy = MinMaxScaler(),MinMaxScaler()
        Xs = sX.fit_transform(X)
        ys = sy.fit_transform(y.reshape(-1,1)).ravel()
        split = int(len(Xs)*0.8)
        model = LinearRegression()
        model.fit(Xs[:split],ys[:split])
        vp = sy.inverse_transform(model.predict(Xs[split:]).reshape(-1,1)).ravel()
        va = sy.inverse_transform(ys[split:].reshape(-1,1)).ravel()
        mae  = float(mean_absolute_error(va,vp))
        mape = float(np.mean(np.abs((va-vp)/(np.abs(va)+1e-9)))*100)
        lag_map = {f'lag_{l}':(fcols.index(f'lag_{l}'),l) for l in [1,2,3,5,10] if f'lag_{l}' in fcols}
        last_X = Xs[-1].copy()
        prev   = list(self.df['Close'].values[-15:])
        preds  = []
        for _ in range(days):
            pp = float(sy.inverse_transform([[model.predict(last_X.reshape(1,-1))[0]]])[0][0])
            preds.append(pp); prev.append(pp)
            for nm,(idx,lag_n) in lag_map.items():
                rv  = prev[-(lag_n+1)]
                rng = sX.data_max_[idx]-sX.data_min_[idx]
                last_X[idx] = (rv-sX.data_min_[idx])/rng if rng>0 else 0.5
        return preds,mae,mape

    def _arima_forecast(self, days: int):
        if not ARIMA_AVAILABLE: return None,None,None
        try:
            prices = self.df['Close'].values[-180:].astype(np.float64)
            fitted = ARIMA(prices,order=(5,1,2)).fit()
            fc     = list(fitted.forecast(steps=days))
            resid  = np.abs(fitted.resid[-30:])
            base   = np.abs(prices[-30:])+1e-9
            return fc, float(np.mean(resid)), float(np.mean(resid/base)*100)
        except Exception: return None,None,None

    @staticmethod
    def _ensemble(lr,ar,w_lr=0.55,w_ar=0.45):
        if lr is None and ar is None: return None
        if lr is None: return ar
        if ar is None: return lr
        n = min(len(lr),len(ar))
        return [w_lr*lr[i]+w_ar*ar[i] for i in range(n)]

    @staticmethod
    def _apply_sentiment(preds, score: float):
        if not preds or score==0: return preds
        n   = len(preds)
        max_drift = 0.015*abs(score)
        d   = 1 if score>0 else -1
        return [p*(1+d*max_drift*((i+1)/n)) for i,p in enumerate(preds)]

    @staticmethod
    def _future_dates(days: int):
        dates, cur = [], datetime.now()
        while len(dates)<days:
            cur += timedelta(days=1)
            if cur.weekday()<5: dates.append(cur)
        return dates

    def _buy_signal(self, current_price, end_price, sentiment, lr_mape=None) -> dict:
        pct  = (end_price - current_price) / current_price * 100
        conf = max(0, 100 - (lr_mape or 5))
        if   pct>3  and sentiment>=0 and conf>=90: signal,reason = "Strong Buy 🟢🟢","Strong upside forecast + positive news sentiment."
        elif pct>1.5 or (pct>0 and sentiment>0.2): signal,reason = "Buy 🟢","Moderate upside predicted. Conditions look favourable."
        elif pct<-3 and sentiment<=0:              signal,reason = "Strong Sell 🔴🔴","Significant downside + negative news."
        elif pct<-1.5 or (pct<0 and sentiment<-0.2):signal,reason= "Sell 🔴","Moderate downside predicted. Consider waiting."
        else:                                      signal,reason = "Hold 🟡","Market looks sideways. Wait for a clearer signal."
        return {"signal":signal,"reason":reason,
                "predicted_change":round(pct,2),"sentiment":round(sentiment,3),"confidence":round(conf,1)}

    def predict(self, days: int = 30, sentiment_score: float = 0.0) -> dict:
        current_price = self._live_price  # always live

        lr_preds, lr_mae, lr_mape       = self._lr_forecast(days)
        arima_preds, arima_mae, arima_mape = self._arima_forecast(days)
        ensemble = self._ensemble(lr_preds, arima_preds)
        if ensemble and sentiment_score!=0:
            ensemble = self._apply_sentiment(ensemble, sentiment_score)

        future_dates = self._future_dates(days)

        weekly_table = []
        for w in range(min(4, days//5)):
            i_s,i_e = w*5, min(w*5+4,days-1)
            row = {"week":f"Week {w+1}",
                   "date_range":(f"{future_dates[i_s].strftime('%b %d')} – "
                                  f"{future_dates[i_e].strftime('%b %d')}")}
            if lr_preds    and i_e<len(lr_preds):    row["lr"]      =round(lr_preds[i_e],2)
            if arima_preds and i_e<len(arima_preds): row["arima"]   =round(arima_preds[i_e],2)
            if ensemble    and i_e<len(ensemble):    row["ensemble"]=round(ensemble[i_e],2)
            weekly_table.append(row)

        end_price  = (ensemble or lr_preds or arima_preds or [current_price])[-1]
        pct_change = (end_price-current_price)/current_price*100
        direction  = "rise" if pct_change>0 else "fall"
        snote      = ""
        if sentiment_score!=0:
            snote = f" News sentiment ({LiveNewsAgent._label(sentiment_score)}: {sentiment_score:+.2f}) applied."

        summary = (f"Nivesh AI forecasts {self.ticker} to {direction} from "
                   f"₹{current_price:.2f} → ₹{end_price:.2f} "
                   f"({pct_change:+.2f}%) over the next {days} trading days.{snote}")

        return {
            "ticker":            self.ticker,
            "current_price":     current_price,
            "days_ahead":        days,
            "future_dates":      future_dates,
            "lr_predictions":    lr_preds,
            "arima_predictions": arima_preds,
            "ensemble":          ensemble,
            "lr_mae":            lr_mae,
            "lr_mape":           lr_mape,
            "arima_mae":         arima_mae,
            "arima_mape":        arima_mape,
            "weekly_table":      weekly_table,
            "summary":           summary,
            "sentiment_score":   sentiment_score,
            "buy_now":           self._buy_signal(current_price, end_price, sentiment_score, lr_mape),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER AGENT
# ══════════════════════════════════════════════════════════════════════════════

class MasterAgent:
    def __init__(self, ticker: str, gemini_api_key: str = None):
        self.ticker     = ticker
        self.api_key    = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.ta_agent   = TechnicalAnalysisAgent(ticker)
        self.fa_agent   = FundamentalAnalysisAgent(ticker)
        self.pred_agent = PricePredictionAgent(ticker)
        self.news_agent = LiveNewsAgent(ticker, self.api_key)
        self.tracker    = PredictionOverrideTracker()
        self._news_cache     = None
        self._analysis_cache = None

    def _vote(self, a, b):
        if a==b: return a
        if "Hold" in [a,b]: return a if b=="Hold" else b
        return "Hold"

    def get_final_verdict(self) -> dict:
        ta = self.ta_agent.analyze()
        fa = self.fa_agent.analyze()
        return {"ticker":self.ticker,
                "verdict":{"short_term":self._vote(ta["verdict"]["short_term"],fa["verdict"]["short_term"]),
                           "long_term": self._vote(ta["verdict"]["long_term"], fa["verdict"]["long_term"])},
                "rationale":{"technical":ta["rationale"],"fundamental":fa["rationale"]}}

    def get_live_news(self) -> dict:
        if self._news_cache is None:
            self._news_cache = self.news_agent.fetch_all_news()
        return self._news_cache

    def get_news_analysis(self) -> dict:
        if self._analysis_cache is None:
            self._analysis_cache = self.news_agent.analyze_news(
                self.get_live_news(), self.pred_agent._live_price)
        return self._analysis_cache

    def get_price_prediction(self, days: int = 30,
                              use_news_sentiment: bool = True) -> dict:
        sentiment = 0.0
        if use_news_sentiment:
            try: sentiment = float(self.get_news_analysis().get("sentiment_score",0.0))
            except Exception: pass
        result = self.pred_agent.predict(days, sentiment_score=sentiment)
        if result.get("ensemble"):
            try:
                pid = self.tracker.save_prediction(
                    self.ticker, result["current_price"],
                    result["ensemble"][-1], days, sentiment,
                    self.get_news_analysis().get("summary",""))
                result["prediction_id"] = pid
            except Exception: pass
        return result

    def get_master_summary(self) -> dict:
        ta    = self.ta_agent.analyze()
        fa    = self.fa_agent.analyze()
        news  = self.get_live_news()
        na    = self.get_news_analysis()
        pred  = self.get_price_prediction(30)
        stats = self.tracker.get_stats(self.ticker)
        tv,fv,nv = (ta["verdict"]["short_term"], fa["verdict"]["long_term"],
                    na.get("buy_signal","Hold"))
        votes = [tv,fv,nv]
        buy_v = sum(1 for v in votes if "Buy" in v)
        sell_v= sum(1 for v in votes if "Sell" in v)
        overall = "Buy 🟢" if buy_v>=2 else "Sell 🔴" if sell_v>=2 else "Hold 🟡"
        return {
            "ticker":                  self.ticker,
            "live_price":              pred["current_price"],
            "timestamp":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_verdict":         overall,
            "verdicts":{"technical_short":tv,"fundamental_long":fv,"news_based":nv},
            "technical_rationale":     ta["rationale"],
            "fundamental_rationale":   fa["rationale"],
            "news_summary":            na.get("summary",""),
            "india_impact":            na.get("india_impact",""),
            "global_impact":           na.get("global_impact",""),
            "sentiment_score":         na.get("sentiment_score",0),
            "sentiment_label":         na.get("sentiment_label","Neutral"),
            "key_themes":              na.get("key_themes",[]),
            "risk_factors":            na.get("risk_factors",[]),
            "positive_catalysts":      na.get("positive_catalysts",[]),
            "price_impact_estimate":   na.get("price_impact_estimate",0),
            "buy_now":                 pred.get("buy_now",{}),
            "prediction": {
                "30d_target":  round(pred["ensemble"][-1],2) if pred.get("ensemble") else None,
                "change_pct":  round((pred["ensemble"][-1]-pred["current_price"])/pred["current_price"]*100,2) if pred.get("ensemble") else None,
                "summary":     pred["summary"],
                "prediction_id":pred.get("prediction_id"),
            },
            "prediction_accuracy": stats,
            "news_counts": {
                "india_stock": len(news.get("india_stock_news",[])),
                "india_macro": len(news.get("india_macro_news",[])),
                "global":      len(news.get("global_news",[])),
            },
        }

    def mark_prediction(self,pid,actual_price,outcome="auto",note=""):
        return self.tracker.mark_outcome(pid,actual_price,outcome,note)
    def get_prediction_history(self,n=10): return self.tracker.get_recent(self.ticker,n)
    def get_pending_predictions(self): return self.tracker.get_pending(self.ticker)
    def get_accuracy_stats(self): return self.tracker.get_stats(self.ticker)


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MasterAgent("ITC.NS")
    ms    = agent.get_master_summary()
    print(f"Overall : {ms['overall_verdict']}")
    print(f"Price   : ₹{ms['live_price']}")
    print(f"Buy Now : {ms['buy_now'].get('signal')} — {ms['buy_now'].get('reason')}")

    pred = agent.get_price_prediction(30)
    calc = calculate_investment(10000, pred["current_price"],
                                pred["ensemble"], pred["future_dates"])
    print(f"\n₹10,000 → ₹{calc['end_value']} ({calc['pl_pct']:+.1f}%)")