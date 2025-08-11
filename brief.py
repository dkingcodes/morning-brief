# brief.py — v2
# Morning Daily Brief → Discord (7:30am ET)
# Adds: detailed Regime Classifier + Playbook cues, Macro calendar hook, optional breadth, IV30/skew proxy.

import os, math, traceback, random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf

ET = ZoneInfo("America/New_York")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
if not WEBHOOK:
    print("❌ DISCORD_WEBHOOK_URL is missing (GitHub secret not set or not passed).")
    raise SystemExit(2)

# ---------------- helpers ----------------
def pct(a, b):
    if a is None or b is None or b == 0: return None
    return 100.0 * (a / b - 1.0)

def fmt(x, n=2):
    return "na" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{n}f}"

def calc_atr14(df):
    if df is None or df.empty or len(df) < 15:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

def load_macro_today():
    """
    Optional file: macro.yml in repo root:
      2025-08-11:
        - "CPI 08:30 ET"
        - "3y auction 13:00 ET"
    If not found, return [].
    """
    p = "macro.yml"
    if not os.path.exists(p): return []
    try:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        key = datetime.now(ET).strftime("%Y-%m-%d")
        return list(y.get(key, []))
    except Exception:
        return []

def iv30_and_skew_spy(prior_close):
    """
    Quick IV30/skew proxy from SPY option chain:
    - choose expiry closest to 30 calendar days
    - IV30 ≈ ATM IV (avg of near-ATM call/put)
    - Skew ≈ IV(OTM put @ ~90% spot) − IV(OTM call @ ~110% spot)  (vol points)
    If chain unavailable, fall back to VIX as IV30 proxy and 'mild' skew text.
    """
    try:
        spy = yf.Ticker("SPY")
        opts = spy.options
        if not opts: return None, "mild"
        # pick expiry closest to 30d
        today = datetime.now(ET).date()
        def d2(exp):
            dt = datetime.strptime(exp, "%Y-%m-%d").date()
            return abs((dt - today).days - 30)
        expiry = sorted(opts, key=d2)[0]
        chain = spy.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        if calls.empty or puts.empty: return None, "mild"
        calls["dist"] = (calls["strike"] - prior_close).abs()
        puts["dist"]  = (puts["strike"]  - prior_close).abs()
        # ATM IV
        atm_call = float(calls.sort_values("dist").iloc[0]["impliedVolatility"])
        atm_put  = float(puts.sort_values("dist").iloc[0]["impliedVolatility"])
        iv30 = (atm_call + atm_put) / 2.0
        # OTM strikes ~ +/-10%
        k_put  = prior_close * 0.90
        k_call = prior_close * 1.10
        put_otm  = puts.iloc[(puts["strike"] - k_put).abs().argsort()[:1]]
        call_otm = calls.iloc[(calls["strike"] - k_call).abs().argsort()[:1]]
        iv_put  = float(put_otm["impliedVolatility"].iloc[0])
        iv_call = float(call_otm["impliedVolatility"].iloc[0])
        skew_pts = (iv_put - iv_call) * 100.0  # vol points
        skew_note = "put-heavy" if skew_pts > 1.0 else ("balanced" if abs(skew_pts) <= 1.0 else "call-tilt")
        return iv30, skew_note
    except Exception:
        # fallback: use VIX
        try:
            vix = float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1]) / 100.0
            return vix, "mild"
        except Exception:
            return None, "mild"

def regime_classifier(prior_close, atr_points, vix, vix3m, px_close):
    # Simple & transparent:
    # MR if term>1 and ATR% not high; TREND if term<=1 and |20d slope| strong; else CHOPPY
    term = None if (vix is None or vix == 0 or vix3m is None) else (vix3m / vix)
    atr_pct = None if (prior_close in (None, 0) or atr_points is None) else (atr_points / prior_close) * 100.0
    slope_pct_per_day = None
    if px_close is not None and len(px_close) >= 21:
        last20 = px_close.tail(20)
        slope_pct_per_day = ((last20.iloc[-1] - last20.iloc[0]) / 20.0) / px_close.iloc[-2] * 100.0

    label = "CHOPPY"; size = 0.5; drivers = []
    if term is not None and term > 1.0 and (atr_pct or 0) < 1.6:
        label = "MEAN-REVERT"; size = 0.8; drivers = ["term>1 (contango)", "ATR% mid"]
    elif term is not None and term <= 1.0 and slope_pct_per_day is not None and abs(slope_pct_per_day) > 0.12:
        label = "TREND"; size = 1.0; drivers = ["term<=1", f"slope {slope_pct_per_day:+.2f}%/day"]
    else:
        drivers = ["neutral mix"]
    return label, size, term, atr_pct, slope_pct_per_day, drivers

def spx_breadth(use_wiki=False, sample=120, seed=42):
    """
    Optional breadth: % of SPX above 20/50dma via Wikipedia tickers (sampled to keep it light).
    Set env USE_WIKI_BREADTH=true in workflow to enable.
    """
    try:
        if not use_wiki: return None, None
        import requests, bs4
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=20).text
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"id":"constituents"})
        syms = [r.find_all("td")[0].text.strip().replace(".", "-") for r in table.find_all("tr")[1:]]
        if len(syms) < 50: return None, None
        random.Random(seed).shuffle(syms)
        syms = syms[:sample]
        df = yf.download(syms, period="70d", interval="1d", progress=False, group_by="ticker", auto_adjust=False)["Close"]
        above20 = []
        above50 = []
        for s in syms:
            try:
                c = df[s].dropna()
                if len(c) >= 50:
                    ma20 = c.rolling(20).mean().iloc[-1]
                    ma50 = c.rolling(50).mean().iloc[-1]
                    above20.append(float(c.iloc[-1] > ma20))
                    above50.append(float(c.iloc[-1] > ma50))
            except Exception:
                continue
        p20 = round(100.0 * np.mean(above20), 1) if above20 else None
        p50 = round(100.0 * np.mean(above50), 1) if above50 else None
        return p20, p50
    except Exception:
        return None, None

# ---------------- main ----------------
def main():
    now_et = datetime.now(ET)

    # Core prices
    spy = yf.Ticker("SPY")
    daily = spy.history(period="60d", interval="1d", auto_adjust=False)
    if daily.empty: raise RuntimeError("No SPY daily data")

    prev = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    prior_close = float(prev["Close"]); prior_high = float(prev["High"]); prior_low = float(prev["Low"])
    prior_vwap = float((prev["High"] + prev["Low"] + prev["Close"]) / 3.0)  # proxy VWAP

    atr_points = calc_atr14(daily)

    pm = spy.history(period="1d", interval="1m", prepost=True)
    pre_last = float(pm["Close"].dropna().iloc[-1]) if not pm.empty else prior_close
    o_low  = float(pm["Low"].min()) if not pm.empty else prior_low
    o_high = float(pm["High"].max()) if not pm.empty else prior_high

    # Vol & term
    vixh = yf.Ticker("^VIX").history(period="2d", interval="1d")
    vix3h = yf.Ticker("^VIX3M").history(period="2d", interval="1d")
    vix = None if vixh.empty else float(vixh["Close"].iloc[-1])
    vix3 = None if vix3h.empty else float(vix3h["Close"].iloc[-1])

    iv30, skew_note = iv30_and_skew_spy(prior_close)

    # Rates/FX + guardrails
    tnxh = yf.Ticker("^TNX").history(period="2d", interval="1d")
    tnx = None if tnxh.empty else float(tnxh["Close"].iloc[-1])  # *10
    ust10y = None if tnx is None else tnx / 10.0

    dxyh = yf.Ticker("DX-Y.NYB").history(period="3d", interval="1d")
    dxy = None if dxyh.empty else float(dxyh["Close"].iloc[-1])
    dxy_prev = None if len(dxyh) < 2 else float(dxyh["Close"].iloc[-2])
    dxy_chg_pct = None if (dxy is None or dxy_prev is None or dxy_prev==0) else (dxy - dxy_prev) / dxy_prev * 100.0

    guard = (vix is not None and vix > 20.0) or (dxy_chg_pct is not None and dxy_chg_pct > 0.5)

    # Regime
    label, size_mult, term, atr_pct, slope_pct_day, drivers = regime_classifier(
        prior_close, atr_points, vix, vix3, daily["Close"]
    )
    base_label = label
    if guard: label = label + " (FLAT*)"

    # Bands
    plus1 = prior_close + (atr_points or 0.0); minus1 = prior_close - (atr_points or 0.0)
    plus1_5 = prior_close + 1.5 * (atr_points or 0.0); minus1_5 = prior_close - 1.5 * (atr_points or 0.0)

    # Macro (optional yaml)
    macro_today = load_macro_today()

    # Breadth (optional)
    use_breadth = (os.getenv("USE_WIKI_BREADTH", "false").lower() == "true")
    p20, p50 = spx_breadth(use_breadth)

    # ----- Build Discord embed -----
    color = 0x8C52FF  # Xenith purple
    fields = [
        {"name":"SPY pre-market",
         "value": f"{fmt(pre_last)} ({(pct(pre_last, prior_close) or 0):+.2f}%) · O/N {fmt(o_low)}–{fmt(o_high)}",
         "inline": False},
        {"name":"Key levels",
         "value": f"H/L {fmt(prior_high)}/{fmt(prior_low)} · Close {fmt(prior_close)} · VWAP~{fmt(prior_vwap)}\n"
                  f"ATR14≈{fmt(atr_points)} → ±1× {fmt(plus1)}/{fmt(minus1)} · ±1.5× {fmt(plus1_5)}/{fmt(minus1_5)}",
         "inline": False},
        {"name":"Vol snapshot",
         "value": f"VIX {fmt(vix,1)} · 3M/1M {fmt(term,2)} · IV30~{fmt(iv30,3)} · skew {skew_note}",
         "inline": True},
        {"name":"Rates/FX",
         "value": f"UST10Y {fmt(ust10y,2)}% · DXY {fmt(dxy,2)} ({fmt(dxy_chg_pct,2)}%) · "
                  f"Guardrails {'TRIGGERED' if guard else 'ok'}",
         "inline": True},
    ]

    if p20 is not None or p50 is not None:
        fields.append({"name":"Breadth",
                       "value": f">%20dma {fmt(p20,1)}% · >%50dma {fmt(p50,1)}%",
                       "inline": True})

    if macro_today:
        fields.append({"name":"Macro calendar (today)", "value": "• " + "\n• ".join(macro_today), "inline": False})

    # Regime + playbook (detailed)
    play_lines = []
    if "MEAN-REVERT" in base_label:
        play_lines = [
            "Primary: 0–2 DTE **iron condors / credit spreads** at ±1.0× ATR bands",
            "Timing: enter after **10:00 ET** on **second probe**; avoid first impulse",
            "Size: **0.8×** baseline (1.0× if breadth >60% & term stays >1)",
            "Avoid: breakout chases unless breadth expands & rates confirm",
        ]
    elif "TREND" in base_label:
        play_lines = [
            "Primary: **directional debit verticals / breakouts** with follow-through",
            "Timing: use pullbacks to 0.5–0.7× ATR; cut on regime slip",
            "Size: **1.0×** baseline (reduce if guardrails trigger)",
            "Avoid: range-selling unless term>1 and momentum wanes",
        ]
    else:
        play_lines = [
            "Primary: **small / delta-neutral**; scalps toward VWAP only",
            "Size: **0.5×** baseline",
            "Avoid: strong directional bets; wait for clarity",
        ]

    fields.append({
        "name":"Regime Classifier",
        "value": f"**{label}** — base={base_label} • size: {size_mult:.1f}×\n"
                 f"Drivers: {', '.join(drivers)}\n"
                 + "\n".join("• " + s for s in play_lines),
        "inline": False
    })

    payload = {
        "embeds": [{
            "title": f"Morning Daily Brief — {now_et.strftime('%a, %b %d, %Y')} · 7:30am ET",
            "color": color,
            "fields": fields,
            "footer": {"text": "Xenith Trading • Automated Daily Brief"}
        }]
    }

    requests.post(WEBHOOK, json=payload, timeout=30).raise_for_status()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            msg = f"❌ Brief failed: {type(e).__name__}: {e}\n```\n{traceback.format_exc()[:1500]}\n```"
            requests.post(WEBHOOK, json={"content": msg}, timeout=15)
        except Exception:
            pass
        raise
