# brief.py
# Sends a Morning Daily Brief to a Discord webhook at 7:30am ET on weekdays.
# Data sources: Yahoo Finance via yfinance. Robust to partial data; always posts something.

import os, json, math, traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf

import os
# ...
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
if not WEBHOOK:
    print("❌ DISCORD_WEBHOOK_URL is missing (GitHub secret not set or not passed).")
    raise SystemExit(2)


ET = ZoneInfo("America/New_York")

def pct(a, b):
    if a is None or b is None or b == 0: return None
    return 100.0 * (a / b - 1.0)

def fmt(x, n=2):
    return "na" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"

def safe_last(series):
    return None if series is None or len(series)==0 else float(series.iloc[-1])

def calc_atr14(df):
    # True Range ATR (points) from daily OHLC
    if df is None or df.empty or len(df) < 15:
        return None
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    return float(atr)

def regime_classifier(prior_close, atr_points, vix, vix3m, px_close):
    # Simple, transparent rules:
    # - MEAN-REVERT if term > 1 and ATR% not high
    # - TREND if term <= 1 and 20d slope is strong
    # - Else CHOPPY
    term = None if (vix is None or vix == 0 or vix3m is None) else (vix3m / vix)
    atr_pct = None if (prior_close in (None, 0) or atr_points is None) else (atr_points / prior_close) * 100.0

    slope_pct_per_day = None
    if px_close is not None and len(px_close) >= 21:
        last20 = px_close.tail(20)
        # simple slope: (y_t - y_0) / 20 days as pct of prior_close
        slope_pct_per_day = ((last20.iloc[-1] - last20.iloc[0]) / 20.0) / px_close.iloc[-2] * 100.0

    label = "CHOPPY"
    size = 0.5
    drivers = []

    if term is not None and term > 1.0 and (atr_pct or 0) < 1.6:
        label = "MEAN-REVERT"; size = 0.8
        drivers = ["term>1 (contango)", "ATR% mid"]
    elif term is not None and term <= 1.0 and slope_pct_per_day is not None and abs(slope_pct_per_day) > 0.12:
        label = "TREND"; size = 1.0
        drivers = ["term<=1", f"slope {slope_pct_per_day:+.2f}%/day"]
    else:
        drivers = ["neutral mix"]

    return label, size, term, atr_pct, slope_pct_per_day

def main():
    now_et = datetime.now(ET)
    # --- Fetch core series ---
    spy = yf.Ticker("SPY")
    daily = spy.history(period="60d", interval="1d", auto_adjust=False)
    if daily.empty:
        raise RuntimeError("No SPY daily data")

    # Prior session
    prev = daily.iloc[-2] if len(daily) >= 2 else daily.iloc[-1]
    prior_close = float(prev["Close"])
    prior_high  = float(prev["High"])
    prior_low   = float(prev["Low"])
    prior_vwap  = float((prev["High"] + prev["Low"] + prev["Close"]) / 3.0)  # proxy

    # ATR14
    atr_points = calc_atr14(daily)

    # Premarket snapshot (best effort)
    pre = spy.history(period="1d", interval="1m", prepost=True)
    pre_last = float(pre["Close"].dropna().iloc[-1]) if not pre.empty else prior_close
    ovr_low  = float(pre["Low"].min()) if not pre.empty else prior_low
    ovr_high = float(pre["High"].max()) if not pre.empty else prior_high

    # Vol & term
    vixh  = yf.Ticker("^VIX").history(period="2d", interval="1d")
    vix3h = yf.Ticker("^VIX3M").history(period="2d", interval="1d")
    vix   = None if vixh.empty else float(vixh["Close"].iloc[-1])
    vix3m = None if vix3h.empty else float(vix3h["Close"].iloc[-1])

    # Rates & FX
    tnxh = yf.Ticker("^TNX").history(period="2d", interval="1d")
    tnx = None if tnxh.empty else float(tnxh["Close"].iloc[-1])  # *10
    ust10y = None if tnx is None else tnx / 10.0

    dxyh = yf.Ticker("DX-Y.NYB").history(period="3d", interval="1d")
    dxy = None if dxyh.empty else float(dxyh["Close"].iloc[-1])
    dxy_prev = None if len(dxyh) < 2 else float(dxyh["Close"].iloc[-2])
    dxy_chg_pct = None if (dxy is None or dxy_prev is None or dxy_prev==0) else (dxy - dxy_prev) / dxy_prev * 100.0

    # Guardrails
    guard = (vix is not None and vix > 20.0) or (dxy_chg_pct is not None and dxy_chg_pct > 0.5)

    # Regime
    label, size_mult, term, atr_pct, slope_pct_day = regime_classifier(
        prior_close, atr_points, vix, vix3m, daily["Close"]
    )
    if guard:
        label = label + " (FLAT*)"

    # Bands
    plus1 = prior_close + (atr_points or 0.0)
    minus1 = prior_close - (atr_points or 0.0)
    plus1_5 = prior_close + 1.5 * (atr_points or 0.0)
    minus1_5 = prior_close - 1.5 * (atr_points or 0.0)

    # Build embed
    color = 0x00A67E if "TREND" in label else (0xEAB308 if "MEAN" in label else 0x6B7280)
    fields = [
        {"name":"SPY pre-market",
         "value": f"{fmt(pre_last)} ({(pct(pre_last, prior_close) or 0):+.2f}%) · O/N {fmt(ovr_low)}–{fmt(ovr_high)}",
         "inline": False},
        {"name":"Key levels",
         "value": f"H/L {fmt(prior_high)}/{fmt(prior_low)} · Close {fmt(prior_close)} · VWAP~{fmt(prior_vwap)}\n"
                  f"ATR14≈{fmt(atr_points)} → ±1× {fmt(plus1)}/{fmt(minus1)} · ±1.5× {fmt(plus1_5)}/{fmt(minus1_5)}",
         "inline": False},
        {"name":"Vol snapshot",
         "value": f"VIX {fmt(vix,1)} · VIX3M/VIX {fmt(term,2)}",
         "inline": True},
        {"name":"Rates/FX",
         "value": f"UST10Y {fmt(ust10y,2)}% · DXY {fmt(dxy,2)} ({fmt(dxy_chg_pct,2)}%) · Guardrails "
                  f"{'TRIGGERED' if guard else 'ok'}",
         "inline": True},
        {"name":"Regime Classifier",
         "value": f"**{label}** — size: {size_mult:.1f}×\n"
                  f"Favor: condors/credit spreads in MR; debit verticals in Trend; smaller/neutral in Choppy.",
         "inline": False}
    ]
    payload = {
        "embeds": [{
            "title": f"Morning Daily Brief — {now_et.strftime('%a, %b %d, %Y')} · 7:30am ET",
            "color": color,
            "fields": fields,
            "footer": {"text": "Automated brief • data via yfinance"}
        }]
    }
    requests.post(WEBHOOK, json=payload, timeout=30).raise_for_status()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Always report an error to Discord so you see failures
        try:
            msg = f"❌ Brief failed: {type(e).__name__}: {e}\n```\n{traceback.format_exc()[:1500]}\n```"
            requests.post(WEBHOOK, json={"content": msg}, timeout=15)
        except Exception:
            pass
        raise
