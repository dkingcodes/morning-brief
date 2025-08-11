# brief.py — v4 (SPX/NQ + DUP/DDP/WUP/WDP + ATR-filtered targets)
import os, math, traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import yaml

ET = ZoneInfo("America/New_York")
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
if not WEBHOOK:
    print("❌ DISCORD_WEBHOOK_URL is missing.")
    raise SystemExit(2)

# ---------- helpers ----------
def pct(a, b):
    if a is None or b is None or b == 0: return None
    return 100.0 * (a / b - 1.0)

def fmt(x, n=2):
    return "na" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{x:.{n}f}"

def calc_atr14(df):
    if df is None or df.empty or len(df) < 15: return None
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

def iv30_and_skew_proxy(prior_close):
    # Best-effort: SPY options; fallback to VIX.
    try:
        spy = yf.Ticker("SPY")
        opts = spy.options
        if not opts: raise RuntimeError("no options")
        today = datetime.now(ET).date()
        expiry = sorted(opts, key=lambda e: abs((datetime.strptime(e,"%Y-%m-%d").date()-today).days-30))[0]
        chain = spy.option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        calls["dist"] = (calls["strike"]-prior_close).abs()
        puts["dist"]  = (puts["strike"] -prior_close).abs()
        atm_iv = float(calls.sort_values("dist").iloc[0]["impliedVolatility"] + puts.sort_values("dist").iloc[0]["impliedVolatility"]) / 2.0
        k_put, k_call = prior_close*0.90, prior_close*1.10
        iv_put  = float(puts.iloc[(puts["strike"]-k_put ).abs().argsort()[:1]]["impliedVolatility"].iloc[0])
        iv_call = float(calls.iloc[(calls["strike"]-k_call).abs().argsort()[:1]]["impliedVolatility"].iloc[0])
        skew_note = "put-heavy" if (iv_put-iv_call)*100.0 > 1.0 else ("balanced" if abs(iv_put-iv_call)*100.0 <= 1.0 else "call-tilt")
        return atm_iv, skew_note
    except Exception:
        try:
            vix = float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1]) / 100.0
            return vix, "mild"
        except Exception:
            return None, "mild"

def regime_classifier(prior_close, atr_points, vix, vix3m, px_close):
    term = None if (vix is None or vix==0 or vix3m is None) else (vix3m / vix)
    atr_pct = None if (prior_close in (None,0) or atr_points is None) else (atr_points/prior_close)*100.0
    slope_pct_per_day = None
    if px_close is not None and len(px_close) >= 21:
        last20 = px_close.tail(20)
        slope_pct_per_day = ((last20.iloc[-1]-last20.iloc[0])/20.0) / px_close.iloc[-2] * 100.0
    label, size, drivers = "CHOPPY", 0.5, []
    if term is not None and term > 1.0 and (atr_pct or 0) < 1.6:
        label, size, drivers = "MEAN-REVERT", 0.8, ["term>1 (contango)","ATR% mid"]
    elif term is not None and term <= 1.0 and slope_pct_per_day is not None and abs(slope_pct_per_day) > 0.12:
        label, size, drivers = "TREND", 1.0, ["term<=1", f"slope {slope_pct_per_day:+.2f}%/day"]
    else:
        drivers = ["neutral mix"]
    return label, size, term, atr_pct, slope_pct_per_day, drivers

def filter_targets(pivot, candidates, band, direction):
    """Keep only levels within ±band of pivot and on the requested side."""
    if pivot is None or band is None: return []
    x = [v for v in (candidates or []) if v is not None]
    if direction == "above":
        keep = [v for v in x if pivot < v <= pivot + band]
        return sorted(keep)
    else:
        keep = [v for v in x if pivot - band <= v < pivot]
        return sorted(keep, reverse=True)

def extract_pivots(entry):
    """Supports new schema and legacy r/s keys."""
    if not entry: return None
    dup = entry.get("dup"); ddp = entry.get("ddp")
    wup = entry.get("wup"); wdp = entry.get("wdp")
    above = list(entry.get("above", []))
    below = list(entry.get("below", []))
    # legacy add-ins
    for k in ("r1","r2","r3"):
        if k in entry and entry[k] is not None: above.append(entry[k])
    for k in ("s1","s2","s3"):
        if k in entry and entry[k] is not None: below.append(entry[k])
    return dup, ddp, wup, wdp, above, below

# ---------- per-symbol block ----------
def build_symbol_block(name, cfg, today):
    ticker = cfg.get("ticker", "ES=F" if name=="SPX" else "NQ=F")
    day_mult  = float(cfg.get("day_atr_mult", 1.0))
    week_mult = float(cfg.get("week_atr_mult", 1.0))

    t = yf.Ticker(ticker)
    daily = t.history(period="90d", interval="1d", auto_adjust=False)
    if daily.empty: raise RuntimeError(f"No data for {name} ({ticker})")
    prev = daily.iloc[-2] if len(daily)>=2 else daily.iloc[-1]
    prior_close = float(prev["Close"]); prior_high=float(prev["High"]); prior_low=float(prev["Low"])
    prior_vwap = float((prev["High"]+prev["Low"]+prev["Close"])/3.0)
    pm = t.history(period="1d", interval="1m", prepost=True)
    pre_last = float(pm["Close"].dropna().iloc[-1]) if not pm.empty else prior_close
    o_low  = float(pm["Low"].min()) if not pm.empty else prior_low
    o_high = float(pm["High"].max()) if not pm.empty else prior_high

    atr = calc_atr14(daily)
    week = t.history(period="400d", interval="1wk", auto_adjust=False)
    watr = calc_atr14(week)

    # pivots from YAML
    d_key = today.strftime("%Y-%m-%d")
    w_key = f"{today.isocalendar().year}-W{int(today.isocalendar().week):02d}"
    d_ent = (cfg.get("daily") or {}).get(d_key, {})
    w_ent = (cfg.get("weekly") or {}).get(w_key, {})

    dup, ddp, _, _, d_above_raw, d_below_raw = extract_pivots(d_ent)
    _,   _, wup, wdp, w_above_raw, w_below_raw = extract_pivots(w_ent)

    # filter to ATR bands
    d_above = filter_targets(dup, d_above_raw, (atr or 0)*day_mult, "above") if dup is not None else []
    d_below = filter_targets(ddp, d_below_raw, (atr or 0)*day_mult, "below") if ddp is not None else []
    w_above = filter_targets(wup, w_above_raw, (watr or 0)*week_mult, "above") if wup is not None else []
    w_below = filter_targets(wdp, w_below_raw, (watr or 0)*week_mult, "below") if wdp is not None else []

    # format block
    hdr = f"**{name}** ({ticker}) pre-mkt"
    pre = f"{fmt(pre_last)} ({(pct(pre_last, prior_close) or 0):+.2f}%) · O/N {fmt(o_low)}–{fmt(o_high)}"
    keys = f"H/L {fmt(prior_high)}/{fmt(prior_low)} · Close {fmt(prior_close)} · VWAP~{fmt(prior_vwap)}"
    atrs = f"ATR14≈{fmt(atr)}  |  WATR14≈{fmt(watr)}"
    d_line = (f"DUP {fmt(dup)} → ▲ {', '.join(fmt(x) for x in d_above) if d_above else '—'}\n"
              f"DDP {fmt(ddp)} → ▼ {', '.join(fmt(x) for x in d_below) if d_below else '—'}")
    w_line = (f"WUP {fmt(wup)} → ▲ {', '.join(fmt(x) for x in w_above) if w_above else '—'}\n"
              f"WDP {fmt(wdp)} → ▼ {', '.join(fmt(x) for x in w_below) if w_below else '—'}")

    return [
        {"name": hdr, "value": pre, "inline": False},
        {"name": f"{name} key levels",
         "value": keys + "\n" + atrs + "\n" + d_line + "\n" + w_line,
         "inline": False},
    ], prior_close

# ---------- main ----------
def main():
    today = datetime.now(ET)

    with open("levels.yml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    symbols_cfg = (cfg.get("symbols") or {})

    # Global regime snapshot off ES (SPX proxy)
    es_tkr = symbols_cfg.get("SPX", {}).get("ticker", "ES=F")
    es = yf.Ticker(es_tkr)
    es_daily = es.history(period="90d", interval="1d", auto_adjust=False)
    es_prev_close = float(es_daily.iloc[-2]["Close"]) if len(es_daily)>=2 else float(es_daily.iloc[-1]["Close"])
    atr_points = calc_atr14(es_daily)
    vix  = float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1]) if not yf.Ticker("^VIX").history(period="2d").empty else None
    vix3 = float(yf.Ticker("^VIX3M").history(period="2d")["Close"].iloc[-1]) if not yf.Ticker("^VIX3M").history(period="2d").empty else None
    iv30, skew_note = iv30_and_skew_proxy(es_prev_close)

    label, size_mult, term, atr_pct, slope_pct_day, drivers = regime_classifier(
        es_prev_close, atr_points, vix, vix3, es_daily["Close"] if not es_daily.empty else None
    )
    # Guardrails
    dxyh = yf.Ticker("DX-Y.NYB").history(period="3d", interval="1d")
    dxy = None if dxyh.empty else float(dxyh["Close"].iloc[-1])
    dxy_prev = None if len(dxyh)<2 else float(dxyh["Close"].iloc[-2])
    dxy_chg_pct = None if (dxy is None or dxy_prev in (None,0)) else (dxy-dxy_prev)/dxy_prev*100.0
    guard = (vix is not None and vix > 20.0) or (dxy_chg_pct is not None and dxy_chg_pct > 0.5)
    base_label = label
    if guard: label = label + " (FLAT*)"

    # Build symbol fields
    fields = []
    for sym in ("SPX","NQ"):
        if sym in symbols_cfg:
            blk, _ = build_symbol_block(sym, symbols_cfg[sym], today)
            fields.extend(blk)

    # Insert global vol/regime
    term_ratio = None if (vix is None or vix==0 or vix3 is None) else vix3/vix
    fields.insert(2, {
        "name":"Vol snapshot",
        "value": f"VIX {fmt(vix,1)} · 3M/1M {fmt(term_ratio,2)} · IV30~{fmt(iv30,3)} · skew {skew_note}",
        "inline": False
    })

    # Regime + Playbook
    if "MEAN-REVERT" in base_label:
        play = [
            "Primary: 0–2 DTE **iron condors / credit spreads** at ±1.0× ATR bands",
            "Timing: enter after **10:00 ET** on **second probe**; avoid first impulse",
            "Size: **0.8×** baseline (1.0× if breadth >60% & term>1)",
            "Avoid: breakout chases unless breadth expands & rates confirm"
        ]
    elif "TREND" in base_label:
        play = [
            "Primary: **directional debit verticals / breakouts** with follow-through",
            "Timing: use pullbacks to 0.5–0.7× ATR; cut on regime slip",
            "Size: **1.0×** baseline (reduce if guardrails trigger)",
            "Avoid: range-selling unless term>1 and momentum wanes"
        ]
    else:
        play = [
            "Primary: **small / delta-neutral**; scalps toward VWAP only",
            "Size: **0.5×** baseline",
            "Avoid: strong directional bets; wait for clarity"
        ]

    fields.append({
        "name":"Regime Classifier",
        "value": f"**{label}** — base={base_label} • size: {size_mult:.1f}×\n"
                 f"Drivers: {', '.join(drivers)}\n" + "\n".join("• "+s for s in play),
        "inline": False
    })

    payload = {
        "embeds": [{
            "title": f"Morning Daily Brief — {today.strftime('%a, %b %d, %Y')} · 7:30am ET",
            "color": 0x8C52FF,
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
