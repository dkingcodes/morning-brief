# brief.py — v6
# Morning Daily Brief to Discord (7:30am ET)
# - Daily/Weekly bands use options-implied Expected Move (EM) instead of ADR/ATR.
# - EM(SPX) from VIX; EM(NQ) from VXN. EM(days) = Price * IV_annual * sqrt(days/252).
# - Daily targets filtered around DUP/DDP with ±(EM_1D * day_atr_mult).
# - Weekly targets filtered around WUP/WDP with ±(EM_5D * week_atr_mult).
# - Keeps detailed Regime + Playbook cues, Macro block, SPX/NQ sections.

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

def fnum(x):
    """Robust float coercion for YAML: handles '17,650', ' na ', None."""
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x == "" or x.lower() in ("na", "none", "null"):
                return None
        return float(x)
    except Exception:
        return None

def fnum_list(xs):
    out = []
    for v in (xs or []):
        fv = fnum(v)
        if fv is not None:
            out.append(fv)
    return out

def calc_atr14(df):
    """Still compute full-session ATR14 for reference text (not used for bands)."""
    if df is None or df.empty or len(df) < 15: return None
    h,l,c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])

def iv30_and_skew_proxy(prior_close):
    """For the global snapshot (SPX-ish), try SPY options; fallback to VIX."""
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

def expected_move(price, iv_annual, days):
    """EM in points for 'days' trading days."""
    p = fnum(price); iv = fnum(iv_annual)
    if p is None or iv is None: return None
    return p * iv * math.sqrt(days / 252.0)

def iv_proxy_for_symbol(name):
    """
    IV proxy (annualized) per symbol for EM:
      SPX -> VIX
      NQ  -> VXN (fallback to VIX if N/A)
    Returns float in decimals (e.g., 0.18) or None.
    """
    try:
        if name == "SPX":
            h = yf.Ticker("^VIX").history(period="2d", interval="1d")
            return None if h.empty else float(h["Close"].iloc[-1]) / 100.0
        elif name == "NQ":
            hvxn = yf.Ticker("^VXN").history(period="2d", interval="1d")
            if not hvxn.empty:
                return float(hvxn["Close"].iloc[-1]) / 100.0
            # fallback to VIX if VXN missing
            hvix = yf.Ticker("^VIX").history(period="2d", interval="1d")
            return None if hvix.empty else float(hvix["Close"].iloc[-1]) / 100.0
        else:
            return None
    except Exception:
        return None

def filter_targets(pivot, candidates, band, direction):
    """Keep only levels within ±band of pivot and on the requested side."""
    p = fnum(pivot)
    b = fnum(band)
    if p is None or b is None or b <= 0:
        return []
    x = fnum_list(candidates)
    if direction == "above":
        keep = [v for v in x if p < v <= p + b]
        return sorted(set(keep))
    else:
        keep = [v for v in x if p - b <= v < p]
        return sorted(set(keep), reverse=True)

def extract_pivots(entry):
    """Supports DUP/DDP/WUP/WDP + legacy r/s keys; coerces to floats."""
    if not entry:
        return None, None, None, None, [], []
    dup = fnum(entry.get("dup"))
    ddp = fnum(entry.get("ddp"))
    wup = fnum(entry.get("wup"))
    wdp = fnum(entry.get("wdp"))
    above = fnum_list(entry.get("above", []))
    below = fnum_list(entry.get("below", []))
    for k in ("r1","r2","r3"):
        fv = fnum(entry.get(k))
        if fv is not None: above.append(fv)
    for k in ("s1","s2","s3"):
        fv = fnum(entry.get(k))
        if fv is not None: below.append(fv)
    return dup, ddp, wup, wdp, above, below

# ---------- per-symbol block ----------
def build_symbol_block(name, cfg, today):
    ticker = cfg.get("ticker", "ES=F" if name=="SPX" else "NQ=F")
    day_mult  = float(cfg.get("day_atr_mult", 1.0))   # now EM multiplier
    week_mult = float(cfg.get("week_atr_mult", 1.0))  # now EM multiplier

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

    # reference ATRs just for display sanity (not used for bands)
    atr = calc_atr14(daily)
    week = t.history(period="400d", interval="1wk", auto_adjust=False)
    watr = calc_atr14(week)

    # IV proxy -> EM
    iv_annual = iv_proxy_for_symbol(name)  # 0.XX
    em_1d = expected_move(prior_close, iv_annual, 1)
    em_5d = expected_move(prior_close, iv_annual, 5)

    # pivots from YAML
    d_key = today.strftime("%Y-%m-%d")
    w_key = f"{today.isocalendar().year}-W{int(today.isocalendar().week):02d}"
    d_ent = (cfg.get("daily") or {}).get(d_key, {})
    w_ent = (cfg.get("weekly") or {}).get(w_key, {})

    dup, ddp, _, _, d_above_raw, d_below_raw = extract_pivots(d_ent)
    _,   _, wup, wdp, w_above_raw, w_below_raw = extract_pivots(w_ent)

    # filter to EM bands
    d_band = (em_1d or 0) * day_mult
    w_band = (em_5d or 0) * week_mult
    d_above = filter_targets(dup, d_above_raw, d_band, "above") if dup is not None else []
    d_below = filter_targets(ddp, d_below_raw, d_band, "below") if ddp is not None else []
    w_above = filter_targets(wup, w_above_raw, w_band, "above") if wup is not None else []
    w_below = filter_targets(wdp, w_below_raw, w_band, "below") if wdp is not None else []

    # ±1× EM daily band for the playbook cue
    plus1 = prior_close + (em_1d or 0) if prior_close is not None else None
    minus1 = prior_close - (em_1d or 0) if prior_close is not None else None

    # format block
    hdr = f"**{name}** ({ticker}) pre-mkt"
    pre = f"{fmt(pre_last)} ({(pct(pre_last, prior_close) or 0):+.2f}%) · O/N {fmt(o_low)}–{fmt(o_high)}"
    keys = f"H/L {fmt(prior_high)}/{fmt(prior_low)} · Close {fmt(prior_close)} · VWAP~{fmt(prior_vwap)}"
    ranges = f"EM(1D)≈{fmt(em_1d)}  |  EM(5D)≈{fmt(em_5d)}  |  ATR14≈{fmt(atr)}  |  WATR14≈{fmt(watr)}"
    d_line = (f"DUP {fmt(dup)} → ▲ {', '.join(fmt(x) for x in d_above) if d_above else '—'}\n"
              f"DDP {fmt(ddp)} → ▼ {', '.join(fmt(x) for x in d_below) if d_below else '—'}")
    w_line = (f"WUP {fmt(wup)} → ▲ {', '.join(fmt(x) for x in w_above) if w_above else '—'}\n"
              f"WDP {fmt(wdp)} → ▼ {', '.join(fmt(x) for x in w_below) if w_below else '—'}")

    fields = [
        {"name": hdr, "value": pre, "inline": False},
        {"name": f"{name} key levels",
         "value": keys + "\n" + ranges + "\n" + d_line + "\n" + w_line,
         "inline": False},
    ]
    # meta for playbook cues
    meta = {"prior_close": prior_close, "em1": em_1d, "em5": em_5d, "bands": (plus1, minus1)}
    return fields, meta

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

    vixh = yf.Ticker("^VIX").history(period="2d", interval="1d")
    vix3h = yf.Ticker("^VIX3M").history(period="2d", interval="1d")
    vix  = None if vixh.empty else float(vixh["Close"].iloc[-1])
    vix3 = None if vix3h.empty else float(vix3h["Close"].iloc[-1])
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

    # Build symbol fields + meta for cues
    fields = []
    spx_meta, nq_meta = None, None
    if "SPX" in symbols_cfg:
        spx_blk, spx_meta = build_symbol_block("SPX", symbols_cfg["SPX"], today)
        fields.extend(spx_blk)
    if "NQ" in symbols_cfg:
        nq_blk, nq_meta = build_symbol_block("NQ", symbols_cfg["NQ"], today)
        fields.extend(nq_blk)

    # Vol snapshot (show VIX & VXN plus term and IV proxy)
    vxn_h = yf.Ticker("^VXN").history(period="2d", interval="1d")
    vxn = None if vxn_h.empty else float(vxn_h["Close"].iloc[-1])
    term_ratio = None if (vix is None or vix==0 or vix3 is None) else vix3/vix
    fields.insert(2, {
        "name":"Vol snapshot",
        "value": f"VIX {fmt(vix,1)} · VXN {fmt(vxn,1)} · 3M/1M(SPX) {fmt(term_ratio,2)} · IV30~{fmt(iv30,3)} · skew {skew_note}",
        "inline": False
    })

    # --- Macro Calendar (static for now) ---
    macro_lines = [
        "No major U.S. economic releases today.",
        "CPI report scheduled Tuesday at 8:30 am ET"
    ]
    fields.append({
        "name": "Macro Calendar",
        "value": "• " + "\n• ".join(macro_lines),
        "inline": False
    })

    # --- Playbook Cue – SPX (options) ---
    if spx_meta:
        up1, dn1 = spx_meta["bands"]
        fields.append({
            "name": "Playbook Cue – SPX",
            "value": (
                "Strategy: 0–2 DTE **iron condors / credit spreads** anchored at edges of intraday range\n"
                f"Entry: fade moves near **±1× EM** bands (**~{fmt(up1)} / ~{fmt(dn1)}**) after initial 10–30 min\n"
                f"Size Multiplier: **{size_mult:.1f}×** (baseline 0.8× in MR)\n"
                "Tactical Notes: Avoid breakout chasing unless breadth >60% or term flips to backwardation. "
                "Favor fades into early reversals / reversion-to-mean."
            ),
            "inline": False
        })

    # --- Playbook Cue – NQ (futures) ---
    if nq_meta:
        nqu, nqd = nq_meta["bands"]
        bias_txt = ("Bias: *fade extremes* in MR; "
                    "in TREND, look to join direction on pullbacks toward VWAP/0.5–0.7× EM.")
        fields.append({
            "name": "Playbook Cue – NQ (Futures)",
            "value": (
                "Strategy: **intraday long/short** using ±1× EM as action bands\n"
                f"Entry: fade toward mean near **~{fmt(nqu)} / ~{fmt(nqd)}** in MR; "
                "or join trend on controlled pullbacks\n"
                f"Risk/Target: initial stop ~0.3× EM; first target ~0.5× EM\n"
                f"Size Multiplier: **{size_mult:.1f}×**\n" + bias_txt
            ),
            "inline": False
        })

    # Regime + Playbook summary (global)
    if "MEAN-REVERT" in base_label:
        play = [
            "Primary: ranges & fades (use EM bands)",
            "Timing: after 10:00 ET; second probe preferred",
            "Avoid: breakout chases unless breadth expands"
        ]
    elif "TREND" in base_label:
        play = [
            "Primary: directional follow-through; pullback entries",
            "Cut on regime slip or guardrail triggers",
            "Avoid: selling ranges unless momentum wanes"
        ]
    else:
        play = ["Primary: small / neutral", "Scalps to VWAP", "Avoid: big directional bets"]

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
