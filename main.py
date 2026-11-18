#!/usr/bin/env python3
"""Quantile forecaster for polymarket-nvidia-ceo-forecast-01.

Entry point: main.py (stdlib only). Reads /input/<id>.in, writes /output/<id>.out.
Model (per market, per horizon h=1..240):
  median(h) = x + kappa * (mean_win - x) * (h / 240)          [mean reversion]
              + lam * (x / S - x)                             [sum normalization]
  q_tau(h)   = median(h) + z_tau * sigma * h**alpha * tail_tau [volatility-scaled]
  where sigma = stdev of 1-min changes over the last `win` minutes (floored),
  S = sum of all markets' last prices, z_tau = normal quantiles.
Params dict keys: win, sig_floor, alpha, sig_scale, lam, kappa, tail.
Defaults are the tuned values (see tune.py); override via env POLY_* for tuning.

Tuned on the kit's 6 practice cases (p00..p05): pinball loss 0.265019 vs the
house baseline's 0.363589 (-27%), beating the baseline on every case.
Tuned values: win=120, sig_floor=0.0005, alpha=0.35, sig_scale=0.7,
lam=0.6, kappa=0 (mean reversion did not help on practice), tail=1.5,
shrink=0.25 toward the cross-market median sigma.
"""
import json, math, os

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

DEFAULTS = {
    "win": 120,          # minutes of history for sigma + mean
    "sig_floor": 0.0005, # min per-minute sigma
    "alpha": 0.35,       # horizon exponent (random walk = 0.5)
    "sig_scale": 0.7,    # multiplier on estimated sigma
    "lam": 0.6,          # pull toward x/S (sum normalization)
    "kappa": 0.0,        # pull toward recent mean (mean reversion)
    "tail": 1.5,         # extra width on the 5%/95% quantiles
    "shrink": 0.25,      # shrink per-market sigma toward the global median sigma
    "robust": 0.0,       # 1 = use MAD sigma instead of stdev
}

def _stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

def _mad_sigma(xs):
    """Robust sigma: 1.4826 * median(|x - median(x)|)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    med = s[len(s) // 2]
    dev = sorted(abs(x - med) for x in xs)
    return 1.4826 * dev[len(dev) // 2]

def _clamp6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0

def market_stats(prices, win, robust=0.0):
    """Return (sigma, mean) over the last `win` minutes of history."""
    n = len(prices)
    w = min(win, n - 1)
    if w < 10:
        return None, None
    seg = prices[n - 1 - w:]
    rets = [seg[k + 1] - seg[k] for k in range(len(seg) - 1)]
    sig = _mad_sigma(rets) if robust else _stdev(rets)
    return sig, sum(seg) / len(seg)

def forecast_case(case, params):
    p = dict(DEFAULTS)
    p.update(params)
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    # The 24 Yes prices are one mutually-exclusive outcome each, so their sum
    # should be near 1; on practice it runs 0.81-1.09. Pulling each median part
    # way (lam) toward x/S removes the common over-round instead of trusting
    # the raw last price. Likewise each market's sigma is shrunk toward the
    # cross-market median: a 120-minute window is short, and a single jump
    # would otherwise set one market's whole spread.
    s_sum = sum(m["prices"][-1] for m in markets if m["prices"])
    # global fallback sigma: median of per-market sigmas
    sigmas = []
    stats = {}
    for m in markets:
        s, mean = market_stats(m["prices"], p["win"], p["robust"])
        stats[m["id"]] = (s, mean)
        if s is not None:
            sigmas.append(s)
    sigmas.sort()
    global_sig = sigmas[len(sigmas) // 2] if sigmas else 0.0025
    global_sig = max(global_sig, p["sig_floor"])

    out = []
    for m in markets:
        prices = m["prices"]
        if not prices:
            raise ValueError(f"market {m['id']} has no prices")
        x = prices[-1]
        s, mean = stats[m["id"]]
        raw = s if s is not None else global_sig
        raw = (1 - p["shrink"]) * raw + p["shrink"] * global_sig
        sigma = max(raw, p["sig_floor"]) * p["sig_scale"]
        med_base = x + p["lam"] * (x / s_sum - x) if s_sum > 0 else x
        rev = (mean - x) if mean is not None else 0.0
        rows = []
        for h in range(1, horizon + 1):
            med = med_base + p["kappa"] * rev * (h / horizon)
            spread = sigma * (h ** p["alpha"])
            row = []
            for j, z in enumerate(Z):
                w = p["tail"] if j in (0, 4) else 1.0
                row.append(_clamp6(med + z * spread * w))
            rows.append(row)
        out.append({"id": m["id"], "q": rows})
    return {"forecasts": out}

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    params = {}
    for k in DEFAULTS:
        v = os.environ.get("POLY_" + k.upper())
        if v is not None:
            params[k] = float(v) if k != "win" else int(float(v))
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(in_dir)):
        if not fn.endswith(".in"):
            continue
        with open(os.path.join(in_dir, fn)) as f:
            case = json.load(f)
        fc = forecast_case(case, params)
        with open(os.path.join(out_dir, fn[:-3] + ".out"), "w") as f:
            f.write(json.dumps(fc) + "\n")

if __name__ == "__main__":
    main()
