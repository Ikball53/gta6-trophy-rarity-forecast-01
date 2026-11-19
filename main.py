#!/usr/bin/env python3
"""Forecast GTA VI trophy earn-rates at a future checkpoint.

Model: decompose each trophy's earn rate into earners / players_tracked,
project both series to the target time, then recombine. Players grow
(even when the read history looks frozen, because the tracker refreshes
in lagged jumps); earners grow sublinearly with players, with a
trophy-specific elasticity, plus a slow progression term as existing
players keep playing. Quantiles are built on the logit scale with a
spread that widens with horizon and with small earner counts.

Reads every *.in JSON case from $MAKEFUN_INPUT_DIR (default /input/)
and writes <case>.out JSON files to $MAKEFUN_OUTPUT_DIR (default
/output/). Standard library only, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

EPS_P = 1e-6          # clamp for probabilities before logit
Q_MIN = 0.0001        # output clamp, percent
Q_MAX = 99.9999       # output clamp, percent
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()

# Fallback median earn-rate (percent) when a trophy has no usable reads.
TIER_RATE_PRIOR = {"platinum": 0.02, "gold": 0.1, "silver": 0.4, "bronze": 1.2}
# Expected earner fraction among tracked players when last read shows 0.
TIER_ZERO_FRAC = {"platinum": 2e-5, "gold": 1e-4, "silver": 3e-4, "bronze": 6e-4}
TIER_BETA_ADJ = {"platinum": -0.08, "gold": -0.03, "silver": 0.0, "bronze": 0.03}
TIER_PROG = {"bronze": 1.0, "silver": 0.8, "gold": 0.5, "platinum": 0.25}


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def parse_ts(text):
    """Parse an ISO timestamp into epoch seconds (float). None on failure."""
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def logit(p):
    p = clamp(p, EPS_P, 1.0 - EPS_P)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def ols(xs, ys):
    """Least-squares (intercept, slope); None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (my - slope * mx, slope)


def change_points(times, values):
    """Collapse consecutive repeats, keeping the first time of each value."""
    out = []
    for t, v in zip(times, values):
        if not out or v != out[-1][1]:
            out.append((t, v))
    return out


def forecast_players(times_h, players, t_target_h, horizon_h):
    """Project players_tracked to the target. Returns (p_future, frozen)."""
    p_last = float(players[-1])
    pts = change_points(times_h, [float(p) for p in players])
    span = times_h[-1] - times_h[0] if len(times_h) > 1 else 0.0
    lo_p = min(players)
    hi_p = max(players)
    frozen = len(pts) <= 1 or (lo_p > 0 and hi_p / lo_p < 1.02)

    if frozen or span < 0.5 or len(pts) < 3:
        # Dilution prior: tracked players keep growing even when reads
        # look frozen (tracker refreshes lag, then jump). Fast early
        # catch-up plus a slow tail; smaller bases grow faster.
        log_g = 1.75 * (1.0 - math.exp(-horizon_h / 1.2)) \
            + 0.55 * math.log1p(horizon_h / 24.0)
        size_factor = (638.0 / max(p_last, 1.0)) ** 0.15
        log_g *= clamp(size_factor, 0.4, 1.3)
        g = math.exp(log_g)
        if len(pts) >= 2 and pts[0][1] > 0:
            observed = pts[-1][1] / pts[0][1]
            g = max(g, min(observed, 3.0))
        return p_last * max(g, 1.0), True

    t_last = times_h[-1]
    # Fit A: power law, log P against log(1 + t).
    xs = [math.log1p(t) for t, _ in pts]
    ys = [math.log(max(p, 1.0)) for _, p in pts]
    fit_a = ols(xs, ys)
    log_a = math.log(max(p_last, 1.0))
    if fit_a is not None:
        a, b = fit_a
        b = clamp(b, 0.0, 1.3)
        log_a = max(a + b * math.log1p(max(t_target_h, 0.0)),
                    math.log(max(p_last, 1.0)))

    # Fit B: recent exponential growth with decaying rate.
    window = max(6.0, 0.25 * span)
    recent = [(t, p) for t, p in pts if t >= t_last - window]
    if len(recent) < 2:
        recent = pts[-2:]
    fit_b = ols([t for t, _ in recent],
                [math.log(max(p, 1.0)) for _, p in recent])
    log_b = math.log(max(p_last, 1.0))
    if fit_b is not None:
        g_rate = clamp(fit_b[1], 0.0, 0.5)
        tau = 30.0
        log_b = math.log(max(p_last, 1.0)) \
            + g_rate * tau * (1.0 - math.exp(-horizon_h / tau))

    # With a long history the launch surge bends the power law, so lean
    # on the recent-growth fit; with a short one, split evenly.
    w_recent = 0.65 if span >= 24.0 else 0.5
    log_f = (1.0 - w_recent) * log_a + w_recent * log_b
    p_future = math.exp(log_f)
    g_max = 4.0 + horizon_h / 12.0
    p_future = clamp(p_future, p_last, p_last * g_max)
    return p_future, False


def beta_prior(p_frac, tier):
    """Prior elasticity of earners w.r.t. players, from rate level + tier."""
    p = clamp(p_frac, 0.0, 1.0)
    b = 0.10 + 0.88 * (p ** 0.35) + TIER_BETA_ADJ.get(tier, 0.0)
    return clamp(b, 0.02, 0.99)


def fit_earner_elasticities(earners, players, times_h, beta0, c0):
    """Fit log earners on log players (b) and log(1 + t) (c), FWL-style.

    Earners grow because more tracked players arrive and because the
    players already tracked keep playing. The two regressors move
    together, so the time slope is identified only from the part of
    log(1 + t) that log players cannot explain (Frisch-Waugh-Lovell),
    then shrunk toward its prior c0 by how much independent variation
    exists; b is adjusted by the FWL identity and lightly shrunk
    toward beta0. Returns (b, c), or None when there is too little
    variation to fit.
    """
    pts = []
    last_p = None
    for e, p, t in zip(earners, players, times_h):
        if e is None or p is None or e <= 0 or p <= 0:
            continue
        row = (math.log(p), math.log1p(max(t, 0.0)), math.log(e))
        if last_p is not None and p == last_p:
            pts[-1] = row
        else:
            pts.append(row)
            last_p = p
    if len(pts) < 8:
        return None
    x1 = [r[0] for r in pts]
    x2 = [r[1] for r in pts]
    ys = [r[2] for r in pts]
    if max(x1) - min(x1) < 0.3:
        return None
    fy = ols(x1, ys)
    fx = ols(x1, x2)
    if fy is None or fx is None:
        return None
    resid_y = [y - (fy[0] + fy[1] * a) for a, y in zip(x1, ys)]
    resid_x = [b - (fx[0] + fx[1] * a) for a, b in zip(x1, x2)]
    var_rx = sum(r * r for r in resid_x)
    if var_rx < 1e-6:
        c = c0
    else:
        c_raw = sum(a * b for a, b in zip(resid_y, resid_x)) / var_rx
        rel = var_rx / (var_rx + 2.0)
        c = rel * clamp(c_raw, 0.0, 0.6) + (1.0 - rel) * c0
    c = clamp(c, 0.0, 0.35)
    b = fy[1] - c * fx[1]
    b = 0.8 * clamp(b, 0.0, 1.1) + 0.2 * beta0
    return clamp(b, 0.0, 1.1), c


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS

    # --- timeline -----------------------------------------------------
    read_times = []
    players_series = []
    for h in history:
        ts = parse_ts(h.get("at")) or parse_ts(h.get("slot"))
        p = h.get("players_tracked")
        if ts is None or p is None:
            continue
        read_times.append(ts)
        players_series.append(float(p))

    at_ts = parse_ts(case.get("at"))
    target_ts = parse_ts(case.get("target"))
    if target_ts is None:
        base = at_ts if at_ts is not None else (read_times[-1] if read_times else 0.0)
        target_ts = base + float(case.get("horizon_minutes") or 0) * 60.0

    if read_times:
        t0 = read_times[0]
        times_h = [(t - t0) / 3600.0 for t in read_times]
        t_target_h = (target_ts - t0) / 3600.0
        horizon_h = max((target_ts - read_times[-1]) / 3600.0, 0.05)
        p_future, frozen = forecast_players(
            times_h, players_series, t_target_h, horizon_h)
        p_last = players_series[-1]
        g_players = p_future / p_last if p_last > 0 else 1.0
        distinct_p = len(change_points(times_h, players_series))
    else:
        horizon_h = max(float(case.get("horizon_minutes") or 0) / 60.0, 0.05)
        p_future, frozen, p_last, g_players, distinct_p = 0.0, True, 0.0, 1.0, 0

    rows = []
    for idx, trophy in enumerate(trophies):
        tier = str(trophy.get("tier") or "").lower()

        # Per-trophy earner series and latest usable rate.
        earners = []
        last_rate = None
        last_earners = None
        for h, p in zip(history, players_series) if read_times else []:
            rates = h.get("rates") or []
            r = rates[idx] if idx < len(rates) else None
            if r is None:
                earners.append(None)
            else:
                e = float(r) / 100.0 * p
                earners.append(e)
                last_rate = float(r)
                last_earners = e

        if last_rate is None or p_future <= 0:
            # No usable reads: tier prior rate, wide spread.
            p_frac = TIER_RATE_PRIOR.get(tier, 0.5) / 100.0
            sigma = clamp(0.35 + 0.05 * math.sqrt(horizon_h) + 0.5, 0.25, 1.4)
        else:
            p_last_frac = clamp(last_rate / 100.0, 0.0, 1.0)
            prior = beta_prior(p_last_frac, tier)
            gamma0 = 0.22 * TIER_PROG.get(tier, 0.7) \
                * math.sqrt(max(1.0 - p_last_frac, 0.0))
            fit = fit_earner_elasticities(
                earners, players_series, times_h, prior, 0.8 * gamma0)

            if last_earners is None or last_earners <= 0:
                e_future = 0.25 + p_future * TIER_ZERO_FRAC.get(tier, 3e-4)
            elif fit is not None:
                b_slope, c_slope = fit
                d_log_t = math.log1p(max(t_target_h, 0.0)) \
                    - math.log1p(max(times_h[-1], 0.0))
                e_future = last_earners * math.exp(
                    b_slope * math.log(max(g_players, 1e-9))
                    + c_slope * max(d_log_t, 0.0))
            else:
                prog = 1.0 + gamma0 * math.log1p(horizon_h / 24.0)
                e_future = last_earners * (g_players ** prior) * prog
            cap_e = 0.999 * p_future
            e_future = min(e_future, cap_e)
            if last_earners is not None:
                e_future = max(e_future, min(last_earners, cap_e))
            p_frac = e_future / p_future if p_future > 0 else p_last_frac

            sigma = 0.35 + 0.05 * math.sqrt(horizon_h)
            if frozen:
                sigma += 0.25
            elif distinct_p < 3:
                sigma += 0.10
            if last_earners is not None and last_earners < 5:
                sigma += 0.30 * (1.0 - max(last_earners, 0.0) / 5.0)
            sigma = clamp(sigma, 0.25, 1.4)

        mu = logit(p_frac)
        quantiles = []
        for lev in levels:
            z = _NORM.inv_cdf(clamp(float(lev), 1e-9, 1.0 - 1e-9))
            q = sigmoid(mu + z * sigma) * 100.0
            quantiles.append(round(clamp(q, Q_MIN, Q_MAX), 4))
        # Enforce non-decreasing order (levels are ascending per spec).
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": trophy.get("id"), "q": quantiles})

    return {"trophies": rows}


def persistence_fallback(case):
    """Last-resort forecast: last rates held flat with a modest spread."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    last_rates = (history[-1].get("rates") if history else []) or []
    rows = []
    for idx, trophy in enumerate(trophies):
        r = last_rates[idx] if idx < len(last_rates) else None
        p_frac = (float(r) / 100.0) if r is not None else 0.001
        mu = logit(p_frac)
        qs = []
        for lev in levels:
            z = _NORM.inv_cdf(clamp(float(lev), 1e-9, 1.0 - 1e-9))
            qs.append(round(clamp(sigmoid(mu + z * 0.6) * 100.0, Q_MIN, Q_MAX), 4))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy.get("id"), "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        return
    for name in names:
        stem = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            result = forecast_case(case)
        except Exception as exc:  # never lose a whole case to a crash
            print(f"fallback for {name}: {exc}", file=sys.stderr)
            result = persistence_fallback(case)
        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
