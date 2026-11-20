#!/usr/bin/env python3
"""Week-one trophy earn-rate forecaster for Grand Theft Auto VI (PS5).

Reads make.fun case files (schema makefun/gta6-trophies-case/v1) from
$MAKEFUN_INPUT_DIR (default /input) and writes one forecast per case to
$MAKEFUN_OUTPUT_DIR (default /output).  For every trophy it emits five
quantiles of the earn rate (percent of tracked players) at the case target.

Model sketch
------------
Each trophy's logit earn rate is tracked over time (hours since release).
Three forces move it during week one:

* progression  -- the launch cohort plays deeper into the game, so story
  and side-content trophies climb from their launch-night trickle;
* dilution     -- the tracked-player denominator grows fast and late
  arrivals are more casual, dragging hard-trophy rates down;
* page staleness -- Exophase refreshes in jumps, so a run of identical
  reads hides a pending re-basing of every rate.

The median forecast blends a weighted logit-vs-log-time trend fitted on
recent distinct reads with a class-prior drift (story / collectible /
feat / economy / completion ...).  The spread combines a random-walk
term over the horizon, fit residuals, binomial count noise from the
small early player pool, a staleness bonus and a per-class uncertainty.
Everything is deterministic: no randomness anywhere.
"""

import glob
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
EPS_PCT = 0.01          # scorer clamps rates to [0.01, 99.99] for logits
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL = statistics.NormalDist()


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(pct):
    """Percent -> logit, with the scorer's clamping."""
    p = clamp(pct, EPS_PCT, 100.0 - EPS_PCT) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """ISO-8601 UTC string -> aware datetime (None when unusable)."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_since_release(moment):
    if moment is None:
        return 0.0
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def log_clock(hours):
    """Log-time axis: fast early movement, slow later movement."""
    return math.log1p(max(0.0, hours))


# --------------------------------------------------------------------------
# trophy behaviour classes and their priors
# --------------------------------------------------------------------------

# drift: expected logit change per unit of log-time when no trend is fit.
# dilute: share of log player-growth that drags the rate down.
# spread: extra logit uncertainty for the class.
CLASS_PRIORS = {
    "opening":    {"drift": 0.00, "dilute": 0.05, "spread": 0.30},
    "story_early": {"drift": 0.55, "dilute": 0.25, "spread": 0.50},
    "story_mid":  {"drift": 0.45, "dilute": 0.35, "spread": 0.55},
    "story_end":  {"drift": 0.35, "dilute": 0.45, "spread": 0.60},
    "side_strand": {"drift": 0.30, "dilute": 0.45, "spread": 0.60},
    "collectible": {"drift": 0.25, "dilute": 0.55, "spread": 0.65},
    "grind":      {"drift": 0.20, "dilute": 0.55, "spread": 0.65},
    "feat":       {"drift": 0.30, "dilute": 0.40, "spread": 0.60},
    "economy":    {"drift": 0.30, "dilute": 0.45, "spread": 0.60},
    "completion": {"drift": 0.15, "dilute": 0.60, "spread": 0.70},
    "platinum":   {"drift": 0.10, "dilute": 0.60, "spread": 0.70},
    "misc":       {"drift": 0.30, "dilute": 0.45, "spread": 0.60},
}

TIER_FALLBACK_PCT = {"platinum": 0.10, "gold": 0.40, "silver": 0.60, "bronze": 1.20}


def classify(trophy):
    """Map a trophy to a behaviour class from its tier and text."""
    tier = str(trophy.get("tier", "")).lower()
    name = str(trophy.get("name", "")).lower()
    desc = str(trophy.get("description", "")).lower()
    text = name + " | " + desc

    if tier == "platinum":
        return "platinum"
    if "100% completion" in text:
        return "completion"
    if "gone fishing" in text or "in the beginning" in text:
        return "opening"
    if "final mission" in text and " for " not in desc:
        return "story_end"
    if "final mission for" in text or "strangers" in text or "mission strands" in text:
        return "side_strand"
    if "chapter 1" in text or "chapter 2" in text:
        return "story_early"
    if "chapter" in text:
        return "story_mid"
    collect_words = ("collectible", "discover 100", "explore all", "purchase all",
                     "every location", "every spot", "100 locations", "sets")
    if any(w in text for w in collect_words):
        return "collectible"
    grind_words = ("250 stunts", "gold medal in all", "50 challenges",
                   "win 20", "try each type", "register 10")
    if any(w in text for w in grind_words):
        return "grind"
    if "$" in desc or "spend" in desc or "earn $" in desc or "laundering" in text:
        return "economy"
    feat_words = ("rob ", "escape", "win ", "beat ", "complete an", "take down",
                  "lose a", "hunt ", "meditate", "bowl", "smash", "enjoy a",
                  "follow-through", "make an unscheduled", "choices made")
    if any(w in text for w in feat_words):
        return "feat"
    if desc.startswith("complete "):
        return "story_mid"
    return "misc"


# --------------------------------------------------------------------------
# series handling
# --------------------------------------------------------------------------

def build_series(case, index):
    """(hours, rate_pct, players) points for one trophy, oldest first."""
    points = []
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            continue
        moment = parse_time(read.get("at")) or parse_time(read.get("slot"))
        players = read.get("players_tracked")
        try:
            players = float(players) if players is not None else None
        except (TypeError, ValueError):
            players = None
        points.append((hours_since_release(moment), rate, players))
    return points


def collapse_stale(points):
    """Merge consecutive reads with identical (rate, players).

    Returns (distinct_points, stale_hours) where stale_hours is how long
    the page has sat unchanged up to the newest read.
    """
    if not points:
        return [], 0.0
    distinct = [points[0]]
    for pt in points[1:]:
        prev = distinct[-1]
        if pt[1] == prev[1] and pt[2] == prev[2]:
            distinct[-1] = (pt[0], pt[1], pt[2])  # keep newest time of the run
        else:
            distinct.append(pt)
    stale = 0.0
    if len(points) >= 2:
        last_change = points[0][0]
        for i in range(1, len(points)):
            if points[i][1] != points[i - 1][1] or points[i][2] != points[i - 1][2]:
                last_change = points[i][0]
        stale = max(0.0, points[-1][0] - last_change)
    return distinct, stale


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_std) or None."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum(w * (y - (intercept + slope * x)) ** 2
                          for x, y, w in zip(xs, ys, ws)) / sw)
    return intercept, slope, resid


def fit_trend(distinct, target_hours):
    """Logit-vs-log-time trend over the last 24h of distinct reads.

    Returns (predicted_logit, slope, resid_std, weight) or None.
    """
    if len(distinct) < 3:
        return None
    newest = distinct[-1][0]
    recent = [p for p in distinct if p[0] >= newest - 24.0]
    if len(recent) < 3:
        recent = distinct
    span = recent[-1][0] - recent[0][0]
    if span < 1.0:
        return None
    xs = [log_clock(p[0]) for p in recent]
    ys = [logit_pct(p[1]) for p in recent]
    # recency weights: newest read counts most
    ws = [0.35 + 0.65 * (p[0] - recent[0][0]) / span for p in recent]
    fit = weighted_line(xs, ys, ws)
    if fit is None:
        return None
    intercept, slope, resid = fit
    slope = clamp(slope, -2.5, 2.5)
    x_target = log_clock(target_hours)
    x_last = xs[-1]
    # shrink long extrapolations: trust the slope less the further we go
    reach = x_target - x_last
    shrink = 1.0 / (1.0 + 0.35 * max(0.0, reach))
    predicted = ys[-1] + slope * reach * shrink
    weight = (len(recent) / (len(recent) + 4.0)) * min(1.0, span / 12.0)
    return predicted, slope, resid, clamp(weight, 0.0, 0.85)


def fit_player_growth(case, target_hours):
    """Predicted tracked-player growth factor to the target (>= 1).

    Returns (growth, last_players, observed).  When the history shows
    fewer than three distinct player counts (a stale launch-night page),
    fall back to a launch power-law prior: the tracked pool grows roughly
    like (1 + hours)^2 in the first days as syncs catch up.
    """
    seen = []
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        if players is None:
            continue
        try:
            players = float(players)
        except (TypeError, ValueError):
            continue
        if players <= 0:
            continue
        moment = parse_time(read.get("at")) or parse_time(read.get("slot"))
        seen.append((hours_since_release(moment), players))
    if not seen:
        return 1.0, None, False
    # distinct player counts only
    distinct = [seen[0]]
    for pt in seen[1:]:
        if pt[1] != distinct[-1][1]:
            distinct.append(pt)
    last_hours, last_players = seen[-1]
    if len(distinct) < 3:
        ratio = (1.0 + target_hours) / (1.0 + last_hours)
        prior_growth = clamp(ratio ** 2.0, 1.0, 15.0)
        return prior_growth, last_players, False
    xs = [log_clock(p[0]) for p in distinct]
    ys = [math.log(p[1]) for p in distinct]
    ws = [1.0] * len(distinct)
    fit = weighted_line(xs, ys, ws)
    if fit is None:
        return 1.0, last_players, True
    intercept, slope, _ = fit
    slope = clamp(slope, 0.0, 3.0)
    predicted = math.exp(intercept + slope * log_clock(target_hours))
    growth = clamp(predicted / last_players, 1.0, 40.0)
    return growth, last_players, True


def effective_dilution(base, horizon_days):
    """How much of the log player-growth drags a rate down.

    Over minutes-to-hours, new arrivals have earned almost nothing, so
    dilution is nearly one-for-one.  Over days they catch up on easy
    content, and the pass-through decays toward the class base rate.
    The opening trophy is exempt: being tracked at all means having
    started the game, so its earners arrive with the denominator.
    """
    if base < 0.1:
        return base
    return base + (1.0 - base) * math.exp(-max(horizon_days, 0.0) / 2.0)


# --------------------------------------------------------------------------
# per-trophy forecast
# --------------------------------------------------------------------------

def forecast_trophy(case, trophy, index, target_hours, horizon_days,
                    growth, last_players, growth_observed, levels):
    prior = CLASS_PRIORS[classify(trophy)]
    points = build_series(case, index)
    distinct, stale_hours = collapse_stale(points)

    if not distinct:
        tier = str(trophy.get("tier", "bronze")).lower()
        median_pct = TIER_FALLBACK_PCT.get(tier, 1.0)
        median_logit = logit_pct(median_pct)
        sigma = 1.2 + 0.5 * math.sqrt(max(horizon_days, 0.0))
        return quantiles_from(median_logit, sigma, levels)

    last_hours, last_rate, _ = distinct[-1]
    last_logit = logit_pct(last_rate)
    dilute = effective_dilution(prior["dilute"], horizon_days)

    # --- median -----------------------------------------------------------
    log_reach = max(0.0, log_clock(target_hours) - log_clock(last_hours))
    prior_move = prior["drift"] * log_reach
    # expected dilution from denominator growth over the horizon
    dilution_move = -dilute * math.log(max(growth, 1.0))

    trend = fit_trend(distinct, target_hours)
    if trend is not None:
        trend_logit, _slope, resid, w = trend
        # the fitted trend already reflects past dilution; add only the
        # part of expected future dilution beyond a mild baseline
        extra_dilution = -dilute * max(0.0, math.log(max(growth, 1.0)) - 0.4)
        median_logit = w * trend_logit + (1.0 - w) * (last_logit + prior_move + dilution_move)
        median_logit += w * extra_dilution
    else:
        resid = 0.0
        median_logit = last_logit + prior_move + dilution_move

    # a stale page hides a refresh: when growth was never observed, the
    # power-law prior already prices the jump; when it was observed but
    # the tail went stale, trust the fit less and widen below
    stale_factor = clamp(stale_hours / 3.0, 0.0, 1.0)
    if stale_factor > 0 and not growth_observed:
        median_logit -= dilute * math.log(max(growth, 1.0)) * 0.25 * stale_factor

    median_logit = clamp(median_logit, last_logit - 3.0, last_logit + 3.0)

    # earners never un-earn: the median may not imply fewer earners than
    # the last read already counted (allowing 5% slack for pool churn)
    if last_players:
        earners_last = last_rate / 100.0 * last_players
        players_pred = last_players * max(growth, 1.0)
        floor_pct = 100.0 * earners_last * 0.95 / players_pred
        median_logit = max(median_logit, logit_pct(floor_pct))

    # --- spread -----------------------------------------------------------
    walk = 0.5 * math.sqrt(max(horizon_days, 0.0))
    earners = max(1.0, last_rate / 100.0 * (last_players or 1000.0))
    others = max(1.0, (last_players or 1000.0) - earners)
    count_noise = math.sqrt(1.0 / earners + 1.0 / others)
    stale_bonus = 1.0 * stale_factor
    sigma = math.sqrt(walk ** 2 + resid ** 2 + (0.6 * count_noise) ** 2
                      + stale_bonus ** 2 + prior["spread"] ** 2)
    # near 100% the logit scale stretches enormously: a dip from 100 to
    # 98 is five logit units, so a saturated trophy needs a wide band
    if last_rate >= 99.5:
        sigma = max(sigma, 2.2 + 0.4 * math.sqrt(max(horizon_days, 0.0)))
    sigma = clamp(sigma, 0.18, 3.5)

    return quantiles_from(median_logit, sigma, levels)


def quantiles_from(median_logit, sigma, levels):
    row = []
    for tau in levels:
        z = NORMAL.inv_cdf(clamp(tau, 1e-6, 1.0 - 1e-6))
        row.append(pct_from_logit(median_logit + z * sigma))
    # clamp, round, enforce non-decreasing
    out = []
    prev = 0.0
    for value in row:
        value = clamp(round(value, 4), 0.0, 100.0)
        if value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


# --------------------------------------------------------------------------
# case driver
# --------------------------------------------------------------------------

def process_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    levels = [float(x) for x in levels]
    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    horizon_minutes = case.get("horizon_minutes")
    if target is not None:
        target_hours = hours_since_release(target)
    elif at is not None and horizon_minutes is not None:
        target_hours = hours_since_release(at) + float(horizon_minutes) / 60.0
    else:
        target_hours = hours_since_release(at)
    if horizon_minutes is not None:
        horizon_days = float(horizon_minutes) / 1440.0
    elif at is not None and target is not None:
        horizon_days = max(0.0, (target - at).total_seconds() / 86400.0)
    else:
        horizon_days = 0.0

    growth, last_players, growth_observed = fit_player_growth(case, target_hours)

    rows = []
    for index, trophy in enumerate(case.get("trophies") or []):
        try:
            q = forecast_trophy(case, trophy, index, target_hours,
                                horizon_days, growth, last_players,
                                growth_observed, levels)
        except Exception:
            q = [1.0] * len(levels)
        rows.append({"id": trophy.get("id", index + 1), "q": q})
    return {"trophies": rows}


def fallback_case(case):
    """Last-resort forecast: last read's rates (or tier priors), wide band."""
    levels = case.get("levels") or DEFAULT_LEVELS
    levels = [float(x) for x in levels]
    rows = []
    history = case.get("history") or []
    last_rates = (history[-1].get("rates") or []) if history else []
    for index, trophy in enumerate(case.get("trophies") or []):
        rate = last_rates[index] if index < len(last_rates) else None
        if rate is None:
            tier = str(trophy.get("tier", "bronze")).lower()
            rate = TIER_FALLBACK_PCT.get(tier, 1.0)
        q = quantiles_from(logit_pct(float(rate)), 1.0, levels)
        rows.append({"id": trophy.get("id", index + 1), "q": q})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(input_dir, "*.in"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            continue
        try:
            result = process_case(case)
        except Exception:
            result = fallback_case(case)
        out_path = os.path.join(output_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"), ensure_ascii=False)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
