[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ikball53/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ikball53/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts, for every Grand Theft Auto VI (PS5) trophy, five quantiles of its
Exophase earn rate at a case's target time. Built for make.fun's
"gta6-trophy-rarity-forecast-01" bounty.

## Model

Week-one earn rates are driven by three forces, and the forecaster prices
each one on the logit scale (the scale the scorer uses):

- **Progression** — the launch cohort plays deeper over time. Each trophy is
  classified from its tier and description (opening mission, early/mid/late
  story, side strands, collectibles, grinds, one-off feats, economy goals,
  100% completion, platinum) and gets a class prior drift per unit of
  log-time since release.
- **Dilution** — the tracked-player denominator grows fast and new arrivals
  are more casual. Player growth is fitted from the case history (or a
  launch power-law prior when the page has not refreshed yet); the
  pass-through to rates is nearly one-for-one over short horizons and
  decays over days as newcomers catch up. The opening trophy is exempt:
  being tracked means having started the game.
- **Page staleness** — Exophase refreshes in jumps. Runs of identical reads
  are collapsed, the staleness is measured, and the forecast band widens
  while the median prices the coming re-basing.

The median blends a recency-weighted logit-vs-log-time trend over the last
24 hours of distinct reads (with shrinkage on long extrapolations) against
the class-prior path, and is floored so it never implies fewer earners than
the last read already counted. The spread combines a random-walk term over
the horizon, fit residuals, binomial count noise from the small early
player pool, a staleness bonus and per-class uncertainty; trophies sitting
at ~100% get an extra-wide band because the logit scale stretches
enormously near saturation. Trophies with no usable history fall back to
tier priors. The program is fully deterministic (no randomness) and uses
only the Python standard library.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory (in name order) and writes
one `<id>.out` forecast per case. With the environment variables unset it
uses `/input` and `/output`, matching the grader.
