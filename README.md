[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=badge&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ikball53/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=badge&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ikball53/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=badge&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts how many GTA VI players will have earned each PS5 trophy at a
future checkpoint, as five quantiles (5/25/50/75/95%) of the earn rate.

The forecaster decomposes every trophy's earn rate into earners divided
by tracked players and projects the two series separately: tracked
players keep growing even when the read history looks frozen (the
tracker refreshes in lagged jumps, which dilutes rates), while earners
grow sublinearly with players plus a slow progression term as existing
players keep playing. The two elasticities are fit per trophy with a
Frisch-Waugh-Lovell split shrunk toward rate- and tier-based priors.
Quantiles are built on the logit scale with a spread that widens with
horizon and with small earner counts.

Pure Python 3 standard library, deterministic, no network. Reads
`$MAKEFUN_INPUT_DIR` (default `/input/`) and writes one `.out` JSON
file per case to `$MAKEFUN_OUTPUT_DIR` (default `/output/`).
