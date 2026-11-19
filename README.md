# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Klowor2002/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Klowor2002/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Week-One Trophy Rarity Forecaster: a stdlib-only Python program that forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of Exophase-tracked players who will have earned it one week after release. It works on the logit scale: per trophy it fits a recency-weighted line of logit(rate) against log-time on count-smoothed rates, shrinks the slope toward its tier's median, anchors the level on the last read, damps the extrapolation (growth decelerates toward week-one ceilings, which also cap the median by tier), and spreads the five quantiles by a sigma combined from the fit residual, the largest jump between distinct reads (capped), a random-walk floor, and a count-noise term. A trophy the page has only ever shown at 100% is pinned near 100. An earners/tracked decomposition was built as a challenger and lost the synthetic backtests, so the rate fit is the shipped path. Practice answers are not available locally, so validation is format/determinism checks on the kit's practice cases plus synthetic backtests with simulated growth, refresh jumps and dilution.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ebgtn94a&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
