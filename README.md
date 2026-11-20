[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/asuquostephen/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/asuquostephen/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA 6 Trophy Rarity Forecast

An entry for make.fun's **gta6-trophy-rarity-forecast-01** bounty: forecast,
for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on
Exophase at the case's target — five quantiles (5%, 25%, 50%, 75%, 95%) per
trophy, scored by mean pinball loss on the logit scale.

## The idea

A printed rate is a ratio: **earners / players tracked**. The house starter
forecasts the ratio itself (persistence plus a random-walk spread). This
entry forecasts the two counts separately and divides, because the thing
that actually moves the rates is Exophase refreshing its counts: a refresh
lands a far larger tracked pool whose newer players have earned much less,
so nearly every rate falls at once. On launch night the pool went from 638
to 3,868 players and every rate but the one at 100% dropped — straight
through the starter's band.

## The model

1,600 seeded Monte Carlo draws per trophy (`random.Random(20261119)`, so
repeat runs are byte-identical); the written row is the draws' empirical
quantiles. Two regimes:

- **Flat** — no refresh anywhere in the history. A mixture: with probability
  `0.45 * exp(-H/4h)` no refresh lands before the target and the answer is
  the last read verbatim; that spike forms the top of the band. Otherwise
  the pool grows by a lognormal catch-up factor (median growth 0.38 log/h)
  over the span the printed pool size has already stood, plus the horizon,
  less a refresh lag of up to ~2 h; existing non-earners convert at a small
  hazard (0.08/day × tier speed × accessibility from the current rate); and
  the newly tracked players earn at κ × the current rate, κ centred on 0.15
  and cut further for platinum and gold.
- **Trending** — refreshes in the history. Log-linear slopes for earners
  (last-24 h blended with the full span, shrunk 65/35 toward the whole-game
  `trophies_earned` slope) and for players tracked (12 h window blended with
  the full span), projected over a damped horizon of `48 * (1 - exp(-H/48))`
  hours. Draw spreads grow with the horizon: σ_pool ≈ 0.16 + 0.16·√days
  (+0.15 if the last refresh is over 6 h stale), σ_earners ≈ 0.16 +
  0.26·√days + 0.35/√E. Earners are capped at the pool size in every draw.

A trophy at 100% is pinned to `[99.85, 99.97, 100, 100, 100]` — its earners
are the whole pool, so no refresh can dilute it.

Tier speeds: platinum 0.15, gold 0.55, silver 0.8, bronze 1.0.

## Running it

`main.py` at the repository root, Python 3.13, standard library only, no
network. It reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`)
and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`):

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Practice behaviour

On the kit's launch-night practice cases (flat histories, 90-minute
horizons) the bands are wide and shifted down from the last rates — medians
below the last rate for nearly every non-100% trophy — built to contain the
refresh the rules describe (pool ×6.06, all rates down) that the starter's
narrow persistence band missed entirely.
