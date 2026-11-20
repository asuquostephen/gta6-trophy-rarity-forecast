"""gta6-trophy-rarity-forecast-01 entry: a counts-based Monte Carlo forecast.

The house starter forecasts persistence: every trophy's earn rate stays at the
last read, with a random-walk spread on the logit scale. This entry instead
forecasts the two counts a rate is made of -- earners and players tracked --
and divides, because the thing that moves a printed rate is Exophase
refreshing its counts: a refresh lands a much larger tracked pool whose newer
players have earned far less, so nearly every rate falls at once (the launch
night refresh took the pool from 638 to 3,868 and every rate but the 100% one
went down). Two regimes:

  - Flat (no refresh anywhere in the history): a mixture. With probability
    0.45 * exp(-H/4h) no refresh lands before the target and the answer is the
    last read verbatim -- that spike forms the top of the band. Otherwise the
    pool grows by a lognormal catch-up factor over the span the printed pool
    size has already stood plus the horizon (less a refresh lag of up to
    ~2 h), existing non-earners convert at a small hazard, and the newly
    tracked players earn at kappa x the current rate, kappa centred on 0.15.
  - Trending (refreshes in the history): log-linear slopes for earners and
    for players tracked (recent windows blended with the full span, the
    earners slope shrunk toward the whole-game trophies_earned slope),
    projected over a damped horizon 48 * (1 - exp(-H/48)) hours, with
    lognormal draw spreads that grow with the horizon.

1,600 seeded draws per trophy (random.Random(20261119)); the rows written are
the draws' empirical quantiles at the case's levels. A trophy at 100% is
pinned to [99.85, 99.97, 100, 100, 100]: its earners are the whole pool, so
no refresh can dilute it.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only, no network. Outside the grader,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
DRAWS = 1600
SEED = 20261119

# Conversion speed by tier: a platinum is earned slowly, a bronze quickly.
TIER_SPEED = {"platinum": 0.15, "gold": 0.55, "silver": 0.8, "bronze": 1.0}
# Newly tracked players earn at kappa x the current rate; the rare tiers are
# cut further because a fresh pool holds almost no completionists.
KAPPA_CUT = {"platinum": 0.35, "gold": 0.6, "silver": 0.85, "bronze": 1.0}

PIN_100 = [99.85, 99.97, 100, 100, 100]


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> float:
    """An ISO-8601 timestamp to epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y against x; 0 when x never moves or n < 2."""
    n = len(points)
    if n < 2:
        return 0.0
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0:
        return 0.0
    return sum((p[0] - mx) * (p[1] - my) for p in points) / sxx


def quantile(sorted_draws: list[float], tau: float) -> float:
    """Empirical quantile of sorted draws, linear interpolation on (n - 1)."""
    n = len(sorted_draws)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_draws[lo] * (1 - frac) + sorted_draws[hi] * frac


def finish(draws: list[float]) -> list:
    """Sorted draws to the five rounded, non-decreasing output quantiles."""
    draws.sort()
    row = []
    for tau in LEVELS:
        value = min(100.0, max(0.01, quantile(draws, tau)))
        row.append(round4(value))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def flat_draws(rng: random.Random, rate: float, pool: float, earners: float,
               tier: str, span_h: float, horizon_h: float) -> list[float]:
    """The flat-regime mixture: a no-refresh spike at the last rate, else a
    catch-up refresh that dilutes the rate."""
    p_no_refresh = 0.45 * math.exp(-horizon_h / 4.0)
    speed = TIER_SPEED.get(tier, 1.0)
    kappa_cut = KAPPA_CUT.get(tier, 1.0)
    # How reachable the trophy is to the pool that has not earned it yet.
    accessibility = min(1.0, rate / 50.0)
    conv = 1.0 - math.exp(-0.08 * speed * accessibility * (horizon_h / 24.0))
    converted = (pool - earners) * conv
    draws = []
    for _ in range(DRAWS):
        if rng.random() < p_no_refresh:
            draws.append(rate)
            continue
        lag = 2.0 * rng.random()
        window = max(span_h + horizon_h - lag, 0.05)
        growth = 0.38 * math.exp(0.55 * rng.gauss(0.0, 1.0))
        new_pool = pool * math.exp(growth * window)
        kappa = 0.15 * kappa_cut * math.exp(0.5 * rng.gauss(0.0, 1.0))
        new_earners = earners + converted + (new_pool - pool) * min(kappa * rate, 100.0) / 100.0
        new_earners = min(new_earners, new_pool)
        draws.append(100.0 * new_earners / new_pool)
    return draws


def trending_draws(rng: random.Random, case: dict, idx: int, tier: str,
                   horizon_h: float) -> list[float]:
    """The trending-regime projection: log-linear slopes for both counts over
    a damped horizon, with horizon-widened lognormal spreads."""
    history = case["history"]
    times = [parse_time(read["at"]) for read in history]
    t0 = times[0]
    hours = [(t - t0) / 3600.0 for t in times]
    pools = [float(read["players_tracked"]) for read in history]
    last = len(history) - 1
    pool_last = pools[last]
    earners_last = case["_earners"][idx][last]

    days = horizon_h / 24.0
    damped = 48.0 * (1.0 - math.exp(-horizon_h / 48.0))

    # Players tracked: 12 h recent window blended with the full span.
    pool_pts = [(hours[i], math.log(max(pools[i], 1.0))) for i in range(len(history))]
    recent_pool = [p for p in pool_pts if p[0] >= hours[last] - 12.0]
    b_pool = 0.65 * slope(recent_pool) + 0.35 * slope(pool_pts)

    # Earners: last-24 h and full-span slopes blended, then shrunk 65/35
    # toward the whole-game trophies_earned slope.
    earn_pts = [(hours[i], math.log(max(case["_earners"][idx][i], 0.5)))
                for i in range(len(history)) if case["_earners"][idx][i] is not None]
    recent_earn = [p for p in earn_pts if p[0] >= hours[last] - 24.0]
    b_earn_blend = 0.6 * slope(recent_earn) + 0.4 * slope(earn_pts)
    game_pts = [(hours[i], math.log(max(float(history[i]["trophies_earned"]), 1.0)))
                for i in range(len(history))]
    b_earn = 0.65 * b_earn_blend + 0.35 * slope(game_pts)

    # Hours since the tracked pool last changed before the checkpoint.
    stale_h = 0.0
    for i in range(last, 0, -1):
        if pools[i - 1] != pool_last:
            stale_h = (parse_time(case["at"]) - times[i]) / 3600.0
            break
    else:
        stale_h = (parse_time(case["at"]) - times[0]) / 3600.0

    sigma_pool = 0.16 + 0.16 * math.sqrt(days) + (0.15 if stale_h > 6.0 else 0.0)
    sigma_earn = 0.16 + 0.26 * math.sqrt(days) + 0.35 / math.sqrt(max(earners_last, 1.0))

    ln_pool = math.log(max(pool_last, 1.0)) + b_pool * damped
    ln_earn = math.log(max(earners_last, 0.5)) + b_earn * damped
    draws = []
    for _ in range(DRAWS):
        pool_draw = math.exp(ln_pool + sigma_pool * rng.gauss(0.0, 1.0))
        earn_draw = math.exp(ln_earn + sigma_earn * rng.gauss(0.0, 1.0))
        earn_draw = min(earn_draw, pool_draw)
        draws.append(100.0 * earn_draw / pool_draw)
    return draws


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_h = case["horizon_minutes"] / 60.0

    # Earners per trophy per read: rate/100 x players tracked (None where a
    # read lacked the trophy).
    case["_earners"] = [
        [None if read["rates"][i] is None else read["rates"][i] / 100.0 * read["players_tracked"]
         for read in history]
        for i in range(len(trophies))
    ]

    refreshed = any(history[i]["players_tracked"] != history[i - 1]["players_tracked"]
                    for i in range(1, len(history)))

    # The span the printed pool size has already stood: the constant-pool
    # suffix of the history, up to the checkpoint.
    pool_last = float(history[-1]["players_tracked"])
    span_h = 0.0
    if not refreshed:
        start = len(history) - 1
        while start > 0 and float(history[start - 1]["players_tracked"]) == pool_last:
            start -= 1
        span_h = max((parse_time(case["at"]) - parse_time(history[start]["at"])) / 3600.0, 0.0)

    rng = random.Random(SEED)
    rows = []
    for i, trophy in enumerate(trophies):
        # The last read that lists this trophy.
        rate = None
        pool = pool_last
        earners = 0.0
        for j in range(len(history) - 1, -1, -1):
            if history[j]["rates"][i] is not None:
                rate = float(history[j]["rates"][i])
                pool = float(history[j]["players_tracked"])
                earners = case["_earners"][i][j]
                break
        if rate is None:
            rows.append({"id": trophy["id"], "q": [0.01, 0.01, 0.01, 0.01, 0.01]})
            continue
        if rate >= 100.0 or earners >= pool:
            rows.append({"id": trophy["id"], "q": list(PIN_100)})
            continue
        tier = str(trophy.get("tier", "")).lower()
        if refreshed:
            draws = trending_draws(rng, case, i, tier, horizon_h)
        else:
            draws = flat_draws(rng, rate, pool, earners, tier, span_h, horizon_h)
        rows.append({"id": trophy["id"], "q": finish(draws)})
    del case["_earners"]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
