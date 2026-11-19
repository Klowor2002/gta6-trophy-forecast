"""Week-one trophy rarity forecaster for GTA VI (PS5) on Exophase.

Reads every case file in the input directory and, for each of the case's
trophies, writes five quantiles of the earn rate (percent of tracked players
who have earned it) at the case's target instant.

Method, in one paragraph: work on the logit scale. For each trophy, fit a
weighted straight line of logit(rate) against ln(1 + hours since release),
giving recent reads more weight because launch-week dynamics shift fast.
Shrink each trophy's slope toward the median slope of its tier, anchor the
line's level on the last observed rate, read the line at the target, cap the
median at a tier ceiling, and spread the quantiles by an honest sigma built
from the fit residual, the largest jump between distinct reads, and a
random-walk floor. A trophy already sitting at 100 percent in every read is
pinned near 100 instead of being dragged down by a fit.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

RATE_LO = 0.01
RATE_HI = 99.99
HALF_LIFE_HOURS = 12.0
ANCHOR_WEIGHT = 0.7  # how much the last read, vs the fitted line, sets the level
WALK_PER_SQRT_DAY = 0.5
# Growth decelerates as a trophy approaches its week-one ceiling, so a
# slope fitted over the first days overstates the next days. Extrapolation
# uses only this share of the fitted slope (calibrated on synthetic
# backtests with saturating and log-time growth families; see README).
EXTRAP_DAMP = 0.35
# One largest jump is an outlier statistic (early reads rest on a handful
# of earners); cap its contribution to the spread.
JUMP_CAP = 0.7
TIER_CEILING = {"platinum": 30.0, "gold": 45.0, "silver": 65.0, "bronze": 85.0}
PINNED_RATE = 99.99


def clamp_rate(pct):
    return min(RATE_HI, max(RATE_LO, pct))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def parse_instant(text):
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def build_series(case, index):
    """One trophy's cleaned history: (hours, rate, tracked) skipping nulls."""
    points = []
    for read in case["history"]:
        rate = read["rates"][index]
        if rate is None:
            continue
        when = parse_instant(read["at"])
        points.append((hours_since_release(when), float(rate),
                       float(read.get("players_tracked") or 0)))
    return points


def distinct_points(points):
    """Collapse consecutive reads that repeat the same rate and tracked count.

    Exophase refreshes on its own schedule, so long runs of identical reads
    carry one piece of information, not many.
    """
    kept = []
    for point in points:
        if kept and kept[-1][1] == point[1] and kept[-1][2] == point[2]:
            kept[-1] = point  # keep the latest timestamp of the same page
        else:
            kept.append(point)
    return kept


def weighted_line_fit(xs, ys, weights):
    """Weighted least squares y ~ a + b*x; returns (a, b, residual_std)."""
    total_w = sum(weights)
    if total_w <= 0:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mean_x = sum(w * x for x, w in zip(xs, weights)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, weights)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, weights))
    if sxx <= 1e-12:
        return mean_y, 0.0, 0.0
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for x, y, w in zip(xs, ys, weights))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    resid_std = math.sqrt(sum(w * r * r for r, w in zip(resid, weights))
                          / total_w * len(xs) / dof) if len(xs) > 2 else 0.0
    return intercept, slope, resid_std


def fit_trophy(points, target_hours):
    """V1 fit for one trophy over its distinct reads.

    Returns a dict with the raw slope, residual std, largest jump between
    distinct reads (logit scale), the last rate, and how many distinct reads
    backed the fit.
    """
    distinct = distinct_points(points)
    last_rate = points[-1][1]
    if not distinct:
        return {"slope": 0.0, "resid": 0.0, "jump": 0.0,
                "last_rate": last_rate, "last_logit": logit_of(last_rate),
                "intercept": logit_of(last_rate), "n_distinct": 0,
                "last_x": math.log1p(max(0.0, target_hours))}
    xs = [math.log1p(max(0.0, t)) for t, _, _ in distinct]
    # Fit on count-smoothed rates: a printed 0.00% at a few hundred tracked
    # players means "fewer than one earner so far", not a logit of -9.2, and
    # letting it stand would inflate every slope. The anchor below still
    # uses the actual last rate; only the fit sees the smoothed series.
    ys = []
    for _, rate, tracked in distinct:
        if tracked > 0:
            earners = rate * tracked / 100.0
            ys.append(logit_of(100.0 * (earners + 0.5) / (tracked + 1.0)))
        else:
            ys.append(logit_of(rate))
    t_last = distinct[-1][0]
    weights = [0.5 ** (max(0.0, t_last - t) / HALF_LIFE_HOURS)
               for t, _, _ in distinct]
    intercept, slope, resid = weighted_line_fit(xs, ys, weights)
    jump = 0.0
    for before, after in zip(ys, ys[1:]):
        jump = max(jump, abs(after - before))
    return {"slope": slope, "resid": resid, "jump": jump,
            "last_rate": last_rate, "last_logit": logit_of(last_rate),
            "intercept": intercept, "n_distinct": len(distinct),
            "last_x": xs[-1]}


def monotone_earners(points):
    """Earners implied by each read, forced non-decreasing over time.

    A refresh can only add earners to the tracked set in expectation; when
    the implied count dips, the tracked pool grew faster than the earners
    (dilution), so hold the count at its running maximum.
    """
    counts = []
    running = 0.0
    for _, rate, tracked in points:
        implied = rate * tracked / 100.0
        running = max(running, implied)
        counts.append(running)
    return counts


def fit_log_growth(times, values, t_last):
    """Weighted line of log(value) on ln(1+t); returns (intercept, slope, resid)."""
    xs = [math.log1p(max(0.0, t)) for t in times]
    ys = [math.log(max(1e-9, v)) for v in values]
    weights = [0.5 ** (max(0.0, t_last - t) / HALF_LIFE_HOURS) for t in times]
    return weighted_line_fit(xs, ys, weights)


def tracked_forecast(case, target_hours):
    """Log-linear forecast of the tracked-player count at the target."""
    times = []
    counts = []
    for read in case["history"]:
        tracked = read.get("players_tracked")
        if not tracked:
            continue
        times.append(hours_since_release(parse_instant(read["at"])))
        counts.append(float(tracked))
    if not counts:
        return 0.0, 0.0, 1.0
    # Collapse repeats the same way as the per-trophy series.
    kept_t, kept_c = [], []
    for t, c in zip(times, counts):
        if kept_c and kept_c[-1] == c:
            kept_t[-1] = t
        else:
            kept_t.append(t)
            kept_c.append(c)
    intercept, slope, resid = fit_log_growth(kept_t, kept_c, kept_t[-1])
    target_x = math.log1p(max(0.0, target_hours))
    last_x = math.log1p(max(0.0, kept_t[-1]))
    level = ANCHOR_WEIGHT * math.log(kept_c[-1]) + \
        (1.0 - ANCHOR_WEIGHT) * (intercept + slope * last_x)
    forecast = math.exp(level + EXTRAP_DAMP * slope * (target_x - last_x))
    growth = kept_c[-1] / kept_c[0] if kept_c[0] > 0 else 1.0
    return forecast, resid, growth


def v2_median_logit(points, tracked_target, target_x):
    """Earners/tracked decomposition: grow earners, divide by tracked.

    Returns the median on the logit scale, or None when the decomposition
    has nothing to stand on (no tracked forecast, too few distinct reads,
    too few earners, or a log-earners fit too noisy to trust), so the
    caller can fall back to V1.
    """
    distinct = distinct_points(points)
    if len(distinct) < 3 or tracked_target <= 0:
        return None
    earners = monotone_earners(distinct)
    if earners[-1] < 3.0:
        return None
    times = [t for t, _, _ in distinct]
    intercept, slope, resid = fit_log_growth(
        times, [e + 0.5 for e in earners], times[-1])
    if resid > 0.6:
        return None
    last_x = math.log1p(max(0.0, times[-1]))
    level = ANCHOR_WEIGHT * math.log(earners[-1] + 0.5) + \
        (1.0 - ANCHOR_WEIGHT) * (intercept + slope * last_x)
    earners_target = math.exp(
        level + EXTRAP_DAMP * slope * (target_x - last_x)) - 0.5
    if earners_target <= 0:
        return None
    rate = 100.0 * earners_target / tracked_target
    return logit_of(rate)


def v1_median_logit(fit, tier_median_slope, target_x):
    """Shrink the slope toward the tier median, anchor the level, extrapolate."""
    noise = fit["resid"] + 0.35
    shrink = fit["n_distinct"] / (fit["n_distinct"] + 2.0 + 4.0 * noise)
    slope = shrink * fit["slope"] + (1.0 - shrink) * tier_median_slope
    fitted_at_last = fit["intercept"] + fit["slope"] * fit["last_x"]
    level = ANCHOR_WEIGHT * fit["last_logit"] + (1.0 - ANCHOR_WEIGHT) * fitted_at_last
    return level + EXTRAP_DAMP * slope * (target_x - fit["last_x"]), slope


def spread_sigma(fit, horizon_days, last_earners):
    """Honest spread, combined in quadrature.

    Parts: the fit residual, the largest jump between distinct reads (a
    refresh moves every rate at once), a random-walk floor over the
    horizon, and a count-noise term: a rate built on a handful of earners
    can swing far on the logit scale when a few more players earn it.
    """
    walk = WALK_PER_SQRT_DAY * math.sqrt(max(0.0, horizon_days))
    count_noise = 1.0 / math.sqrt(max(1.0, last_earners))
    jump = min(fit["jump"], JUMP_CAP)
    sigma = math.sqrt(fit["resid"] ** 2 + jump ** 2 + walk ** 2
                      + count_noise ** 2)
    return min(3.0, max(walk, sigma))


def quantiles_from(median_logit, sigma):
    row = []
    for z in Z_SCORES:
        row.append(round4(clamp_rate(pct_of_logit(median_logit + z * sigma))))
    row.sort()
    return row


def forecast_case(case):
    trophies = case["trophies"]
    target_hours = hours_since_release(parse_instant(case["target"]))
    target_x = math.log1p(max(0.0, target_hours))
    horizon_days = case["horizon_minutes"] / 1440.0

    all_points = [build_series(case, i) for i in range(len(trophies))]
    fits = [fit_trophy(points, target_hours) for points in all_points]

    tier_slopes = {}
    for trophy, fit in zip(trophies, fits):
        tier_slopes.setdefault(trophy["tier"], []).append(fit["slope"])
    tier_median = {tier: median_of(slopes)
                   for tier, slopes in tier_slopes.items()}

    rows = []
    for trophy, points, fit in zip(trophies, all_points, fits):
        rates_seen = [rate for _, rate, _ in points]
        if rates_seen and all(rate >= 99.99 for rate in rates_seen):
            # Pinned at the top: the page has never shown anything else.
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(logit_of(PINNED_RATE), 0.05)})
            continue
        median_logit, _slope = v1_median_logit(
            fit, tier_median.get(trophy["tier"], 0.0), target_x)
        # The earners/tracked decomposition (V2, kept above) was built as
        # a challenger and lost the synthetic backtests to V1 on both the
        # tuning and held-out seed blocks (see the entry's build notes):
        # its tracked-pool forecast error swamps the dilution signal at
        # this horizon. V1 is therefore the shipped path for every trophy.
        median_pct = pct_of_logit(median_logit)
        ceiling = TIER_CEILING.get(trophy["tier"], 85.0)
        if median_pct > ceiling:
            median_logit = logit_of(ceiling)
        last_earners = 0.0
        if points:
            last_earners = points[-1][1] * points[-1][2] / 100.0
        sigma = spread_sigma(fit, horizon_days, last_earners)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(median_logit, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
