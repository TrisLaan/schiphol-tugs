"""Metrics: waiting times, distance/energy totals, and zone-formation measures.

Waiting time is defined as (tow_start - request creation time): the seconds
the aircraft sits waiting until a tug has arrived and the tow begins.  Only
requests whose tow actually started are counted (requests created minutes
before the simulation ends may still be in flight; both methods are treated
identically, so comparisons stay fair).
"""

import numpy as np
import pandas as pd

from .config import REGIONS


def wait_times(requests) -> np.ndarray:
    """Waiting time (s) of every request whose tow started."""
    return np.array(
        [r.tow_start - r.time for r in requests if r.tow_start is not None],
        dtype=float,
    )


def summarize(sim) -> dict:
    """Headline metrics for one finished simulation run."""
    w = wait_times(sim.requests)
    served = int(w.size)
    total = len(sim.requests)
    return {
        "requests_total": total,
        "requests_served": served,
        "mean_wait_s": float(np.mean(w)) if served else float("nan"),
        "p95_wait_s": float(np.percentile(w, 95)) if served else float("nan"),
        "total_distance_km": float(sum(t.total_distance for t in sim.tugs) / 1000.0),
        # Energy proxy: battery percentage-points consumed by driving,
        # summed over the fleet (drain rates are per-metre, so this is
        # proportional to true energy for a fixed pack size).
        "total_energy_pct": float(sum(t.energy_used for t in sim.tugs) * 100.0),
    }


def specialization_index(theta: np.ndarray) -> float:
    """Mean over piers of the coefficient of variation of thresholds.

    ``theta`` has shape (n_tugs, n_regions).  A uniform fleet gives 0; the
    index rises as individual tugs specialise on particular piers.
    """
    mean = theta.mean(axis=0)
    std = theta.std(axis=0)
    cv = np.where(mean > 0, std / mean, 0.0)
    return float(cv.mean())


def specialization_series(snapshots) -> tuple[np.ndarray, np.ndarray]:
    """(times_s, index) from the simulator's threshold snapshots."""
    times = np.array([t for t, _ in snapshots], dtype=float)
    idx = np.array([specialization_index(arr) for _, arr in snapshots])
    return times, idx


def snapshot_at(snapshots, t_target: float) -> np.ndarray:
    """The threshold matrix from the snapshot closest to ``t_target``."""
    best = min(snapshots, key=lambda s: abs(s[0] - t_target))
    return best[1]


def rolling_wait(requests, duration_s: int, window_s: int = 3600,
                 step_s: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean waiting time over the day.

    At each grid time t the mean is taken over requests whose tow started in
    (t - window_s, t].  Empty windows yield NaN (matplotlib leaves gaps).
    """
    starts = np.array(
        [(r.tow_start, r.tow_start - r.time) for r in requests if r.tow_start is not None],
        dtype=float,
    )
    grid = np.arange(window_s, duration_s + 1, step_s, dtype=float)
    out = np.full(grid.shape, np.nan)
    if starts.size:
        for i, t in enumerate(grid):
            mask = (starts[:, 0] > t - window_s) & (starts[:, 0] <= t)
            if mask.any():
                out[i] = starts[mask, 1].mean()
    return grid, out


def region_of(regions_index: int) -> str:
    return REGIONS[regions_index]


def theta_snapshots_dataframe(snapshots) -> pd.DataFrame:
    """Long-format per-tug threshold history from the simulator's snapshot
    list, independent of visual recording (mirrors :meth:`Simulation.
    contests_dataframe`, which is likewise always available)."""
    rows = []
    for t, arr in snapshots:
        for tug_i in range(arr.shape[0]):
            rows.append((t, tug_i, *arr[tug_i]))
    return pd.DataFrame(rows, columns=["time", "tug_id", *REGIONS])


def dominance_hierarchy(contests_df: pd.DataFrame, fleet_size: int) -> pd.DataFrame:
    """Per-tug wasp-contest record: how often each tug entered a multi-
    responder contest and how often it won, sorted by win rate descending.

    A flat win rate across tugs means no dominance hierarchy emerged; a
    skewed one (see :func:`gini`) means a subset of tugs consistently beats
    the rest, e.g. because they hold a proximity or specialization advantage.
    """
    wins = np.zeros(fleet_size, dtype=int)
    appearances = np.zeros(fleet_size, dtype=int)
    for _, row in contests_df.iterrows():
        responders = [int(v) for v in str(row["responders"]).split(",")]
        for r in responders:
            appearances[r] += 1
        wins[int(row["winner"])] += 1
    win_rate = np.divide(wins, appearances,
                          out=np.zeros(fleet_size), where=appearances > 0)
    return pd.DataFrame({
        "tug_id": np.arange(fleet_size), "contests": appearances,
        "wins": wins, "win_rate": win_rate,
    }).sort_values("win_rate", ascending=False, kind="mergesort").reset_index(drop=True)


def gini(values) -> float:
    """Gini coefficient of a non-negative array (0 = perfectly equal,
    towards 1 = concentrated in a few entries). Used on contest win rates to
    quantify how skewed the emergent dominance hierarchy is."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * v) / (n * v.sum()))


def primary_region(theta_snapshot: np.ndarray) -> np.ndarray:
    """Index into REGIONS of each tug's lowest-threshold (most specialised)
    pier. ``theta_snapshot`` has shape (n_tugs, n_regions)."""
    return theta_snapshot.argmin(axis=1)


def division_of_labor_counts(theta_snapshot: np.ndarray) -> np.ndarray:
    """Number of tugs primarily specialised in each region (length
    len(REGIONS)), from a single threshold snapshot."""
    return np.bincount(primary_region(theta_snapshot), minlength=len(REGIONS))
