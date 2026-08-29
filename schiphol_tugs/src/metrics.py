"""Metrics: waiting times and distance/energy totals.

Waiting time is defined as (tow_start - request creation time): the seconds
the aircraft sits waiting until a tug has arrived and the tow begins.  Only
requests whose tow actually started are counted (requests created minutes
before the simulation ends may still be in flight; both methods are treated
identically, so comparisons stay fair).
"""

import numpy as np


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
        # Physical energy consumed by driving, summed over the fleet (kWh).
        "total_energy_kwh": float(sum(t.energy_used for t in sim.tugs)),
    }
