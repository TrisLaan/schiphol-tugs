"""Parameter sensitivity analysis: one-at-a-time sweeps (tornado plot), a 2D
interaction sweep (heatmap), and environment-parameter sweeps -- all with
multiple seeds per cell so sweep results carry uncertainty like every other
figure in this project (see AUDIT.md category 6).

Sweeps run on a short (``ExperimentConfig.sens_hours``) morning-peak window,
not the full 24 h day, so the whole sensitivity analysis finishes in well
under a minute even though it covers 8 algorithm/environment parameters. All
sweeps start from the *calibrated* configuration (passed in as ``cfg``) and
vary exactly one field away from it, which is what makes this a meaningful
one-at-a-time sensitivity analysis rather than 8 independent calibrations.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, plotting
from .airport import Airport, generate_requests
from .config import Config, ExperimentConfig, make_rng
from .simulator import Simulation


def _run_cell(airport: Airport, cfg: Config, ec: ExperimentConfig, seed_base: int,
             duration_s: int, rate_multiplier=None) -> tuple[float, float]:
    """(mean_wait_s, specialization_end), each averaged over ec.sens_seeds seeds."""
    waits, specs = [], []
    for k in range(ec.sens_seeds):
        reqs = generate_requests(airport, cfg, make_rng(seed_base + 10 * k),
                                 duration_s, start_hour=ec.sens_start_hour,
                                 rate_multiplier=rate_multiplier)
        sim = Simulation(airport, cfg, reqs, make_rng(seed_base + 10 * k + 1),
                         mode="swarm", duration_s=duration_s)
        sim.run()
        stats = metrics.summarize(sim)
        waits.append(stats["mean_wait_s"])
        _, spec = metrics.specialization_series(sim.snapshots)
        specs.append(float(spec[-1]) if len(spec) else float("nan"))
    return float(np.nanmean(waits)), float(np.nanmean(specs))


def run_sensitivity(airport: Airport, cfg: Config, ec: ExperimentConfig,
                    results_dir: Path, figures_dir: Path) -> dict:
    duration_s = int(ec.sens_hours * 3600)
    rows: list[dict] = []
    stream = 90_000

    def record(param, value, mean_wait, spec):
        rows.append({"parameter": param, "value": value, "mean_wait_s": mean_wait,
                    "specialization_end": spec, "n_seeds": ec.sens_seeds})

    def sweep_cfg_field(param_label, field, grid):
        nonlocal stream
        vals = []
        for v in grid:
            c = replace(cfg, **{field: v})
            mw, sp = _run_cell(airport, c, ec, stream, duration_s)
            stream += 100
            record(param_label, v, mw, sp)
            vals.append(mw)
        return list(grid), vals

    def sweep_transform(param_label, grid, transform, rebuild_airport=False):
        nonlocal stream
        vals = []
        for v in grid:
            c = transform(cfg, v)
            a = Airport(c) if rebuild_airport else airport
            mw, sp = _run_cell(a, c, ec, stream, duration_s)
            stream += 100
            record(param_label, v, mw, sp)
            vals.append(mw)
        return list(grid), vals

    tornado: dict[str, tuple[list, list]] = {}
    # --- algorithm parameters --------------------------------------------
    tornado["xi (specialization gain)"] = sweep_cfg_field("xi", "xi", ec.xi_sens_grid)
    tornado["phi (forgetting rate)"] = sweep_cfg_field("phi", "phi", ec.phi_sens_grid)
    tornado["n (Hill exponent)"] = sweep_cfg_field("n_exp", "n_exp", ec.n_exp_sens_grid)
    tornado["w_d (distance weight)"] = sweep_cfg_field("w_d", "w_d", ec.w_d_sens_grid)

    def theta_bounds(c, mult):
        return replace(c, theta_min=max(0.5, Config().theta_min / mult),
                       theta_max=Config().theta_max * mult)
    tornado["theta bounds (spread ×)"] = sweep_transform(
        "theta_bounds_mult", ec.theta_spread_sens_grid, theta_bounds)

    # --- environment parameters -------------------------------------------
    tornado["fleet_size"] = sweep_cfg_field("fleet_size", "fleet_size", ec.fleet_sens_grid)

    demand_vals = []
    for mult in ec.demand_mult_sens_grid:
        mw, sp = _run_cell(airport, cfg, ec, stream, duration_s,
                           rate_multiplier=lambda t, m=mult: m)
        stream += 100
        record("demand_multiplier", mult, mw, sp)
        demand_vals.append(mw)
    tornado["demand rate (×)"] = (list(ec.demand_mult_sens_grid), demand_vals)

    def n_chargers(c, k):
        return replace(c, n_charging_stations=k)
    tornado["n_charging_stations"] = sweep_transform(
        "n_charging_stations", ec.n_chargers_sens_grid, n_chargers, rebuild_airport=True)

    baseline_wait, _ = _run_cell(airport, cfg, ec, stream, duration_s)
    stream += 100

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "sensitivity.csv", index=False)

    names = list(tornado)
    lo = [min(v) for _, v in tornado.values()]
    hi = [max(v) for _, v in tornado.values()]
    plotting.plot_tornado(names, lo, hi, baseline_wait,
                          figures_dir / "sensitivity_tornado.png")

    # --- 2D interaction sweep: xi x demand multiplier, heatmap ------------
    x_grid = list(ec.heatmap_xi_grid)
    y_grid = list(ec.heatmap_demand_grid)
    z_mean = np.zeros((len(y_grid), len(x_grid)))
    for iy, mult in enumerate(y_grid):
        for ix, xi_val in enumerate(x_grid):
            c = replace(cfg, xi=xi_val)
            mw, _ = _run_cell(airport, c, ec, stream, duration_s,
                              rate_multiplier=lambda t, m=mult: m)
            stream += 100
            z_mean[iy, ix] = mw
    plotting.plot_sensitivity_heatmap(
        x_grid, y_grid, "ξ (specialization gain)", "Demand multiplier (×)",
        z_mean, "Mean waiting time (s)",
        figures_dir / "sensitivity_heatmap_xi_demand.png", n_seeds=ec.sens_seeds)

    ranges = {name: float(max(v) - min(v)) for name, (_, v) in tornado.items()}
    most_impactful = max(ranges, key=ranges.get)
    return {
        "n_seeds_per_cell": ec.sens_seeds,
        "sweep_window_hours": ec.sens_hours,
        "baseline_mean_wait_s": round(baseline_wait, 2),
        "parameter_impact_ranges_s": {k: round(v, 2) for k, v in ranges.items()},
        "most_impactful_parameter": most_impactful,
    }
