"""Single entry point: calibration, baseline comparison, sensitivity, summary.

Run from the repo root:  python run_all.py

Everything is seeded from config.SEED (= 42) through named rng "streams", so
two consecutive runs produce byte-identical CSV/JSON outputs. Every headline
experiment is replicated over EXPERIMENT.n_seeds independent seeds and every
figure that reports a metric shows mean +/- 95% CI -- see docs/notation.md,
"Uncertainty reporting", for the statistical policy this implements.

Stream id ranges (each is a distinct, reproducible rng stream, see
config.make_rng):

    10       calibration request stream (cal_hours, starting at cal_start_hour)
    11+      calibration xi/alpha decision streams
    20000+k  main-day request stream, seed k
    21000+k / 22000+k / 23000+k / 24000+k   swarm/baseline/random/optimal decisions, seed k
    90000+   sensitivity-sweep streams (see src/sensitivity.py)
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import metrics, plotting, sensitivity
from src import stats as statlib
from src.airport import Airport, generate_requests
from src.config import DEFAULT, EXPERIMENT, make_rng
from src.simulator import Simulation

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

DAY = 24 * 3600

MODE_LABEL = {
    "swarm": "Swarm (VRTM + wasp)",
    "baseline": "Centralized baseline",
    "random": "Random assignment",
    "optimal": "Centralized optimal",
}


def run_sim(airport, cfg, requests, stream, mode="swarm", duration=DAY, **kw):
    sim = Simulation(airport, cfg, requests, make_rng(stream), mode=mode,
                     duration_s=duration, **kw)
    return sim.run()


# ---------------------------------------------------------------------------
def calibrate(airport, cfg, ec):
    """Sweep (xi, alpha) jointly at the fixed fleet size (29, literature-sourced
    -- not a decision variable, see config.Config.fleet_size). Selection
    criterion: minimal mean waiting time, total distance as the tiebreaker
    (rounded to break float noise only). The xi/alpha grid is also rendered
    as a heatmap figure (the sensitivity rubric's "at least one 2D
    interaction sweep" requirement)."""
    rows = []
    cal_s = int(ec.cal_hours * 3600)
    cal_requests = generate_requests(airport, cfg, make_rng(10), cal_s,
                                     start_hour=ec.cal_start_hour)

    grid = [(xi, alpha) for xi in ec.xi_grid for alpha in ec.alpha_grid]
    stream = 11
    for xi, alpha in tqdm(grid, desc="calibrating xi/alpha"):
        c = replace(cfg, xi=xi, alpha=alpha)
        sim = run_sim(airport, c, cal_requests, stream, duration=cal_s)
        stream += 1
        st = metrics.summarize(sim)
        rows.append({"sweep": "xi_alpha", "xi": xi, "alpha": alpha,
                     "fleet_size": c.fleet_size, **st})
    best = min(rows, key=lambda r: (round(r["mean_wait_s"], 6),
                                    round(r["total_distance_km"], 6)))
    best_xi, best_alpha = best["xi"], best["alpha"]

    xi_vals, alpha_vals = list(ec.xi_grid), list(ec.alpha_grid)
    lookup = {(r["xi"], r["alpha"]): r["mean_wait_s"] for r in rows if r["sweep"] == "xi_alpha"}
    z = np.array([[lookup[(xi, a)] for xi in xi_vals] for a in alpha_vals])
    plotting.plot_sensitivity_heatmap(
        xi_vals, alpha_vals, "ξ (specialization gain)", "α (stimulus growth, s/s)",
        z, f"Mean waiting time (s), {ec.cal_hours:g} h calibration window",
        FIGURES / "calibration_heatmap.png", n_seeds=1)

    pd.DataFrame(rows).to_csv(RESULTS / "calibration.csv", index=False)
    print(f"Calibration chose xi={best_xi}, alpha={best_alpha} "
          f"(fleet_size fixed at {cfg.fleet_size})")
    return replace(cfg, xi=best_xi, alpha=best_alpha)


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    cfg = DEFAULT
    ec = EXPERIMENT
    N = ec.n_seeds
    airport = Airport(cfg)
    seed_rows: list[dict] = []  # raw per-seed results -> results/seed_results.csv

    # -- layout figure -------------------------------------------------------
    plotting.plot_airport_layout(airport, FIGURES / "airport_layout.png")

    # -- calibration ----------------------------------------------------------
    cfg = calibrate(airport, cfg, ec)

    # -- experiment 1: swarm vs 3 baselines, N seeds -------------------------
    modes = ["swarm", "baseline", "random", "optimal"]
    stats_by_mode = {m: [] for m in modes}

    for k in tqdm(range(N), desc="main comparison (N seeds x 4 modes)"):
        reqs = generate_requests(airport, cfg, make_rng(20000 + k), DAY)
        for i, m in enumerate(modes):
            sim = run_sim(airport, cfg, reqs, 21000 + 1000 * i + k, mode=m)
            st = metrics.summarize(sim)
            stats_by_mode[m].append(st)
            seed_rows.append({"experiment": "main", "seed": k, "mode": m, **st})

    def agg_dict(dicts, keys):
        return {k: statlib.mean_ci([d[k] for d in dicts]) for k in keys}

    wait_agg = {MODE_LABEL[m]: agg_dict(stats_by_mode[m], ["mean_wait_s", "p95_wait_s"])
               for m in modes}
    plotting.plot_wait_comparison(wait_agg, N, FIGURES / "wait_time_comparison.png")

    energy_agg = {MODE_LABEL[m]: agg_dict(stats_by_mode[m],
                                          ["total_energy_kwh", "total_distance_km"])
                 for m in modes}
    plotting.plot_energy_comparison(energy_agg, N, FIGURES / "energy_comparison.png")

    swarm_means = [d["mean_wait_s"] for d in stats_by_mode["swarm"]]
    base_means = [d["mean_wait_s"] for d in stats_by_mode["baseline"]]
    sig_vs_baseline = statlib.paired_test(swarm_means, base_means)

    # -- sensitivity analysis -------------------------------------------------
    print("\nRunning sensitivity sweeps...")
    sens_summary = sensitivity.run_sensitivity(airport, cfg, ec, RESULTS, FIGURES)

    # -- persist raw per-seed results -----------------------------------------
    pd.DataFrame(seed_rows).to_csv(RESULTS / "seed_results.csv", index=False)

    # -- summary --------------------------------------------------------------
    def agg_row(dicts, key):
        m, ci = statlib.mean_ci([d[key] for d in dicts])
        return {"mean": round(m, 2), "ci95_halfwidth": round(ci, 2) if ci == ci else None}

    gap_pct = 100.0 * (np.mean(swarm_means) / np.mean(base_means) - 1.0)
    narrative = (
        f"Over a simulated 24-hour day, replicated across N={N} independent "
        f"seeds, the decentralized swarm achieved a mean aircraft wait of "
        f"{np.mean(swarm_means):.1f} s (95% CI +/-{statlib.mean_ci(swarm_means)[1]:.1f} s) "
        f"versus {np.mean(base_means):.1f} s for the centralized greedy "
        f"baseline ({gap_pct:+.1f} %), "
        f"{'within' if abs(gap_pct) <= 20 else 'outside'} the 20 % "
        f"competitiveness target (Wilcoxon signed-rank p={sig_vs_baseline['p_value']:.4f}, "
        f"matched-pairs rank-biserial effect size={sig_vs_baseline['effect_size_rank_biserial']:.2f}). "
        f"The random-assignment floor and centralized-optimal ceiling give this "
        f"comparison context (see wait_time_comparison.png). "
        f"Sensitivity analysis identifies '{sens_summary['most_impactful_parameter']}' "
        f"as the parameter with the largest impact on mean wait "
        f"(see sensitivity_tornado.png)."
    )
    summary = {
        "n_seeds": N,
        "calibration": {"xi": cfg.xi, "alpha": cfg.alpha, "fleet_size": cfg.fleet_size},
        "experiment_1_dispatch_policy_comparison": {
            MODE_LABEL[m]: {
                "mean_wait_s": agg_row(stats_by_mode[m], "mean_wait_s"),
                "p95_wait_s": agg_row(stats_by_mode[m], "p95_wait_s"),
                "total_distance_km": agg_row(stats_by_mode[m], "total_distance_km"),
                "total_energy_kwh": agg_row(stats_by_mode[m], "total_energy_kwh"),
            } for m in modes
        },
        "swarm_vs_baseline_significance": sig_vs_baseline,
        "sensitivity": sens_summary,
        "narrative": narrative,
    }
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n" + "=" * 72)
    print(narrative)
    print("=" * 72)
    print(f"\nDone in {time.time() - t_start:.1f} s. "
          f"Figures in {FIGURES}, data in {RESULTS}.")


if __name__ == "__main__":
    main()
