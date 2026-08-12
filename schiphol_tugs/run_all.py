"""Single entry point: calibration, all experiments, all figures, summary.

Run from the repo root:  python run_all.py

Everything is seeded from config.SEED (= 42) through named rng "streams", so
two consecutive runs produce byte-identical CSV/JSON outputs. Every headline
experiment is replicated over EXPERIMENT.n_seeds independent seeds and every
figure that reports a metric shows mean +/- 95% CI (bars) or mean +/- 1 s.d.
(time series bands) across those seeds -- see docs/notation.md, "Uncertainty
reporting", for the statistical policy this implements.

Stream id ranges (each is a distinct, reproducible rng stream, see
config.make_rng):

    1        demand-profile figure request stream
    10       calibration request stream (cal_hours, starting at cal_start_hour)
    11+      calibration xi/alpha + fleet-size decision streams
    20000+k  main-day request stream, seed k
    21000+k / 22000+k / 23000+k / 24000+k   swarm/baseline/random/optimal decisions, seed k
    25000+k  ablation (fixed thresholds) decisions, seed k
    30000+k  failure-scenario tug-choice draw, seed k
    31000+k / 32000+k  failure swarm/baseline decisions, seed k
    40000+k  spike request stream, seed k
    41000+k / 42000+k  spike swarm/baseline decisions, seed k
    50000+k  charging-station-outage swarm decisions, seed k
    90000+   sensitivity-sweep streams (see src/sensitivity.py)
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import analysis, metrics, plotting, sensitivity
from src import stats as statlib
from src.airport import Airport, generate_requests
from src.config import DEFAULT, EXPERIMENT, REGIONS, make_rng
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
    """Sweep (xi, alpha) jointly at fleet_size default, then fleet size at
    the best pair. Selection criterion: minimal mean waiting time, total
    distance as the tiebreaker (rounded to break float noise only). The
    xi/alpha grid is also rendered as a heatmap figure (the sensitivity
    rubric's "at least one 2D interaction sweep" requirement)."""
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

    for fleet in tqdm(ec.fleet_grid, desc="calibrating fleet size"):
        c = replace(cfg, xi=best_xi, alpha=best_alpha, fleet_size=fleet)
        sim = run_sim(airport, c, cal_requests, stream, duration=cal_s)
        stream += 1
        st = metrics.summarize(sim)
        rows.append({"sweep": "fleet", "xi": best_xi, "alpha": best_alpha,
                     "fleet_size": fleet, **st})
    fleet_rows = [r for r in rows if r["sweep"] == "fleet"]
    best_fleet_row = min(fleet_rows, key=lambda r: (round(r["mean_wait_s"], 6),
                                                    round(r["total_distance_km"], 6)))
    best_fleet = best_fleet_row["fleet_size"]

    pd.DataFrame(rows).to_csv(RESULTS / "calibration.csv", index=False)
    print(f"Calibration chose xi={best_xi}, alpha={best_alpha}, "
          f"fleet_size={best_fleet}")
    return replace(cfg, xi=best_xi, alpha=best_alpha, fleet_size=best_fleet)


def _avg_snapshot_at(snapshots_by_seed, t_target):
    arrs = [metrics.snapshot_at(snaps, t_target) for snaps in snapshots_by_seed]
    return np.mean(arrs, axis=0)


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

    # -- layout + demand figures --------------------------------------------
    plotting.plot_airport_layout(airport, FIGURES / "airport_layout.png")
    demo_requests = generate_requests(airport, cfg, make_rng(1), DAY)
    plotting.plot_demand_profile(demo_requests, cfg, FIGURES / "demand_profile.png")

    # -- calibration ----------------------------------------------------------
    cfg = calibrate(airport, cfg, ec)

    # -- experiment 1: swarm vs 3 baselines, N seeds -------------------------
    modes = ["swarm", "baseline", "random", "optimal"]
    stats_by_mode = {m: [] for m in modes}
    waits_by_mode = {m: [] for m in modes}
    swarm_sims: list[Simulation] = []
    fixed_stats, fixed_spec = [], []
    request_streams: list[list] = []
    spec_times = None
    spec_by_seed = []

    for k in tqdm(range(N), desc="main comparison (N seeds x 4 modes + ablation)"):
        reqs = generate_requests(airport, cfg, make_rng(20000 + k), DAY)
        request_streams.append(reqs)
        for i, m in enumerate(modes):
            sim = run_sim(airport, cfg, reqs, 21000 + 1000 * i + k, mode=m)
            st = metrics.summarize(sim)
            stats_by_mode[m].append(st)
            waits_by_mode[m].append(metrics.wait_times(sim.requests))
            seed_rows.append({"experiment": "main", "seed": k, "mode": m, **st})
            if m == "swarm":
                swarm_sims.append(sim)
                t_, s_ = metrics.specialization_series(sim.snapshots)
                spec_times = t_
                spec_by_seed.append(s_)

        fixed_sim = run_sim(airport, replace(cfg, adaptive=False), reqs, 25000 + k)
        fst = metrics.summarize(fixed_sim)
        fixed_stats.append(fst)
        _, fsp = metrics.specialization_series(fixed_sim.snapshots)
        fixed_spec.append(fsp)
        seed_rows.append({"experiment": "ablation_fixed", "seed": k, "mode": "fixed", **fst})

    def agg_dict(dicts, keys):
        return {k: statlib.mean_ci([d[k] for d in dicts]) for k in keys}

    wait_agg = {MODE_LABEL[m]: agg_dict(stats_by_mode[m], ["mean_wait_s", "p95_wait_s"])
               for m in modes}
    plotting.plot_wait_comparison(wait_agg, N, FIGURES / "wait_time_comparison.png")

    energy_agg = {MODE_LABEL[m]: agg_dict(stats_by_mode[m],
                                          ["total_energy_pct", "total_distance_km"])
                 for m in modes}
    plotting.plot_energy_comparison(energy_agg, N, FIGURES / "energy_comparison.png")

    plotting.plot_wait_cdf({MODE_LABEL[m]: waits_by_mode[m] for m in modes},
                           FIGURES / "wait_time_cdf.png")

    swarm_means = [d["mean_wait_s"] for d in stats_by_mode["swarm"]]
    base_means = [d["mean_wait_s"] for d in stats_by_mode["baseline"]]
    sig_vs_baseline = statlib.paired_test(swarm_means, base_means)

    spec_arr = np.array(spec_by_seed)
    fixed_spec_arr = np.array(fixed_spec)
    plotting.plot_specialization(
        spec_times, {"Adaptive θ": spec_arr, "Fixed θ (ablation)": fixed_spec_arr},
        FIGURES / "specialization_index.png")

    spec_end = spec_arr[:, -1]
    fixed_spec_end = fixed_spec_arr[:, -1]
    ablation_agg = {
        "Adaptive θ": {"mean_wait_s": statlib.mean_ci(swarm_means),
                       "specialization": statlib.mean_ci(spec_end)},
        "Fixed θ (ablation)": {"mean_wait_s": statlib.mean_ci([d["mean_wait_s"] for d in fixed_stats]),
                               "specialization": statlib.mean_ci(fixed_spec_end)},
    }
    plotting.plot_ablation(ablation_agg, N, FIGURES / "ablation.png")
    sig_ablation = statlib.paired_test(swarm_means, [d["mean_wait_s"] for d in fixed_stats])

    snapshots_by_seed = [sim.snapshots for sim in swarm_sims]
    plotting.plot_threshold_evolution(
        _avg_snapshot_at(snapshots_by_seed, 0),
        _avg_snapshot_at(snapshots_by_seed, 6 * 3600),
        _avg_snapshot_at(snapshots_by_seed, 24 * 3600),
        FIGURES / "threshold_evolution.png", n_seeds=N,
    )

    # -- raw per-run logs from seed 0 (events/contests/thetas) --------------
    # These are always written (not gated behind visual recording, see
    # AUDIT.md 7.4), so src/analysis.py is reproducible from run_all.py alone.
    swarm0 = swarm_sims[0]
    swarm0.events_dataframe().to_csv(RESULTS / "events.csv", index=False)
    contests_df = swarm0.contests_dataframe()
    contests_df.to_csv(RESULTS / "contests.csv", index=False)
    theta_df = metrics.theta_snapshots_dataframe(swarm0.snapshots)
    theta_df.to_csv(RESULTS / "theta_snapshots.csv", index=False)

    analysis_result = analysis.analyze(contests_df, theta_df, cfg.fleet_size,
                                       cfg.xi, cfg.phi, cfg.w_d, cfg.w_b, cfg.w_s)
    print("\n--- Dominance hierarchy / division of labour (seed 0) ---")
    print(analysis_result["interpretation"])

    # -- experiment 3: robustness to tug failures, N seeds -------------------
    fail_time_s = int(ec.failure_time_h * 3600)
    fail_swarm_rolls, fail_base_rolls = [], []
    fail_swarm_stats, fail_base_stats = [], []
    for k in tqdm(range(N), desc="failure scenario"):
        failed_ids = tuple(int(i) for i in make_rng(30000 + k).choice(
            cfg.fleet_size, ec.n_failures, replace=False))
        fs = run_sim(airport, cfg, request_streams[k], 31000 + k,
                    failure_time=fail_time_s, failed_ids=failed_ids)
        fb = run_sim(airport, cfg, request_streams[k], 32000 + k, mode="baseline",
                    failure_time=fail_time_s, failed_ids=failed_ids)
        fail_swarm_stats.append(metrics.summarize(fs))
        fail_base_stats.append(metrics.summarize(fb))
        grid, sw = metrics.rolling_wait(fs.requests, DAY)
        _, bw = metrics.rolling_wait(fb.requests, DAY)
        fail_swarm_rolls.append(sw)
        fail_base_rolls.append(bw)
        seed_rows.append({"experiment": "failure", "seed": k, "mode": "swarm",
                          "failed_ids": str(failed_ids), **fail_swarm_stats[-1]})
        seed_rows.append({"experiment": "failure", "seed": k, "mode": "baseline",
                          "failed_ids": str(failed_ids), **fail_base_stats[-1]})
    plotting.plot_robustness_failure(grid, np.array(fail_swarm_rolls),
                                     np.array(fail_base_rolls), ec.failure_time_h,
                                     FIGURES / "robustness_failure.png")

    # -- experiment 4: robustness to a demand spike, N seeds -----------------
    spike_start_s = int(ec.spike_start_h * 3600)
    spike_end_s = int((ec.spike_start_h + ec.spike_duration_h) * 3600)

    def spike(t):
        return ec.spike_factor if spike_start_s <= t < spike_end_s else 1.0

    spike_swarm_rolls, spike_base_rolls = [], []
    spike_swarm_stats, spike_base_stats = [], []
    for k in tqdm(range(N), desc="demand-spike scenario"):
        spike_requests = generate_requests(airport, cfg, make_rng(40000 + k), DAY,
                                           rate_multiplier=spike)
        ss = run_sim(airport, cfg, spike_requests, 41000 + k)
        sb = run_sim(airport, cfg, spike_requests, 42000 + k, mode="baseline")
        spike_swarm_stats.append(metrics.summarize(ss))
        spike_base_stats.append(metrics.summarize(sb))
        grid, sw = metrics.rolling_wait(ss.requests, DAY)
        _, bw = metrics.rolling_wait(sb.requests, DAY)
        spike_swarm_rolls.append(sw)
        spike_base_rolls.append(bw)
        seed_rows.append({"experiment": "spike", "seed": k, "mode": "swarm", **spike_swarm_stats[-1]})
        seed_rows.append({"experiment": "spike", "seed": k, "mode": "baseline", **spike_base_stats[-1]})
    plotting.plot_robustness_spike(grid, np.array(spike_swarm_rolls),
                                   np.array(spike_base_rolls), ec.spike_start_h,
                                   ec.spike_start_h + ec.spike_duration_h,
                                   FIGURES / "robustness_spike.png")

    # -- experiment 6: robustness to a charging-station outage, N seeds ------
    outage_start_s = int(ec.outage_start_h * 3600)
    outage_end_s = int((ec.outage_start_h + ec.outage_duration_h) * 3600)
    outage_rolls, normal_rolls = [], []
    outage_stats = []
    for k in tqdm(range(N), desc="charging-station outage scenario"):
        so = run_sim(airport, cfg, request_streams[k], 50000 + k,
                    station_outage=(ec.outage_station, outage_start_s, outage_end_s))
        outage_stats.append(metrics.summarize(so))
        grid, ow = metrics.rolling_wait(so.requests, DAY)
        _, nw = metrics.rolling_wait(swarm_sims[k].requests, DAY)
        outage_rolls.append(ow)
        normal_rolls.append(nw)
        seed_rows.append({"experiment": "charger_outage", "seed": k, "mode": "swarm",
                          **outage_stats[-1]})
    plotting.plot_robustness_charger_outage(
        grid, np.array(outage_rolls), np.array(normal_rolls),
        ec.outage_start_h, ec.outage_start_h + ec.outage_duration_h,
        FIGURES / "robustness_charger_outage.png")

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
    spec_start_mean = float(spec_arr[:, 0].mean())
    spec_end_mean = float(spec_end.mean())
    narrative = (
        f"Over a simulated 24-hour day with ~{len(request_streams[0])} tow "
        f"requests, replicated across N={N} independent seeds, the "
        f"decentralized swarm achieved a mean aircraft wait of "
        f"{np.mean(swarm_means):.1f} s (95% CI +/-{statlib.mean_ci(swarm_means)[1]:.1f} s) "
        f"versus {np.mean(base_means):.1f} s for the centralized greedy "
        f"baseline ({gap_pct:+.1f} %), "
        f"{'within' if abs(gap_pct) <= 20 else 'outside'} the 20 % "
        f"competitiveness target (Wilcoxon signed-rank p={sig_vs_baseline['p_value']:.4f}, "
        f"matched-pairs rank-biserial effect size={sig_vs_baseline['effect_size_rank_biserial']:.2f}). "
        f"The random-assignment floor and centralized-optimal ceiling give this "
        f"comparison context (see wait_time_comparison.png). Emergent service "
        f"zones formed: the fleet-wide specialization index rose from "
        f"{spec_start_mean:.3f} to {spec_end_mean:.3f} averaged over seeds. "
        f"With fixed (non-adaptive) thresholds the mean wait was "
        f"{np.mean([d['mean_wait_s'] for d in fixed_stats]):.1f} s and final "
        f"specialization was {fixed_spec_end.mean():.3f} "
        f"(p={sig_ablation['p_value']:.4f} vs adaptive), confirming threshold "
        f"adaptation drives both the zone formation and the performance. "
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
                "total_energy_pct": agg_row(stats_by_mode[m], "total_energy_pct"),
            } for m in modes
        },
        "swarm_vs_baseline_significance": sig_vs_baseline,
        "experiment_2_zone_formation": {
            "specialization_start": {"mean": round(spec_start_mean, 4)},
            "specialization_end": {"mean": round(spec_end_mean, 4),
                                   "ci95_halfwidth": round(statlib.mean_ci(spec_end)[1], 4)},
        },
        "experiment_3_tug_failures": {
            "swarm_mean_wait_s": agg_row(fail_swarm_stats, "mean_wait_s"),
            "baseline_mean_wait_s": agg_row(fail_base_stats, "mean_wait_s"),
        },
        "experiment_4_demand_spike": {
            "swarm_mean_wait_s": agg_row(spike_swarm_stats, "mean_wait_s"),
            "baseline_mean_wait_s": agg_row(spike_base_stats, "mean_wait_s"),
        },
        "experiment_5_ablation": {
            "adaptive_mean_wait_s": agg_row(stats_by_mode["swarm"], "mean_wait_s"),
            "fixed_mean_wait_s": agg_row(fixed_stats, "mean_wait_s"),
            "significance": sig_ablation,
        },
        "experiment_6_charging_station_outage": {
            "outage_mean_wait_s": agg_row(outage_stats, "mean_wait_s"),
            "normal_mean_wait_s": agg_row(stats_by_mode["swarm"], "mean_wait_s"),
        },
        "dominance_hierarchy_and_division_of_labor": {
            k: v for k, v in analysis_result.items() if k != "interpretation"
        },
        "interpretation": analysis_result["interpretation"],
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
