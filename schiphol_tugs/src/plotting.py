"""All figures for the project (static PNGs at 300 DPI).

Colors follow a validated brand-neutral palette: categorical slot 1 (blue)
for the swarm, slot 2 (aqua) for the centralized baseline, slot 5 (violet)
and slot 3 (yellow) for the extra random/optimal baselines, a single-hue
blue sequential ramp for heatmap magnitude, and the reserved status red only
for failure/disruption event markers.  Chrome (grid, axes, labels) stays in
recessive grays so the data ink dominates.

Every multi-seed figure shows mean +/- 95% CI (bars: error bars; time series:
shaded band of +/- 1 s.d. across seeds) and states N in its title, per the
project's statistics policy (see docs/notation.md, "Uncertainty reporting").
"""

import matplotlib

matplotlib.use("Agg")  # headless rendering; run_all works without a display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import REGIONS

DPI = 300

# ---- palette (light mode) --------------------------------------------------
C_SWARM = "#2a78d6"      # categorical slot 1: blue
C_BASELINE = "#1baf7a"   # categorical slot 2: aqua (sub-3:1 => direct labels)
C_RANDOM = "#c3423f"     # categorical slot 4: warm red-brown (random floor)
C_ACCENT = "#4a3aa7"     # categorical slot 5: violet (charging stations / optimal)
C_YELLOW = "#eda100"     # categorical slot 3 (runway threshold markers)
C_EVENT = "#d03b3b"      # status critical: disruption markers only
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Mode -> color, used everywhere a figure compares dispatch policies.
MODE_COLOR = {
    "Swarm (VRTM + wasp)": C_SWARM,
    "Centralized baseline": C_BASELINE,
    "Random assignment": C_RANDOM,
    "Centralized optimal": C_ACCENT,
    "Adaptive θ": C_SWARM,
    "Fixed θ (ablation)": C_BASELINE,
}

# Single-hue sequential ramp (blue steps 100 -> 700) for heatmaps.
_SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQ_STEPS)
# Diverging ramp for tornado / sensitivity plots (low -> neutral -> high).
_DIV_STEPS = ["#1baf7a", "#bfe3d3", "#f4f3ef", "#f6cf9e", "#d0742a"]
DIV_CMAP = LinearSegmentedColormap.from_list("div_seq", _DIV_STEPS)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE_AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
})


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _bar_labels(ax, bars, fmt="{:.0f}"):
    """Direct value labels on bars (relief rule for low-contrast fills)."""
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK_2,
                    xytext=(0, 2), textcoords="offset points")


def _color_for(label, fallback_cycle=(C_SWARM, C_BASELINE, C_RANDOM, C_ACCENT)):
    if label in MODE_COLOR:
        return MODE_COLOR[label]
    return fallback_cycle[hash(label) % len(fallback_cycle)]


# ---------------------------------------------------------------------------
def plot_airport_layout(airport, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    from .airport import PIERS, RUNWAYS  # static layout tables

    # Taxiway edges as a faint background network.
    for a, b in airport.graph.edges:
        xa, ya = airport.xy(a)
        xb, yb = airport.xy(b)
        ax.plot([xa, xb], [ya, yb], color=GRID, lw=0.9, zorder=1)

    # Runways as heavy dark lines with name labels.
    for name, (na, xya, nb, xyb) in RUNWAYS.items():
        ax.plot([xya[0], xyb[0]], [xya[1], xyb[1]], color=INK_2, lw=4,
                solid_capstyle="round", zorder=2)
        mx, my = (xya[0] + xyb[0]) / 2, (xya[1] + xyb[1]) / 2
        ax.annotate(name, (mx, my), fontsize=8, color=INK,
                    xytext=(6, 6), textcoords="offset points")

    gates = airport.all_gates()
    gx = [airport.xy(g)[0] for g in gates]
    gy = [airport.xy(g)[1] for g in gates]
    ax.scatter(gx, gy, s=14, color=C_SWARM, zorder=3, label="Gates")
    tx = [airport.xy(n)[0] for n in airport.runway_thresholds]
    ty = [airport.xy(n)[1] for n in airport.runway_thresholds]
    ax.scatter(tx, ty, s=30, marker="s", color=C_YELLOW, edgecolor=INK_2,
               linewidth=0.5, zorder=4, label="Runway thresholds")
    cx = [airport.xy(c)[0] for c in airport.charging_stations]
    cy = [airport.xy(c)[1] for c in airport.charging_stations]
    ax.scatter(cx, cy, s=90, marker="^", color=C_ACCENT, zorder=5,
               label="Charging stations")

    for pier, (hub, _) in PIERS.items():
        ax.annotate(f"Pier {pier}", hub, fontsize=9, color=INK,
                    fontweight="bold", xytext=(-4, -16),
                    textcoords="offset points")

    ax.set_xlabel("x (m, east of terminal)")
    ax.set_ylabel("y (m, north of terminal)")
    ax.set_title("Simplified Schiphol layout: piers, runways, taxiways, chargers")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    _save(fig, path)


def plot_demand_profile(requests, cfg, path):
    hours = np.arange(24)
    counts = np.zeros(24)
    for r in requests:
        counts[int(r.time // 3600) % 24] += 1
    lam = [cfg.hourly_rate(h + 0.5) for h in hours]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hours + 0.5, counts, width=0.9, color=C_SWARM, label="Realized requests")
    ax.plot(hours + 0.5, lam, color=INK, lw=2, label="Expected rate λ(t)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Tow requests per hour")
    ax.set_title("Diurnal demand profile (Poisson tow-request stream)")
    ax.set_xticks(range(0, 25, 3))
    ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------- bars
def _grouped_ci_bars(ax, agg, metric_keys, metric_labels, fmt="{:.0f}"):
    """agg: {label: {metric_key: (mean, ci_halfwidth)}}. Draws one group of
    bars per metric, one bar per label within the group, with error bars."""
    labels = list(agg)
    x = np.arange(len(metric_keys))
    n = len(labels)
    w = 0.8 / max(n, 1)
    for i, label in enumerate(labels):
        means = [agg[label][m][0] for m in metric_keys]
        cis = [0.0 if np.isnan(agg[label][m][1]) else agg[label][m][1] for m in metric_keys]
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, means, w, yerr=cis, capsize=3,
                      color=_color_for(label), label=label,
                      error_kw={"elinewidth": 1.1, "ecolor": INK_2})
        _bar_labels(ax, bars, fmt)
    ax.set_xticks(x, metric_labels)
    return labels


def plot_wait_comparison(agg, n_seeds, path):
    """agg: {label: {"mean_wait_s": (mean, ci), "p95_wait_s": (mean, ci)}}"""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _grouped_ci_bars(ax, agg, ["mean_wait_s", "p95_wait_s"],
                     ["Mean wait", "95th-pct wait"])
    ax.set_ylabel("Waiting time (s)")
    ax.set_title(f"Aircraft waiting time by dispatch policy "
                 f"(mean ± 95% CI, N = {n_seeds} seeds)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_energy_comparison(agg, n_seeds, path):
    """agg: {label: {"total_energy_pct": (mean, ci), "total_distance_km": (mean, ci)}}"""
    labels = list(agg)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, ylabel, title in [
        (axes[0], "total_energy_pct", "Battery percentage-points consumed (fleet total)", "Energy proxy"),
        (axes[1], "total_distance_km", "Total fleet distance (km)", "Distance driven"),
    ]:
        means = [agg[l][key][0] for l in labels]
        cis = [0.0 if np.isnan(agg[l][key][1]) else agg[l][key][1] for l in labels]
        colors = [_color_for(l) for l in labels]
        bars = ax.bar(labels, means, width=0.55, yerr=cis, capsize=3, color=colors,
                      error_kw={"elinewidth": 1.1, "ecolor": INK_2})
        _bar_labels(ax, bars)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle(f"Fleet energy and distance over the 24 h day "
                 f"(mean ± 95% CI, N = {n_seeds} seeds)", color=INK)
    fig.tight_layout()
    _save(fig, path)


def plot_wait_cdf(waits_by_label, path):
    """waits_by_label: {label: [wait_array_seed0, wait_array_seed1, ...]}."""
    all_max = max((w.max() for arrs in waits_by_label.values() for w in arrs if w.size),
                  default=1.0)
    grid = np.linspace(0.0, all_max, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    n_seeds = 0
    for label, arrs in waits_by_label.items():
        n_seeds = max(n_seeds, len(arrs))
        cdfs = np.array([[(w <= g).mean() if w.size else np.nan for g in grid]
                         for w in arrs])
        mean_cdf = np.nanmean(cdfs, axis=0)
        std_cdf = np.nanstd(cdfs, axis=0)
        color = _color_for(label)
        ax.plot(grid, mean_cdf, color=color, lw=2, label=label)
        ax.fill_between(grid, np.clip(mean_cdf - std_cdf, 0, 1),
                        np.clip(mean_cdf + std_cdf, 0, 1), color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Waiting time (s)")
    ax.set_ylabel("Fraction of requests served within")
    ax.set_title(f"Waiting-time CDF (mean ± 1 s.d. across N = {n_seeds} seeds)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def plot_threshold_evolution(snap0, snap6, snap24, path, n_seeds=1):
    snaps = [(snap0, "t = 0 h"), (snap6, "t = 6 h"), (snap24, "t = 24 h")]
    vmax = max(s.max() for s, _ in snaps)
    vmin = min(s.min() for s, _ in snaps)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (snap, label) in zip(axes, snaps):
        im = ax.imshow(snap, aspect="auto", cmap=SEQ_CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(REGIONS)), REGIONS)
        ax.set_title(label, color=INK)
        ax.set_xlabel("Pier region")
        ax.grid(False)
    axes[0].set_ylabel("Tug id")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Response threshold θ (low = specialist)")
    seed_note = f", averaged over N = {n_seeds} seeds" if n_seeds > 1 else ""
    fig.suptitle("Emergent zone formation: per-tug thresholds by pier over the day"
                 + seed_note, color=INK)
    _save(fig, path)


def plot_specialization(times_s, series_by_label, path):
    """series_by_label: {label: 2D array [n_seeds, n_times]} (mean +/- s.d. band)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    n_seeds = 0
    for label, arr in series_by_label.items():
        arr = np.atleast_2d(arr)
        n_seeds = max(n_seeds, arr.shape[0])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = _color_for(label)
        ax.plot(times_s / 3600.0, mean, color=color, lw=2, label=label)
        if arr.shape[0] > 1:
            ax.fill_between(times_s / 3600.0, mean - std, mean + std,
                            color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Simulated time (h)")
    ax.set_ylabel("Specialization index (mean CV of θ across tugs)")
    ax.set_title(f"Fleet specialization rises as service zones emerge "
                 f"(mean ± 1 s.d., N = {n_seeds} seeds)")
    ax.set_xlim(0, times_s[-1] / 3600.0)
    ax.legend()
    _save(fig, path)


def _band(ax, grid_s, arr2d, color, label):
    arr2d = np.atleast_2d(np.asarray(arr2d, dtype=float))
    mean = np.nanmean(arr2d, axis=0)
    ax.plot(grid_s / 3600.0, mean, color=color, lw=2, label=label)
    if arr2d.shape[0] > 1:
        std = np.nanstd(arr2d, axis=0)
        ax.fill_between(grid_s / 3600.0, mean - std, mean + std, color=color,
                        alpha=0.15, lw=0)
    return arr2d.shape[0]


def plot_robustness_failure(grid_s, swarm_2d, base_2d, failure_h, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n1 = _band(ax, grid_s, swarm_2d, C_SWARM, "Swarm (VRTM + wasp)")
    n2 = _band(ax, grid_s, base_2d, C_BASELINE, "Centralized baseline")
    ax.axvline(failure_h, color=C_EVENT, lw=1.5, ls="--")
    ax.annotate("5 tugs fail", (failure_h, ax.get_ylim()[1]), color=C_EVENT,
                fontsize=9, xytext=(6, -14), textcoords="offset points")
    ax.set_xlabel("Simulated time (h)")
    ax.set_ylabel("Rolling mean wait (s, 1 h window)")
    ax.set_title(f"Robustness to tug failures (5 of the fleet lost at 12:00, "
                 f"mean ± 1 s.d., N = {max(n1, n2)} seeds)")
    ax.legend()
    _save(fig, path)


def plot_robustness_spike(grid_s, swarm_2d, base_2d, spike_start_h, spike_end_h, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n1 = _band(ax, grid_s, swarm_2d, C_SWARM, "Swarm (VRTM + wasp)")
    n2 = _band(ax, grid_s, base_2d, C_BASELINE, "Centralized baseline")
    ax.axvspan(spike_start_h, spike_end_h, color=C_EVENT, alpha=0.12, lw=0)
    ax.axvline(spike_start_h, color=C_EVENT, lw=1.5, ls="--")
    ax.annotate("demand ×3 for 30 min", (spike_start_h, ax.get_ylim()[1]),
                color=C_EVENT, fontsize=9, xytext=(6, -14),
                textcoords="offset points")
    ax.set_xlabel("Simulated time (h)")
    ax.set_ylabel("Rolling mean wait (s, 1 h window)")
    ax.set_title(f"Robustness to a demand spike (rate tripled 10:00–10:30, "
                 f"mean ± 1 s.d., N = {max(n1, n2)} seeds)")
    ax.legend()
    _save(fig, path)


def plot_robustness_charger_outage(grid_s, outage_2d, normal_2d, start_h, end_h, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n1 = _band(ax, grid_s, outage_2d, C_SWARM, "Swarm, station CS1 down")
    n2 = _band(ax, grid_s, normal_2d, C_ACCENT, "Swarm, all stations up")
    ax.axvspan(start_h, end_h, color=C_EVENT, alpha=0.12, lw=0)
    ax.axvline(start_h, color=C_EVENT, lw=1.5, ls="--")
    ax.annotate("CS1 outage", (start_h, ax.get_ylim()[1]), color=C_EVENT,
                fontsize=9, xytext=(6, -14), textcoords="offset points")
    ax.set_xlabel("Simulated time (h)")
    ax.set_ylabel("Rolling mean wait (s, 1 h window)")
    ax.set_title(f"Robustness to a charging-station outage "
                 f"(mean ± 1 s.d., N = {max(n1, n2)} seeds)")
    ax.legend()
    _save(fig, path)


def plot_ablation(agg, n_seeds, path):
    """agg: {"Adaptive θ": {"mean_wait_s": (m,ci), "specialization": (m,ci)},
             "Fixed θ (ablation)": {...}}"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    labels = list(agg)
    colors = [_color_for(l) for l in labels]

    means = [agg[l]["mean_wait_s"][0] for l in labels]
    cis = [0.0 if np.isnan(agg[l]["mean_wait_s"][1]) else agg[l]["mean_wait_s"][1] for l in labels]
    b = axes[0].bar(labels, means, width=0.55, yerr=cis, capsize=3, color=colors)
    _bar_labels(axes[0], b)
    axes[0].set_ylabel("Mean waiting time (s)")
    axes[0].set_title("Waiting time")

    means = [agg[l]["specialization"][0] for l in labels]
    cis = [0.0 if np.isnan(agg[l]["specialization"][1]) else agg[l]["specialization"][1] for l in labels]
    b = axes[1].bar(labels, means, width=0.55, yerr=cis, capsize=3, color=colors)
    _bar_labels(axes[1], b, fmt="{:.3f}")
    axes[1].set_ylabel("Final specialization index")
    axes[1].set_title("Specialization")

    fig.suptitle(f"Ablation: adaptive vs fixed response thresholds "
                 f"(mean ± 95% CI, N = {n_seeds} seeds)", color=INK)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------- analysis
def plot_dominance_hierarchy(hierarchy_df, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(hierarchy_df))
    ax.bar(x, hierarchy_df["win_rate"], color=C_SWARM)
    ax.set_xticks(x, hierarchy_df["tug_id"], fontsize=6.5, rotation=90)
    ax.set_xlabel("Tug id (sorted by win rate, descending)")
    ax.set_ylabel("Wasp-contest win rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Dominance hierarchy: per-tug win rate in multi-responder contests")
    _save(fig, path)


def plot_division_of_labor(counts_start, counts_end, path):
    x = np.arange(len(REGIONS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2, counts_start, w, color=C_BASELINE, label="t = 0 h")
    b2 = ax.bar(x + w / 2, counts_end, w, color=C_SWARM, label="t = 24 h")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.set_xticks(x, REGIONS)
    ax.set_xlabel("Pier region")
    ax.set_ylabel("Number of tugs primarily specialised there")
    ax.set_title("Division of labour: tug count by most-specialised pier")
    ax.legend()
    _save(fig, path)


# ------------------------------------------------------------- sensitivity
def plot_sensitivity_heatmap(x_vals, y_vals, x_name, y_name, z_mean, z_label, path,
                             n_seeds=1):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(z_mean, aspect="auto", cmap=SEQ_CMAP, origin="lower")
    ax.set_xticks(range(len(x_vals)), [f"{v:g}" for v in x_vals])
    ax.set_yticks(range(len(y_vals)), [f"{v:g}" for v in y_vals])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.grid(False)
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            ax.annotate(f"{z_mean[i, j]:.0f}", (j, i), ha="center", va="center",
                        fontsize=7.5, color=INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(z_label)
    ax.set_title(f"2D sensitivity sweep (mean of N = {n_seeds} seeds per cell)")
    _save(fig, path)


def plot_tornado(names, low, high, baseline, path, xlabel="Mean waiting time (s)"):
    """One-at-a-time sensitivity: for each parameter, the range of the output
    metric spanned when the parameter is swept across its grid, all other
    parameters held at the calibrated default. Sorted by range (impact)."""
    order = np.argsort(np.abs(np.asarray(high) - np.asarray(low)))
    names = [names[i] for i in order]
    low = [low[i] for i in order]
    high = [high[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.5))
    y = np.arange(len(names))
    for yi, lo, hi in zip(y, low, high):
        left, right = min(lo, hi), max(lo, hi)
        ax.barh(yi, right - left, left=left, height=0.55,
               color=C_SWARM if hi >= lo else C_RANDOM, alpha=0.85)
    ax.axvline(baseline, color=INK, lw=1.2, ls="--", label="calibrated default")
    ax.set_yticks(y, names)
    ax.set_xlabel(xlabel)
    ax.set_title("Parameter sensitivity ranking (one-at-a-time sweep range)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)
