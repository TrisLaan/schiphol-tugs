"""All figures for the project (static PNGs at 300 DPI).

Colors follow a validated brand-neutral palette: categorical slot 1 (blue)
for the swarm, slot 2 (aqua) for the centralized baseline, slot 5 (violet)
and slot 4 (red) for the extra random/optimal baselines, and a single-hue
blue sequential ramp for heatmap magnitude.  Chrome (grid, axes, labels)
stays in recessive grays so the data ink dominates.

Every multi-seed figure shows mean +/- 95% CI (error bars) and states N in
its title, per the project's statistics policy (see docs/notation.md,
"Uncertainty reporting").
"""

import matplotlib

matplotlib.use("Agg")  # headless rendering; run_all works without a display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DPI = 300

# ---- palette (light mode) --------------------------------------------------
C_SWARM = "#2a78d6"      # categorical slot 1: blue
C_BASELINE = "#1baf7a"   # categorical slot 2: aqua (sub-3:1 => direct labels)
C_RANDOM = "#c3423f"     # categorical slot 4: warm red-brown (random floor)
C_ACCENT = "#4a3aa7"     # categorical slot 5: violet (charging stations / optimal)
C_YELLOW = "#eda100"     # categorical slot 3 (runway threshold markers)
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
}

# Single-hue sequential ramp (blue steps 100 -> 700) for heatmaps.
_SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _SEQ_STEPS)

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
    """agg: {label: {"total_energy_kwh": (mean, ci), "total_distance_km": (mean, ci)}}"""
    labels = list(agg)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, key, ylabel, title in [
        (axes[0], "total_energy_kwh", "Fleet energy consumed (kWh)", "Energy use"),
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
