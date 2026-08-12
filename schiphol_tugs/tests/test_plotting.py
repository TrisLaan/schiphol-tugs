"""Smoke tests for every figure-producing function in src/plotting.py.

Each test only checks that the function runs without error against small
synthetic data and writes a real (non-empty) file -- it does not inspect
pixel content. Before this file existed, plotting.py had 0% test coverage
(see AUDIT.md category 4.7); these tests are the fix.
"""

import numpy as np
import pytest

from src import plotting
from src.airport import Airport, generate_requests
from src.config import Config, REGIONS, make_rng


@pytest.fixture(scope="module")
def airport():
    return Airport(Config())


def _assert_written(path):
    assert path.exists()
    assert path.stat().st_size > 500


def test_plot_airport_layout(airport, tmp_path):
    path = tmp_path / "layout.png"
    plotting.plot_airport_layout(airport, path)
    _assert_written(path)


def test_plot_demand_profile(airport, tmp_path):
    cfg = Config()
    reqs = generate_requests(airport, cfg, make_rng(701), 3600)
    path = tmp_path / "demand.png"
    plotting.plot_demand_profile(reqs, cfg, path)
    _assert_written(path)


def _fake_agg(labels, keys):
    return {l: {k: (100.0 + i, 5.0) for i, k in enumerate(keys)} for l in labels}


def test_plot_wait_comparison(tmp_path):
    agg = _fake_agg(["Swarm (VRTM + wasp)", "Centralized baseline"],
                    ["mean_wait_s", "p95_wait_s"])
    path = tmp_path / "wait_comparison.png"
    plotting.plot_wait_comparison(agg, n_seeds=4, path=path)
    _assert_written(path)


def test_plot_energy_comparison(tmp_path):
    agg = _fake_agg(["Swarm (VRTM + wasp)", "Centralized baseline"],
                    ["total_energy_pct", "total_distance_km"])
    path = tmp_path / "energy.png"
    plotting.plot_energy_comparison(agg, n_seeds=4, path=path)
    _assert_written(path)


def test_plot_wait_cdf(tmp_path):
    rng = np.random.default_rng(1)
    waits = {
        "Swarm (VRTM + wasp)": [rng.uniform(0, 300, 50) for _ in range(3)],
        "Centralized baseline": [rng.uniform(0, 250, 50) for _ in range(3)],
    }
    path = tmp_path / "cdf.png"
    plotting.plot_wait_cdf(waits, path)
    _assert_written(path)


def test_plot_threshold_evolution(tmp_path):
    rng = np.random.default_rng(2)
    snap0 = np.full((6, len(REGIONS)), 50.0)
    snap6 = rng.uniform(5, 100, size=(6, len(REGIONS)))
    snap24 = rng.uniform(5, 200, size=(6, len(REGIONS)))
    path = tmp_path / "theta.png"
    plotting.plot_threshold_evolution(snap0, snap6, snap24, path, n_seeds=3)
    _assert_written(path)


def test_plot_specialization(tmp_path):
    times = np.arange(0, 3601, 600, dtype=float)
    series = {
        "Adaptive θ": np.tile(np.linspace(0, 0.3, times.size), (4, 1)),
        "Fixed θ (ablation)": np.zeros((4, times.size)),
    }
    path = tmp_path / "spec.png"
    plotting.plot_specialization(times, series, path)
    _assert_written(path)


def test_plot_robustness_failure(tmp_path):
    grid = np.arange(3600, 7201, 300, dtype=float)
    swarm = np.tile(np.linspace(100, 200, grid.size), (3, 1))
    base = np.tile(np.linspace(90, 180, grid.size), (3, 1))
    path = tmp_path / "fail.png"
    plotting.plot_robustness_failure(grid, swarm, base, 2.0, path)
    _assert_written(path)


def test_plot_robustness_spike(tmp_path):
    grid = np.arange(3600, 7201, 300, dtype=float)
    swarm = np.tile(np.linspace(100, 200, grid.size), (3, 1))
    base = np.tile(np.linspace(90, 180, grid.size), (3, 1))
    path = tmp_path / "spike.png"
    plotting.plot_robustness_spike(grid, swarm, base, 1.0, 1.5, path)
    _assert_written(path)


def test_plot_robustness_charger_outage(tmp_path):
    grid = np.arange(3600, 7201, 300, dtype=float)
    outage = np.tile(np.linspace(100, 220, grid.size), (3, 1))
    normal = np.tile(np.linspace(90, 180, grid.size), (3, 1))
    path = tmp_path / "outage.png"
    plotting.plot_robustness_charger_outage(grid, outage, normal, 1.0, 2.0, path)
    _assert_written(path)


def test_plot_ablation(tmp_path):
    agg = {
        "Adaptive θ": {"mean_wait_s": (200.0, 10.0), "specialization": (0.3, 0.02)},
        "Fixed θ (ablation)": {"mean_wait_s": (230.0, 12.0), "specialization": (0.0, 0.0)},
    }
    path = tmp_path / "ablation.png"
    plotting.plot_ablation(agg, n_seeds=5, path=path)
    _assert_written(path)


def test_plot_dominance_hierarchy(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"tug_id": [0, 1, 2], "contests": [10, 8, 5],
                       "wins": [6, 3, 1], "win_rate": [0.6, 0.375, 0.2]})
    path = tmp_path / "dominance.png"
    plotting.plot_dominance_hierarchy(df, path)
    _assert_written(path)


def test_plot_division_of_labor(tmp_path):
    counts0 = np.zeros(len(REGIONS))
    counts1 = np.arange(len(REGIONS))
    path = tmp_path / "division.png"
    plotting.plot_division_of_labor(counts0, counts1, path)
    _assert_written(path)


def test_plot_sensitivity_heatmap(tmp_path):
    z = np.random.default_rng(3).uniform(100, 300, size=(4, 5))
    path = tmp_path / "heatmap.png"
    plotting.plot_sensitivity_heatmap([1, 2, 3, 4, 5], [0.5, 1.0, 1.5, 2.0],
                                      "x", "y", z, "value", path, n_seeds=3)
    _assert_written(path)


def test_plot_tornado(tmp_path):
    names = ["a", "b", "c"]
    low = [190.0, 180.0, 210.0]
    high = [250.0, 300.0, 215.0]
    path = tmp_path / "tornado.png"
    plotting.plot_tornado(names, low, high, baseline=200.0, path=path)
    _assert_written(path)
