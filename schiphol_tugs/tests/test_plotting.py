"""Smoke tests for every figure-producing function in src/plotting.py.

Each test only checks that the function runs without error against small
synthetic data and writes a real (non-empty) file -- it does not inspect
pixel content.
"""

import numpy as np
import pytest

from src import plotting
from src.airport import Airport
from src.config import Config


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
                    ["total_energy_kwh", "total_distance_km"])
    path = tmp_path / "energy.png"
    plotting.plot_energy_comparison(agg, n_seeds=4, path=path)
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
