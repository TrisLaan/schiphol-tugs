"""Unit tests for src/metrics.py: waits, summaries, and the analysis metrics
used by src/analysis.py (dominance hierarchy, division of labour, Gini)."""

import numpy as np
import pandas as pd
import pytest

from src import metrics
from src.airport import Airport, generate_requests
from src.config import Config, REGIONS, make_rng
from src.simulator import Simulation


@pytest.fixture(scope="module")
def airport():
    return Airport(Config())


@pytest.fixture(scope="module")
def sim(airport):
    cfg = Config()
    reqs = generate_requests(airport, cfg, make_rng(601), 3600, start_hour=7.0)
    s = Simulation(airport, cfg, reqs, make_rng(602), duration_s=3600)
    s.run()
    return s


def test_wait_times_only_counts_started_tows(sim):
    w = metrics.wait_times(sim.requests)
    n_started = sum(1 for r in sim.requests if r.tow_start is not None)
    assert w.size == n_started
    assert (w >= 0).all()


def test_summarize_keys_and_values(sim):
    st = metrics.summarize(sim)
    assert set(st) == {"requests_total", "requests_served", "mean_wait_s",
                       "p95_wait_s", "total_distance_km", "total_energy_pct"}
    assert st["requests_served"] <= st["requests_total"]
    assert st["total_distance_km"] >= 0.0


def test_specialization_index_zero_when_uniform():
    theta = np.full((5, len(REGIONS)), 50.0)
    assert metrics.specialization_index(theta) == pytest.approx(0.0)


def test_specialization_index_positive_when_varied():
    rng = np.random.default_rng(0)
    theta = rng.uniform(5, 200, size=(10, len(REGIONS)))
    assert metrics.specialization_index(theta) > 0.0


def test_specialization_series_matches_snapshot_count(sim):
    times, idx = metrics.specialization_series(sim.snapshots)
    assert times.size == len(sim.snapshots)
    assert idx.size == times.size


def test_snapshot_at_picks_closest(sim):
    snap = metrics.snapshot_at(sim.snapshots, 0)
    assert snap.shape == (sim.cfg.fleet_size, len(REGIONS))


def test_rolling_wait_shape(sim):
    grid, out = metrics.rolling_wait(sim.requests, 3600, window_s=600, step_s=300)
    assert grid.shape == out.shape


def test_theta_snapshots_dataframe_columns(sim):
    df = metrics.theta_snapshots_dataframe(sim.snapshots)
    assert list(df.columns) == ["time", "tug_id", *REGIONS]
    assert len(df) == len(sim.snapshots) * sim.cfg.fleet_size


def test_gini_zero_for_equal_values():
    assert metrics.gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_gini_high_for_concentrated_values():
    assert metrics.gini([0.0, 0.0, 0.0, 10.0]) > 0.5


def test_gini_empty_or_zero_sum_is_zero():
    assert metrics.gini([]) == 0.0
    assert metrics.gini([0.0, 0.0]) == 0.0


def test_dominance_hierarchy_from_synthetic_contests():
    df = pd.DataFrame({
        "time": [10, 20, 30],
        "request_id": [0, 1, 2],
        "responders": ["0,1", "0,1,2", "1,2"],
        "forces": ["0.5,0.3", "0.5,0.3,0.1", "0.3,0.1"],
        "winner": [0, 0, 1],
    })
    h = metrics.dominance_hierarchy(df, fleet_size=3)
    assert list(h.columns) == ["tug_id", "contests", "wins", "win_rate"]
    row0 = h[h["tug_id"] == 0].iloc[0]
    assert row0["contests"] == 2
    assert row0["wins"] == 2
    assert row0["win_rate"] == pytest.approx(1.0)
    # sorted descending by win rate
    assert (h["win_rate"].diff().dropna() <= 1e-12).all()


def test_primary_region_and_division_of_labor_counts():
    theta = np.array([
        [10, 50, 50, 50, 50, 50, 50, 50],   # specialises in region 0 (B)
        [50, 50, 10, 50, 50, 50, 50, 50],   # region 2 (D)
    ])
    idx = metrics.primary_region(theta)
    assert list(idx) == [0, 2]
    counts = metrics.division_of_labor_counts(theta)
    assert counts.sum() == 2
    assert counts[0] == 1 and counts[2] == 1
