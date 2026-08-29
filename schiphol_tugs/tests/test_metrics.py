"""Unit tests for src/metrics.py: waiting times and run summaries."""

import pytest

from src import metrics
from src.airport import Airport, generate_requests
from src.config import Config, make_rng
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
                       "p95_wait_s", "total_distance_km", "total_energy_kwh"}
    assert st["requests_served"] <= st["requests_total"]
    assert st["total_distance_km"] >= 0.0
