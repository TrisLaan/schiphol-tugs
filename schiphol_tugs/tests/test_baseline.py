"""Unit tests for the three centralized dispatcher baselines."""

from dataclasses import replace

import pytest

from src import baseline
from src.airport import Airport, generate_requests
from src.config import Config, make_rng
from src.simulator import Simulation


@pytest.fixture(scope="module")
def airport():
    return Airport(Config())


def _reqs(airport, cfg, stream=501, duration_s=1800):
    return generate_requests(airport, cfg, make_rng(stream), duration_s, start_hour=7.0)


def test_assign_random_uses_only_eligible_tugs(airport):
    cfg = Config()
    reqs = _reqs(airport, cfg)
    sim = Simulation(airport, cfg, reqs, make_rng(502), duration_s=1)
    sim._activate_requests()
    pairs = baseline.assign_random(sim.open_requests, sim.tugs, make_rng(1))
    eligible_ids = {t.id for t in sim.tugs if t.eligible}
    for _, tug in pairs:
        assert tug.id in eligible_ids
    # no tug used twice in the same step
    assert len({tug.id for _, tug in pairs}) == len(pairs)


def test_assign_random_is_reproducible_given_same_rng_stream(airport):
    cfg = Config()
    reqs = _reqs(airport, cfg)
    sim = Simulation(airport, cfg, reqs, make_rng(503), duration_s=1)
    sim._activate_requests()
    a = baseline.assign_random(sim.open_requests, sim.tugs, make_rng(77))
    b = baseline.assign_random(sim.open_requests, sim.tugs, make_rng(77))
    assert [(r.req_id, t.id) for r, t in a] == [(r.req_id, t.id) for r, t in b]


def test_assign_optimal_matches_greedy_or_better_total_distance(airport):
    """The joint (Hungarian) matching should never have a larger total
    matched distance than the request-by-request greedy dispatcher on the
    same instantaneous pool -- it is a joint optimum by construction."""
    cfg = Config()
    reqs = _reqs(airport, cfg, stream=504, duration_s=3600)
    sim = Simulation(airport, cfg, reqs, make_rng(505), duration_s=1)
    sim._activate_requests()
    if len(sim.open_requests) < 2:
        pytest.skip("need >= 2 simultaneously open requests for this check")

    greedy = baseline.assign_nearest(sim.open_requests, sim.tugs, sim.t)
    optimal = baseline.assign_optimal(sim.open_requests, sim.tugs, sim.t)
    greedy_total = sum(t.dist_to(r.origin) for r, t in greedy)
    optimal_total = sum(t.dist_to(r.origin) for r, t in optimal)
    assert len(optimal) == len(greedy)
    assert optimal_total <= greedy_total + 1e-6


def test_assign_optimal_empty_inputs(airport):
    assert baseline.assign_optimal([], [], 0) == []


def test_simulator_random_and_optimal_modes_serve_requests(airport):
    cfg = Config()
    reqs = _reqs(airport, cfg, stream=506, duration_s=3600)
    for mode in ("random", "optimal"):
        sim = Simulation(airport, cfg, reqs, make_rng(507), mode=mode,
                         duration_s=2 * 3600)
        sim.run()
        served = sum(1 for r in sim.requests if r.tow_end is not None)
        assert served > 0


def test_simulator_rejects_unknown_mode(airport):
    cfg = Config()
    reqs = _reqs(airport, cfg, stream=508, duration_s=60)
    with pytest.raises(AssertionError):
        Simulation(airport, cfg, reqs, make_rng(509), mode="not_a_mode")
