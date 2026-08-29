"""End-to-end simulator invariants."""

from dataclasses import replace

import pytest

from src import metrics
from src.airport import Airport, generate_requests
from src.config import Config, make_rng
from src.simulator import Simulation
from src.tug import IDLE, TO_CHARGER


@pytest.fixture(scope="module")
def airport():
    return Airport(Config())


def _run(airport, cfg, req_stream, sim_stream, mode="swarm",
         request_s=3600, duration_s=3600):
    reqs = generate_requests(airport, cfg, make_rng(req_stream), request_s,
                             start_hour=7.0)  # busy window => real load
    sim = Simulation(airport, cfg, reqs, make_rng(sim_stream), mode=mode,
                     duration_s=duration_s)
    return sim.run()


def test_no_negative_battery(airport):
    cfg = Config()
    sim = _run(airport, cfg, 201, 202, duration_s=2 * 3600)
    for tug in sim.tugs:
        assert tug.min_battery >= 0.0
        assert 0.0 <= tug.battery <= cfg.battery_capacity_kwh


def test_every_request_eventually_served(airport):
    # One hour of peak-time requests, three hours of simulation to drain the
    # queue: a generous limit.
    cfg = Config()
    sim = _run(airport, cfg, 203, 204, request_s=3600, duration_s=3 * 3600)
    assert len(sim.requests) > 10
    for r in sim.requests:
        assert r.tow_end is not None, f"request {r.req_id} never served"


def test_tug_count_conserved(airport):
    cfg = Config()
    sim = _run(airport, cfg, 205, 206, duration_s=2 * 3600)
    assert len(sim.tugs) == cfg.fleet_size
    assert len({t.id for t in sim.tugs}) == cfg.fleet_size


def test_reproducibility_same_seed_same_events(airport):
    cfg = Config()
    a = _run(airport, cfg, 207, 208, request_s=1800, duration_s=1800)
    b = _run(airport, cfg, 207, 208, request_s=1800, duration_s=1800)
    assert a.events == b.events


def test_smoke_one_hour(airport):
    """1-hour smoke test: things happen and stay sane."""
    cfg = Config()
    sim = _run(airport, cfg, 209, 210, duration_s=3600)
    df = sim.events_dataframe()
    assert len(df) > 0
    kinds = set(df["event"])
    assert "request_created" in kinds
    assert "tug_assigned" in kinds
    assert "tow_completed" in kinds
    stats = metrics.summarize(sim)
    assert stats["requests_served"] > 0
    assert stats["mean_wait_s"] > 0
    assert stats["total_distance_km"] > 0
    # Waits are finite and positive.
    w = metrics.wait_times(sim.requests)
    assert (w >= 0).all()


def test_baseline_serves_requests(airport):
    cfg = Config()
    sim = _run(airport, cfg, 211, 212, mode="baseline", duration_s=2 * 3600)
    stats = metrics.summarize(sim)
    assert stats["requests_served"] > 0


def test_contest_log_always_populated_without_recording(airport):
    """Contest logging must not depend on record_snapshots: the visual replay
    logs (src/visualizer.py) consume the contest log, which is populated on
    every swarm run regardless of whether high-frequency recording is on."""
    cfg = Config()
    assert cfg.record_snapshots is False
    reqs = generate_requests(airport, cfg, make_rng(215), 3600, start_hour=8.0)
    sim = Simulation(airport, cfg, reqs, make_rng(216), duration_s=3600)
    sim.run()
    df = sim.contests_dataframe()
    assert list(df.columns) == ["time", "request_id", "responders", "forces", "winner"]
    # A busy morning hour with a full fleet should produce at least one
    # multi-responder contest.
    assert len(df) > 0


def test_charging_bay_throttle_limits_concurrent_dispatch(airport):
    """Only bays_per_station soft-call candidates should be dispatched per
    station per tick, even when more of them are waiting."""
    cfg = replace(Config(), bays_per_station=1)
    sim = Simulation(airport, cfg, [], make_rng(230), duration_s=1)
    station = sim.tugs[0].home_station
    same_home = [t for t in sim.tugs if t.home_station == station]
    assert len(same_home) >= 3, "need >=3 tugs sharing a home station for this test"
    candidates = same_home[:3]
    for t in candidates:
        t.battery = 0.4 * cfg.battery_capacity_kwh  # inside [recharge_trigger, charge_call_threshold)
    sim._run_charging_dispatch()
    dispatched = [t for t in candidates if t.state == TO_CHARGER]
    still_idle = [t for t in candidates if t.state == IDLE]
    assert len(dispatched) == 1
    assert len(still_idle) == 2


def test_charging_dispatch_prefers_lowest_battery(airport):
    """The reversed contest should favour the most urgent (lowest-battery)
    candidate when distance is equal."""
    cfg = replace(Config(), bays_per_station=1, contest_mode="deterministic")
    sim = Simulation(airport, cfg, [], make_rng(231), duration_s=1)
    station = sim.tugs[0].home_station
    same_home = [t for t in sim.tugs if t.home_station == station]
    assert len(same_home) >= 2
    low, high = same_home[0], same_home[1]
    low.node = station  # equal distance to the station: only battery differs
    high.node = station
    low.battery = 0.3 * cfg.battery_capacity_kwh
    high.battery = 0.45 * cfg.battery_capacity_kwh
    sim._run_charging_dispatch()
    assert low.state == TO_CHARGER
    assert high.state == IDLE


def test_hard_floor_bypasses_bay_cap(airport):
    """A tug below recharge_trigger must go charge immediately, even if the
    soft-call throttle would otherwise have no free bays -- the safety
    override is unconditional."""
    cfg = replace(Config(), bays_per_station=0)  # no bay capacity at all
    sim = Simulation(airport, cfg, [], make_rng(232), duration_s=1)
    tug = sim.tugs[0]
    tug.battery = 0.1 * cfg.battery_capacity_kwh  # below recharge_trigger (0.25)
    sim._update_tug(tug)
    assert tug.state == TO_CHARGER


def test_large_fleet_runs(airport):
    """The swarm must actually scale to a fleet large enough to call a
    'swarm', not just the calibrated default of ~20-30."""
    cfg = replace(Config(), fleet_size=150)
    reqs = generate_requests(airport, cfg, make_rng(219), 3600, start_hour=8.0)
    sim = Simulation(airport, cfg, reqs, make_rng(220), duration_s=3600)
    sim.run()
    assert len(sim.tugs) == 150
    stats = metrics.summarize(sim)
    assert stats["requests_served"] > 0
