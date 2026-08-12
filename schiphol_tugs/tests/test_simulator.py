"""End-to-end simulator invariants."""

from dataclasses import replace

import pytest

from src import metrics
from src.airport import Airport, generate_requests
from src.config import Config, make_rng
from src.simulator import Simulation


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
        assert 0.0 <= tug.battery <= 1.0


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


def test_failed_tugs_stop_and_requests_recover(airport):
    cfg = Config()
    reqs = generate_requests(airport, cfg, make_rng(213), 3600, start_hour=7.0)
    sim = Simulation(airport, cfg, reqs, make_rng(214), duration_s=3 * 3600,
                     failure_time=600, failed_ids=(0, 1, 2))
    sim.run()
    for tug in sim.tugs:
        if tug.id in (0, 1, 2):
            assert tug.state == "failed"
    # The rest of the fleet still clears the backlog.
    served = sum(1 for r in sim.requests if r.tow_end is not None)
    assert served == len(sim.requests)


def test_contest_log_always_populated_without_recording(airport):
    """Contest logging must not depend on record_snapshots (AUDIT.md 7.4):
    the main run_all.py swarm run doesn't record snapshots but still needs
    contests.csv for the dominance-hierarchy analysis."""
    cfg = Config()
    assert cfg.record_snapshots is False
    reqs = generate_requests(airport, cfg, make_rng(215), 3600, start_hour=8.0)
    sim = Simulation(airport, cfg, reqs, make_rng(216), duration_s=3600)
    sim.run()
    df = sim.contests_dataframe()
    assert list(df.columns) == ["time", "request_id", "responders", "forces", "winner"]
    # A busy morning hour with 20 tugs should produce at least one
    # multi-responder contest.
    assert len(df) > 0


def test_charging_destination_reroutes_during_outage(airport):
    """Direct, deterministic check of the rerouting rule itself (a full-day
    simulation may or may not naturally drive a tug's battery low enough to
    exercise this path within a short window)."""
    cfg = Config()
    sim = Simulation(airport, cfg, [], make_rng(218), duration_s=1,
                     station_outage=("CS1", 0, 3 * 3600))
    tug = sim.tugs[0]
    tug.home_station = "CS1"

    sim.t = 100  # inside the outage window
    dest = sim._charging_destination(tug)
    assert dest != "CS1"
    assert dest in airport.charging_stations

    sim.t = 4 * 3600  # outside the outage window
    assert sim._charging_destination(tug) == "CS1"


def test_charging_station_outage_no_charging_events_at_downed_station(airport):
    cfg = Config()
    reqs = generate_requests(airport, cfg, make_rng(217), 3 * 3600, start_hour=6.0)
    outage = ("CS1", 0, 3 * 3600)  # CS1 down for the whole run
    sim = Simulation(airport, cfg, reqs, make_rng(218), duration_s=3 * 3600,
                     station_outage=outage)
    sim.run()
    # No tug should ever be *charging* at CS1 while the outage is active.
    charging_at_cs1 = [
        e for e in sim.events
        if e[1] == "tug_charging_started" and e[4] == "CS1"
    ]
    assert charging_at_cs1 == []


def test_large_fleet_runs(airport):
    """The swarm must actually scale to a fleet large enough to call a
    'swarm' (AUDIT.md 2.4), not just the calibrated default of ~20-30."""
    cfg = replace(Config(), fleet_size=150)
    reqs = generate_requests(airport, cfg, make_rng(219), 3600, start_hour=8.0)
    sim = Simulation(airport, cfg, reqs, make_rng(220), duration_s=3600)
    sim.run()
    assert len(sim.tugs) == 150
    stats = metrics.summarize(sim)
    assert stats["requests_served"] > 0
