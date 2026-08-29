"""Structural tests for the Schiphol graph."""

import networkx as nx
import pytest

from src.airport import Airport, generate_requests
from src.config import Config, REGIONS, make_rng


@pytest.fixture(scope="module")
def airport():
    return Airport(Config())


def test_graph_is_connected(airport):
    assert nx.is_connected(airport.graph)


def test_every_gate_reaches_every_runway_threshold(airport):
    for gate in airport.all_gates():
        for thr in airport.runway_thresholds:
            assert thr in airport.dist[gate]
            assert airport.dist[gate][thr] > 0


def test_every_gate_reaches_a_charging_station(airport):
    for gate in airport.all_gates():
        assert any(cs in airport.dist[gate] for cs in airport.charging_stations)


def test_coordinates_in_plausible_bounding_box(airport):
    # Everything should fit inside ~7 km west, ~3 km east/south, ~5 km north
    # of the terminal (metres).
    for node, data in airport.graph.nodes(data=True):
        x, y = data["xy"]
        assert -7000 <= x <= 4000, node
        assert -4000 <= y <= 6000, node


def test_all_regions_have_gates(airport):
    for pier in REGIONS:
        assert len(airport.gates[pier]) >= 6


def test_request_generation_is_reasonable(airport):
    cfg = Config()
    reqs = generate_requests(airport, cfg, make_rng(123), 24 * 3600)
    # Expected total for the default diurnal profile is a few hundred a day.
    assert 300 <= len(reqs) <= 1500
    for r in reqs:
        assert 0 <= r.time < 24 * 3600
        assert r.region in REGIONS
        if r.kind == "departure":
            assert r.origin in airport.all_gates()
            assert r.dest in airport.runway_thresholds
        else:
            assert r.origin in airport.runway_thresholds
            assert r.dest in airport.all_gates()
    times = [r.time for r in reqs]
    assert times == sorted(times)


def test_requests_are_heterogeneous_by_aircraft_class(airport):
    """Job types differ: wide-bodies carry a higher VRTM stimulus priority
    than narrow-bodies, drawn at Config.p_widebody."""
    cfg = Config()
    reqs = generate_requests(airport, cfg, make_rng(124), 24 * 3600)
    classes = {r.aircraft_class for r in reqs}
    assert classes == {"narrowbody", "widebody"}
    for r in reqs:
        expected = (cfg.widebody_priority if r.aircraft_class == "widebody"
                    else cfg.narrowbody_priority)
        assert r.priority == expected
    widebody_frac = sum(r.aircraft_class == "widebody" for r in reqs) / len(reqs)
    assert abs(widebody_frac - cfg.p_widebody) < 0.05
