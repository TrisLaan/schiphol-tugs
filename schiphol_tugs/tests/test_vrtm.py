"""Unit tests for the response-threshold mathematics."""

import pytest

from src import vrtm
from src.config import Config

CFG = Config()


def test_probability_half_when_s_equals_theta():
    assert vrtm.response_probability(50.0, 50.0, CFG) == pytest.approx(0.5, abs=1e-9)


def test_probability_approaches_one_for_large_stimulus():
    assert vrtm.response_probability(10_000.0, 10.0, CFG) > 0.999


def test_probability_approaches_zero_for_small_stimulus():
    assert vrtm.response_probability(0.1, 100.0, CFG) < 0.001


def test_stimulus_grows_and_saturates():
    assert vrtm.stimulus(0.0, CFG) == CFG.s0
    assert vrtm.stimulus(10.0, CFG) == pytest.approx(CFG.s0 + 10 * CFG.alpha)
    assert vrtm.stimulus(1e6, CFG) == CFG.s_max


def test_stimulus_priority_scales_growth_not_floor():
    # priority leaves s0 (wait=0) untouched but scales the growth rate, so a
    # higher-priority (e.g. wide-body) job's stimulus rises
    # faster per second of waiting than a default-priority job's.
    assert vrtm.stimulus(0.0, CFG, priority=1.8) == CFG.s0
    assert vrtm.stimulus(10.0, CFG, priority=1.8) == pytest.approx(
        CFG.s0 + 10 * CFG.alpha * 1.8
    )
    assert vrtm.stimulus(10.0, CFG, priority=1.8) > vrtm.stimulus(10.0, CFG, priority=1.0)


def test_threshold_decreases_on_performance():
    assert vrtm.update_performer(50.0, CFG) == pytest.approx(50.0 - CFG.xi)


def test_threshold_increases_on_non_performance():
    assert vrtm.update_non_performer(50.0, CFG) == pytest.approx(50.0 + CFG.phi)


def test_threshold_stays_within_bounds():
    theta = 50.0
    for _ in range(10_000):
        theta = vrtm.update_performer(theta, CFG)
    assert theta == CFG.theta_min
    for _ in range(100_000):
        theta = vrtm.update_non_performer(theta, CFG)
    assert theta == CFG.theta_max
