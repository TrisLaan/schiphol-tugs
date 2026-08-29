"""Unit tests for the wasp dominance contest."""

from dataclasses import replace

import pytest

from src import wasp
from src.config import Config, make_rng

CFG = Config()


def test_higher_force_wins_deterministic():
    responders = ["a", "b", "c"]
    forces = [0.3, 0.9, 0.5]
    assert wasp.contest(responders, forces, CFG) == "b"


def test_single_responder_returned_immediately():
    # Works even in stochastic mode with no rng: no draw is needed.
    cfg = replace(CFG, contest_mode="stochastic")
    assert wasp.contest(["only"], [0.42], cfg, rng=None) == "only"


def test_stochastic_equal_forces_are_fair():
    cfg = replace(CFG, contest_mode="stochastic")
    rng = make_rng(999)
    wins = sum(
        1 for _ in range(10_000)
        if wasp.contest(["a", "b"], [0.7, 0.7], cfg, rng) == "a"
    )
    assert 4700 <= wins <= 5300  # ~50/50 within ~3 sigma


def test_force_prefers_near_full_battery_idle():
    near = wasp.force(0.05, 1.0, busy=False, cfg=CFG)
    far_low_busy = wasp.force(0.9, 0.3, busy=True, cfg=CFG)
    assert near > far_low_busy


def test_contest_rejects_bad_input():
    with pytest.raises(ValueError):
        wasp.contest([], [], CFG)
    with pytest.raises(ValueError):
        wasp.contest(["a"], [0.1, 0.2], CFG)


def test_charge_force_prefers_low_battery_near_station():
    # Reversed force: low battery + close to the station should outrank
    # high battery + far from it -- the mirror image of the job force.
    urgent_near = wasp.charge_force(0.05, 0.1, cfg=CFG)
    fine_far = wasp.charge_force(0.9, 0.9, cfg=CFG)
    assert urgent_near > fine_far


def test_charge_force_monotonic_in_urgency():
    near = 0.2
    low_batt = wasp.charge_force(near, 0.1, cfg=CFG)
    high_batt = wasp.charge_force(near, 0.45, cfg=CFG)
    assert low_batt > high_batt
