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
