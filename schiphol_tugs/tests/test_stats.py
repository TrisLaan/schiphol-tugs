"""Unit tests for the multi-seed statistics helpers."""

import numpy as np
import pytest

from src import stats


def test_mean_ci_basic():
    mean, half = stats.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == pytest.approx(3.0)
    assert half > 0.0


def test_mean_ci_single_value_has_nan_halfwidth():
    mean, half = stats.mean_ci([7.0])
    assert mean == 7.0
    assert np.isnan(half)


def test_mean_ci_empty_is_nan():
    mean, half = stats.mean_ci([])
    assert np.isnan(mean)
    assert np.isnan(half)


def test_paired_test_detects_systematic_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(10.0, 0.1, size=20)
    b = rng.normal(5.0, 0.1, size=20)
    result = stats.paired_test(a, b)
    assert result["p_value"] < 0.01
    assert result["effect_size_rank_biserial"] == pytest.approx(1.0)


def test_paired_test_no_difference_gives_high_p():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    result = stats.paired_test(a, b)
    assert result["p_value"] == 1.0
    assert result["effect_size_rank_biserial"] == 0.0
