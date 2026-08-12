"""Tests for the interpretive dominance-hierarchy / division-of-labour
analysis (src/analysis.py)."""

import pandas as pd
import pytest

from src import analysis
from src.config import REGIONS


@pytest.fixture
def synthetic_data():
    fleet_size = 4
    contests_df = pd.DataFrame({
        "time": [10, 20, 30, 40],
        "request_id": [0, 1, 2, 3],
        "responders": ["0,1", "0,1,2", "1,2,3", "0,3"],
        "forces": ["0.5,0.3", "0.5,0.3,0.1", "0.3,0.1,0.2", "0.4,0.2"],
        "winner": [0, 0, 2, 0],
    })
    rows = []
    for tug in range(fleet_size):
        for region_i, region in enumerate(REGIONS):
            theta0 = 50.0
            theta1 = 10.0 if region_i == tug % len(REGIONS) else 60.0
            rows.append({"time": 0, "tug_id": tug, **{r: theta0 for r in REGIONS}})
            rows.append({"time": 3600, "tug_id": tug,
                        **{r: (theta1 if r == region else 60.0) for r in REGIONS}})
    theta_df = pd.DataFrame(rows).drop_duplicates(subset=["time", "tug_id"])
    return contests_df, theta_df, fleet_size


def test_analyze_returns_expected_fields(synthetic_data, tmp_path, monkeypatch):
    contests_df, theta_df, fleet_size = synthetic_data
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    monkeypatch.setattr(analysis, "FIGURES", figures_dir)

    result = analysis.analyze(contests_df, theta_df, fleet_size,
                              xi=1.0, phi=0.1, w_d=0.6, w_b=0.3, w_s=0.1)
    assert set(result) >= {
        "n_contested_tugs", "win_rate_gini", "top_tug_id", "top_tug_win_rate",
        "division_of_labor_gini_end", "empty_piers_start", "empty_piers_end",
        "interpretation",
    }
    assert 0.0 <= result["win_rate_gini"] <= 1.0
    assert isinstance(result["interpretation"], str) and len(result["interpretation"]) > 100
    assert (figures_dir / "dominance_hierarchy.png").exists()
    assert (figures_dir / "division_of_labor.png").exists()
