"""Smoke test for the sensitivity-sweep pipeline (src/sensitivity.py).

Uses tiny grids and a short window so the test runs in a couple of seconds
while still exercising every sweep branch (algorithm params, environment
params including an airport rebuild, the tornado plot, and the 2D heatmap).
"""

from dataclasses import replace

import pytest

from src import sensitivity
from src.airport import Airport
from src.config import Config, ExperimentConfig


@pytest.fixture(scope="module")
def airport():
    return Airport(Config())


def test_run_sensitivity_smoke(airport, tmp_path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "results" / "figures"
    figures_dir.mkdir(parents=True)

    tiny = ExperimentConfig(
        sens_hours=0.2, sens_start_hour=7.0, sens_seeds=1,
        xi_sens_grid=(0.5, 2.0), phi_sens_grid=(0.05, 0.2),
        n_exp_sens_grid=(1.0, 2.0), theta_spread_sens_grid=(0.5, 1.0),
        w_d_sens_grid=(0.3, 0.6),
        demand_mult_sens_grid=(0.5, 1.0), n_chargers_sens_grid=(2, 3),
        heatmap_xi_grid=(0.5, 2.0), heatmap_demand_grid=(0.5, 1.0),
    )
    cfg = Config()
    summary = sensitivity.run_sensitivity(airport, cfg, tiny, results_dir, figures_dir)

    assert (results_dir / "sensitivity.csv").exists()
    assert (figures_dir / "sensitivity_tornado.png").exists()
    assert (figures_dir / "sensitivity_heatmap_xi_demand.png").exists()
    assert "most_impactful_parameter" in summary
    assert summary["n_seeds_per_cell"] == 1
