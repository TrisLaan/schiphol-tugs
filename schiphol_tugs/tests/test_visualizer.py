"""Structural tests for the replay logs and a headless render smoke test."""

import matplotlib

matplotlib.use("Agg")  # headless: never open a window from the test suite

from dataclasses import replace

import pandas as pd
import pytest

from src.airport import Airport, generate_requests
from src.config import Config, make_rng
from src.simulator import Simulation
from src.visualizer import Visualizer


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A 10-minute busy-window recording, saved to a temp results dir."""
    out = tmp_path_factory.mktemp("results")
    cfg = replace(Config(), record_snapshots=True)
    airport = Airport(cfg)
    reqs = generate_requests(airport, cfg, make_rng(301), 600, start_hour=8.0)
    sim = Simulation(airport, cfg, reqs, make_rng(302), mode="swarm",
                     duration_s=600)
    sim.run()
    sim.save_visual_logs(out)
    return out, cfg, airport, sim


def test_snapshots_columns_and_no_nans(recorded):
    out, cfg, _, _ = recorded
    df = pd.read_csv(out / "snapshots.csv")
    assert list(df.columns) == ["time", "tug_id", "x", "y", "state",
                                "battery", "assigned_request_id"]
    # assigned_request_id is legitimately empty when a tug has no job;
    # everything else must be fully populated.
    for col in ["time", "tug_id", "x", "y", "state", "battery"]:
        assert not df[col].isna().any(), col
    assert df["battery"].between(0.0, 1.0).all()
    assert set(df["state"]).issubset({"idle", "moving_to_aircraft", "towing",
                                      "moving_to_charger", "charging", "failed"})
    # One row per tug per snapshot second.
    assert len(df) == cfg.fleet_size * df["time"].nunique()


def test_snapshot_tug_ids_match_fleet(recorded):
    out, cfg, _, sim = recorded
    df = pd.read_csv(out / "snapshots.csv")
    assert set(df["tug_id"].unique()) == {t.id for t in sim.tugs}
    assert len(sim.tugs) == cfg.fleet_size


def test_assigned_requests_exist_in_timeline(recorded):
    out, _, _, _ = recorded
    snaps = pd.read_csv(out / "snapshots.csv")
    timeline = pd.read_csv(out / "requests_timeline.csv")
    assigned = set(snaps["assigned_request_id"].dropna().astype(int))
    assert assigned <= set(timeline["request_id"])
    assert list(timeline.columns) == [
        "request_id", "appear_time", "origin_x", "origin_y",
        "dest_x", "dest_y", "region", "resolve_time", "assigned_tug"]


def test_contest_winner_among_responders(recorded):
    out, _, _, _ = recorded
    con = pd.read_csv(out / "contests.csv")
    assert list(con.columns) == ["time", "request_id", "responders",
                                 "forces", "winner"]
    for _, row in con.iterrows():
        responders = [int(v) for v in str(row["responders"]).split(",")]
        forces = [float(v) for v in str(row["forces"]).split(",")]
        assert len(responders) >= 2          # single-responder cases not logged
        assert len(forces) == len(responders)
        assert int(row["winner"]) in responders


def test_headless_render_produces_file(recorded):
    out, cfg, airport, _ = recorded
    # Render the recorded 10 minutes as a handful of frames, save-only.
    viz_cfg = replace(cfg, start_time_hours=0.0, end_time_hours=600 / 3600.0,
                      playback_speed=1200.0, fps=5, show_live=False)
    viz = Visualizer(airport, out, viz_cfg, mode="inspection")
    path = viz.save()
    assert path.exists()
    assert path.stat().st_size > 10_000  # a real video/gif, not an empty stub
