"""Visual replay of the tug-swarm simulation.

    python visualize.py --mode overview     # full 24 h day at 300x speed
    python visualize.py --mode inspection   # 08:00-10:00 at 60x, event feed

If the replay logs (results/snapshots.csv etc.) are missing, don't cover the
requested window, or --fresh is given, a fresh swarm simulation is recorded
first.  The replayed run uses the calibrated parameters from
results/summary.json when available and the same rng streams as run_all.py,
so what you watch is the same day the report describes.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PRESETS = {
    # start_h, end_h, speed, captions come from the mode; CLI flags override.
    "overview": {"start": 0.0, "end": 24.0, "speed": 300.0},
    "inspection": {"start": 8.0, "end": 10.0, "speed": 60.0},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(PRESETS), default="overview",
                   help="overview: whole day, no captions; inspection: short "
                        "window with a scrolling event feed")
    p.add_argument("--start", type=float, default=None,
                   help="window start in hours (overrides the mode preset)")
    p.add_argument("--end", type=float, default=None,
                   help="window end in hours (overrides the mode preset)")
    p.add_argument("--speed", type=float, default=None,
                   help="playback speed multiplier (overrides the mode preset)")
    p.add_argument("--no-save", action="store_true",
                   help="skip writing results/animation.mp4 (or .gif)")
    p.add_argument("--no-show", action="store_true",
                   help="skip the live matplotlib window (headless render)")
    p.add_argument("--fresh", action="store_true",
                   help="re-record the simulation even if logs exist")
    return p.parse_args()


def build_cfg(args):
    from dataclasses import replace

    from src.config import DEFAULT

    preset = PRESETS[args.mode]
    start = preset["start"] if args.start is None else args.start
    end = preset["end"] if args.end is None else args.end
    speed = preset["speed"] if args.speed is None else args.speed
    if not 0.0 <= start < end <= 24.0:
        raise SystemExit(f"invalid window: --start {start} --end {end}")

    # Replay the calibrated configuration from run_all.py when available.
    cal = {}
    summary = RESULTS / "summary.json"
    if summary.exists():
        cal = json.loads(summary.read_text(encoding="utf-8"))["calibration"]
    return replace(
        DEFAULT,
        xi=cal.get("xi", DEFAULT.xi),
        alpha=cal.get("alpha", DEFAULT.alpha),
        fleet_size=cal.get("fleet_size", DEFAULT.fleet_size),
        record_snapshots=True,
        playback_speed=speed,
        start_time_hours=start,
        end_time_hours=end,
        save_video=not args.no_save,
        show_live=not args.no_show,
    )


def ensure_logs(cfg, fresh: bool) -> None:
    """Reuse recorded logs when they cover the window; otherwise record."""
    w0 = int(cfg.start_time_hours * 3600)
    w1 = int(cfg.end_time_hours * 3600)
    meta_path = RESULTS / "visual_meta.json"
    logs = ["snapshots.csv", "requests_timeline.csv", "contests.csv",
            "theta_snapshots.csv"]
    if not fresh and meta_path.exists() and all((RESULTS / f).exists() for f in logs):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta["window_start_s"] <= w0 and meta["window_end_s"] >= w1
                and meta["fleet_size"] == cfg.fleet_size):
            print("Reusing recorded replay logs in results/ "
                  "(pass --fresh to re-record).")
            return
    print(f"Recording a fresh swarm simulation (0–{w1 / 3600:.1f} h, snapshots "
          f"{w0 / 3600:.1f}–{w1 / 3600:.1f} h)…")
    from src.airport import Airport, generate_requests
    from src.config import make_rng
    from src.simulator import Simulation

    airport = Airport(cfg)
    # Streams 20/21 = the main-day run of run_all.py (recording draws no
    # extra random numbers, so the replayed day matches the report).
    requests = generate_requests(airport, cfg, make_rng(20), 24 * 3600)
    sim = Simulation(airport, cfg, requests, make_rng(21), mode="swarm",
                     duration_s=w1, record_window=(w0, w1))
    sim.run()
    RESULTS.mkdir(exist_ok=True)
    sim.save_visual_logs(RESULTS)
    print(f"Replay logs written to {RESULTS}")


def main():
    args = parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")  # decide the backend before pyplot is imported

    cfg = build_cfg(args)
    ensure_logs(cfg, args.fresh)

    from src.airport import Airport
    from src.visualizer import Visualizer

    viz = Visualizer(Airport(cfg), RESULTS, cfg, mode=args.mode)
    print(f"Animating {cfg.start_time_hours:.1f}–{cfg.end_time_hours:.1f} h at "
          f"{cfg.playback_speed:.0f}× ({viz.n_frames} frames @ {cfg.fps} fps)")
    if cfg.save_video:
        path = viz.save()
        print(f"Saved animation: {path}")
    if cfg.show_live:
        print("Opening live window (close it to finish)…")
        viz.show()


if __name__ == "__main__":
    main()
