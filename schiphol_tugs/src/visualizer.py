"""Matplotlib replay animation of a recorded simulation run.

Reads the replay logs written by ``Simulation.save_visual_logs`` —
``snapshots.csv`` (per-tug positions/states/batteries per simulated second),
``requests_timeline.csv``, ``contests.csv`` and ``theta_snapshots.csv`` —
and renders a 16:9 animation: airport map with moving colour-coded tugs,
battery arcs, waiting-aircraft triangles, assignment lines, wasp-contest
flashes, plus a live sidebar (clock, state counts, metrics, threshold
heatmap and, in inspection mode, a scrolling event feed).

This module must NOT call ``matplotlib.use`` — the entry point decides the
backend (Agg when ``--no-show``), so live display keeps working.
"""

import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation as manim
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from .airport import PIERS, RUNWAYS, Airport
from .config import Config, REGIONS

# ---- palette ----------------------------------------------------------------
# Tug state colours follow the visual spec (light blue / orange / red /
# yellow / green); chrome greys match the report figures.  Constants are
# duplicated from plotting.py on purpose: importing plotting would force the
# Agg backend and kill the live window.
STATE_COLORS = {
    "idle": "#6db3f2",
    "moving_to_aircraft": "#eb6834",
    "towing": "#d03b3b",
    "moving_to_charger": "#eda100",
    "charging": "#0ca30c",
    "failed": "#898781",
}
STATE_NAMES = ["idle", "moving_to_aircraft", "towing",
               "moving_to_charger", "charging", "failed"]
STATE_CODE = {s: i for i, s in enumerate(STATE_NAMES)}
STATE_RGBA = np.array([to_rgba(STATE_COLORS[s]) for s in STATE_NAMES])

REQ_GRAY, REQ_YELLOW, REQ_RED = "#898781", "#eda100", "#d03b3b"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
CHARGER_GREEN = "#0ca30c"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])

BATT_OK, BATT_LOW = "#52514e", "#d03b3b"
ASSIGN_DASHED, ASSIGN_SOLID = "#eb6834", "#d03b3b"
FLASH_YELLOW, FLASH_GREEN = "#eda100", "#0ca30c"


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary (PATH first, then the winget install dir)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget.exists():
        hits = sorted(winget.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"))
        if hits:
            return str(hits[0])
    return None


def _clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


class ReplayData:
    """The four replay CSVs, loaded into frame-lookup-friendly arrays."""

    def __init__(self, results_dir: Path):
        results_dir = Path(results_dir)
        snaps = pd.read_csv(
            results_dir / "snapshots.csv",
            dtype={"time": np.int32, "tug_id": np.int16, "x": np.float64,
                   "y": np.float64, "state": "category",
                   "battery": np.float64, "assigned_request_id": "Int32"},
        ).sort_values(["time", "tug_id"], kind="mergesort")
        self.n_tugs = int(snaps["tug_id"].nunique())
        self.tug_ids = np.sort(snaps["tug_id"].unique())
        # Rows come in complete per-time blocks of n_tugs (one per tug),
        # so frame lookup is a searchsorted plus a slice.
        self.times = snaps["time"].to_numpy()[:: self.n_tugs]
        self.x = snaps["x"].to_numpy()
        self.y = snaps["y"].to_numpy()
        self.state = snaps["state"].map(STATE_CODE).to_numpy(np.int8)
        self.battery = snaps["battery"].to_numpy()
        self.req = snaps["assigned_request_id"].fillna(-1).to_numpy(np.int32)

        tl = pd.read_csv(results_dir / "requests_timeline.csv",
                         dtype={"resolve_time": "Int32", "assigned_tug": "Int16"})
        ids = tl["request_id"].to_numpy()
        size = int(ids.max()) + 1 if len(ids) else 1
        self.appear = np.full(size, np.inf)
        self.resolve = np.full(size, np.inf)   # inf = never picked up
        self.ox = np.zeros(size)
        self.oy = np.zeros(size)
        self.dx = np.zeros(size)
        self.dy = np.zeros(size)
        self.region = np.empty(size, dtype=object)
        self.appear[ids] = tl["appear_time"].to_numpy()
        resolved = tl["resolve_time"].notna().to_numpy()
        self.resolve[ids[resolved]] = tl["resolve_time"].dropna().to_numpy(np.float64)
        self.ox[ids] = tl["origin_x"].to_numpy()
        self.oy[ids] = tl["origin_y"].to_numpy()
        self.dx[ids] = tl["dest_x"].to_numpy()
        self.dy[ids] = tl["dest_y"].to_numpy()
        self.region[ids] = tl["region"].to_numpy()
        self.timeline = tl

        # Tow completion (aircraft delivered): last snapshot in which some tug
        # was towing the request.  Derived here so the timeline CSV can keep
        # exactly the columns the spec asks for.
        self.tow_end = np.full(size, np.inf)
        towing = snaps[(snaps["state"] == "towing")
                       & snaps["assigned_request_id"].notna()]
        if len(towing):
            done = towing.groupby("assigned_request_id", observed=True)["time"].max()
            self.tow_end[done.index.to_numpy(np.int64)] = done.to_numpy(np.float64)
        # First moment a tug is seen working on the request (for the event feed).
        assigned = snaps[snaps["assigned_request_id"].notna()]
        firsts = assigned.groupby("assigned_request_id", observed=True).first()
        self.first_assign = {int(k): (int(v["time"]), int(v["tug_id"]))
                             for k, v in firsts.iterrows()}

        # Pre-sorted resolved waits so "mean wait so far" is O(log n) a frame.
        mask = np.isfinite(self.resolve)
        order = np.argsort(self.resolve[mask])
        self._res_times = self.resolve[mask][order]
        self._res_waits_cum = np.cumsum((self.resolve - self.appear)[mask][order])
        self._tow_end_sorted = np.sort(self.tow_end[np.isfinite(self.tow_end)])

        con = pd.read_csv(results_dir / "contests.csv")
        self.contest_time = con["time"].to_numpy(np.float64) if len(con) else np.empty(0)
        self.contest_resp = [
            [int(v) for v in str(s).split(",")] for s in con["responders"]
        ] if len(con) else []
        self.contest_winner = con["winner"].to_numpy(np.int64) if len(con) else np.empty(0, np.int64)
        self.contests = con

        th = pd.read_csv(results_dir / "theta_snapshots.csv")
        self.theta_times = np.sort(th["time"].unique())
        self.theta = {
            t: g.sort_values("tug_id")[list(REGIONS)].to_numpy()
            for t, g in th.groupby("time")
        }

    # ------------------------------------------------------------------ query
    def frame_slice(self, t: float) -> slice:
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        i = max(0, i)
        return slice(i * self.n_tugs, (i + 1) * self.n_tugs)

    def mean_wait_until(self, t: float) -> float | None:
        k = int(np.searchsorted(self._res_times, t, side="right"))
        return (self._res_waits_cum[k - 1] / k) if k else None

    def tows_completed_until(self, t: float) -> int:
        return int(np.searchsorted(self._tow_end_sorted, t, side="right"))

    def events(self, w0: float, w1: float) -> tuple[np.ndarray, list[str]]:
        """Chronological event feed for inspection mode."""
        ev: list[tuple[float, int, str]] = []
        for rid in range(len(self.appear)):
            a = self.appear[rid]
            if not np.isfinite(a) or not (w0 <= a <= w1):
                continue
            ev.append((a, 0, f"Request R{rid:04d} created in zone {self.region[rid]}"))
        for rid, (t, tug) in self.first_assign.items():
            if w0 <= t <= w1:
                ev.append((t, 1, f"Tug T{tug:02d} assigned to R{rid:04d}"))
        crids = (self.contests["request_id"].to_numpy(np.int64)
                 if len(self.contests) else np.empty(0, np.int64))
        for t, resp, win, rid in zip(self.contest_time, self.contest_resp,
                                     self.contest_winner, crids):
            if w0 <= t <= w1:
                names = " vs ".join(f"T{r:02d}" for r in resp)
                ev.append((t, 2, f"Contest for R{rid:04d}: {names} → T{win:02d} wins"))
        for rid in range(len(self.resolve)):
            t = self.resolve[rid]
            if np.isfinite(t) and w0 <= t <= w1:
                ev.append((t, 3, f"Tow of R{rid:04d} starts"))
            te = self.tow_end[rid]
            if np.isfinite(te) and w0 <= te <= w1:
                ev.append((te, 4, f"Tow of R{rid:04d} completed"))
        ev.sort(key=lambda e: (e[0], e[1]))
        times = np.array([e[0] for e in ev])
        texts = [f"{_clock(e[0])}  {e[2]}" for e in ev]
        return times, texts


class Visualizer:
    """Builds the animation figure and drives FuncAnimation."""

    def __init__(self, airport: Airport, results_dir, cfg: Config,
                 mode: str = "overview"):
        self.airport = airport
        self.cfg = cfg
        self.mode = mode
        self.captions = mode == "inspection"
        self.results_dir = Path(results_dir)
        self.data = ReplayData(self.results_dir)

        d = self.data
        self.w0 = max(cfg.start_time_hours * 3600.0, float(d.times[0]))
        self.w1 = min(cfg.end_time_hours * 3600.0, float(d.times[-1]))
        self.spf = cfg.playback_speed / cfg.fps  # sim seconds per frame
        self.n_frames = max(2, int((self.w1 - self.w0) / self.spf) + 1)

        # Deterministic per-tug display offset so tugs parked on the same
        # node stay individually visible (concentric rings of 10).
        ring, pos = np.divmod(d.tug_ids.astype(int), 10)
        ang = 2 * np.pi * pos / 10.0
        rad = 60.0 + 80.0 * ring
        self.jx = rad * np.cos(ang)
        self.jy = rad * np.sin(ang)

        if self.captions:
            self.evt_times, self.evt_texts = d.events(self.w0, self.w1)
        self._theta_key = None
        self._anim = None
        self._build_figure()

    # ------------------------------------------------------------ static part
    def _build_figure(self):
        a = self.airport
        self.fig = plt.figure(figsize=(16, 9), dpi=100)
        self.fig.patch.set_facecolor(SURFACE)
        ax = self.fig.add_axes([0.015, 0.03, 0.655, 0.90])
        self.ax = ax
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        xmin, ymin, xmax, ymax = a.bbox
        ax.set_xlim(xmin - 350, xmax + 350)
        ax.set_ylim(ymin - 350, ymax + 350)
        ax.set_aspect("equal")
        self.fig.text(0.015, 0.955,
                      "Schiphol tug swarm — decentralized replay "
                      f"({self.mode} mode, {self.cfg.playback_speed:.0f}× speed)",
                      fontsize=14, color=INK, fontweight="bold")

        # Taxiways.
        segs = [(a.xy(u), a.xy(v)) for u, v in a.graph.edges]
        ax.add_collection(LineCollection(segs, colors=GRID, lw=0.9, zorder=1))
        # Runways + names.
        for name, (na, xya, nb, xyb) in RUNWAYS.items():
            ax.plot([xya[0], xyb[0]], [xya[1], xyb[1]], color=INK_2, lw=5,
                    solid_capstyle="round", zorder=2)
            ax.annotate(name, ((xya[0] + xyb[0]) / 2, (xya[1] + xyb[1]) / 2),
                        fontsize=8, color=INK_2, xytext=(6, 6),
                        textcoords="offset points", zorder=2)
        # Gates as hollow circles.
        gates = a.all_gates()
        ax.scatter([a.xy(g)[0] for g in gates], [a.xy(g)[1] for g in gates],
                   s=16, facecolors="none", edgecolors=MUTED, lw=0.8, zorder=3)
        # Pier labels.
        for pier, (hub, _) in PIERS.items():
            ax.annotate(pier, hub, fontsize=11, color=INK, fontweight="bold",
                        xytext=(-14, -6), textcoords="offset points", zorder=3)
        # Charging stations.
        for cs in a.charging_stations:
            x, y = a.xy(cs)
            ax.scatter([x], [y], s=110, marker="s", color=CHARGER_GREEN,
                       edgecolor=INK_2, lw=0.6, zorder=4)
            ax.annotate(f"{cs} (charging)", (x, y), fontsize=7, color="#006300",
                        xytext=(8, -12), textcoords="offset points", zorder=4)
        # North arrow (top-left) and 500 m scale bar (bottom-left).
        ax.annotate("N", xy=(0.03, 0.97), xytext=(0.03, 0.89),
                    xycoords="axes fraction", textcoords="axes fraction",
                    ha="center", fontsize=12, fontweight="bold", color=INK,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
        sx, sy = xmin - 100, ymin - 150
        ax.plot([sx, sx + 500], [sy, sy], color=INK, lw=3, solid_capstyle="butt")
        ax.annotate("500 m", (sx + 250, sy), ha="center", va="bottom",
                    fontsize=8, color=INK, xytext=(0, 4),
                    textcoords="offset points")

        # ---------------------------------------------------- dynamic artists
        n = self.data.n_tugs
        self.tug_dots = ax.scatter(np.zeros(n), np.zeros(n), s=58, zorder=8,
                                   edgecolors=INK_2, linewidths=0.4)
        self.tug_labels = [ax.text(0, 0, f"{i}", fontsize=5.5, color=INK_2,
                                   zorder=9, ha="left", va="bottom")
                           for i in self.data.tug_ids]
        self.batt_arcs = LineCollection([], lw=1.5, zorder=9)
        ax.add_collection(self.batt_arcs)
        self.req_tris = ax.scatter([], [], marker="^", s=95, zorder=6,
                                   edgecolors=INK_2, linewidths=0.5)
        self._req_texts: dict[int, plt.Text] = {}
        self.lines_dashed = LineCollection([], colors=ASSIGN_DASHED, lw=1.0,
                                           linestyles="--", zorder=5)
        self.lines_solid = LineCollection([], colors=ASSIGN_SOLID, lw=1.6,
                                          linestyles="-", zorder=5)
        ax.add_collection(self.lines_dashed)
        ax.add_collection(self.lines_solid)
        self.ring_yellow = ax.scatter([], [], s=300, facecolors="none",
                                      edgecolors=FLASH_YELLOW, linewidths=2.4,
                                      zorder=10)
        self.ring_green = ax.scatter([], [], s=360, facecolors="none",
                                     edgecolors=FLASH_GREEN, linewidths=2.6,
                                     zorder=10)

        # ----------------------------------------------------------- sidebar
        fx = 0.695
        self.txt_clock = self.fig.text(fx, 0.90, "", fontsize=26, color=INK,
                                       family="monospace", fontweight="bold")
        self.txt_states = []
        for i, s in enumerate(STATE_NAMES[:5]):
            self.txt_states.append(self.fig.text(
                fx, 0.845 - 0.028 * i, "", fontsize=10,
                color=STATE_COLORS[s], family="monospace"))
        self.txt_metrics = self.fig.text(fx, 0.685, "", fontsize=10, color=INK,
                                         family="monospace", va="top")
        if self.captions:
            self.fig.text(fx, 0.545, "Recent events", fontsize=9, color=INK_2,
                          fontweight="bold")
            self.txt_events = self.fig.text(fx, 0.53, "", fontsize=7.3,
                                            color=INK, family="monospace",
                                            va="top")
            heat_rect = [0.71, 0.045, 0.235, 0.20]
        else:
            heat_rect = [0.71, 0.045, 0.235, 0.30]
        axh = self.fig.add_axes(heat_rect)
        self.axh = axh
        all_theta = np.concatenate([m.ravel() for m in self.data.theta.values()])
        self.im = axh.imshow(
            list(self.data.theta.values())[0], aspect="auto", cmap=SEQ_CMAP,
            vmin=float(all_theta.min()), vmax=float(all_theta.max()),
            interpolation="nearest")
        axh.set_xticks(range(len(REGIONS)), REGIONS, fontsize=7)
        axh.set_yticks([])
        axh.set_ylabel("tug", fontsize=7, color=INK_2)
        axh.set_title("Thresholds θ by pier (30-min updates; light = specialist)",
                      fontsize=8, color=INK_2)

    # ------------------------------------------------------------- per frame
    def _batt_segments(self, x, y, frac):
        segs, colors = [], []
        for cx, cy, b in zip(x, y, frac):
            npts = max(3, int(26 * b) + 1)
            ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * b, npts)
            segs.append(np.column_stack([cx + 92 * np.cos(ang),
                                         cy + 92 * np.sin(ang)]))
            colors.append(BATT_LOW if b < self.cfg.recharge_trigger else BATT_OK)
        return segs, colors

    def _update(self, frame: int):
        d, cfg = self.data, self.cfg
        t = self.w0 + frame * self.spf
        sl = d.frame_slice(t)
        x = d.x[sl] + self.jx
        y = d.y[sl] + self.jy
        states = d.state[sl]
        batt = d.battery[sl]
        reqs = d.req[sl]

        self.tug_dots.set_offsets(np.column_stack([x, y]))
        self.tug_dots.set_facecolors(STATE_RGBA[states])
        for xi, yi, txt in zip(x, y, self.tug_labels):
            txt.set_position((xi + 70, yi + 70))
        segs, cols = self._batt_segments(x, y, batt)
        self.batt_arcs.set_segments(segs)
        self.batt_arcs.set_colors(cols)

        # Waiting aircraft (triangles + wait labels).
        active = np.flatnonzero((d.appear <= t) & (d.resolve > t))
        if len(active):
            waits = t - d.appear[active]
            colors = np.where(waits >= 120, REQ_RED,
                              np.where(waits >= 60, REQ_YELLOW, REQ_GRAY))
            self.req_tris.set_offsets(np.column_stack([d.ox[active], d.oy[active]]))
            self.req_tris.set_facecolors(colors)
        else:
            self.req_tris.set_offsets(np.empty((0, 2)))
        active_set = set(int(i) for i in active)
        for rid in [r for r in self._req_texts if r not in active_set]:
            self._req_texts.pop(rid).remove()
        for rid in active_set:
            if rid not in self._req_texts:
                self._req_texts[rid] = self.ax.text(
                    d.ox[rid] + 90, d.oy[rid] - 40, "", fontsize=6.5,
                    color=INK_2, zorder=7)
            self._req_texts[rid].set_text(f"{int(t - d.appear[rid])} s")

        # Assignment lines: dashed while heading to the aircraft, solid
        # (towards the destination) while towing.
        dash, solid = [], []
        for i in range(d.n_tugs):
            rid = reqs[i]
            if rid < 0:
                continue
            if states[i] == STATE_CODE["moving_to_aircraft"]:
                dash.append([(x[i], y[i]), (d.ox[rid], d.oy[rid])])
            elif states[i] == STATE_CODE["towing"]:
                solid.append([(x[i], y[i]), (d.dx[rid], d.dy[rid])])
        self.lines_dashed.set_segments(dash)
        self.lines_solid.set_segments(solid)

        # Contest flashes: responders ringed yellow for 2 s of animation time
        # (= 2 * playback_speed sim seconds), then the winner green for 1 s.
        y_pts, g_pts = [], []
        t_yellow = 2.0 * cfg.playback_speed
        t_green = 1.0 * cfg.playback_speed
        lo = int(np.searchsorted(d.contest_time, t - t_yellow - t_green, "left"))
        hi = int(np.searchsorted(d.contest_time, t, "right"))
        id_row = {int(tid): k for k, tid in enumerate(d.tug_ids)}
        for c in range(lo, hi):
            age = t - d.contest_time[c]
            if age < t_yellow:
                for r in d.contest_resp[c]:
                    k = id_row.get(r)
                    if k is not None:
                        y_pts.append((x[k], y[k]))
            elif age < t_yellow + t_green:
                k = id_row.get(int(d.contest_winner[c]))
                if k is not None:
                    g_pts.append((x[k], y[k]))
        self.ring_yellow.set_offsets(np.array(y_pts) if y_pts else np.empty((0, 2)))
        self.ring_green.set_offsets(np.array(g_pts) if g_pts else np.empty((0, 2)))

        # ------------------------------------------------------------ sidebar
        self.txt_clock.set_text(_clock(t))
        counts = np.bincount(states, minlength=6)
        for i, s in enumerate(STATE_NAMES[:5]):
            self.txt_states[i].set_text(f"● {s}: {counts[i]}")
        mean_w = d.mean_wait_until(t)
        longest = (t - d.appear[active]).max() if len(active) else 0.0
        self.txt_metrics.set_text(
            f"Active requests : {len(active)}\n"
            f"Mean wait so far: {'—' if mean_w is None else f'{mean_w:5.0f} s'}\n"
            f"Longest current : {longest:5.0f} s\n"
            f"Tows completed  : {d.tows_completed_until(t)}"
        )
        if self.captions:
            k = int(np.searchsorted(self.evt_times, t, side="right"))
            self.txt_events.set_text("\n".join(self.evt_texts[max(0, k - 8):k]))
        # Threshold heatmap, refreshed every simulated 30 minutes.
        key_candidates = d.theta_times[d.theta_times <= max(t, d.theta_times[0])]
        key = key_candidates[key_candidates % 1800 == 0]
        key = float(key[-1]) if len(key) else float(d.theta_times[0])
        if key != self._theta_key:
            self._theta_key = key
            self.im.set_data(d.theta[key])
        return []

    # -------------------------------------------------------------- playback
    def _ensure_anim(self):
        if self._anim is None:
            self._anim = manim.FuncAnimation(
                self.fig, self._update, frames=self.n_frames,
                interval=1000.0 / self.cfg.fps, blit=False, repeat=False)
        return self._anim

    def save(self) -> Path:
        """Write the animation; MP4 via ffmpeg, GIF via Pillow as fallback."""
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
            writer = manim.FFMpegWriter(fps=self.cfg.fps, bitrate=3000)
            path = self.results_dir / "animation.mp4"
        else:
            print("WARNING: ffmpeg not found — saving animated GIF via Pillow "
                  "instead (bigger file, slower to write). Install ffmpeg to "
                  "get results/animation.mp4.")
            writer = manim.PillowWriter(fps=self.cfg.fps)
            path = self.results_dir / "animation.gif"
        anim = self._ensure_anim()
        total = self.n_frames

        def progress(i, n):
            if i % 250 == 0 or i == n - 1:
                print(f"  rendering frame {i + 1}/{total}", flush=True)

        anim.save(str(path), writer=writer, progress_callback=progress)
        return path

    def show(self) -> None:
        self._ensure_anim()
        plt.show()
