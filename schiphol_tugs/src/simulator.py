"""Discrete-event simulator (fixed 1-second time step).

Each step:
  1. update tug positions, batteries and states,
  2. activate requests created at this time,
  3. determine responders per open request (VRTM in swarm mode, nearest
     available in baseline mode),
  4. resolve multi-responder conflicts with the wasp dominance contest,
  5. assign winners (losers stay eligible for the next request),
  6. adapt thresholds (swarm mode),
and every event is appended to an in-memory log that can be written to CSV.
"""

import copy

import numpy as np
import pandas as pd

from . import baseline, vrtm, wasp
from .airport import Airport, Request
from .config import Config, REGIONS
from .tug import (CHARGING, FAILED, IDLE, TO_AIRCRAFT, TO_CHARGER, TOWING, Tug)


class Simulation:
    """One simulation run (swarm or baseline) over a fixed request stream."""

    def __init__(
        self,
        airport: Airport,
        cfg: Config,
        requests: list[Request],
        rng: np.random.Generator,
        mode: str = "swarm",
        duration_s: int = 24 * 3600,
        record_window: tuple[int, int] | None = None,
    ):
        assert mode in ("swarm", "baseline", "random", "optimal")
        self.airport = airport
        self.cfg = cfg
        self.mode = mode
        self.rng = rng
        self.duration_s = duration_s
        # Each run works on its own copies, so paired experiments (swarm and
        # baseline on the same stream) never contaminate each other's records.
        self.requests = [copy.copy(r) for r in requests]
        for r in self.requests:
            r.reset_assignment()
        # Fleet: home stations assigned round-robin across charging stations.
        stations = airport.charging_stations
        self.tugs = [
            Tug(i, stations[i % len(stations)], airport, cfg)
            for i in range(cfg.fleet_size)
        ]
        self.events: list[tuple] = []       # (time, event, tug_id, req_id, node)
        self.snapshots: list[tuple] = []    # (time, theta array [n_tugs, n_regions])
        self._next_req = 0                  # index into the time-sorted request list
        self.open_requests: list[Request] = []
        self.t = 0
        # Optional high-frequency recording for the visual replay
        # (src/visualizer.py).  Recording draws no random numbers, so a run
        # with record_snapshots=True is event-identical to one without.
        self.contest_log: list[tuple] = []
        self._rec = None
        if cfg.record_snapshots:
            w0, w1 = record_window if record_window is not None else (0, duration_s)
            self._rec_w0 = max(0, int(w0))
            self._rec_w1 = min(int(w1), duration_s)
            self._rec_step = max(1, int(cfg.snapshot_interval_s))
            n_rows = ((self._rec_w1 - self._rec_w0) // self._rec_step + 1) * cfg.fleet_size
            self._rec = {
                "time": np.empty(n_rows, np.int32),
                "tug_id": np.empty(n_rows, np.int16),
                "x": np.empty(n_rows, np.float64),
                "y": np.empty(n_rows, np.float64),
                "state": np.empty(n_rows, np.int8),
                "battery": np.empty(n_rows, np.float64),
                "req": np.empty(n_rows, np.int32),
            }
            self._rec_i = 0

    # Fixed state encoding used by the visual logs.
    STATE_NAMES = [IDLE, TO_AIRCRAFT, TOWING, TO_CHARGER, CHARGING, FAILED]
    _STATE_CODE = {name: i for i, name in enumerate(STATE_NAMES)}

    # ------------------------------------------------------------------- log
    def log(self, event: str, tug_id=None, req_id=None, node=""):
        self.events.append((self.t, event, tug_id, req_id, node))

    def events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.events, columns=["time", "event", "tug_id", "request_id", "node"]
        )

    def contests_dataframe(self) -> pd.DataFrame:
        """Every multi-responder wasp contest, independent of visual
        recording (see :attr:`contest_log`; always populated)."""
        return pd.DataFrame(
            self.contest_log,
            columns=["time", "request_id", "responders", "forces", "winner"],
        )

    # ------------------------------------------------------------------ steps
    def _reposition_target(self, tug: Tug) -> str:
        """Pier hub an idle tug drifts back to after finishing a job.

        Swarm tugs head for the hub of their most-specialised (lowest-theta)
        region -- this is what turns threshold specialisation into physical
        service zones.  Ties (and the baseline, which has no thresholds) fall
        back to the hub closest to the tug's home charging station, so the
        round-robin homes spread the fleet around the terminal.
        """
        if self.mode == "swarm":
            t_min = min(tug.theta.values())
            candidates = [r for r in REGIONS if tug.theta[r] <= t_min + 1e-9]
        else:
            candidates = REGIONS
        return min(
            (f"PIER_{r}" for r in candidates),
            key=lambda hub: self.airport.dist[tug.home_station][hub],
        )

    def _go_idle(self, tug: Tug) -> None:
        tug.state = IDLE
        target = self._reposition_target(tug)
        if target != tug.node:
            tug.set_destination(target)
            tug.repositioning = True
            self.log("tug_repositioning", tug.id, node=target)

    def _send_to_charger(self, tug: Tug, dest: str | None = None) -> None:
        """Dispatch ``tug`` to a charging station, unconditionally.

        Used both by the hard safety floor (recharge_trigger, always wins
        regardless of bay availability) and by the soft-call contest in
        :meth:`_run_charging_dispatch` (only called once a bay is free)."""
        tug.repositioning = False
        tug.state = TO_CHARGER
        dest = dest or tug.home_station
        tug.charging_station = dest
        tug.set_destination(dest)
        self.log("tug_to_charger", tug.id, node=dest)

    def _run_charging_dispatch(self) -> None:
        """Soft charging-bay queue: idle tugs between recharge_trigger and
        charge_call_threshold contest for a free bay at their (rerouted-if-
        outaged) destination station via a reversed wasp force that favours
        low battery and proximity (wasp.charge_force) -- the mirror image of
        the job-assignment contest. Bay capacity (cfg.bays_per_station) is
        the actual throttle: it staggers a synchronized low-battery wave
        instead of letting every candidate cross a single hard threshold in
        the same tick. Tugs that hit the hard recharge_trigger floor bypass
        this entirely via _send_to_charger in _update_tug."""
        cfg = self.cfg
        candidates = [
            tug for tug in self.tugs
            if tug.state == IDLE and not tug.repositioning
            and cfg.recharge_trigger <= tug.battery_frac < cfg.charge_call_threshold
        ]
        if not candidates:
            return
        by_station: dict[str, list[Tug]] = {}
        for tug in candidates:
            by_station.setdefault(tug.home_station, []).append(tug)
        for station, pool in by_station.items():
            occupied = sum(
                1 for t in self.tugs
                if t.charging_station == station and t.state in (TO_CHARGER, CHARGING)
            )
            free = max(0, cfg.bays_per_station - occupied)
            pool = list(pool)
            for _ in range(min(free, len(pool))):
                forces = [
                    wasp.charge_force(tug.dist_to(station) / self.airport.diag,
                                      tug.battery_frac, cfg)
                    for tug in pool
                ]
                winner = wasp.contest(pool, forces, cfg, self.rng)
                pool.remove(winner)
                self._send_to_charger(winner, station)

    def _update_tug(self, tug: Tug) -> None:
        cfg = self.cfg
        if tug.state in (FAILED,):
            return
        if tug.state == CHARGING:
            cap = cfg.battery_capacity_kwh
            tug.battery = min(cfg.charge_target * cap, tug.battery + cfg.recharge_rate * cfg.dt)
            if tug.battery >= cfg.charge_target * cap:
                self.log("tug_charging_ended", tug.id, node=tug.node)
                tug.charging_station = None
                self._go_idle(tug)
            return
        if tug.state == IDLE:
            if tug.battery_frac < cfg.recharge_trigger:
                self._send_to_charger(tug)
            elif tug.repositioning:
                if tug.advance(cfg.dt):  # still interruptible: idle => eligible
                    tug.repositioning = False
            elif cfg.idle_topup > 0.0 and tug.battery_frac < cfg.idle_topup:
                # Nothing to do and battery below the top-up point: charge now
                # rather than being forced to at 25 % during a peak.
                self._send_to_charger(tug)
            return
        # Moving states.
        arrived = tug.advance(cfg.dt)
        if not arrived:
            return
        if tug.state == TO_AIRCRAFT:
            req = tug.request
            tug.state = TOWING
            req.tow_start = self.t
            tug.set_destination(req.dest)
            self.log("tow_started", tug.id, req.req_id, node=req.origin)
        elif tug.state == TOWING:
            req = tug.request
            req.tow_end = self.t
            tug.request = None
            self.log("tow_completed", tug.id, req.req_id, node=req.dest)
            if tug.battery_frac < cfg.recharge_trigger:
                self._send_to_charger(tug)
            else:
                self._go_idle(tug)
        elif tug.state == TO_CHARGER:
            tug.state = CHARGING
            self.log("tug_charging_started", tug.id, node=tug.node)

    def _activate_requests(self) -> None:
        while (
            self._next_req < len(self.requests)
            and self.requests[self._next_req].time <= self.t
        ):
            req = self.requests[self._next_req]
            self.open_requests.append(req)
            self.log("request_created", req_id=req.req_id, node=req.origin)
            self._next_req += 1

    def _assign(self, req: Request, tug: Tug) -> None:
        req.tug_id = tug.id
        tug.repositioning = False  # a job interrupts any idle drift
        tug.request = req
        tug.state = TO_AIRCRAFT
        tug.set_destination(req.origin)
        self.open_requests.remove(req)
        self.log("tug_assigned", tug.id, req.req_id, node=req.origin)

    def _assign_swarm(self) -> None:
        cfg = self.cfg
        # Oldest request first; a tug assigned earlier in the step is no
        # longer idle for later requests, losers remain eligible.
        for req in list(self.open_requests):
            eligible = [tug for tug in self.tugs if tug.eligible]
            if not eligible:
                break
            s = vrtm.stimulus(self.t - req.time, cfg, priority=req.priority)
            responders = []
            for tug in eligible:  # tug order is fixed by id => reproducible draws
                p = vrtm.response_probability(s, tug.theta[req.region], cfg)
                if self.rng.random() < p:
                    responders.append(tug)
            if not responders:
                continue
            forces = [
                wasp.force(tug.dist_to(req.origin) / self.airport.diag,
                           tug.battery_frac, busy=False, cfg=cfg)
                for tug in responders
            ]
            winner = wasp.contest(responders, forces, cfg, self.rng)
            if len(responders) > 1:
                self.contest_log.append((
                    self.t, req.req_id,
                    ",".join(str(tg.id) for tg in responders),
                    ",".join(f"{f:.4f}" for f in forces),
                    winner.id,
                ))
            self._assign(req, winner)
            winner.theta[req.region] = vrtm.update_performer(
                winner.theta[req.region], cfg)
            for tug in eligible:
                if tug is not winner:
                    tug.theta[req.region] = vrtm.update_non_performer(
                        tug.theta[req.region], cfg)

    def _assign_baseline(self) -> None:
        for req, tug in baseline.assign_nearest(self.open_requests, self.tugs, self.t):
            self._assign(req, tug)

    def _assign_random(self) -> None:
        for req, tug in baseline.assign_random(self.open_requests, self.tugs, self.rng):
            self._assign(req, tug)

    def _assign_optimal(self) -> None:
        for req, tug in baseline.assign_optimal(self.open_requests, self.tugs, self.t):
            self._assign(req, tug)

    def _snapshot(self) -> None:
        arr = np.array([[tug.theta[r] for r in REGIONS] for tug in self.tugs])
        self.snapshots.append((self.t, arr))

    # -------------------------------------------------------------------- run
    def run(self) -> "Simulation":
        for self.t in range(0, self.duration_s + 1):
            if self.t % self.cfg.snapshot_interval == 0:
                self._snapshot()
            for tug in self.tugs:
                self._update_tug(tug)
            if self.mode == "swarm":
                # Reversed-contest charging queue is a swarm-side coordination
                # protocol (broadcast + contest, no central dispatcher), so it
                # only applies here -- baseline/random/optimal keep the
                # unconditional hard-trigger charging behaviour unchanged.
                self._run_charging_dispatch()
            self._activate_requests()
            if self.open_requests:
                if self.mode == "swarm":
                    self._assign_swarm()
                elif self.mode == "baseline":
                    self._assign_baseline()
                elif self.mode == "random":
                    self._assign_random()
                else:  # "optimal"
                    self._assign_optimal()
            if (
                self._rec is not None
                and self._rec_w0 <= self.t <= self._rec_w1
                and (self.t - self._rec_w0) % self._rec_step == 0
            ):
                self._record_frame()
        return self

    # -------------------------------------------------------- visual logging
    def _record_frame(self) -> None:
        """One position/state/battery row per tug (end-of-step state)."""
        i, r = self._rec_i, self._rec
        for tug in self.tugs:
            x, y = tug.position()
            r["time"][i] = self.t
            r["tug_id"][i] = tug.id
            r["x"][i] = x
            r["y"][i] = y
            r["state"][i] = self._STATE_CODE[tug.state]
            r["battery"][i] = tug.battery_frac
            r["req"][i] = tug.request.req_id if tug.request is not None else -1
            i += 1
        self._rec_i = i

    def save_visual_logs(self, results_dir) -> None:
        """Write the replay logs consumed by src/visualizer.py.

        ``resolve_time`` in requests_timeline.csv is the pickup moment (the
        tow starts and the aircraft stops waiting); empty if the aircraft was
        never picked up inside the simulated horizon.
        """
        import json
        from pathlib import Path

        assert self._rec is not None, "run with cfg.record_snapshots=True first"
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        n = self._rec_i
        assigned = pd.array(self._rec["req"][:n], dtype="Int32")
        assigned[assigned < 0] = pd.NA
        pd.DataFrame({
            "time": self._rec["time"][:n],
            "tug_id": self._rec["tug_id"][:n],
            "x": np.round(self._rec["x"][:n], 1),
            "y": np.round(self._rec["y"][:n], 1),
            "state": pd.Categorical.from_codes(
                self._rec["state"][:n], categories=self.STATE_NAMES),
            "battery": np.round(self._rec["battery"][:n], 4),
            "assigned_request_id": assigned,
        }).to_csv(results_dir / "snapshots.csv", index=False)

        rows = []
        for r in self.requests:
            ox, oy = self.airport.xy(r.origin)
            dx, dy = self.airport.xy(r.dest)
            rows.append((r.req_id, r.time, round(ox, 1), round(oy, 1),
                         round(dx, 1), round(dy, 1), r.region,
                         r.tow_start, r.tug_id))
        timeline = pd.DataFrame(rows, columns=[
            "request_id", "appear_time", "origin_x", "origin_y",
            "dest_x", "dest_y", "region", "resolve_time", "assigned_tug"])
        timeline["resolve_time"] = timeline["resolve_time"].astype("Int32")
        timeline["assigned_tug"] = timeline["assigned_tug"].astype("Int16")
        timeline.to_csv(results_dir / "requests_timeline.csv", index=False)

        self.contests_dataframe().to_csv(results_dir / "contests.csv", index=False)

        theta_rows = []
        for t, arr in self.snapshots:
            for tug_i in range(arr.shape[0]):
                theta_rows.append((t, tug_i, *np.round(arr[tug_i], 2)))
        pd.DataFrame(theta_rows, columns=["time", "tug_id", *REGIONS]).to_csv(
            results_dir / "theta_snapshots.csv", index=False)

        meta = {
            "window_start_s": self._rec_w0,
            "window_end_s": self._rec_w1,
            "interval_s": self._rec_step,
            "fleet_size": self.cfg.fleet_size,
            "duration_s": self.duration_s,
        }
        (results_dir / "visual_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
