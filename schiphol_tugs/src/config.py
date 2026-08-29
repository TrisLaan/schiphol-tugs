"""Central configuration for the Schiphol tug-swarm simulation.

All tunable parameters live here.  Every random draw in the project goes
through a numpy Generator produced by :func:`make_rng`, which derives its
seed deterministically from the single project SEED (= 42).  Different
"streams" (request generation, swarm decisions, calibration runs, ...) get
different, but fully reproducible, generators so that paired experiments
(e.g. swarm vs. baseline on the *same* request stream) are possible while
run_all.py stays byte-reproducible end to end.

Python's built-in ``random`` module is not used anywhere in the project.
"""

from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported for callers)

import numpy as np

# The one and only project seed (see "Reproducibility" in the README).
SEED = 42


def make_rng(stream: int = 0) -> np.random.Generator:
    """Return a deterministic Generator for the given logical stream.

    ``default_rng([SEED, stream])`` hashes the pair through SeedSequence, so
    every (SEED, stream) combination yields an independent but reproducible
    stream of draws.
    """
    return np.random.default_rng([SEED, stream])


# Fixed order of pier regions used everywhere (threshold maps, heatmaps, ...).
REGIONS = ["B", "C", "D", "E", "F", "G", "H", "M"]


@dataclass
class Config:
    """All simulation parameters.  Defaults follow the project spec."""

    # ------------------------------------------------------------------ fleet
    fleet_size: int = 29               # literature-sourced fleet size, fixed (not calibrated)
    speed_empty: float = 10.0          # m/s when driving without an aircraft
    speed_towing: float = 5.0          # m/s when towing
    battery_capacity_kwh: float = 300.0  # physical pack size per tug
    # Drain rates in kWh per metre driven, derived from the same relative
    # cost (0.5 %/km empty, 1.5 %/km towing -- towing costs 3x) scaled onto
    # a 300 kWh pack: 1.5 kWh/km empty, 4.5 kWh/km towing.
    drain_empty: float = 1.5 / 1000    # kWh per metre, empty
    drain_towing: float = 4.5 / 1000   # kWh per metre, towing
    charger_power_kw: float = 150.0    # DC fast-charging power per station
    recharge_rate: float = 150.0 / 3600.0  # kWh per second (== charger_power_kw)
    recharge_trigger: float = 0.25     # hard floor: below this SoC a tug goes charging
                                        # immediately, bypassing the bay throttle below
    charge_target: float = 1.0         # charge until full (SoC fraction) before rejoining
    # Opportunistic top-up: an idle tug that has finished drifting back to its
    # hub goes to charge when below this fraction (0.0 disables).  Off by
    # default: experiments showed it removes tugs from the eligible pool at
    # the wrong moments and slightly worsens waiting times.
    idle_topup: float = 0.0
    # Soft charging call: an idle (non-repositioning) tug below this SoC but
    # still above recharge_trigger becomes a *candidate* for a charging bay,
    # resolved by a reversed wasp contest against the station's bay capacity
    # (see Simulation._run_charging_dispatch) instead of going unconditionally.
    # This staggers charging departures instead of letting the whole fleet
    # cross a single hard threshold in the same narrow window.
    charge_call_threshold: float = 0.5

    # --------------------------------------------------------------- airport
    n_charging_stations: int = 4
    # Physical concurrent-charger capacity per station. 3 bays x 4 stations =
    # 12 concurrent chargers fleet-wide (~41% of a 29-tug fleet) -- enough for
    # ordinary turnover, but a real cap that forces the reversed contest to
    # stagger a synchronized low-battery wave rather than letting every
    # candidate through at once.
    bays_per_station: int = 3
    # Station positions (metres, terminal-centred).  Length must equal
    # n_charging_stations; they are placed near the pier clusters.
    charging_station_xy: tuple = (
        (-650.0, -100.0),   # near piers B/C (west side of terminal)
        (-100.0, -950.0),   # near piers D/E (south)
        (700.0, -650.0),    # near piers F/G (south-east)
        (1250.0, 350.0),    # near piers H/M (east)
    )

    # ---------------------------------------------------------------- demand
    # Diurnal request intensity: base + two Gaussian peaks (~08:00 & ~18:00).
    base_rate: float = 3.0     # requests/hour in the middle of the night
    morning_peak: float = 45.0  # extra requests/hour at the morning peak
    evening_peak: float = 42.0  # extra requests/hour at the evening peak
    morning_hour: float = 8.0
    evening_hour: float = 18.0
    morning_width: float = 2.0  # std-dev of the peak, in hours
    evening_width: float = 2.5
    p_departure: float = 0.5   # probability a request is a departure
    # Heterogeneous job types: a request is drawn wide-body with probability
    # p_widebody (schedule-critical connecting bank), else narrow-body.
    # priority multiplies the VRTM stimulus growth rate alpha (vrtm.stimulus)
    # so a wide-body's stimulus rises faster per second of waiting than a
    # narrow-body's -- tugs perceive it as more urgent at the same wait time.
    p_widebody: float = 0.2
    widebody_priority: float = 1.8
    narrowbody_priority: float = 1.0

    # ------------------------------------------------------------------ VRTM
    theta0: float = 50.0   # initial response threshold, all tugs / regions
    theta_min: float = 5.0
    theta_max: float = 200.0
    s0: float = 30.0       # stimulus at zero waiting time
    alpha: float = 2.0     # stimulus growth per second of aircraft waiting
    s_max: float = 200.0   # stimulus ceiling
    n_exp: float = 2.0     # Hill exponent n in the response probability
    xi: float = 1.0        # threshold decrease for the tug that performs
    phi: float = 0.1       # threshold increase for eligible non-performers

    # ------------------------------------------------------------------ wasp
    w_d: float = 0.6       # weight of the distance term in the force
    w_b: float = 0.3       # weight of the battery term
    w_s: float = 0.1       # weight of the not-busy term
    contest_mode: str = "deterministic"  # or "stochastic"

    # ------------------------------------------------------ charging contest
    # Reversed dominance force for the charging-bay queue (see wasp.charge_force):
    # favours low battery (urgent) and proximity to the station (cheap to send
    # now) -- the mirror image of the job contest's w_d/w_b weighting. No
    # "not busy" term: only idle, non-repositioning tugs are ever candidates.
    w_d_charge: float = 0.6   # weight of proximity to the charging station
    w_u_charge: float = 0.4   # weight of urgency, (1 - battery_frac)

    # ------------------------------------------------------------- simulator
    dt: float = 1.0                # simulation time step, seconds
    snapshot_interval: int = 600   # threshold snapshot cadence, seconds

    # ---------------------------------------------------------- visualization
    record_snapshots: bool = False   # write visual replay logs (see visualizer)
    snapshot_interval_s: int = 1     # visual snapshot cadence, sim seconds
    playback_speed: float = 300.0    # sim seconds per wall-clock second
    fps: int = 30                    # animation frames per second
    start_time_hours: float = 0.0    # animation window start (sim clock)
    end_time_hours: float = 24.0     # animation window end (sim clock)
    save_video: bool = True          # write results/animation.mp4 (gif fallback)
    show_live: bool = True           # also open an interactive window

    def hourly_rate(self, hour: float) -> float:
        """Expected tow requests per hour at clock time ``hour`` (0-24)."""
        m = self.morning_peak * np.exp(
            -0.5 * ((hour - self.morning_hour) / self.morning_width) ** 2
        )
        e = self.evening_peak * np.exp(
            -0.5 * ((hour - self.evening_hour) / self.evening_width) ** 2
        )
        return float(self.base_rate + m + e)


# A ready-to-use default configuration.
DEFAULT = Config()


@dataclass
class ExperimentConfig:
    """Every experiment-driver constant used by run_all.py: calibration
    windows, sweep grids, and the multi-seed replication count.  Kept out of
    run_all.py itself so no algorithm or experiment-design parameter is a bare
    magic number in module scope."""

    # -------------------------------------------------------- replication
    n_seeds: int = 6              # independent replicates per experiment

    # ------------------------------------------------------- calibration
    cal_hours: float = 4.0        # calibration horizon (h)
    cal_start_hour: float = 6.0   # window start (so it contains the morning peak)
    xi_grid: tuple = (0.5, 1.0, 2.0, 5.0)
    alpha_grid: tuple = (0.5, 1.0, 2.0, 5.0)

    # ------------------------------------------------------- sensitivity
    # One-at-a-time sweep grids for the tornado plot (algorithm params).
    sens_hours: float = 2.0       # short window so the sweep stays fast
    sens_start_hour: float = 7.0
    sens_seeds: int = 3           # seeds per sweep cell (smaller: many cells)
    xi_sens_grid: tuple = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    phi_sens_grid: tuple = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    n_exp_sens_grid: tuple = (1.0, 1.5, 2.0, 3.0, 4.0)
    theta_spread_sens_grid: tuple = (0.5, 1.0, 2.0, 4.0)  # multiplies [theta_min,theta_max] width
    w_d_sens_grid: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)
    demand_mult_sens_grid: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)
    n_chargers_sens_grid: tuple = (1, 2, 3, 4)
    # 2D interaction sweep (xi x demand multiplier), plotted as a heatmap.
    heatmap_xi_grid: tuple = (0.5, 1.0, 2.0, 5.0, 10.0)
    heatmap_demand_grid: tuple = (0.5, 1.0, 1.5, 2.0, 2.5)


EXPERIMENT = ExperimentConfig()
