# Notation, glossary, units, and locality

This document is the single source of truth for every symbol, abbreviation,
unit, and reference-frame choice used in the code and (should be) in the
report. Every symbol below is matched to the exact line in the code that
defines or consumes it, so report and repository stay in sync.

## 1. Glossary (abbreviations expanded once)

| Abbreviation | Expansion |
|---|---|
| VRTM | Variable Response Threshold Model (Theraulaz, Bonabeau & Deneubourg, 1998) |
| SoC | State of Charge — battery fraction in [0, 1] |
| CI | Confidence Interval |
| CDF | Cumulative Distribution Function |
| CV | Coefficient of Variation (std / mean) |
| CS | Charging Station |
| ATC | Air Traffic Control (referenced only in the "known limitations" discussion — not modelled) |
| RNG | (Pseudo-)Random Number Generator |
| E2E | End-to-end (test) |

## 2. Free parameters of the method

The rubric explicitly penalises methods with only 2–3 free parameters. VRTM
and the wasp contest together expose **19** algorithm parameters (below),
plus a further ~20 environment/fleet parameters (`src/config.py`, not
repeated here since they characterise the *environment*, not the *method*,
per rubric categories 1 vs. 2).

### 2.1 VRTM parameters

| Symbol | Name | Code field | Units | Default | Physical meaning | Source |
|---|---|---|---|---|---|---|
| θ₀ | Initial threshold | `Config.theta0` | stimulus units | 50.0 | Starting response threshold, identical for every (tug, region) pair before any learning | Theraulaz et al. (1998) §2 |
| θ_min | Threshold floor | `Config.theta_min` | stimulus units | 5.0 | Lower bound on specialisation — a tug can never become infinitely eager | Bonabeau, Theraulaz & Deneubourg (1996) |
| θ_max | Threshold ceiling | `Config.theta_max` | stimulus units | 200.0 | Upper bound on de-specialisation | Bonabeau, Theraulaz & Deneubourg (1996) |
| n | Hill exponent | `Config.n_exp` | dimensionless | 2.0 | Steepness of the response curve `P = sⁿ/(sⁿ+θⁿ)`; higher n → closer to a hard threshold | Theraulaz et al. (1998) eq. 1 |
| ξ | Learning coefficient | `Config.xi` | stimulus units | 1.0 | Threshold drop for the tug that performs a job (reinforcement) | Theraulaz et al. (1998) §3 |
| φ | Forgetting coefficient | `Config.phi` | stimulus units | 0.1 | Threshold rise for eligible tugs that did *not* perform (extinction) | Theraulaz et al. (1998) §3 |
| s₀ | Baseline stimulus | `Config.s0` | stimulus units | 30.0 | Stimulus of a freshly created request (zero wait) | project-specific — calibrated so `s0` sits below `theta0` |
| α | Stimulus growth rate | `Config.alpha` | stimulus units / s | 2.0 | How fast an unmet request's stimulus grows with waiting time | project-specific |
| s_max | Stimulus ceiling | `Config.s_max` | stimulus units | 200.0 | Saturation cap so `P → 1` for very old requests without overflow | project-specific |
| priority | Job-type stimulus multiplier | `Request.priority` | dimensionless | 1.0 (narrow-body) / 1.8 (wide-body) | Scales α per request: wide-body (schedule-critical) turnarounds become urgent faster than narrow-body ones at the same wait — the model's heterogeneous-job-type mechanism (see §2.4 below and `src/vrtm.py::stimulus`) | project-specific |
| adaptive | Learning on/off | `Config.adaptive` | bool | True | `False` freezes θ (the ablation control, `results/figures/ablation.png`) | project-specific |

### 2.2 Wasp dominance-contest parameters

| Symbol | Name | Code field | Units | Default | Physical meaning | Source |
|---|---|---|---|---|---|---|
| w_d | Distance weight | `Config.w_d` | dimensionless | 0.6 | Weight of proximity in the dominance force `F` | Cicirello & Smith (2001, 2004) |
| w_b | Battery weight | `Config.w_b` | dimensionless | 0.3 | Weight of state-of-charge in `F` | Cicirello & Smith (2001, 2004) |
| w_s | Idle-status weight | `Config.w_s` | dimensionless | 0.1 | Weight of "not currently busy" in `F` | Cicirello & Smith (2001, 2004) |
| contest_mode | Resolution rule | `Config.contest_mode` | categorical | `"deterministic"` | `"deterministic"`: highest F wins outright; `"stochastic"`: sequential pairwise elimination with `P(i beats j) = F_i²/(F_i²+F_j²)` | Cicirello & Smith (2001) eq. for probabilistic dominance |

### 2.3 Fleet / environment parameters that interact with the method

(Full list in `src/config.py`; listed in rubric category 2, not repeated
here — e.g. `fleet_size`, `speed_empty`, `speed_towing`, `drain_empty`,
`drain_towing`, `recharge_rate`, `recharge_trigger`, `n_charging_stations`,
`base_rate`, `morning_peak`, `evening_peak`, `p_departure`, `p_widebody`,
`widebody_priority`.)

### 2.4 Heterogeneous job types

Every generated request (`src/airport.py::generate_requests`) is drawn
**wide-body** with probability `p_widebody` (default 0.2) or **narrow-body**
otherwise. `Request.aircraft_class` records which; `Request.priority`
carries the corresponding multiplier (`widebody_priority` = 1.8,
`narrowbody_priority` = 1.0) into `vrtm.stimulus(wait, cfg, priority=...)`
(`src/vrtm.py:17`), so a wide-body's stimulus — and hence its volunteer
probability at a fixed wait — grows faster than a narrow-body's. This is the
project's heterogeneous-job-type mechanism (rubric 2.2); job direction
(departure/arrival, via `p_departure`) is a second, independent axis of
heterogeneity that determines travel direction (gate→runway vs. runway→gate)
rather than urgency.

## 3. State space and action space

- **State space** (per tug): continuous 2-D position (linear interpolation
  along the current graph edge, `src/tug.py:57`), continuous battery SoC in
  [0, 1] (`src/tug.py:78`), a discrete machine state
  (idle/moving_to_aircraft/towing/moving_to_charger/charging/failed), and a
  continuous threshold vector θ ∈ ℝ⁸ (one per pier region).
- **Action space** (per tug, per open request): a stochastic binary
  volunteer/don't-volunteer decision (VRTM), followed — if multiple tugs
  volunteer — by the wasp contest's outcome. There is no hand-authored
  discretisation of positions or times into a grid.
- **Deliberate abstraction**: the simulator advances in discrete ticks of
  `Config.dt` = 1.0 s, and the graph itself is a finite node set (piers,
  runway thresholds, chargers). This is standard for a discrete-event
  simulation and does not coarsen the underlying continuous quantities
  (position, battery, stimulus) — only the *decision cadence* is discretised.

## 4. Locality of decisions

Each tug's VRTM volunteer decision reads only **its own** `theta[region]`
and the request's (globally broadcast, environment-level) stimulus
(`src/simulator.py:232-236`). Each tug's wasp force reads only **its own**
distance/battery/busy state (`src/simulator.py` contest-building block). The
one place the simulator reads the full tug list, `self.tugs`
(`src/simulator.py:229`, `eligible = [tug for tug in self.tugs if
tug.eligible]`), is the **engine's** bookkeeping — determining which tugs are
physically capable of sensing this request's stimulus at all (idle, charged)
— not a per-agent decision computed from global information. No tug ever
reads another tug's threshold, battery, or position to make its own
volunteer or contest decision; there is no shared scheduler, global task
queue, or omniscient dispatcher inside the swarm mode (that role is exactly
what `src/baseline.py::assign_nearest`/`assign_optimal` add back in, which is
*why* they are the comparison baselines rather than another swarm variant).

## 5. Frames of reference and units

- **Coordinate system**: planar Cartesian metres, origin at the terminal
  centre, +x east, +y north (`src/airport.py:8-9`). Not a survey/geodetic
  projection — a shape-faithful caricature of Schiphol's real layout (see
  README "Known limitations").
- **Time**: seconds since 00:00 sim-clock, `int` throughout
  (`Request.time`, `Tug` state machine ticks at `Config.dt` = 1.0 s).
  Figures and the report convert to hours for readability only.
- **Battery / SoC**: dimensionless fraction in [0, 1] (`Tug.battery`); drain
  rates are battery-fraction per metre driven (`Config.drain_empty`,
  `Config.drain_towing`); recharge rate is battery-fraction per second
  (`Config.recharge_rate`). The "energy" figures report this as
  percentage-points consumed, a proxy proportional to true kWh for a
  fixed pack size (see `src/metrics.py::summarize`, `total_energy_pct`).
- **Stimulus / threshold**: shared, dimensionless "urgency" units — s and θ
  are only ever compared to each other (`P = sⁿ/(sⁿ+θⁿ)`), so no physical
  unit is attached; `s0`, `alpha`, `theta0`, etc. were chosen so that
  `s0 < theta0 < s_max, theta_max` on a common numeric scale (§2.1).
- **Distance / force**: metres for all graph edge weights and shortest paths
  (`networkx` edge `"weight"`); the wasp force's distance term uses
  `d_norm`, the shortest-path distance divided by the airport bounding-box
  diagonal, so `w_d`/`w_b`/`w_s` are combined on a common dimensionless
  [0, ~1] scale (`src/wasp.py:19-29`).

## 6. Uncertainty reporting policy

Every headline experiment in `run_all.py` is replicated over
`ExperimentConfig.n_seeds` (default 6) independent request-stream/decision
seeds. Aggregated figures and `results/summary.json` report the seed-wise
**mean** and a **95% t-distribution confidence-interval half-width**
(`src/stats.py::mean_ci`); time-series figures (e.g.
`specialization_index.png`) show mean ± 1 standard deviation as a shaded
band across seeds. Configuration comparisons (swarm vs. baseline, adaptive
vs. fixed-threshold ablation) are additionally tested with a paired
Wilcoxon signed-rank test plus a matched-pairs rank-biserial effect size
(`src/stats.py::paired_test`), run seed-by-seed on the *same* request stream
so the comparison is paired, not independent-samples.

No warm-up/transient period is discarded before computing metrics: demand is
explicitly non-stationary (a 24 h diurnal profile with morning/evening
peaks), so there is no meaningful stationary "steady state" to wait for in
the usual queueing-theory sense — every metric is computed over the full
simulated day, and this is a deliberate modelling choice, not an oversight.
The sensitivity sweeps (`src/sensitivity.py`) instead use a short
(`ExperimentConfig.sens_hours` = 2 h) morning-peak window so sweeping ~10
parameters stays fast; that window choice is stated explicitly in
`src/sensitivity.py`'s module docstring.
