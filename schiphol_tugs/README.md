# Bio-inspired swarm task allocation for electric aircraft tugs at Schiphol

A self-contained Python simulation of an autonomous electric-tug fleet at a
simplified Schiphol Airport, written for the TU Delft course AE4350
(Bio-inspired Intelligence and Learning for Aerospace Applications). Tugs
assign themselves to aircraft tow requests with two bio-inspired mechanisms:

1. **Variable Response Threshold Model (VRTM)** — decides *who volunteers*.
   Each tug keeps a response threshold θ per pier region; a request emits a
   stimulus that grows with the aircraft's waiting time (faster for
   schedule-critical wide-body jobs than narrow-body ones, see
   `Config.p_widebody`), and a tug volunteers with probability
   `P = sⁿ / (sⁿ + θⁿ)` (Hill exponent n). Tugs that perform a job in a
   region lower their threshold there by ξ (specialisation); eligible tugs
   that did not perform raise theirs by φ (forgetting).
2. **Wasp-inspired dominance contest** — decides *who wins* when several tugs
   volunteer. Each responder computes a force
   `F = w_d/(1 + d_norm) + w_b·battery + w_s·(1 − busy)` and the contest is
   resolved deterministically (highest force) or stochastically
   (`P(i beats j) = F_i²/(F_i² + F_j²)`), selectable in `src/config.py`.

The decentralized swarm is validated against **three** controls on identical
request streams — a uniformly random assignment (naive floor), a centralized
greedy nearest-available dispatcher, and a centralized optimal (Hungarian)
per-step matching (ceiling). Every headline result is replicated over
`ExperimentConfig.n_seeds` (default 6) independent seeds and reported as
mean ± 95% CI; see `docs/notation.md` for the full symbol table, glossary,
units, and the project's uncertainty-reporting policy, and `references.bib`
for the bio-inspiration and airport-operations literature.

**Report**: the accompanying AE4350 report (PDF, filename containing the
student's name) is submitted alongside this repository and is not itself
part of it — see `LICENSE` for the code's licence and replace the
placeholder name there before submission.

## Install & run

```bash
pip install -r requirements.txt   # versions pinned; verified clean install
pytest -q                         # 57 tests (56 pass, 1 skipped)
python run_all.py                 # calibration, baseline comparison, sensitivity, figures (~a few minutes)
```

`run_all.py` is fully deterministic (single project seed 42 in
`src/config.py`, all draws through `numpy.random.default_rng` streams); two
consecutive runs produce byte-identical CSV/JSON outputs (verified by
`tests/test_simulator.py::test_reproducibility_same_seed_same_events` and by
re-running the pipeline and diffing `results/summary.json`). Everything
lands in `results/`:

- `results/calibration.csv`, `results/sensitivity.csv` — the parameter
  sweeps and their metrics (multi-seed, with uncertainty)
- `results/seed_results.csv` — every per-seed, per-mode raw metric row
- `results/summary.json` — calibrated values, headline metrics (mean ± CI)
  per policy, significance tests, sensitivity ranking, and a
  natural-language narrative (also printed to stdout)
- `results/figures/*.png` — the six figures below (300 DPI)

`results/animation*.mp4` and `results/snapshots.csv` (the visual-replay log)
are large, fully regeneratable (`python visualize.py`) artefacts and are
excluded from version control by `.gitignore` — see "Visualizing a run".

## Project structure

```
run_all.py          entry point: calibration, baseline comparison, sensitivity, figures, summary
src/config.py       every parameter (Config + ExperimentConfig) + seeded rng factory (seed = 42)
src/airport.py      Schiphol-shaped NetworkX graph, chargers, heterogeneous request stream
src/vrtm.py         threshold response maths (stimulus, P, adaptation)
src/wasp.py         dominance force + deterministic/stochastic contest
src/tug.py          tug agent: state machine, path following, battery
src/simulator.py    1 s discrete-event loop (swarm, baseline, random, optimal modes)
src/baseline.py     random / greedy-nearest / centralized-optimal (Hungarian) dispatch controls
src/metrics.py      waiting-time and energy/distance metrics
src/stats.py        multi-seed mean+CI and paired Wilcoxon significance testing
src/sensitivity.py  one-at-a-time + 2D-interaction parameter sweeps, tornado ranking
src/plotting.py     all figures (300 DPI PNG, labelled axes + units + legends)
src/visualizer.py   animated replay of a recorded run on the airport map
docs/notation.md    symbol table, glossary, units, locality statement, uncertainty policy
references.bib      VRTM / wasp / airport-operations / electric-taxiing citations
tests/               pytest suite: vrtm, wasp, airport, simulator, baseline, stats, sensitivity, metrics, plotting, visualizer
```

## Model summary

- **Airport**: 8 piers (B–H, M; 6–10 gates each) around a central terminal,
  6 runways at roughly real relative positions (Polderbaan far north-west,
  Kaagbaan oblique south-west, Aalsmeerbaan east, Buitenveldertbaan east-west
  north, Oostbaan oblique east, Zwanenburgbaan central), taxiway edges
  weighted by metres, 4 charging stations near the pier clusters.
- **Demand**: inhomogeneous Poisson stream over 24 h, low at night with peaks
  around 08:00 and 18:00 (~50 requests/h at peak). Departures go
  gate → runway threshold, arrivals the reverse; a request's *region* is the
  pier zone its pickup location lies in (each runway threshold belongs to the
  zone of its nearest pier), so an emergent service zone is a pier plus the
  runways it naturally serves.
- **Tugs**: 29 tugs (literature-sourced fleet size, fixed — not calibrated),
  10 m/s empty / 5 m/s towing, 300 kWh battery pack, drain 1.5 kWh/km empty /
  4.5 kWh/km towing, recharge at a 150 kW charger per station (~2 h 0→100 %)
  at their round-robin home station, withdraw immediately below 25 % state of
  charge (hard safety floor). Below 50 % but above the hard floor, an idle
  swarm tug becomes a soft charging candidate: stations broadcast a "low
  battery" signal and a reversed wasp contest (`wasp.charge_force`, favouring
  low battery and proximity, mirroring the job contest) decides who takes a
  free bay (3 per station), staggering charging departures instead of the
  whole fleet crossing one hard threshold in the same window — see
  `Simulation._run_charging_dispatch`, swarm mode only. After finishing a job
  an idle tug drifts back to the hub of its most-specialised (lowest-θ) pier
  — this converts threshold specialisation into physical zone coverage;
  baseline tugs drift to the hub nearest their home station. Drifting tugs
  stay eligible and abort the drift when they win a job. Waiting time is
  measured from request creation until the tow physically starts.
- **Calibration** (in `run_all.py`): ξ and α are swept jointly on a 4 h
  morning-peak window at the fixed fleet size (29); minimal mean wait wins,
  total distance breaks ties.

## Visualizing a run

`visualize.py` replays a recorded simulation as an animation on top of the
airport map, so the model can be verified by inspection:

```bash
python visualize.py --mode overview     # full 24 h day at 300× (default)
python visualize.py --mode inspection   # 08:00–10:00 at 60×, with event feed
```

Two presets:

- **overview** — the whole day at 300× speed (1 sim-hour ≈ 12 s of video).
  Good for watching the diurnal rhythm, charging cycles and zone formation.
- **inspection** — a two-hour peak window at 60× speed with a scrolling
  sidebar feed of the last 8 events (request created, tug assigned, contest
  X vs Y → winner, tow started/completed). For close-inspection validation.

Flags (all optional): `--start H` / `--end H` override the window (hours),
`--speed X` the playback multiplier, `--no-save` skips writing the video,
`--no-show` skips the live window (headless render), `--fresh` forces
re-recording the replay logs even if `results/snapshots.csv` etc. exist and
cover the window. The animation is saved to `results/animation.mp4` (ffmpeg;
falls back to `results/animation.gif` via Pillow with a warning if ffmpeg is
missing). Playback parameters (`fps`, `playback_speed`, `snapshot_interval_s`,
…) live in `src/config.py`.

What you see: tugs are filled circles colour-coded by state (light blue idle,
orange moving-to-aircraft, red towing, yellow moving-to-charger, green
charging) with a battery arc around each that shrinks as charge drops and
turns red below the 25 % recharge trigger. Waiting aircraft are triangles at
their pickup node (gray, then yellow after 60 s, red after 120 s of waiting)
with a live wait label; a dashed line links a tug to its aircraft while
approaching, a solid line while towing. When a wasp contest with ≥ 2
responders fires, all responders flash a yellow ring for 2 s and the winner
keeps a green ring for 1 s. The sidebar shows the simulation clock, live
state counts, running metrics (active requests, mean/longest wait, tows
completed) and a tug × pier threshold heatmap refreshed every simulated
30 minutes. The replay reads `results/snapshots.csv`,
`requests_timeline.csv`, `contests.csv` and `theta_snapshots.csv`, written by
the simulator when `record_snapshots=True`; the replayed day uses the same
rng streams as `run_all.py`, so it is the same day the report describes.

## How to read each figure

| Figure | What it shows / how to interpret it |
|---|---|
| `airport_layout.png` | The graph: gates (blue dots) clustered per labeled pier, runways as heavy lines, thresholds as yellow squares, chargers as violet triangles. Should be recognisably Schiphol-shaped. |
| `wait_time_comparison.png` | Mean and 95th-percentile aircraft wait (mean ± 95% CI across seeds) for all four dispatch policies: random (floor), swarm, centralized greedy (baseline), centralized optimal (ceiling). The swarm is competitive if it sits close to the greedy baseline and well below random. |
| `energy_comparison.png` | Fleet energy consumption (kWh) and kilometres driven. Shows the price (or saving) of decentralization in movement. |
| `calibration_heatmap.png` | 2D interaction sweep: mean wait as a function of ξ and α (stimulus growth rate) on the calibration window — the sensitivity rubric's required 2D heatmap, used here to pick the operating point. |
| `sensitivity_heatmap_xi_demand.png` | A second, independent 2D interaction sweep: mean wait as a function of ξ and demand multiplier, multi-seed per cell. |
| `sensitivity_tornado.png` | One-at-a-time sweep ranges (ξ, φ, n, θ-bound spread, w_d, demand rate, number of charging stations) against the baseline mean wait — a tornado plot ranking which parameters matter most. Fleet size is excluded: it's a fixed literature constant (29), not a decision or algorithm parameter. |

## Known limitations

- The airport geometry is a shape-faithful caricature, not survey data; taxi
  routes ignore runway crossings, one-way rules and other traffic.
- Tow requests are independent Poisson draws; real schedules bank by airline
  and runway configuration changes with wind.
- Hook-up and release times are folded into travel time rather than modelled
  as separate service durations.
- Job heterogeneity is currently one axis (wide-body vs. narrow-body urgency,
  `Config.p_widebody`/`widebody_priority`, see `docs/notation.md` §2.4);
  there is no separate deadline/missed-deadline concept, since real tow
  deadlines are set by gate/runway slot times this model does not schedule.
- No warm-up period is discarded before computing metrics: demand is
  explicitly non-stationary (diurnal), so metrics are computed over the full
  simulated day by design, not because transient effects were overlooked
  (see `docs/notation.md` §6).
