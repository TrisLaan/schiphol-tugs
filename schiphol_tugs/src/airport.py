"""Schiphol-shaped airport graph, charging stations and request generation.

The layout is a deliberately simplified caricature of Schiphol: eight piers
(B, C, D, E, F, G, H, M) fan out south and east of a central terminal, and
the six runways sit at roughly their real relative positions (Polderbaan far
to the north-west, Kaagbaan oblique in the south-west, Aalsmeerbaan in the
east, Buitenveldertbaan east-west in the north, Oostbaan oblique near the
east side, Zwanenburgbaan north-south in the middle).  Coordinates are in
metres, origin at the terminal.  Absolute positions are not survey-accurate;
only the recognisable shape matters.
"""

from dataclasses import dataclass, field
import math

import networkx as nx
import numpy as np

from .config import Config, REGIONS


# ----------------------------------------------------------------------------
# Static layout tables
# ----------------------------------------------------------------------------

# Pier hub position and number of gates.  Piers curve from the west side of
# the terminal (B) around the south to the east (M), as at the real airport.
PIERS = {
    #        hub (x, y)      gates
    "B": ((-550.0, -300.0), 6),
    "C": ((-330.0, -560.0), 7),
    "D": ((0.0, -760.0), 10),
    "E": ((340.0, -660.0), 8),
    "F": ((640.0, -470.0), 7),
    "G": ((900.0, -260.0), 6),
    "H": ((1060.0, -60.0), 6),
    "M": ((1120.0, 200.0), 6),
}

# Runways: name -> (threshold_a, xy_a, threshold_b, xy_b).
RUNWAYS = {
    "Polderbaan":       ("RW18R", (-6000.0, 4800.0), "RW36L", (-6000.0, 1000.0)),
    "Zwanenburgbaan":   ("RW18C", (-2000.0, 2800.0), "RW36C", (-2000.0, -500.0)),
    "Kaagbaan":         ("RW24", (-3000.0, -2600.0), "RW06", (-1000.0, -1000.0)),
    "Aalsmeerbaan":     ("RW18L", (1800.0, 500.0),  "RW36R", (1800.0, -2800.0)),
    "Buitenveldertbaan": ("RW27", (-1500.0, 1500.0), "RW09", (2000.0, 1500.0)),
    "Oostbaan":         ("RW04", (1500.0, -300.0),  "RW22", (2900.0, 900.0)),
}

# Extra taxiway junction nodes that give the network its backbone.
JUNCTIONS = {
    "J_W": (-1000.0, -300.0),    # west of the terminal
    "J_NW": (-2000.0, 200.0),    # towards Zwanenburgbaan / Polderbaan
    "J_PB": (-4000.0, 800.0),    # long taxiway out to the Polderbaan
    "J_S": (-100.0, -1400.0),    # south, towards the Kaagbaan
    "J_E": (1450.0, -150.0),     # east, towards Aalsmeerbaan / Oostbaan
    "J_N": (100.0, 800.0),       # north, towards the Buitenveldertbaan
}


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Request:
    """A single aircraft tow request."""

    req_id: int
    time: int              # creation time, seconds since 00:00
    kind: str              # "departure" (gate -> runway) or "arrival"
    origin: str            # node the aircraft is picked up at
    dest: str              # node the aircraft is towed to
    region: str            # pier zone the aircraft is picked up in
    aircraft_class: str = "narrowbody"  # "narrowbody" or "widebody"
    priority: float = 1.0  # VRTM stimulus multiplier, see aircraft_class
    # Runtime bookkeeping, filled in by the simulator:
    tug_id: int | None = None
    tow_start: int | None = None
    tow_end: int | None = None

    def reset_assignment(self) -> None:
        """Re-open the request (used when its tug fails)."""
        self.tug_id = None
        self.tow_start = None
        self.tow_end = None


class Airport:
    """The taxiway graph plus precomputed shortest-path tables."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graph = nx.Graph()
        self.gates: dict[str, list[str]] = {}      # pier -> gate node names
        self.gate_region: dict[str, str] = {}       # gate node -> pier
        self.runway_thresholds: list[str] = []
        self.charging_stations: list[str] = []
        self._build()
        # Precompute all-pairs shortest paths once; the graph is small
        # (~120 nodes) and the simulator does millions of lookups.
        self.dist: dict[str, dict[str, float]] = {}
        self.path: dict[str, dict[str, list[str]]] = {}
        for src, (dists, paths) in nx.all_pairs_dijkstra(self.graph, weight="weight"):
            self.dist[src] = dists
            self.path[src] = paths
        # Service zone of every node: gates belong to their pier; every other
        # node (runway thresholds, junctions, chargers, terminal) belongs to
        # the zone of its nearest pier hub (shortest-path distance).  A
        # request's region is the zone its *pickup location* lies in, so a
        # zone is a pier plus the runways it naturally serves.
        self.node_region: dict[str, str] = {}
        for node in self.graph.nodes:
            if node in self.gate_region:
                self.node_region[node] = self.gate_region[node]
            else:
                self.node_region[node] = min(
                    REGIONS, key=lambda r: self.dist[node][f"PIER_{r}"]
                )
        xs = [d["xy"][0] for _, d in self.graph.nodes(data=True)]
        ys = [d["xy"][1] for _, d in self.graph.nodes(data=True)]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        # Bounding-box diagonal: normaliser for the wasp distance term.
        self.diag = math.hypot(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])

    # ------------------------------------------------------------------ build
    def _add(self, name: str, xy, kind: str) -> None:
        self.graph.add_node(name, xy=(float(xy[0]), float(xy[1])), kind=kind)

    def _edge(self, a: str, b: str) -> None:
        xa = self.graph.nodes[a]["xy"]
        xb = self.graph.nodes[b]["xy"]
        self.graph.add_edge(a, b, weight=_dist(xa, xb))

    def _build(self) -> None:
        self._add("TERM", (0.0, 0.0), "terminal")

        # Piers: a hub connected to the terminal, gates strung outward from
        # the hub along the hub's radial direction.
        for pier, (hub_xy, n_gates) in PIERS.items():
            hub = f"PIER_{pier}"
            self._add(hub, hub_xy, "pier_hub")
            self._edge("TERM", hub)
            norm = math.hypot(*hub_xy) or 1.0
            ux, uy = hub_xy[0] / norm, hub_xy[1] / norm
            self.gates[pier] = []
            for i in range(n_gates):
                gxy = (hub_xy[0] + ux * 70.0 * (i + 1), hub_xy[1] + uy * 70.0 * (i + 1))
                gname = f"{pier}{i + 1}"
                self._add(gname, gxy, "gate")
                self.gates[pier].append(gname)
                self.gate_region[gname] = pier
                # Chain gates so the pier is a finger, not a star.
                self._edge(gname, self.gates[pier][i - 1] if i else hub)

        # Adjacent pier hubs are linked (the taxiway ring around the terminal).
        order = list(PIERS)
        for a, b in zip(order, order[1:]):
            self._edge(f"PIER_{a}", f"PIER_{b}")

        for name, xy in JUNCTIONS.items():
            self._add(name, xy, "junction")

        for rwy, (na, xya, nb, xyb) in RUNWAYS.items():
            self._add(na, xya, "runway")
            self._add(nb, xyb, "runway")
            self.runway_thresholds += [na, nb]
            self._edge(na, nb)  # taxiway running alongside the runway

        # Backbone taxiways.
        for a, b in [
            ("PIER_B", "J_W"), ("J_W", "J_NW"), ("J_NW", "J_PB"),
            ("J_PB", "RW36L"),                       # out to the Polderbaan
            ("J_NW", "RW36C"), ("J_NW", "RW18C"),    # Zwanenburgbaan
            ("PIER_D", "J_S"), ("J_S", "RW06"), ("J_W", "RW06"),  # Kaagbaan
            ("PIER_H", "J_E"), ("J_E", "RW18L"),     # Aalsmeerbaan
            ("J_E", "RW04"),                         # Oostbaan
            ("PIER_M", "J_N"), ("J_N", "RW27"), ("J_N", "RW09"),  # Buitenveldertbaan
        ]:
            self._edge(a, b)

        # Charging stations, each hooked to its nearest pier hub.
        hubs = [f"PIER_{p}" for p in PIERS]
        for i, xy in enumerate(self.cfg.charging_station_xy[: self.cfg.n_charging_stations]):
            cs = f"CS{i + 1}"
            self._add(cs, xy, "charger")
            nearest = min(hubs, key=lambda h: _dist(self.graph.nodes[h]["xy"], xy))
            self._edge(cs, nearest)
            self.charging_stations.append(cs)

    # ------------------------------------------------------------------ query
    def xy(self, node: str):
        return self.graph.nodes[node]["xy"]

    def all_gates(self) -> list[str]:
        return [g for pier in REGIONS for g in self.gates[pier]]


# ----------------------------------------------------------------------------
# Request generation
# ----------------------------------------------------------------------------

def generate_requests(
    airport: Airport,
    cfg: Config,
    rng: np.random.Generator,
    duration_s: int,
    start_hour: float = 0.0,
    rate_multiplier=None,
) -> list[Request]:
    """Draw a Poisson stream of tow requests with a diurnal intensity profile.

    Requests are drawn per one-minute bin (Poisson count, then uniform
    placement inside the bin), which is equivalent to an inhomogeneous
    Poisson process at the resolution we care about.  ``rate_multiplier`` is
    an optional callable t_seconds -> factor used for the demand-spike
    experiment.
    """
    gates = airport.all_gates()
    thresholds = airport.runway_thresholds
    requests: list[Request] = []
    req_id = 0
    for minute in range(duration_s // 60):
        t0 = minute * 60
        hour = (start_hour + t0 / 3600.0) % 24.0
        lam = cfg.hourly_rate(hour) / 60.0
        if rate_multiplier is not None:
            lam *= rate_multiplier(t0)
        n = int(rng.poisson(lam))
        for _ in range(n):
            t = t0 + int(rng.integers(0, 60))
            gate = gates[int(rng.integers(len(gates)))]
            rwy = thresholds[int(rng.integers(len(thresholds)))]
            if rng.random() < cfg.p_departure:
                kind, origin, dest = "departure", gate, rwy
            else:
                kind, origin, dest = "arrival", rwy, gate
            is_widebody = rng.random() < cfg.p_widebody
            requests.append(
                Request(
                    req_id=req_id,
                    time=t,
                    kind=kind,
                    origin=origin,
                    dest=dest,
                    # The region is where the aircraft *is* (pickup zone):
                    # its pier for departures, the zone of the runway
                    # threshold for arrivals.
                    region=airport.node_region[origin],
                    aircraft_class="widebody" if is_widebody else "narrowbody",
                    priority=cfg.widebody_priority if is_widebody else cfg.narrowbody_priority,
                )
            )
            req_id += 1
    requests.sort(key=lambda r: (r.time, r.req_id))
    return requests
