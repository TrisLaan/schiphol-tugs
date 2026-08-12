"""The autonomous electric tug agent.

A tug is a small state machine (idle / moving_to_aircraft / towing /
moving_to_charger / charging / failed) that moves along shortest paths on
the airport graph.  Progress along the current path is tracked as a scalar
distance; the (x, y) position is interpolated linearly along the current
edge on demand.  Battery drains per metre (more when towing) and recharges
at the tug's home station.
"""

from .airport import Airport
from .config import Config, REGIONS

IDLE = "idle"
TO_AIRCRAFT = "moving_to_aircraft"
TOWING = "towing"
TO_CHARGER = "moving_to_charger"
CHARGING = "charging"
FAILED = "failed"


class Tug:
    def __init__(self, tug_id: int, home_station: str, airport: Airport, cfg: Config):
        self.id = tug_id
        self.cfg = cfg
        self.airport = airport
        self.home_station = home_station   # assigned round-robin by the simulator
        self.node = home_station           # last graph node reached
        self.state = IDLE
        self.repositioning = False   # idle but drifting back to a pier hub
        self.battery = 1.0
        self.min_battery = 1.0             # for the never-negative invariant test
        self.theta = {r: cfg.theta0 for r in REGIONS}
        self.request = None                # Request currently being served
        # Path-following state.
        self._path: list[str] = []
        self._cum: list[float] = [0.0]     # cumulative edge lengths along _path
        self._along = 0.0                  # metres travelled along _path
        # Lifetime statistics.
        self.total_distance = 0.0          # metres driven (empty + towing)
        self.energy_used = 0.0             # battery fraction consumed by driving

    # ------------------------------------------------------------- path logic
    def set_destination(self, node: str) -> None:
        """Start following the shortest path from the current node."""
        self._path = self.airport.path[self.node][node]
        cum = [0.0]
        for a, b in zip(self._path, self._path[1:]):
            cum.append(cum[-1] + self.airport.graph.edges[a, b]["weight"])
        self._cum = cum
        self._along = 0.0

    @property
    def path_remaining(self) -> float:
        return self._cum[-1] - self._along

    def position(self) -> tuple[float, float]:
        """Current (x, y), linearly interpolated along the current edge."""
        if not self._path or self._along >= self._cum[-1]:
            return self.airport.xy(self.node)
        for i in range(len(self._path) - 1):
            if self._along <= self._cum[i + 1]:
                a = self.airport.xy(self._path[i])
                b = self.airport.xy(self._path[i + 1])
                seg = self._cum[i + 1] - self._cum[i]
                f = 0.0 if seg == 0 else (self._along - self._cum[i]) / seg
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        return self.airport.xy(self._path[-1])

    def advance(self, dt: float) -> bool:
        """Move for ``dt`` seconds; drain battery; return True on arrival."""
        speed = self.cfg.speed_towing if self.state == TOWING else self.cfg.speed_empty
        drain = self.cfg.drain_towing if self.state == TOWING else self.cfg.drain_empty
        step = min(speed * dt, self.path_remaining)
        self._along += step
        self.total_distance += step
        used = step * drain
        self.battery = max(0.0, self.battery - used)
        self.energy_used += used
        self.min_battery = min(self.min_battery, self.battery)
        if self.path_remaining <= 1e-9:
            if self._path:
                self.node = self._path[-1]
            return True
        # Snap self.node to the last node passed, so distance queries made
        # mid-path stay meaningful for a moving tug.
        for i in range(len(self._path) - 1):
            if self._along < self._cum[i + 1]:
                self.node = self._path[i]
                break
        return False

    # ------------------------------------------------------------ convenience
    @property
    def eligible(self) -> bool:
        """Idle, healthy, and enough charge to accept work."""
        return self.state == IDLE and self.battery >= self.cfg.recharge_trigger

    def dist_to(self, node: str) -> float:
        """Shortest-path distance from the tug's reference node to ``node``."""
        return self.airport.dist[self.node][node]
