"""Variable Response Threshold Model (VRTM) mathematics.

Pure functions only; the simulator owns the state.  Each tug i keeps a
threshold theta_{i,r} per pier region r.  A request emits a stimulus that
grows with the aircraft's waiting time; the probability that a tug
volunteers follows the classic Hill response curve

    P = s^n / (s^n + theta^n)

(Bonabeau/Theraulaz fixed-threshold model, made adaptive via the
reinforcement rules in :func:`update_performer` / :func:`update_non_performer`).
"""

from .config import Config


def stimulus(waiting_time_s: float, cfg: Config, priority: float = 1.0) -> float:
    """Stimulus of a request whose aircraft has waited ``waiting_time_s``.

    ``priority`` (default 1.0, the historical behaviour) scales how fast the
    stimulus grows with waiting time: it is how a heterogeneous job type
    (e.g. a schedule-critical wide-body turnaround, see
    ``Config.widebody_priority``) becomes more urgent for the same wait than
    a routine one, without changing the ceiling ``s_max`` both share.
    """
    return min(cfg.s_max, cfg.s0 + cfg.alpha * priority * waiting_time_s)


def response_probability(s: float, theta: float, cfg: Config) -> float:
    """Hill response curve: probability that a tug volunteers."""
    sn = s ** cfg.n_exp
    tn = theta ** cfg.n_exp
    return sn / (sn + tn)


def update_performer(theta: float, cfg: Config) -> float:
    """The tug that performed the job specialises: threshold drops by xi."""
    return max(cfg.theta_min, theta - cfg.xi)


def update_non_performer(theta: float, cfg: Config) -> float:
    """Eligible tugs that did not perform de-specialise: threshold rises by phi."""
    return min(cfg.theta_max, theta + cfg.phi)
