"""Wasp-inspired dominance contest.

When several tugs volunteer for the same request, each computes a "force"
(cf. dominance interactions in Polistes wasp colonies, Theraulaz et al.):

    F_i = w_d / (1 + d_norm) + w_b * battery + w_s * (1 - busy)

The contest is resolved either deterministically (highest force wins) or
stochastically via sequential pairwise elimination with

    P(i beats j) = F_i^2 / (F_i^2 + F_j^2).
"""

import numpy as np

from .config import Config


def force(d_norm: float, battery: float, busy: bool, cfg: Config) -> float:
    """Dominance force of one responder.

    ``d_norm`` is the shortest-path distance to the aircraft divided by the
    airport bounding-box diagonal, ``battery`` the charge fraction in [0, 1].
    """
    return (
        cfg.w_d / (1.0 + d_norm)
        + cfg.w_b * battery
        + cfg.w_s * (0.0 if busy else 1.0)
    )


def charge_force(d_norm: float, battery_frac: float, cfg: Config) -> float:
    """Reversed dominance force for the charging-bay queue contest.

    The mirror image of :func:`force`: favours *low* battery (urgent) and
    proximity to the charging station (cheap to send now), instead of high
    battery and proximity to a job. Only idle, non-repositioning tugs are
    ever candidates, so there is no "not busy" term.

    ``d_norm`` is the shortest-path distance to the station divided by the
    airport bounding-box diagonal, ``battery_frac`` the charge fraction in
    [0, 1].
    """
    return (
        cfg.w_d_charge / (1.0 + d_norm)
        + cfg.w_u_charge * (1.0 - battery_frac)
    )


def contest(responders: list, forces: list[float], cfg: Config,
            rng: np.random.Generator | None = None):
    """Resolve a contest; returns the winning element of ``responders``.

    A single responder wins immediately without any random draw.  Ties in
    deterministic mode go to the earliest responder in the list (stable and
    reproducible).
    """
    if len(responders) != len(forces) or not responders:
        raise ValueError("responders and forces must be equal-length, non-empty")
    if len(responders) == 1:
        return responders[0]

    if cfg.contest_mode == "deterministic":
        best = 0
        for i in range(1, len(forces)):
            if forces[i] > forces[best]:
                best = i
        return responders[best]

    if cfg.contest_mode == "stochastic":
        if rng is None:
            raise ValueError("stochastic contest needs an rng")
        champ, f_champ = responders[0], forces[0]
        for cand, f_cand in zip(responders[1:], forces[1:]):
            p_champ = f_champ**2 / (f_champ**2 + f_cand**2)
            if rng.random() >= p_champ:
                champ, f_champ = cand, f_cand
        return champ

    raise ValueError(f"unknown contest mode: {cfg.contest_mode!r}")
