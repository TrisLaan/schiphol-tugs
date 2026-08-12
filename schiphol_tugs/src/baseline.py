"""Centralized dispatcher baselines: the comparison targets for the swarm.

Three controls, spanning a floor-to-ceiling range for "how good is the swarm":
  - :func:`assign_random`   -- a naive lower bound: ignores distance, battery,
    everything. If the swarm cannot beat this, VRTM + wasp add nothing.
  - :func:`assign_nearest`  -- the original greedy baseline: an omniscient
    dispatcher that, the moment a request is open, assigns the idle eligible
    tug with the shortest path to the aircraft.  No thresholds, no contest,
    no adaptation, but each request is handled independently (request-by-
    request greedy).
  - :func:`assign_optimal`  -- a centralized upper bound: a joint (Hungarian)
    minimal-total-distance matching between *all* currently open requests and
    *all* currently eligible tugs.  This is optimal for the instantaneous
    snapshot (it can out-perform the greedy dispatcher when two requests are
    open at once and greedy's request-by-request order sends the wrong tug to
    the wrong aircraft) but is not a globally optimal temporal schedule --
    solving that would require solving a much harder online sequential
    assignment problem, which is out of scope here.
"""

from scipy.optimize import linear_sum_assignment


def assign_nearest(open_requests, tugs, t):
    """Yield (request, winning_tug) pairs for the greedy assignment.

    Requests are served oldest-first; each assignment removes the tug from
    the pool considered for the remaining requests in this step.  Ties on
    distance go to the lowest tug id (deterministic).
    """
    pool = [tug for tug in tugs if tug.eligible]
    assignments = []
    for req in open_requests:
        if not pool:
            break
        best = min(pool, key=lambda tug: (tug.dist_to(req.origin), tug.id))
        pool.remove(best)
        assignments.append((req, best))
    return assignments


def assign_random(open_requests, tugs, rng):
    """Yield (request, winning_tug) pairs: a uniformly random eligible tug
    per request, oldest request first. The naive lower-bound control."""
    pool = [tug for tug in tugs if tug.eligible]
    assignments = []
    for req in open_requests:
        if not pool:
            break
        idx = int(rng.integers(len(pool)))
        assignments.append((req, pool.pop(idx)))
    return assignments


def assign_optimal(open_requests, tugs, t):
    """Yield (request, winning_tug) pairs from a joint min-distance matching.

    Solves assignment via :func:`scipy.optimize.linear_sum_assignment` on the
    full open-requests x eligible-tugs distance matrix. When there are more
    requests than eligible tugs (or vice-versa) the matching only pairs up
    min(n_requests, n_tugs) of them; the rest stay open for the next step,
    same as the other dispatch modes.
    """
    pool = [tug for tug in tugs if tug.eligible]
    if not pool or not open_requests:
        return []
    cost = [[tug.dist_to(req.origin) for tug in pool] for req in open_requests]
    rows, cols = linear_sum_assignment(cost)
    return [(open_requests[i], pool[j]) for i, j in zip(rows, cols)]
