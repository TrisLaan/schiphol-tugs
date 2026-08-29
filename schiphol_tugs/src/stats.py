"""Statistics helpers for multi-seed experiment aggregation.

No new dependency: scipy is already required by the project. These are the
only two things every multi-seed result in run_all.py needs:

    mean_ci(values)     -- (mean, 95% CI half-width) for an error bar
    paired_test(a, b)   -- is the difference between two paired per-seed
                           samples (e.g. swarm vs baseline mean wait, run on
                           the same seeds) larger than seed noise?
"""

import numpy as np
from scipy import stats as _sps


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """(mean, half-width) of a t-distribution confidence interval.

    With a single value the half-width is NaN (undefined dispersion); callers
    plot that as a zero-length error bar.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(v.mean())
    if n == 1:
        return mean, float("nan")
    sem = v.std(ddof=1) / np.sqrt(n)
    half = float(_sps.t.ppf((1 + confidence) / 2, n - 1) * sem)
    return mean, half


def paired_test(a, b) -> dict:
    """Wilcoxon signed-rank test between paired per-seed samples ``a``/``b``,
    plus the matched-pairs rank-biserial correlation as an effect size
    (0 = no systematic difference, +-1 = ``a`` always higher/lower than
    ``b``). Non-parametric: makes no distribution assumption across the
    (typically small) number of seeds.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = a.size
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"test": "wilcoxon", "n": int(n), "p_value": 1.0,
                "effect_size_rank_biserial": 0.0}
    _, p = _sps.wilcoxon(a, b)
    ranks = _sps.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = r_plus + r_minus
    effect = (r_plus - r_minus) / total if total else 0.0
    return {"test": "wilcoxon", "n": int(n), "p_value": float(p),
            "effect_size_rank_biserial": float(effect)}
