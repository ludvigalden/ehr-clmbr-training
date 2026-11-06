"""Paired significance tests for two models on the same subjects.

Provides McNemar's test on paired correctness flags, the
direction-of-shift sign test on paired score differences, and the
Clopper-Pearson exact confidence interval on a binomial proportion.

McNemar's statistic is the continuity-corrected chi-square
``(|b - c| - 1)^2 / (b + c)`` on the two discordant cell counts, with
the exact two-sided binomial p-value ``min(1, 2 * P(X <= min(b, c)))``
under ``X ~ Binomial(b + c, 0.5)`` reported alongside.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, binom, chi2

__all__ = [
    "BinomialCI",
    "McNemarResult",
    "SignTestResult",
    "exact_binomial_ci",
    "mcnemar",
    "sign_test",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class McNemarResult:
    """McNemar test result for paired correctness flags.

    Attributes:
        n_both_correct: Subjects both models classified correctly.
        n_a_only_correct: Subjects only model A classified correctly.
        n_b_only_correct: Subjects only model B classified correctly.
        n_both_wrong: Subjects both models classified incorrectly.
        statistic: Continuity-corrected chi-square statistic
            ``(|b - c| - 1)^2 / (b + c)`` on the discordant counts, 1
            degree of freedom. 0.0 when there are no discordant pairs.
        p_value: Two-sided p-value from ``chi2(1)``. 1.0 when there are
            no discordant pairs.
        p_value_exact: Two-sided exact binomial p-value
            ``min(1, 2 * binom.cdf(min(b, c), b + c, 0.5))``. 1.0 when
            there are no discordant pairs.
    """

    n_both_correct: int
    n_a_only_correct: int
    n_b_only_correct: int
    n_both_wrong: int
    statistic: float
    p_value: float
    p_value_exact: float


@dataclass(frozen=True, kw_only=True, slots=True)
class SignTestResult:
    """Direction-of-shift sign test result on paired differences.

    Attributes:
        n_a_better: Positive differences (model A better).
        n_b_better: Negative differences (model B better).
        n_ties: Zero differences; excluded from the test per the
            standard sign-test convention.
        p_value: Two-sided exact binomial p-value under ``X ~
            Binomial(n_a_better + n_b_better, 0.5)`` with ``X <=
            min(n_a_better, n_b_better)``; 1.0 when every difference is
            a tie.
    """

    n_a_better: int
    n_b_better: int
    n_ties: int
    p_value: float


@dataclass(frozen=True, kw_only=True, slots=True)
class BinomialCI:
    """Clopper-Pearson exact confidence interval on a proportion.

    Attributes:
        n_successes: Number of successes ``k``.
        n: Number of trials.
        low: Lower interval bound; 0.0 when ``k = 0``.
        high: Upper interval bound; 1.0 when ``k = n``.
    """

    n_successes: int
    n: int
    low: float
    high: float


def _as_bool_array(values: NDArray[np.bool_], *, name: str) -> NDArray[np.bool_]:
    """Validate a 1-D, non-empty boolean correctness array.

    Raises:
        ValueError: If ``values`` is not a 1-D non-empty boolean array.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        msg = f"{name} must be 1-D; got shape={arr.shape}"
        raise ValueError(msg)
    if arr.dtype != np.bool_:
        msg = f"{name} must be a boolean array; got dtype={arr.dtype}"
        raise ValueError(msg)
    if arr.size == 0:
        msg = f"{name} must be non-empty"
        raise ValueError(msg)
    return arr


def mcnemar(
    correct_a: NDArray[np.bool_],
    correct_b: NDArray[np.bool_],
) -> McNemarResult:
    """McNemar's test on paired correctness flags.

    Counts the four agreement cells; only the discordant counts
    ``b = n_b_only_correct`` and ``c = n_a_only_correct`` enter the
    test. The asymptotic statistic is the continuity-corrected
    chi-square ``(|b - c| - 1)^2 / (b + c)`` and the exact p-value is
    the two-sided binomial ``min(1, 2 * binom.cdf(min(b, c), b + c,
    0.5))``.

    Args:
        correct_a: Per-subject correctness for model A (boolean).
        correct_b: Per-subject correctness for model B (boolean), same
            length and pairing as ``correct_a``.

    Returns:
        :class:`McNemarResult` with the cell counts, statistic, and
        both p-values.

    Raises:
        ValueError: If either array is not a 1-D non-empty boolean
            array or the lengths differ.
    """
    a = _as_bool_array(correct_a, name="correct_a")
    b = _as_bool_array(correct_b, name="correct_b")
    if a.size != b.size:
        msg = f"correct_a and correct_b must have equal length; got {a.size} and {b.size}"
        raise ValueError(msg)

    n_both_correct = int(np.count_nonzero(a & b))
    n_a_only_correct = int(np.count_nonzero(a & ~b))
    n_b_only_correct = int(np.count_nonzero(~a & b))
    n_both_wrong = int(np.count_nonzero(~a & ~b))
    discordant = n_a_only_correct + n_b_only_correct

    if discordant == 0:
        return McNemarResult(
            n_both_correct=n_both_correct,
            n_a_only_correct=n_a_only_correct,
            n_b_only_correct=n_b_only_correct,
            n_both_wrong=n_both_wrong,
            statistic=0.0,
            p_value=1.0,
            p_value_exact=1.0,
        )

    statistic = float((abs(n_a_only_correct - n_b_only_correct) - 1) ** 2 / discordant)
    p_value = float(chi2.sf(statistic, df=1))
    k = min(n_a_only_correct, n_b_only_correct)
    p_value_exact = float(min(1.0, 2.0 * float(binom.cdf(k, discordant, 0.5))))
    return McNemarResult(
        n_both_correct=n_both_correct,
        n_a_only_correct=n_a_only_correct,
        n_b_only_correct=n_b_only_correct,
        n_both_wrong=n_both_wrong,
        statistic=statistic,
        p_value=p_value,
        p_value_exact=p_value_exact,
    )


def sign_test(diff: Sequence[float] | NDArray[np.floating[Any]]) -> SignTestResult:
    """Direction-of-shift sign test on paired differences.

    Positive differences count toward model A, negative toward model B,
    and zeros are ties dropped from the test. Under the null both
    directions are equally likely, giving the two-sided exact binomial
    p-value ``min(1, 2 * binom.cdf(min(n_a_better, n_b_better),
    n_a_better + n_b_better, 0.5))``.

    Args:
        diff: Per-subject signed differences (model A minus model B) of
            shape ``(n,)``.

    Returns:
        :class:`SignTestResult` with the direction counts and the exact
        two-sided p-value.

    Raises:
        ValueError: If ``diff`` is not a non-empty 1-D array of finite
            values.
    """
    arr = np.asarray(diff, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"diff must be 1-D; got shape={arr.shape}"
        raise ValueError(msg)
    if arr.size == 0:
        msg = "diff must be non-empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "diff must contain only finite values"
        raise ValueError(msg)

    n_a_better = int(np.count_nonzero(arr > 0))
    n_b_better = int(np.count_nonzero(arr < 0))
    n_ties = int(np.count_nonzero(arr == 0))
    n_directional = n_a_better + n_b_better
    if n_directional == 0:
        p_value = 1.0
    else:
        k = min(n_a_better, n_b_better)
        p_value = float(min(1.0, 2.0 * float(binom.cdf(k, n_directional, 0.5))))
    return SignTestResult(
        n_a_better=n_a_better, n_b_better=n_b_better, n_ties=n_ties, p_value=p_value
    )


def exact_binomial_ci(successes: int, n: int, *, alpha: float = 0.05) -> BinomialCI:
    """Clopper-Pearson exact confidence interval on a binomial proportion.

    Lower bound ``beta.ppf(alpha / 2, k, n - k + 1)`` (0.0 when
    ``k = 0``), upper bound ``beta.ppf(1 - alpha / 2, k + 1, n - k)``
    (1.0 when ``k = n``), via ``scipy.stats.beta``.

    Args:
        successes: Number of successes ``k``, ``0 <= k <= n``.
        n: Number of trials, ``>= 1``.
        alpha: Significance level; ``alpha = 0.05`` yields a 95%
            interval.

    Returns:
        :class:`BinomialCI` with the exact two-sided interval.

    Raises:
        ValueError: If ``n < 1``, ``successes`` is outside ``[0, n]``,
            or ``alpha`` lies outside ``(0, 1)``.
    """
    if n < 1:
        msg = f"n must be >= 1; got {n}"
        raise ValueError(msg)
    if not 0 <= successes <= n:
        msg = f"successes must satisfy 0 <= k <= n; got successes={successes}, n={n}"
        raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must lie strictly inside (0, 1); got {alpha}"
        raise ValueError(msg)

    low = float(beta.ppf(alpha / 2.0, successes, n - successes + 1)) if successes > 0 else 0.0
    high = (
        float(beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes)) if successes < n else 1.0
    )
    return BinomialCI(n_successes=successes, n=n, low=low, high=high)
