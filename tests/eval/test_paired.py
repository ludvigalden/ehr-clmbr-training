"""Known-value tests for the paired significance tests.

- McNemar hand table: b = 10 (only model B correct) and c = 2 (only
  model A correct) among 12 discordant pairs. Statistic =
  (|10 - 2| - 1)^2 / 12 = 49/12. Exact p-value = 2 * P(X <= 2) under
  X ~ Binomial(12, 0.5) = 2 * (1 + 12 + 66)/4096 = 2 * 79/4096 =
  79/2048.
- Sign test hand case: differences [+1, +1, +1, +1, +1, -1, 0, 0]
  give n_a_better = 5, n_b_better = 1, n_ties = 2; exact p-value =
  2 * P(X <= 1) under X ~ Binomial(6, 0.5) = 2 * 7/64 = 7/32.
- Clopper-Pearson with k = 0, n = 10, alpha = 0.05: low = 0 and
  high = 1 - 0.025^(1/10).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom, chi2

from ehrfm.eval.metrics import exact_binomial_ci, mcnemar, sign_test


def test_mcnemar_hand_table() -> None:
    # 20 subjects: 10 (False, True), 2 (True, False), 4 (True, True),
    # 4 (False, False).
    correct_a = np.array([False] * 10 + [True] * 2 + [True] * 4 + [False] * 4)
    correct_b = np.array([True] * 10 + [False] * 2 + [True] * 4 + [False] * 4)
    result = mcnemar(correct_a, correct_b)
    assert result.n_both_correct == 4
    assert result.n_a_only_correct == 2
    assert result.n_b_only_correct == 10
    assert result.n_both_wrong == 4
    assert result.statistic == pytest.approx(49.0 / 12.0, abs=1e-12)
    assert result.p_value == pytest.approx(float(chi2.sf(49.0 / 12.0, df=1)), abs=1e-14)
    assert result.p_value_exact == pytest.approx(79.0 / 2048.0, abs=1e-12)
    # Cross-check the exact p-value against the binomial CDF.
    assert result.p_value_exact == pytest.approx(2.0 * float(binom.cdf(2, 12, 0.5)), abs=1e-12)


def test_mcnemar_with_no_discordant_pairs() -> None:
    correct_a = np.array([True, True, False, False])
    correct_b = np.array([True, True, False, False])
    result = mcnemar(correct_a, correct_b)
    assert result.n_a_only_correct == 0
    assert result.n_b_only_correct == 0
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.p_value_exact == 1.0


def test_mcnemar_validates_inputs() -> None:
    a = np.array([True, False])
    b_int = np.array([1, 0])
    with pytest.raises(ValueError, match="boolean"):
        mcnemar(a, b_int)
    with pytest.raises(ValueError, match="equal length"):
        mcnemar(a, np.array([True]))


def test_sign_test_hand_case() -> None:
    diff = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0])
    result = sign_test(diff)
    assert result.n_a_better == 5
    assert result.n_b_better == 1
    assert result.n_ties == 2
    assert result.p_value == pytest.approx(7.0 / 32.0, abs=1e-12)


def test_sign_test_all_ties_gives_p_one() -> None:
    result = sign_test(np.array([0.0, 0.0, 0.0]))
    assert result.n_a_better == 0
    assert result.n_b_better == 0
    assert result.n_ties == 3
    assert result.p_value == 1.0


def test_sign_test_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        sign_test(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        sign_test(np.array([0.1, np.nan]))


def test_exact_binomial_ci_known_bounds() -> None:
    result = exact_binomial_ci(0, 10, alpha=0.05)
    assert result.n_successes == 0
    assert result.n == 10
    assert result.low == 0.0
    assert result.high == pytest.approx(1.0 - 0.025 ** (1.0 / 10.0), abs=1e-12)

    upper = exact_binomial_ci(10, 10, alpha=0.05)
    assert upper.low == pytest.approx(0.025 ** (1.0 / 10.0), abs=1e-12)
    assert upper.high == 1.0


def test_exact_binomial_ci_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        exact_binomial_ci(0, 0)
    with pytest.raises(ValueError, match="0 <= k <= n"):
        exact_binomial_ci(11, 10)
    with pytest.raises(ValueError, match="alpha"):
        exact_binomial_ci(3, 10, alpha=0.0)
