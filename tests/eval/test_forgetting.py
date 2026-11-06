"""Known-value tests for the forgetting summaries.

Sign conventions: ``forgetting`` (base - continued) is positive when
the continued model got worse on a higher-is-better metric;
``backward_transfer`` (mean(continued - base)) is negative for
forgetting, opposite in sign to ``forgetting``; ``loss_increase``
(continued - base) is positive when the held-out loss grew.
"""

from __future__ import annotations

import numpy as np
import pytest

from ehrfm.eval.metrics import backward_transfer, forgetting, loss_increase


def test_forgetting_sign_convention() -> None:
    assert forgetting(0.80, 0.75) == pytest.approx(0.05, abs=1e-12)  # forgetting
    assert forgetting(0.80, 0.85) == pytest.approx(-0.05, abs=1e-12)  # improved
    assert forgetting(0.80, 0.80) == 0.0


def test_backward_transfer_sign_convention() -> None:
    base = [0.80, 0.70, 0.90]
    continued = [0.75, 0.70, 0.85]
    result = backward_transfer(base, continued)
    assert result == pytest.approx((-0.05 + 0.0 - 0.05) / 3.0, abs=1e-12)
    assert result < 0  # negative means forgetting


def test_backward_transfer_is_opposite_sign_to_mean_forgetting() -> None:
    base = [0.80, 0.70, 0.90]
    continued = [0.75, 0.72, 0.88]
    per_task_forgetting = [forgetting(b, c) for b, c in zip(base, continued, strict=True)]
    result = backward_transfer(base, continued)
    assert result == pytest.approx(-float(np.mean(per_task_forgetting)), abs=1e-12)


def test_loss_increase_sign_convention() -> None:
    assert loss_increase(0.30, 0.45) == pytest.approx(0.15, abs=1e-12)  # worse
    assert loss_increase(0.30, 0.25) == pytest.approx(-0.05, abs=1e-12)  # better


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="finite"):
        forgetting(float("nan"), 0.5)
    with pytest.raises(ValueError, match="equal length"):
        backward_transfer([0.8, 0.7], [0.75])
    with pytest.raises(ValueError, match="non-empty"):
        backward_transfer([], [])
    with pytest.raises(ValueError, match="finite"):
        backward_transfer([0.8, float("inf")], [0.75, 0.72])
    with pytest.raises(ValueError, match="finite"):
        loss_increase(0.3, float("nan"))
