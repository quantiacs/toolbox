import numpy as np
import pytest
from numpy.testing import assert_allclose

from qnt.ta.chaikin_ad import chaikin_adl_np_1d
from qnt.ta.change import change_np_1d
from qnt.ta.correlation import covariance_np_1d, std_np_1d, variance_np_1d
from qnt.ta.ema import ema_np_1d
from qnt.ta.obv import obv_np_1d
from qnt.ta.pivot_points import bottom_pivot_points_np_1d, top_pivot_points_np_1d
from qnt.ta.shift import shift_np_1d
from qnt.ta.sma import sma_np_1d
from qnt.ta.stochastic import k_np_1d
from qnt.ta.wma import lwma_np_1d, vwma_np_1d, wma_np_1d
from qnt.tests.hl_support import Floats
from qnt.tests.jit_support import as_jit

N = 120


def noisy(seed: int, scale: float = 1.0, offset: float = 50.0, with_gaps: bool = True) -> Floats:
    rng = np.random.default_rng(seed)
    values = offset + np.cumsum(rng.normal(0.0, scale, N))
    if with_gaps:
        values[rng.choice(N, 8, replace=False)] = np.nan
    return values


def prices() -> tuple[Floats, Floats, Floats, Floats]:
    close = noisy(1, with_gaps=False)
    high = close + np.abs(noisy(2, 0.3, 0.5, with_gaps=False))
    low = close - np.abs(noisy(3, 0.3, 0.5, with_gaps=False))
    volume = np.abs(noisy(4, 5.0, 100.0, with_gaps=False))
    return high, low, close, volume


def assert_jit_equals_python(dispatcher: object, *args: object) -> None:
    jit = as_jit(dispatcher)
    assert_allclose(jit(*args), jit.py_func(*args), rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("periods", [1, 2, 5, 20, N, N + 5])
@pytest.mark.parametrize(
    "func",
    [sma_np_1d, lwma_np_1d, change_np_1d, shift_np_1d, variance_np_1d, std_np_1d],
)
def test_single_series_indicators_match_python(func: object, periods: int) -> None:
    assert_jit_equals_python(func, noisy(5), periods)
    assert_jit_equals_python(func, noisy(6, with_gaps=False), periods)


@pytest.mark.parametrize("periods", [1, 3, 14, 60])
def test_ema_matches_python(periods: int) -> None:
    for warm in (periods, max(1, periods // 2)):
        assert_jit_equals_python(ema_np_1d, noisy(7), periods, warm)


@pytest.mark.parametrize("periods", [2, 10, 40])
def test_two_series_indicators_match_python(periods: int) -> None:
    high, low, close, volume = prices()
    assert_jit_equals_python(covariance_np_1d, noisy(8), noisy(9), periods)
    assert_jit_equals_python(vwma_np_1d, close, volume, periods)
    assert_jit_equals_python(k_np_1d, high, low, close, periods)


def test_volume_and_range_indicators_match_python() -> None:
    high, low, close, volume = prices()
    assert_jit_equals_python(obv_np_1d, close, volume)
    assert_jit_equals_python(chaikin_adl_np_1d, high, low, close, volume)
    assert_jit_equals_python(wma_np_1d, noisy(10), np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.parametrize("periods", [1, 3, 5, 20])
def test_pivot_points_match_python_once_window_is_full(periods: int) -> None:
    series = noisy(11, with_gaps=False)
    for func in (bottom_pivot_points_np_1d, top_pivot_points_np_1d):
        jit = as_jit(func)(series, periods)
        python = as_jit(func).py_func(series, periods)
        start = 2 * periods
        assert_allclose(jit[start:], python[start:], equal_nan=True)


@pytest.mark.parametrize(
    "func",
    [
        bottom_pivot_points_np_1d,
        pytest.param(
            top_pivot_points_np_1d,
            marks=pytest.mark.xfail(strict=True, reason="ValueError: negative dimensions not allowed for a period of 130"),
        ),
    ],
)
def test_pivot_points_period_above_int8_range(func: object) -> None:
    series = 50.0 + np.cumsum(np.random.default_rng(12).normal(0.0, 1.0, 400))
    python = as_jit(func).py_func(series, 130)
    assert_allclose(as_jit(func)(series, 130)[260:], python[260:], equal_nan=True)


@pytest.mark.parametrize("periods", [1, 3, 5, 20])
@pytest.mark.parametrize("func", [bottom_pivot_points_np_1d, top_pivot_points_np_1d])
@pytest.mark.xfail(strict=True, reason="pivot points are not NaN exactly for the first 2*periods values")
def test_pivot_points_are_nan_until_window_is_full(func: object, periods: int) -> None:
    series = noisy(13, with_gaps=False)
    result = as_jit(func)(series, periods)
    assert np.isnan(result[: 2 * periods]).all()
    assert not np.isnan(result[2 * periods :]).any()


@pytest.mark.parametrize("func", [bottom_pivot_points_np_1d, top_pivot_points_np_1d])
@pytest.mark.xfail(strict=False, reason="pivot points compare against the uninitialised np.empty window before it is full: the result is nondeterministic")
def test_pivot_points_do_not_look_ahead(func: object) -> None:
    rng = np.random.default_rng(14)
    series = 50.0 + np.cumsum(rng.normal(0.0, 1.0, 300))
    baseline = as_jit(func)(series, 5)
    for k in (40, 120, 250):
        changed = series.copy()
        changed[k + 1 :] = rng.normal(0.0, 100.0, 300 - k - 1)
        assert_allclose(as_jit(func)(changed, 5)[: k + 1], baseline[: k + 1], equal_nan=True)
