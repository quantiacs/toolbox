import os
import time
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qnt.log import log_info
from qnt.stats import (
    calc_holding_log_np_nb,
    calc_relative_return_np,
    calc_relative_return_np_per_asset,
)
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

# Timing checks are machine dependent, so they run only on request:
#   QNT_RUN_SPEED_TESTS=1 python -m pytest qnt/tests/test_hl_numba_speed.py -s
pytestmark = pytest.mark.skipif(
    os.environ.get("QNT_RUN_SPEED_TESTS") != "1", reason="set QNT_RUN_SPEED_TESTS=1 to run"
)

N_POINTS = 200_000
N_DAYS = 20_000
N_ASSETS = 20
MIN_SPEEDUP = 5.0
# Vectorised numpy bodies (and the per-asset loop over tiny rows) gain little from compilation;
# for them the check is only that compilation does not make the function slower.
MIN_SPEEDUP_BY_NAME = {
    "change": 0.5,
    "obv": 0.5,
    "chaikin_adl": 0.5,
    "relative_return_per_asset": 2.0,
}
REPEATS = 3

Args = tuple[object, ...]


def walk(seed: int, scale: float = 1.0, offset: float = 50.0) -> Floats:
    rng = np.random.default_rng(seed)
    return offset + np.cumsum(rng.normal(0.0, scale, N_POINTS))


def one_series(*extra: int) -> Callable[[], Args]:
    return lambda: (walk(0), *extra)


def prices() -> tuple[Floats, Floats, Floats, Floats]:
    close = walk(1)
    high = close + np.abs(walk(2, 0.3, 0.5))
    low = close - np.abs(walk(3, 0.3, 0.5))
    volume = np.abs(walk(4, 5.0, 100.0))
    return high, low, close, volume


def two_series() -> Args:
    return walk(5), walk(6), 20


def price_volume() -> Args:
    _, _, close, volume = prices()
    return close, volume, 20


def close_volume() -> Args:
    _, _, close, volume = prices()
    return close, volume


def high_low_close_periods() -> Args:
    high, low, close, _ = prices()
    return high, low, close, 14


def high_low_close_volume() -> Args:
    return prices()


def weights_and_weights() -> Args:
    return walk(7), np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def trading_book() -> Args:
    rng = np.random.default_rng(8)
    shape = (N_DAYS, N_ASSETS)
    open_ = 50.0 * np.cumprod(1 + rng.normal(0.0, 0.01, shape), axis=0)
    close = open_ * (1 + rng.normal(0.0, 0.005, shape))
    weight = rng.normal(0.0, 0.05, shape)
    unlocked = rng.random(shape) > 0.05
    slippage = np.full(shape, 0.001)
    divs = np.zeros(shape)
    return weight, unlocked, open_, close, slippage, divs, None, None


def position_history() -> Args:
    rng = np.random.default_rng(9)
    weights = rng.normal(0.0, 0.05, (N_DAYS, N_ASSETS))
    weights[rng.random(weights.shape) < 0.3] = 0.0
    return (weights,)


CASES: list[tuple[str, object, Callable[[], Args]]] = [
    ("sma", sma_np_1d, one_series(20)),
    ("ema", ema_np_1d, one_series(20, 20)),
    ("lwma", lwma_np_1d, one_series(20)),
    ("wma", wma_np_1d, weights_and_weights),
    ("vwma", vwma_np_1d, price_volume),
    ("change", change_np_1d, one_series(5)),
    ("shift", shift_np_1d, one_series(5)),
    ("variance", variance_np_1d, one_series(20)),
    ("std", std_np_1d, one_series(20)),
    ("covariance", covariance_np_1d, two_series),
    ("obv", obv_np_1d, close_volume),
    ("chaikin_adl", chaikin_adl_np_1d, high_low_close_volume),
    ("stochastic_k", k_np_1d, high_low_close_periods),
    ("bottom_pivot", bottom_pivot_points_np_1d, one_series(5)),
    ("top_pivot", top_pivot_points_np_1d, one_series(5)),
    ("relative_return", calc_relative_return_np, trading_book),
    ("relative_return_per_asset", calc_relative_return_np_per_asset, trading_book),
    ("holding_log", calc_holding_log_np_nb, position_history),
]


def best_time(func: Callable[[], object]) -> float:
    times: list[float] = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return min(times)


@pytest.mark.parametrize(("name", "dispatcher", "make_args"), CASES, ids=[c[0] for c in CASES])
def test_compiled_function_is_faster_than_python(
    name: str, dispatcher: object, make_args: Callable[[], Args]
) -> None:
    args = make_args()
    jit = as_jit(dispatcher)
    py_func = jit.py_func
    compiled_result = jit(*args)

    compiled = best_time(lambda: jit(*args))
    interpreted = best_time(lambda: py_func(*args))
    speedup = interpreted / compiled
    log_info(
        f"{name}: compiled={compiled * 1e3:.2f} ms python={interpreted * 1e3:.1f} ms x{speedup:.0f}"
    )
    assert speedup >= MIN_SPEEDUP_BY_NAME.get(name, MIN_SPEEDUP)
    if "pivot" not in name:
        assert_allclose(compiled_result, py_func(*args), rtol=1e-9, atol=1e-12, equal_nan=True)
