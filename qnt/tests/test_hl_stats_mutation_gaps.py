import json
import logging
import math
import re
import urllib.request
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.output as qnout
import qnt.stats as qnstats
from qnt.tests.hl_support import (
    FIELDS,
    Floats,
    business_days,
    field_values,
    make_data,
    make_weights,
    trailing_mean,
    trailing_std,
)

pytestmark = pytest.mark.usefixtures("quiet_tracking")

Bools: TypeAlias = npt.NDArray[np.bool_]
Correlation: TypeAlias = dict[str, object]

EPS = 1e-7
ATOL = 1e-13
N_PIPE = 24
PIPE_FRACT = 0.03
PIPE_ROLL_FRACT = 0.07


def dynamic(name: str) -> str:
    return "".join(list(name))


def named(name: str) -> xr.DataArray:
    return xr.DataArray(0.0, name=dynamic(name))


def frame(
    rows: Sequence[Sequence[float]], start: str = "2020-01-01", names: Sequence[str] = ()
) -> xr.DataArray:
    values = np.array(rows, dtype=np.float64)
    assets = list(names) if names else [f"NYS:A{i}" for i in range(values.shape[1])]
    return xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={"time": business_days(values.shape[0], start), "asset": assets},
    )


def series(values: Sequence[float], start: str = "2020-01-01") -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=np.float64),
        dims=["time"],
        coords={"time": business_days(len(values), start)},
    )


def ratio(numerator: Floats, denominator: Floats) -> Floats:
    with np.errstate(invalid="ignore", divide="ignore"):
        return numerator / denominator


def ffilled(values: Floats) -> Floats:
    return pd.DataFrame(values).ffill().fillna(0.0).to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# independent references of the equity simulation
# ---------------------------------------------------------------------------------------------


def simulate_per_asset(
    weight: Floats,
    unlocked: Bools,
    open_: Floats,
    close: Floats,
    slippage: Floats,
    divs: Floats,
    roll: Floats | None,
    roll_slippage: Floats | None,
) -> Floats:
    """Every asset owns a separate account of 1.0; a locked asset freezes its account."""
    n_days, n_assets = weight.shape
    returns = np.zeros((n_days, n_assets))
    for asset in range(n_assets):
        shares_prev = 0.0
        before_prev = 1.0
        after_prev = 1.0
        tonight_prev = 1.0
        for day in range(n_days):
            if day == 0:
                before = 1.0
            else:
                price_move = open_[day, asset] - open_[day - 1, asset] + divs[day, asset]
                before = after_prev + price_move * shares_prev
            shares = shares_prev
            if unlocked[day, asset]:
                shares = before * weight[day, asset] / open_[day, asset]
            after = before - slippage[day, asset] * abs(shares - shares_prev)
            if roll is not None and roll_slippage is not None and day > 0:
                kept = (
                    min(abs(shares), abs(shares_prev))
                    if np.sign(shares) == np.sign(shares_prev)
                    else 0.0
                )
                after -= float(np.sign(shares)) * kept * roll[day, asset]
                after -= kept * roll_slippage[day, asset]
            tonight = after + (close[day, asset] - open_[day, asset]) * shares
            if not unlocked[day, asset]:
                if day == 0:
                    before, after, tonight, shares = 1.0, 1.0, 1.0, 0.0
                else:
                    before, after, tonight, shares = (
                        before_prev,
                        after_prev,
                        tonight_prev,
                        shares_prev,
                    )
            returns[day, asset] = tonight / tonight_prev - 1.0
            shares_prev, before_prev, after_prev, tonight_prev = shares, before, after, tonight
    return np.where(np.isfinite(returns), returns, 0.0)


def simulate_total(
    weight: Floats,
    unlocked: Bools,
    open_: Floats,
    close: Floats,
    slippage: Floats,
    divs: Floats,
    roll: Floats | None,
    roll_slippage: Floats | None,
) -> Floats:
    """One account; locked assets keep their shares for the day, then the shares are dropped."""
    n_days, n_assets = weight.shape
    returns = np.zeros(n_days)
    shares_prev = np.zeros(n_assets)
    after_prev = 1.0
    tonight_prev = 1.0
    for day in range(n_days):
        locked = ~unlocked[day]
        shares = shares_prev.copy()
        if day == 0:
            before = 1.0
        else:
            moves = (open_[day] - open_[day - 1] + divs[day]) * shares
            before = after_prev + float(np.nansum(moves))
        total_weight = float(np.nansum(np.abs(weight[day])))
        free_cash = max(1.0, total_weight) - total_weight
        operable_weight = float(np.nansum(np.abs(weight[day][unlocked[day]]))) + free_cash
        locked_value = float(np.nansum(open_[day][locked] * np.abs(shares[locked])))
        if operable_weight < EPS:
            after = before
        else:
            shares[unlocked[day]] = (
                (before - locked_value)
                * weight[day][unlocked[day]]
                / (operable_weight * open_[day][unlocked[day]])
            )
            traded = shares[unlocked[day]] - (shares_prev[unlocked[day]] if day > 0 else 0.0)
            after = before - float(np.nansum(slippage[day][unlocked[day]] * np.abs(traded)))
        if roll is not None and roll_slippage is not None and day > 0:
            kept = np.where(
                np.sign(shares) == np.sign(shares_prev),
                np.minimum(np.abs(shares), np.abs(shares_prev)),
                0.0,
            )
            after -= float(
                np.nansum(np.sign(shares) * kept * roll[day] + kept * roll_slippage[day])
            )
        tonight = after + float(np.nansum((close[day] - open_[day]) * shares))
        shares[locked] = 0.0
        returns[day] = tonight / tonight_prev - 1.0
        shares_prev, after_prev, tonight_prev = shares, after, tonight
    return np.where(np.isfinite(returns), returns, 0.0)


@dataclass(frozen=True)
class SimInputs:
    weight: Floats
    unlocked: Bools
    open_: Floats
    close: Floats
    slippage: Floats
    divs: Floats
    roll: Floats | None
    roll_slippage: Floats | None

    def per_asset(self) -> Floats:
        return simulate_per_asset(
            self.weight,
            self.unlocked,
            self.open_,
            self.close,
            self.slippage,
            self.divs,
            self.roll,
            self.roll_slippage,
        )

    def total(self) -> Floats:
        return simulate_total(
            self.weight,
            self.unlocked,
            self.open_,
            self.close,
            self.slippage,
            self.divs,
            self.roll,
            self.roll_slippage,
        )

    def numba_per_asset(self, roll: Floats, roll_slippage: Floats) -> Floats:
        return qnstats.calc_relative_return_np_per_asset(
            self.weight,
            self.unlocked,
            self.open_,
            self.close,
            self.slippage,
            self.divs,
            roll,
            roll_slippage,
        )

    def numba_total(self, roll: Floats, roll_slippage: Floats) -> Floats:
        return qnstats.calc_relative_return_np(
            self.weight,
            self.unlocked,
            self.open_,
            self.close,
            self.slippage,
            self.divs,
            roll,
            roll_slippage,
        )

    def run_numba_per_asset(self, with_roll: bool) -> Floats:
        zeros = np.zeros_like(self.weight)
        roll, roll_slippage = (self.roll, self.roll_slippage) if with_roll else (zeros, zeros)
        if roll is None or roll_slippage is None:
            raise ValueError("inputs have no roll data")
        return self.numba_per_asset(roll, roll_slippage)

    def run_numba_total(self, with_roll: bool) -> Floats:
        zeros = np.zeros_like(self.weight)
        roll, roll_slippage = (self.roll, self.roll_slippage) if with_roll else (zeros, zeros)
        if roll is None or roll_slippage is None:
            raise ValueError("inputs have no roll data")
        return self.numba_total(roll, roll_slippage)

    def without_roll(self) -> "SimInputs":
        return SimInputs(
            self.weight,
            self.unlocked,
            self.open_,
            self.close,
            self.slippage,
            self.divs,
            None,
            None,
        )


def hand_inputs() -> SimInputs:
    """5 days x 3 assets: sign flips, partial exits, dividends, locked assets, a fully locked day."""
    nan = np.nan
    return SimInputs(
        weight=np.array(
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.3, 0.2],
                [-0.4, 0.2, nan],
                [-0.4, -0.3, 0.5],
                [0.2, 0.0, 0.4],
            ]
        ),
        unlocked=np.array(
            [
                [True, True, True],
                [True, True, False],
                [True, False, False],
                [False, False, False],
                [True, True, True],
            ]
        ),
        open_=np.array(
            [[10, 20, 5], [11, 19, 5.5], [10.5, 21, 6], [12, 22, 5.8], [11.5, 20, 6.5]],
            dtype=np.float64,
        ),
        close=np.array(
            [
                [10.8, 19.5, 5.2],
                [10.6, 20, 5.9],
                [11.4, 21.5, 5.7],
                [11.7, 21, 6.1],
                [12.2, 19, 6.4],
            ]
        ),
        slippage=np.asfortranarray(
            [
                [0.5, 0.8, 0.2],
                [0.6, 0.7, 0.3],
                [0.5, 0.9, 0.2],
                [0.4, 0.8, 0.3],
                [0.5, 0.7, 0.4],
            ]
        ),
        divs=np.array(
            [
                [0, 0, 0],
                [0.3, 0, 0.1],
                [0, 0.5, 0],
                [0.2, 0, 0],
                [0, 0.4, 0.05],
            ],
            dtype=np.float64,
        ),
        roll=np.array(
            [
                [0.2, 0.3, 0.1],
                [0.5, 0.4, -0.4],
                [0.3, 0.6, 0.3],
                [0.7, -0.5, 0.2],
                [0.4, 0.3, 0.2],
            ]
        ),
        roll_slippage=np.array(
            [
                [0.1, 0.1, 0.1],
                [0.3, 0.2, 0.2],
                [0.2, 0.25, 0.3],
                [0.35, 0.3, 0.1],
                [0.2, 0.1, 0.15],
            ]
        ),
    )


@pytest.fixture(scope="module", autouse=True)
def warm_up_numba() -> None:
    inputs = hand_inputs()
    for with_roll in (False, True):
        inputs.run_numba_per_asset(with_roll)
        inputs.run_numba_total(with_roll)
    qnstats.calc_holding_log_np_nb(np.zeros((3, 2)))
    qnstats.calc_avg_holding_time(book_for_holding(), points_per_year=252)
    data = market("futures", True)
    book = make_weights(data, seed=8)
    for per_asset in (False, True):
        qnstats.calc_relative_return(data, book, PIPE_FRACT, PIPE_ROLL_FRACT, per_asset=per_asset)


def idle_asset_inputs(
    weight: list[float], open_: list[float], close: list[float], divs: list[float]
) -> SimInputs:
    """One traded asset plus an idle second asset (zero weight, flat price)."""
    days = len(weight)
    return SimInputs(
        weight=np.column_stack([weight, np.zeros(days)]),
        unlocked=np.ones((days, 2), dtype=np.bool_),
        open_=np.column_stack([open_, np.ones(days)]),
        close=np.column_stack([close, np.ones(days)]),
        slippage=np.asfortranarray(np.column_stack([np.full(days, 0.2), np.zeros(days)])),
        divs=np.column_stack([divs, np.zeros(days)]),
        roll=np.column_stack([[0.0, 3.0], np.zeros(days)]),
        roll_slippage=np.column_stack([[0.0, 2.0], np.zeros(days)]),
    )


def test_per_asset_numba_matches_hand_computed_account() -> None:
    inputs = idle_asset_inputs([0.5, 0.5], [10.0, 12.0], [11.0, 12.0], [0.0, 1.0])
    result = inputs.run_numba_per_asset(False)
    # day 0: 0.05 shares, equity 1 - 0.2 * 0.05 + (11 - 10) * 0.05 = 1.04
    # day 1: equity 0.99 + (12 - 10 + 1) * 0.05 = 1.14, 0.0475 shares, sold 0.0025 shares
    assert_allclose(result[:, 0], [0.04, (1.14 - 0.2 * 0.0025) / 1.04 - 1.0], rtol=1e-12)
    assert_array_equal(result[:, 1], [0.0, 0.0])


def test_total_numba_matches_hand_computed_account_with_roll() -> None:
    inputs = idle_asset_inputs([0.5, 0.25], [10.0, 12.0], [11.0, 12.0], [0.0, 0.0])
    plain = inputs.run_numba_total(False)
    rolled = inputs.run_numba_total(True)
    # day 0: 0.05 shares, 0.99 + 0.05 = 1.04; day 1: 1.09 before, 1.09 * 0.25 / 12 shares
    shares = 1.09 * 0.25 / 12.0
    after_buy = 1.09 - 0.2 * (0.05 - shares)
    assert_allclose(plain[1], after_buy / 1.04 - 1.0, rtol=1e-12)
    assert_allclose(rolled[1], (after_buy - shares * 3.0 - shares * 2.0) / 1.04 - 1.0, rtol=1e-12)


@pytest.mark.parametrize("with_roll", [False, True])
def test_per_asset_numba_matches_independent_simulation(with_roll: bool) -> None:
    inputs = hand_inputs() if with_roll else hand_inputs().without_roll()
    expected = inputs.per_asset()
    actual = inputs.run_numba_per_asset(with_roll)
    assert_allclose(actual, expected, rtol=1e-11, atol=ATOL)
    assert np.abs(expected).max() > 0.01


@pytest.mark.parametrize("with_roll", [False, True])
def test_total_numba_matches_independent_simulation(with_roll: bool) -> None:
    inputs = hand_inputs() if with_roll else hand_inputs().without_roll()
    expected = inputs.total()
    actual = inputs.run_numba_total(with_roll)
    assert_allclose(actual, expected, rtol=1e-11, atol=ATOL)
    assert np.abs(expected).max() > 0.01


def test_roll_costs_change_results() -> None:
    inputs = hand_inputs()
    plain_pa = inputs.run_numba_per_asset(False)
    plain_total = inputs.run_numba_total(False)
    assert np.abs(inputs.run_numba_per_asset(True) - plain_pa).max() > 1e-3
    assert np.abs(inputs.run_numba_total(True) - plain_total).max() > 1e-3


def test_total_numba_skips_trading_when_operable_weight_equals_eps() -> None:
    weight = np.array([[EPS, 1.0], [EPS, 1.0]])
    inputs = SimInputs(
        weight=weight,
        unlocked=np.array([[True, False], [True, False]]),
        open_=np.array([[10.0, 20.0], [11.0, 19.0]]),
        close=np.array([[10.5, 20.5], [11.5, 19.5]]),
        slippage=np.asfortranarray([[0.3, 0.3], [0.3, 0.3]]),
        divs=np.zeros((2, 2)),
        roll=None,
        roll_slippage=None,
    )
    actual = inputs.run_numba_total(False)
    assert_allclose(actual, inputs.total(), rtol=1e-11, atol=ATOL)
    assert_allclose(actual[0], 0.02, rtol=1e-11)


def test_total_numba_ignores_nan_weights_and_slippage_of_locked_assets() -> None:
    inputs = hand_inputs().without_roll()
    weight = inputs.weight.copy()
    weight[1, 2] = np.nan
    slippage = np.asfortranarray(inputs.slippage.copy())
    slippage[1:3, 2] = np.nan
    changed = SimInputs(
        weight,
        inputs.unlocked,
        inputs.open_,
        inputs.close,
        slippage,
        inputs.divs,
        None,
        None,
    )
    actual = changed.run_numba_total(False)
    assert np.isfinite(actual).all()
    assert_allclose(actual, changed.total(), rtol=1e-11, atol=ATOL)


# ---------------------------------------------------------------------------------------------
# calc_relative_return end to end
# ---------------------------------------------------------------------------------------------


def market(name: str, extras: bool, gap: bool = True, n_days: int = N_PIPE) -> xr.DataArray:
    """Synthetic prices; extras adds non-zero dividends and a roll field, gap blanks an open price."""
    base = make_data(n_assets=3, n_days=n_days, name=dynamic(name), seed=17)
    values = base.values.copy()
    if gap:
        values[16:18, 0, 1] = np.nan
    fields = list(FIELDS)
    if extras:
        rng = np.random.default_rng(3)
        values[:, 5, :] = rng.uniform(0.2, 2.0, (n_days, 3))
        roll = rng.choice([0.0, 1.5, -1.0, 2.5], size=(n_days, 1, 3))
        values = np.concatenate([values, roll], axis=1)
        fields.append("roll")
    else:
        keep = [i for i, field in enumerate(FIELDS) if field != "divs"]
        values = values[:, keep, :]
        fields = [FIELDS[i] for i in keep]
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={
            "time": base.coords["time"].values,
            "field": fields,
            "asset": base.coords["asset"].values,
        },
        name=base.name,
    )


def normalize_rows(values: Floats, per_asset: bool) -> Floats:
    clean = np.nan_to_num(values)
    if per_asset:
        return np.clip(clean, -1.0, 1.0).astype(np.float64)
    scale = np.maximum(np.abs(clean).sum(axis=1), 1.0)
    return np.asarray(clean / scale[:, None], dtype=np.float64)


def pipeline_inputs(
    data: xr.DataArray, weights: xr.DataArray, fract: float, roll_fract: float, per_asset: bool
) -> SimInputs:
    fields = list(data.coords["field"].values)
    open_ = field_values(data, "open")
    high = field_values(data, "high")
    low = field_values(data, "low")
    close = field_values(data, "close")
    prev_raw = np.vstack([np.full((1, open_.shape[1]), np.nan), close[:-1]])
    true_range = np.maximum.reduce([high - low, np.abs(high - prev_raw), np.abs(prev_raw - low)])
    atr = trailing_mean(true_range, 14, 14)
    slippage = atr * fract
    divs = np.nan_to_num(field_values(data, "divs")) if "divs" in fields else np.zeros_like(open_)
    roll = np.nan_to_num(field_values(data, "roll")) if "roll" in fields else None
    roll_slippage = (
        np.where(roll != 0, np.nan_to_num(atr * roll_fract), 0.0) if roll is not None else None
    )
    shifted = np.vstack([np.zeros((1, open_.shape[1])), np.nan_to_num(weights.values)[:-1]])
    prev_close = pd.DataFrame(close).shift(1).ffill().to_numpy(dtype=np.float64)
    target = normalize_rows(shifted, per_asset)
    target = normalize_rows(np.nan_to_num(target * open_ / prev_close), per_asset)
    open_sim = ffilled(open_)
    unlocked = np.isfinite(slippage) & np.isfinite(open_) & np.isfinite(close) & (open_sim > EPS)
    return SimInputs(
        target, unlocked, open_sim, ffilled(close), slippage, divs, roll, roll_slippage
    )


@pytest.mark.parametrize("per_asset", [False, True])
def test_relative_return_matches_independent_pipeline(per_asset: bool) -> None:
    data = market("futures", True)
    raw = make_weights(data, seed=8) * (0.6 if per_asset else 0.5)
    before = raw.values.copy()
    actual = qnstats.calc_relative_return(
        data, raw, PIPE_FRACT, PIPE_ROLL_FRACT, per_asset=per_asset
    )
    inputs = pipeline_inputs(data, raw, PIPE_FRACT, PIPE_ROLL_FRACT, per_asset)
    expected = inputs.per_asset() if per_asset else inputs.total()
    assert actual.dims == (("time", "asset") if per_asset else ("time",))
    assert_allclose(actual.values, expected, rtol=1e-9, atol=1e-12)
    assert np.abs(expected[15:]).max() > 0.001
    assert_array_equal(raw.values, before)
    plain = pipeline_inputs(data, raw, PIPE_FRACT, PIPE_FRACT, per_asset)
    other = plain.per_asset() if per_asset else plain.total()
    assert np.abs(expected - other).max() > 1e-4


def test_relative_return_uses_separate_atr_factor_for_roll_slippage() -> None:
    data = market("futures", True, gap=False)
    raw = make_weights(data, seed=2) * 0.5
    same = qnstats.calc_relative_return(data, raw, 0.03, 0.03)
    other = qnstats.calc_relative_return(data, raw, 0.03, 0.12)
    expected_same = pipeline_inputs(data, raw, 0.03, 0.03, False).total()
    expected_other = pipeline_inputs(data, raw, 0.03, 0.12, False).total()
    assert_allclose(same.values, expected_same, rtol=1e-9, atol=1e-12)
    assert_allclose(other.values, expected_other, rtol=1e-9, atol=1e-12)
    assert np.abs(expected_same - expected_other).max() > 1e-4


# ---------------------------------------------------------------------------------------------
# defaults per market type
# ---------------------------------------------------------------------------------------------

CRYPTODAILY_NAMES = ("cryptodaily", "crypto_daily", "crypto_daily_long", "crypto_daily_long_short")

SLIPPAGE_ROWS: list[tuple[str, str, float]] = [
    ("stocks_s&p500", "SL_STOCKS_S&P500", 0.05),
    ("stocks_nasdaq100", "SL_STOCKS_NASDAQ100", 0.05),
    ("stocks", "SL_STOCKS", 0.05),
    ("futures", "SL_FUTURES", 0.04),
    ("cryptofutures", "SL_CRYPTOFUTURES", 0.04),
    ("crypto_futures", "SL_CRYPTOFUTURES", 0.04),
    *[(name, "SL_CRYPTODAILY", 0.04) for name in CRYPTODAILY_NAMES],
    ("crypto", "SL_CRYPTO", 0.05),
]
IS_PERIOD_ROWS: list[tuple[str, str, int]] = [
    ("stocks_s&p500", "IS_STOCKS_S&P500", 4536),
    ("stocks_nasdaq100", "IS_STOCKS_NASDAQ100", 3528),
    ("stocks", "IS_STOCKS", 1512),
    ("stocks_long", "IS_STOCKS", 1512),
    ("futures", "IS_FUTURES", 3528),
    ("cryptofutures", "IS_CRYPTOFUTURES", 1764),
    ("crypto_futures", "IS_CRYPTOFUTURES", 1764),
    *[(name, "IS_CRYPTOFUTURES", 1764) for name in CRYPTODAILY_NAMES],
    ("crypto", "IS_CRYPTO", 60000),
]
START_DATE_ROWS: list[tuple[str, str, str]] = [
    ("stocks_s&p500", "SD_STOCKS_S&P500", "2006-01-01"),
    ("stocks_nasdaq100", "SD_STOCKS_NASDAQ100", "2006-01-01"),
    ("stocks", "SD_STOCKS", "2015-01-01"),
    ("stocks_long", "SD_STOCKS", "2015-01-01"),
    ("futures", "SD_FUTURES", "2006-01-01"),
    ("cryptofutures", "SD_CRYPTOFUTURES", "2014-01-01"),
    ("crypto_futures", "SD_CRYPTOFUTURES", "2014-01-01"),
    *[(name, "SD_CRYPTODAILY", "2016-01-01") for name in CRYPTODAILY_NAMES],
    ("crypto", "SD_CRYPTO", "2014-01-01"),
]
MAX_EXPOSURE_ROWS: list[tuple[str, str, float]] = [
    ("stocks_s&p500", "ME_STOCKS_S&P500", 0.1),
    ("stocks_nasdaq100", "ME_STOCKS_NASDAQ100", 0.1),
]
UNKNOWN_NAMES = ["stocks_long_short", "unknown", "", "bitcoin", "stocks_s&p501", "zzz"]
ENV_KEYS = sorted(
    {row[1] for row in SLIPPAGE_ROWS}
    | {row[1] for row in IS_PERIOD_ROWS}
    | {row[1] for row in START_DATE_ROWS}
    | {row[1] for row in MAX_EXPOSURE_ROWS}
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def distinct_values(prefix: str) -> dict[str, str]:
    return {key: f"{prefix}{index}" for index, key in enumerate(ENV_KEYS) if key.startswith(prefix)}


@pytest.mark.parametrize(("name", "_key", "expected"), SLIPPAGE_ROWS)
def test_default_slippage_table_is_silent(
    capsys: pytest.CaptureFixture[str],
    name: str,
    _key: str,
    expected: float,
) -> None:
    assert qnstats.get_default_slippage(named(name)) == expected
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("name", "key", "_expected"), SLIPPAGE_ROWS)
def test_default_slippage_env_override_per_type(
    clean_env: pytest.MonkeyPatch, name: str, key: str, _expected: float
) -> None:
    for index, env_key in enumerate(k for k in ENV_KEYS if k.startswith("SL_")):
        clean_env.setenv(env_key, f"0.{index + 11}")
    expected = float(f"0.{[k for k in ENV_KEYS if k.startswith('SL_')].index(key) + 11}")
    assert qnstats.get_default_slippage(named(name)) == expected


@pytest.mark.parametrize("name", UNKNOWN_NAMES)
def test_default_slippage_of_unknown_type_ignores_environment(
    clean_env: pytest.MonkeyPatch, name: str
) -> None:
    for env_key in ENV_KEYS:
        clean_env.setenv(env_key, "0.77")
    assert qnstats.get_default_slippage(named(name)) == 0.05
    assert qnstats.get_default_slippage(xr.DataArray(0.0)) == 0.05


def test_default_slippage_of_stocks_long_ignores_stocks_variable_characterization(
    clean_env: pytest.MonkeyPatch,
) -> None:
    # unlike the period and start-date tables, stocks_long is not routed to the stocks variable
    clean_env.setenv("SL_STOCKS", "0.77")
    assert qnstats.get_default_slippage(named("stocks_long")) == 0.05
    assert qnstats.get_default_slippage(named("stocks")) == 0.77


@pytest.mark.parametrize(("name", "_key", "expected"), IS_PERIOD_ROWS)
def test_default_is_period_table_is_silent(
    capsys: pytest.CaptureFixture[str],
    name: str,
    _key: str,
    expected: int,
) -> None:
    result = qnstats.get_default_is_period_for_type(dynamic(name))
    assert result == expected
    assert isinstance(result, int)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("name", "key", "_expected"), IS_PERIOD_ROWS)
def test_default_is_period_env_override_per_type(
    clean_env: pytest.MonkeyPatch, name: str, key: str, _expected: int
) -> None:
    keys = [k for k in ENV_KEYS if k.startswith("IS_")]
    for index, env_key in enumerate(keys):
        clean_env.setenv(env_key, str(100 + index))
    assert qnstats.get_default_is_period_for_type(dynamic(name)) == 100 + keys.index(key)


def test_default_is_period_of_cryptodaily_reads_cryptofutures_env_characterization(
    clean_env: pytest.MonkeyPatch,
) -> None:
    # looks like a bug: the cryptodaily family reads IS_CRYPTOFUTURES, not a cryptodaily variable
    clean_env.setenv("IS_CRYPTOFUTURES", "123")
    for name in CRYPTODAILY_NAMES:
        assert qnstats.get_default_is_period_for_type(dynamic(name)) == 123


@pytest.mark.parametrize("name", UNKNOWN_NAMES)
def test_default_is_period_and_start_date_of_unknown_type_are_none(
    clean_env: pytest.MonkeyPatch, name: str
) -> None:
    for env_key in ENV_KEYS:
        clean_env.setenv(env_key, "5")
    assert qnstats.get_default_is_period_for_type(dynamic(name)) is None
    assert qnstats.get_default_is_start_date_for_type(dynamic(name)) is None


def test_get_default_is_period_falls_back_to_five_years_of_points() -> None:
    unknown = make_data(n_assets=1, n_days=300, name=dynamic("mystery"))
    assert qnstats.get_default_is_period(unknown) == 5 * qnstats.calc_avg_points_per_year(unknown)
    known = make_data(n_assets=1, n_days=300, name=dynamic("futures"))
    assert qnstats.get_default_is_period(known) == 3528


@pytest.mark.parametrize(("name", "_key", "expected"), START_DATE_ROWS)
def test_default_start_date_table_is_silent(
    capsys: pytest.CaptureFixture[str],
    name: str,
    _key: str,
    expected: str,
) -> None:
    assert qnstats.get_default_is_start_date_for_type(dynamic(name)) == expected
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("name", "key", "_expected"), START_DATE_ROWS)
def test_default_start_date_env_override_per_type(
    clean_env: pytest.MonkeyPatch, name: str, key: str, _expected: str
) -> None:
    keys = [k for k in ENV_KEYS if k.startswith("SD_")]
    for index, env_key in enumerate(keys):
        clean_env.setenv(env_key, f"20{index + 10}-02-03")
    expected = f"20{keys.index(key) + 10}-02-03"
    assert qnstats.get_default_is_start_date_for_type(dynamic(name)) == expected


@pytest.mark.parametrize(("name", "_key", "expected"), MAX_EXPOSURE_ROWS)
def test_default_max_exposure_table_is_silent(
    capsys: pytest.CaptureFixture[str],
    name: str,
    _key: str,
    expected: float,
) -> None:
    assert qnstats.get_default_max_exposure(dynamic(name)) == expected
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("name", "key", "_expected"), MAX_EXPOSURE_ROWS)
def test_default_max_exposure_env_override_per_type(
    clean_env: pytest.MonkeyPatch, name: str, key: str, _expected: float
) -> None:
    keys = [k for k in ENV_KEYS if k.startswith("ME_")]
    for index, env_key in enumerate(keys):
        clean_env.setenv(env_key, f"0.{index + 31}")
    assert qnstats.get_default_max_exposure(dynamic(name)) == float(f"0.{keys.index(key) + 31}")


@pytest.mark.parametrize("name", ["stocks", "futures", "crypto", "stocks_long", "unknown", ""])
def test_default_max_exposure_of_other_types_is_one(
    clean_env: pytest.MonkeyPatch, name: str
) -> None:
    for env_key in ENV_KEYS:
        clean_env.setenv(env_key, "0.33")
    assert qnstats.get_default_max_exposure(dynamic(name)) == 1.0


# ---------------------------------------------------------------------------------------------
# check_exposure and calc_exposure
# ---------------------------------------------------------------------------------------------


def reference_exposure_ok(
    weights: Floats,
    soft: float,
    hard: float,
    days_tolerance: float,
    excess_tolerance: float,
    avg_period: int,
    check_period: int,
) -> bool:
    total = np.abs(weights).sum(axis=1)
    exposure = np.abs(weights) / np.where(total > EPS, total, 1.0)[:, None]
    max_exposure = exposure.max(axis=1)
    periods = min(avg_period, weights.shape[0])

    def window_means(values: Floats) -> Floats:
        tail = values[-check_period:]
        return np.array(
            [tail[i - periods + 1 : i + 1].mean() for i in range(periods - 1, len(tail))]
        )

    days_ok = not (window_means((max_exposure > soft).astype(np.float64)) > days_tolerance).any()
    excess = np.maximum(exposure - soft, 0.0).sum(axis=1)
    excess_ok = not (window_means(excess) > excess_tolerance).any()
    hard_ok = not (max_exposure > hard).any()
    return hard_ok and (days_ok or excess_ok)


def exposure_book(rows: Sequence[Sequence[float]]) -> xr.DataArray:
    return frame(rows, names=["A0", "A1", "A2"][: len(rows[0])])


def balanced_with_spikes(n_days: int, spikes: Mapping[int, float]) -> list[list[float]]:
    rows = [[0.5, 0.5] for _ in range(n_days)]
    for day, heavy in spikes.items():
        rows[day] = [heavy, 1.0 - heavy]
    return rows


@dataclass(frozen=True)
class ExposureCase:
    rows: list[list[float]]
    expected: bool
    soft: float = 0.5
    hard: float = 0.9
    days_tolerance: float = 0.25
    excess_tolerance: float = 0.046875
    avg_period: int = 4
    check_period: int = 8
    hard_state: str = "Ok."
    days_state: str = "Ok."
    excess_state: str = "Ok."


EXPOSURE_CASES = {
    "hard_limit_equal_is_ok": ExposureCase(balanced_with_spikes(6, {}), True, soft=0.5, hard=0.5),
    "days_share_equal_to_tolerance": ExposureCase(
        balanced_with_spikes(8, {5: 0.75}), True, excess_state="Failed."
    ),
    "excess_equal_to_tolerance": ExposureCase(
        balanced_with_spikes(8, {5: 0.75}),
        True,
        days_tolerance=0.1,
        excess_tolerance=0.0625,
        days_state="Failed.",
    ),
    "days_and_excess_fail": ExposureCase(
        balanced_with_spikes(8, {5: 0.75}),
        False,
        days_tolerance=0.1,
        excess_tolerance=0.03125,
        days_state="Failed.",
        excess_state="Failed.",
    ),
    "hard_limit_exceeded": ExposureCase(
        balanced_with_spikes(20, {10: 0.95}),
        False,
        days_tolerance=0.1,
        excess_tolerance=0.05,
        avg_period=20,
        check_period=20,
        hard_state="Failed.",
    ),
    "old_bad_day_is_outside_check_period": ExposureCase(
        balanced_with_spikes(12, {3: 0.75}),
        True,
        days_tolerance=0.2,
        excess_tolerance=0.05,
        avg_period=2,
        check_period=8,
    ),
    "bad_day_at_start_of_check_period": ExposureCase(
        balanced_with_spikes(12, {4: 0.75}),
        False,
        days_tolerance=0.2,
        excess_tolerance=0.05,
        avg_period=2,
        check_period=8,
        days_state="Failed.",
        excess_state="Failed.",
    ),
    "average_period_longer_than_history": ExposureCase(
        balanced_with_spikes(6, {2: 0.75}),
        False,
        days_tolerance=0.1,
        excess_tolerance=0.04,
        avg_period=252,
        check_period=1260,
        days_state="Failed.",
        excess_state="Failed.",
    ),
    "average_period_longer_than_history_excess_ok": ExposureCase(
        balanced_with_spikes(6, {2: 0.75}),
        True,
        days_tolerance=0.1,
        excess_tolerance=0.05,
        avg_period=252,
        check_period=1260,
        days_state="Failed.",
    ),
}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("case_name", list(EXPOSURE_CASES))
def test_check_exposure_branches_and_boundaries(
    capsys: pytest.CaptureFixture[str], case_name: str, reverse: bool
) -> None:
    case = EXPOSURE_CASES[case_name]
    book = exposure_book(case.rows)
    if reverse:
        book = book.isel(time=slice(None, None, -1))
    weights = np.array(case.rows)
    assert (
        reference_exposure_ok(
            weights,
            case.soft,
            case.hard,
            case.days_tolerance,
            case.excess_tolerance,
            case.avg_period,
            case.check_period,
        )
        is case.expected
    )
    result = qnstats.check_exposure(
        book,
        case.soft,
        case.hard,
        case.days_tolerance,
        case.excess_tolerance,
        case.avg_period,
        case.check_period,
    )
    report = capsys.readouterr()
    assert result is case.expected
    over_limit = (weights.max(axis=1) / weights.sum(axis=1) > case.soft).any()
    assert ("Positions with max exposure over the limit:" in report.out) is bool(over_limit)
    if over_limit:
        assert "A0" in report.out
        assert "exposure" in report.out
    if case.expected:
        assert "Ok. The exposure check succeed." in report.out
        assert "WARNING" not in report.err
        assert "Hard limit check" not in report.out
    else:
        assert "Ok. The exposure check succeed." not in report.out
        assert "WARNING! The exposure check failed." in report.err
        assert re.search(rf"Hard limit check:\s+{re.escape(case.hard_state)}", report.out)
        assert re.search(rf"Days check:\s+{re.escape(case.days_state)}", report.out)
        assert re.search(rf"Excess check:\s+{re.escape(case.excess_state)}", report.out)


def test_check_exposure_of_empty_positions_is_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert qnstats.check_exposure(frame([[0.0, 0.0]] * 5)) is True
    report = capsys.readouterr()
    assert "Positions with max exposure" not in report.out


def test_calc_exposure_threshold_for_tiny_positions() -> None:
    at_eps = qnstats.calc_exposure(frame([[EPS, 0.0]]))
    assert_allclose(at_eps.values, [[EPS, 0.0]], rtol=1e-12, atol=0)
    above = qnstats.calc_exposure(frame([[3 * EPS, -EPS]]))
    assert_allclose(above.values, [[0.75, 0.25]], rtol=1e-12)
    short = qnstats.calc_exposure(frame([[-2.0, 6.0, 0.0]]))
    assert_allclose(short.values, [[0.25, 0.75, 0.0]], rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# calc_stat
# ---------------------------------------------------------------------------------------------


def stat_field(stat: xr.DataArray, field: str) -> Floats:
    return stat.sel(field=field).values


def running_max(values: Floats) -> Floats:
    return np.maximum.accumulate(values, axis=0)


def expected_equity_fields(returns: Floats) -> tuple[Floats, Floats, Floats]:
    equity = np.cumprod(1.0 + returns, axis=0).astype(np.float64)
    underwater = equity / running_max(equity) - 1.0
    drawdown = np.minimum.accumulate(underwater, axis=0).astype(np.float64)
    return equity, underwater, drawdown


def reference_turnover(
    weights: Floats,
    equity: Floats,
    open_: Floats,
    window: int,
    min_periods: int,
    per_asset: bool,
) -> Floats:
    held = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    held_before = np.vstack([np.zeros((1, weights.shape[1])), held[:-1]])
    open_before = np.vstack([open_[:1], open_[:-1]])
    wide = equity if equity.ndim == 2 else equity[:, None]
    equity_before = np.vstack([np.ones((1, wide.shape[1])), wide[:-1]])
    drift = held_before * equity_before * open_ / (open_before * wide)
    turnover = np.abs(held - drift)
    if not per_asset:
        turnover = np.nansum(turnover, axis=1)[:, None]
    result = trailing_mean(turnover, window, min_periods)
    return result if per_asset else result[:, 0]


def test_calc_stat_explicit_arguments_drive_every_field() -> None:
    data = market("stocks", True, gap=False)
    book = qnout.normalize(make_weights(data, seed=9))
    stat = qnstats.calc_stat(
        data,
        book,
        slippage_factor=PIPE_FRACT,
        roll_slippage_factor=PIPE_ROLL_FRACT,
        min_periods=3,
        max_periods=6,
        points_per_year=100,
    )
    assert stat.dims == ("time", "field")
    returns = pipeline_inputs(data, book, PIPE_FRACT, PIPE_ROLL_FRACT, False).total()
    equity, underwater, drawdown = expected_equity_fields(returns)
    volatility = trailing_std(returns[:, None], 6, 3)[:, 0] * math.sqrt(100)
    log_mean = trailing_mean(np.log1p(returns)[:, None], 6, 3)[:, 0]
    mean_return = np.exp(log_mean) ** 100 - 1.0
    assert_allclose(stat_field(stat, "relative_return"), returns, rtol=1e-9, atol=1e-12)
    assert_allclose(stat_field(stat, "equity"), equity, rtol=1e-9)
    assert_allclose(stat_field(stat, "volatility"), volatility, rtol=1e-8, equal_nan=True)
    assert_allclose(stat_field(stat, "underwater"), underwater, rtol=1e-9, atol=1e-12)
    assert_allclose(stat_field(stat, "max_drawdown"), drawdown, rtol=1e-9, atol=1e-12)
    assert_allclose(stat_field(stat, "mean_return"), mean_return, rtol=1e-8, equal_nan=True)
    assert_allclose(
        stat_field(stat, "sharpe_ratio"), ratio(mean_return, volatility), rtol=1e-7, equal_nan=True
    )
    weights = book.values
    bias = weights.sum(axis=1) / np.abs(weights).sum(axis=1)
    assert_allclose(stat_field(stat, "bias"), bias, rtol=1e-9)
    ever_traded = np.maximum.accumulate(weights != 0, axis=0).sum(axis=1)
    assert_array_equal(stat_field(stat, "instruments"), ever_traded)
    turnover = reference_turnover(weights, equity, field_values(data, "open"), 6, 3, False)
    assert_allclose(stat_field(stat, "avg_turnover"), turnover, rtol=1e-8, equal_nan=True)
    holding = qnstats.calc_avg_holding_time(
        book, max_periods=6, min_periods=3, per_asset=False, points_per_year=100
    )
    assert_allclose(stat_field(stat, "avg_holding_time"), holding.values, equal_nan=True)


def test_calc_stat_defaults_for_cryptodaily_use_arithmetic_estimators() -> None:
    data = market("cryptodaily", True)
    book = qnout.normalize(make_weights(data, seed=4))
    stat = qnstats.calc_stat(data, book)
    returns = pipeline_inputs(data, book, 0.04, 0.04, False).total()
    ppy = 365
    mean_return = trailing_mean(returns[:, None], N_PIPE, 1)[:, 0] * ppy
    volatility = trailing_std(returns[:, None], N_PIPE, 1)[:, 0] * math.sqrt(ppy)
    arithmetic = trailing_mean(returns[:, None], N_PIPE, 2)[:, 0] * ppy
    spread = trailing_std(returns[:, None], N_PIPE, 2)[:, 0] * math.sqrt(ppy)
    assert_allclose(stat_field(stat, "relative_return"), returns, rtol=1e-9, atol=1e-12)
    assert_allclose(stat_field(stat, "mean_return"), mean_return, rtol=1e-9, equal_nan=True)
    assert_allclose(stat_field(stat, "volatility"), volatility, rtol=1e-8, equal_nan=True)
    assert_allclose(
        stat_field(stat, "sharpe_ratio"), ratio(arithmetic, spread), rtol=1e-7, equal_nan=True
    )
    geometric = np.exp(trailing_mean(np.log1p(returns)[:, None], N_PIPE, 1)[:, 0]) ** ppy - 1.0
    assert np.abs(geometric[-1] - mean_return[-1]) > 1e-3


def test_calc_stat_per_asset_keeps_clipped_weights_and_separate_accounts() -> None:
    data = market("stocks", True, gap=False)
    pattern = np.where(np.arange(N_PIPE)[:, None] % 5 == 0, 0.0, [[0.6, -0.7, 0.8]])
    book = xr.DataArray(
        pattern,
        dims=["time", "asset"],
        coords={"time": data.coords["time"].values, "asset": data.coords["asset"].values},
    )
    stat = qnstats.calc_stat(data, book, per_asset=True, points_per_year=252)
    assert stat.dims == ("time", "field", "asset")
    returns = pipeline_inputs(data, book, 0.05, 0.05, True).per_asset()
    equity, underwater, _ = expected_equity_fields(returns)
    assert_allclose(stat_field(stat, "relative_return"), returns, rtol=1e-9, atol=1e-12)
    assert_allclose(stat_field(stat, "equity"), equity, rtol=1e-9)
    assert_allclose(stat_field(stat, "underwater"), underwater, rtol=1e-9, atol=1e-12)
    assert_allclose(stat_field(stat, "bias"), pattern)
    assert_array_equal(stat_field(stat, "instruments"), np.ones((N_PIPE, 3)))
    turnover = reference_turnover(pattern, equity, field_values(data, "open"), N_PIPE, 1, True)
    assert_allclose(stat_field(stat, "avg_turnover"), turnover, rtol=1e-8, equal_nan=True)


def test_calc_stat_warnings_about_empty_missed_and_non_liquid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = market("stocks", True, gap=False, n_days=30)
    weights = qnout.normalize(make_weights(data, seed=2)).values.copy()
    weights[:6, 0] = 0.0
    clean = xr.DataArray(
        weights,
        dims=["time", "asset"],
        coords={"time": data.coords["time"].values, "asset": data.coords["asset"].values},
    )
    qnstats.calc_stat(data, clean)
    assert "WARNING" not in capsys.readouterr().err

    qnstats.calc_stat(data, clean.isel(time=[i for i in range(30) if i != 12]))
    err = capsys.readouterr().err
    assert "some dates are missed" in err
    assert "non-liquid" not in err

    illiquid = clean.copy(deep=True)
    illiquid[2, 0] = 0.4
    qnstats.calc_stat(data, illiquid)
    err = capsys.readouterr().err
    assert "Strategy trades non-liquid assets" in err
    assert "some dates are missed" not in err

    with pytest.raises(IndexError):
        qnstats.calc_stat(data, clean.isel(time=slice(0, 0)))
    assert "Output is empty" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------------
# check_correlation (the network call is replaced)
# ---------------------------------------------------------------------------------------------


@dataclass
class CorrelationSpy:
    calls: list[tuple[xr.DataArray, bool, Hashable | None]]


def install_correlation(
    monkeypatch: pytest.MonkeyPatch, result: list[Correlation] | None
) -> CorrelationSpy:
    spy = CorrelationSpy([])

    def fake(
        relative_returns: xr.DataArray,
        suppress_exception: bool = True,
        competition_type: Hashable | None = None,
    ) -> list[Correlation]:
        spy.calls.append((relative_returns, suppress_exception, competition_type))
        if result is None:
            raise ConnectionError("offline")
        return result

    monkeypatch.setattr(qnstats, "calc_correlation", fake)
    return spy


def correlation_inputs(name: str) -> tuple[xr.DataArray, xr.DataArray]:
    data = market(name, True, gap=False, n_days=30)
    return data, qnout.normalize(make_weights(data, seed=6))


def entry(
    ident: int, name: str, cofactor: float, sharpe: float, my: bool, template: bool
) -> Correlation:
    return {
        "id": ident,
        "name": name,
        "my": my,
        "template": template,
        "cofactor": cofactor,
        "sharpe_ratio": sharpe,
    }


@pytest.mark.parametrize("print_stack_trace", [True, False])
def test_check_correlation_reports_failed_lookup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    print_stack_trace: bool,
) -> None:
    spy = install_correlation(monkeypatch, None)
    data, weights = correlation_inputs("stocks")
    with caplog.at_level(logging.ERROR):
        qnstats.check_correlation(weights, data, print_stack_trace)
    report = capsys.readouterr()
    logged = [r for r in caplog.records if r.getMessage() == "Correlation check failed."]
    assert len(spy.calls) == 1
    assert ("Correlation check failed." in report.err) is (not print_stack_trace)
    assert len(logged) == (1 if print_stack_trace else 0)
    assert report.out == ""


def test_check_correlation_passes_returns_flag_and_type_to_lookup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spy = install_correlation(monkeypatch, [])
    data, weights = correlation_inputs("futures")
    qnstats.check_correlation(weights, data)
    report = capsys.readouterr()
    returns, suppress, kind = spy.calls[0]
    assert suppress is False
    assert kind == "futures"
    assert_allclose(
        returns.values, qnstats.calc_relative_return(data, weights).values, atol=ATOL
    )
    assert "Ok. This strategy does not correlate with other strategies." in report.out
    assert "WARNING" not in report.err
    assert "correlates" not in report.out


def current_sharpe(output: str) -> float:
    match = re.search(r"Current sharpe ratio\(3y\):\s+(\S+)", output)
    assert match is not None
    return float(match.group(1))


@pytest.mark.parametrize(("kind", "arithmetic"), [("stocks", False), ("cryptodaily", True)])
def test_check_correlation_lists_correlated_systems(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    arithmetic: bool,
) -> None:
    found = [
        entry(1, "mine_one", 0.93, 1.4, True, False),
        entry(2, "tmpl_two", 0.97, 2.2, False, True),
        entry(3, "plain_three", 0.91, 0.8, False, False),
    ]
    install_correlation(monkeypatch, found)
    data, weights = correlation_inputs(kind)
    qnstats.check_correlation(weights, data)
    report = capsys.readouterr()
    assert (
        "WARNING! This strategy correlates with other strategies and will be rejected."
        in report.err
    )
    assert "Modify the strategy to produce the different output." in report.err
    assert "correlation larger than 0.9: 3" in report.out
    assert re.search(r"systems with a larger Sharpe ratio\):\s+0\.97\b", report.out)
    assert "My correlated submissions" in report.out
    assert "Correlated examples" in report.out
    assert "mine_one" in report.out
    assert "tmpl_two" in report.out
    assert "plain_three" not in report.out
    assert "Name" in report.out
    assert "Coefficient" in report.out
    assert "Sharpe ratio" in report.out

    # the sharpe ratio is annualised with the points per year taken from the timestamps of the
    # unnamed returns series, not with the 251 or 365 of the market type: characterization
    steps = np.diff(data.coords["time"].values).astype("timedelta64[s]").astype(np.float64)
    points_per_year = round(365.25 * 24 / (steps.mean() / 3600.0))
    returns = qnstats.calc_relative_return(data, weights).values
    mean_log = trailing_mean(np.log1p(returns)[:, None], 30, 2)[-1, 0]
    mean = trailing_mean(returns[:, None], 30, 2)[-1, 0]
    spread = trailing_std(returns[:, None], 30, 2)[-1, 0]
    annual_spread = spread * math.sqrt(points_per_year)
    annual_mean = (
        mean * points_per_year if arithmetic else math.exp(mean_log) ** points_per_year - 1
    )
    assert_allclose(current_sharpe(report.out), annual_mean / annual_spread, rtol=1e-8)


@pytest.mark.parametrize(
    ("found", "my_shown", "examples_shown"),
    [
        ([entry(1, "alpha", 0.95, 1.0, True, False)], True, False),
        ([entry(2, "beta", 0.96, 1.0, False, True)], False, True),
        ([entry(3, "gamma", 0.92, 1.0, False, False)], False, False),
    ],
)
def test_check_correlation_shows_tables_only_when_non_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    found: list[Correlation],
    my_shown: bool,
    examples_shown: bool,
) -> None:
    install_correlation(monkeypatch, found)
    data, weights = correlation_inputs("stocks")
    qnstats.check_correlation(weights, data)
    report = capsys.readouterr()
    assert ("My correlated submissions" in report.out) is my_shown
    assert ("Correlated examples" in report.out) is examples_shown
    assert "correlation larger than 0.9: 1" in report.out


# ---------------------------------------------------------------------------------------------
# calc_correlation details (the network call is replaced)
# ---------------------------------------------------------------------------------------------


@dataclass
class StubResponse:
    payload: bytes

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def install_descending_server(monkeypatch: pytest.MonkeyPatch, ids: list[int]) -> None:
    submissions = [{"id": i, "name": f"s{i}", "my": False, "template": False} for i in ids]

    def fake_urlopen(_url: str, payload: bytes | None = None) -> StubResponse:
        if payload is None:
            return StubResponse(json_bytes(submissions))
        asked = [str(i) for i in loads_list(payload, "submission_ids")]
        cofactors = [
            {"id": i, "cofactor": int(i) / 100, "sharpe_ratio": int(i) / 10} for i in asked
        ]
        return StubResponse(json_bytes(cofactors))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode()


def loads_list(payload: bytes, key: str) -> list[int]:
    loaded: dict[str, list[int]] = json.loads(payload)
    return loaded[key]


def test_calc_correlation_matches_cofactors_by_id_not_by_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SUBMISSION_ID", raising=False)
    install_descending_server(monkeypatch, [30, 20, 10])
    result = qnstats.calc_correlation(series([0.01, -0.02, 0.03] * 4), False, "stocks")
    assert [(r["id"], r["name"], r["cofactor"], r["sharpe_ratio"]) for r in result] == [
        (30, "s30", 0.3, 3.0),
        (20, "s20", 0.2, 2.0),
        (10, "s10", 0.1, 1.0),
    ]


def test_calc_correlation_with_empty_submission_id_still_checks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_descending_server(monkeypatch, [7])
    monkeypatch.setenv("SUBMISSION_ID", "")
    result = qnstats.calc_correlation(series([0.01, -0.02, 0.03] * 4), False, "stocks")
    assert [r["id"] for r in result] == [7]
    assert "correlation check disabled" not in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# find_missed_dates and its layout test
# ---------------------------------------------------------------------------------------------


def layered(fields: Sequence[str], layers: Sequence[Floats], n_assets: int = 2) -> xr.DataArray:
    values = np.stack(layers, axis=0)
    return xr.DataArray(
        values,
        dims=["field", "time", "asset"],
        coords={
            "field": list(fields),
            "time": business_days(values.shape[1], "2020-01-01"),
            "asset": [f"a{i}" for i in range(n_assets)],
        },
    )


NAN = np.nan
LIQUID = np.array(
    [[1, 1], [0, 0], [1, 0], [0, 0], [1, 1], [0, 1], [-1, -1], [1, 0]], dtype=np.float64
)
CLOSES = np.array(
    [[5, 5], [0, 0], [NAN, 3], [NAN, NAN], [-1, 2], [0, 0.5], [-2, -2], [7, NAN]],
    dtype=np.float64,
)
OPENS = np.full((8, 2), 4.0)
OPENS_GAPPY = OPENS.copy()
OPENS_GAPPY[5] = NAN


def missed_book(use_liquid: bool, plain: bool) -> xr.DataArray:
    if use_liquid:
        fields, layers = ["open", "is_liquid"], [OPENS_GAPPY, LIQUID]
    else:
        fields, layers = ["open", "close"], [OPENS, CLOSES]
    if not plain:
        fields = [*fields, "open"]
        layers = [*layers, OPENS]
    return layered(fields, layers)


def output_on(book: xr.DataArray, days: list[int]) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((len(days), 2)),
        dims=["time", "asset"],
        coords={"time": book.coords["time"].values[days], "asset": ["a0", "a1"]},
    )


@pytest.mark.parametrize("plain", [True, False])
@pytest.mark.parametrize("use_liquid", [True, False])
@pytest.mark.parametrize(
    ("days", "expected_days"),
    [([0, 4], [2, 5, 7]), ([2, 4], [5, 7]), ([7, 4, 0], [2, 5]), ([0, 2, 4, 5, 7], [])],
)
def test_find_missed_dates_counts_only_tradable_days_after_first_output(
    plain: bool, use_liquid: bool, days: list[int], expected_days: list[int]
) -> None:
    book = missed_book(use_liquid, plain)
    missed = qnstats.find_missed_dates(output_on(book, days), book)
    assert_array_equal(missed, book.coords["time"].values[expected_days])


def test_find_missed_dates_is_empty_when_nothing_tradable_remains() -> None:
    book = missed_book(False, True)
    missed = qnstats.find_missed_dates(output_on(book, [0, 7]), book)
    assert missed.size == 3
    assert qnstats.find_missed_dates(output_on(book, [7]), book).size == 0


# ---------------------------------------------------------------------------------------------
# layout checks of arrange_data and the liquidity helpers
# ---------------------------------------------------------------------------------------------


def plain_data_and_weights() -> tuple[xr.DataArray, xr.DataArray]:
    data = make_data(n_assets=3, n_days=6, name=dynamic("stocks"), seed=2)
    return data, make_weights(data, seed=1)


def test_arrange_data_drops_empty_labels_but_keeps_requested_order() -> None:
    data, weights = plain_data_and_weights()
    holed = data.copy(deep=True)
    holed[:, :, 1] = np.nan
    adjusted, target = qnstats.arrange_data(holed, weights, False)
    assert list(adjusted.coords["asset"].values) == ["NYS:A0", "NYS:A2"]
    assert list(target.coords["asset"].values) == ["NYS:A0", "NYS:A2"]
    assert adjusted.dims == ("field", "time", "asset")
    total = np.abs(target.values).sum(axis=1)
    assert (total <= 1.0 + 1e-12).all()
    clipped_adjusted, clipped_target = qnstats.arrange_data(data, weights * 3.0, True)
    assert np.abs(clipped_target.values).max() == 1.0
    assert adjusted.shape[0] == len(FIELDS)
    assert clipped_adjusted.shape[0] == len(FIELDS)


def liquidity_data(fields: list[str], layers: list[Floats]) -> xr.DataArray:
    return xr.DataArray(
        np.stack(layers, axis=1),
        dims=["time", "field", "asset"],
        coords={
            "time": business_days(4, "2020-01-01"),
            "field": fields,
            "asset": ["a", "b", "c"],
        },
    )


def test_calc_non_liquid_selects_zero_flags_and_normalizes_weights() -> None:
    prices = np.full((4, 3), 5.0)
    flags = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    data = liquidity_data(["open", "is_liquid"], [prices, flags])
    weights = frame(
        [[0.75, 0.75, 0.5], [0.75, 0.75, 0.5], [0.75, 0.75, 0.5], [0.0, 0.0, 0.0]],
        "2020-01-01",
        ["a", "b", "c"],
    )
    before = weights.values.copy()
    result = qnstats.calc_non_liquid(data, weights)
    assert list(result.coords["asset"].values) == ["a", "b"]
    assert_array_equal(result.coords["time"].values, data.coords["time"].values[:2])
    assert_allclose(result.values, [[NAN, 0.375], [0.375, 0.375]], equal_nan=True)
    assert_array_equal(weights.values, before)


def test_calc_non_liquid_without_liquidity_field_is_empty() -> None:
    closes = liquidity_data(["close"], [np.full((4, 3), 5.0)])
    weights = frame([[0.5, 0.25, 0.25]] * 4, "2020-01-01", ["a", "b", "c"])
    result = qnstats.calc_non_liquid(closes, weights)
    assert result.sizes["time"] == 0
    assert result.sizes["asset"] == 0


# ---------------------------------------------------------------------------------------------
# means, sharpe ratio and the estimator switch
# ---------------------------------------------------------------------------------------------

RETURNS = [0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, 0.005, -0.015, 0.04]
RETURNS_WITH_GAPS = [0.01, NAN, 0.03, -0.02, NAN, -0.01, 0.02, 0.005, NAN, 0.04]


def test_geometric_and_simple_means_skip_nan_returns() -> None:
    returns = np.array(RETURNS_WITH_GAPS)
    rr = series(RETURNS_WITH_GAPS)
    geometric = qnstats.calc_mean_return(rr, max_periods=4, min_periods=2, points_per_year=252)
    expected = np.exp(trailing_mean(np.log1p(returns)[:, None], 4, 2)[:, 0]) - 1.0
    assert_allclose(geometric.values, expected, rtol=1e-12, atol=1e-14, equal_nan=True)
    simple = qnstats.calc_simple_mean_return_annualized(rr, 4, 2, 100)
    assert_allclose(
        simple.values,
        trailing_mean(returns[:, None], 4, 2)[:, 0] * 100,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    annual = qnstats.calc_mean_return_annualized(rr, 4, 2, 100)
    assert_allclose(
        annual.values, (1.0 + expected) ** 100 - 1.0, rtol=1e-10, atol=1e-12, equal_nan=True
    )
    assert np.isnan(expected[1])
    assert np.isfinite(expected[2])
    assert np.isfinite(expected[4])


def test_mean_return_with_explicit_points_per_year_works_on_a_single_point() -> None:
    one = series([0.02])
    assert_allclose(
        qnstats.calc_mean_return(one, points_per_year=252).values, [0.02], rtol=1e-12
    )
    assert_allclose(
        qnstats.calc_simple_mean_return_annualized(one, points_per_year=10).values,
        [0.2],
        rtol=1e-12,
    )


@pytest.mark.parametrize("estimator", ["geometric", "arithmetic"])
def test_sharpe_ratio_estimators_match_numpy(estimator: str) -> None:
    returns = np.array(RETURNS)
    rr = series(RETURNS)
    ppy = 50
    result = qnstats.calc_sharpe_ratio_annualized(
        rr, 6, 3, points_per_year=ppy, mean_estimator=dynamic(estimator)
    )
    spread = trailing_std(returns[:, None], 6, 3)[:, 0] * math.sqrt(ppy)
    if estimator == "arithmetic":
        mean = trailing_mean(returns[:, None], 6, 3)[:, 0] * ppy
    else:
        mean = np.exp(trailing_mean(np.log1p(returns)[:, None], 6, 3)[:, 0]) ** ppy - 1.0
    assert_allclose(result.values, mean / spread, rtol=1e-9, equal_nan=True)


@pytest.mark.parametrize("estimator", ["aardvark", "zzz", "Geometric", "", "arithmetic "])
def test_sharpe_ratio_rejects_unknown_estimator(estimator: str) -> None:
    with pytest.raises(ValueError, match="Unknown mean_estimator"):
        qnstats.calc_sharpe_ratio_annualized(series(RETURNS), mean_estimator=dynamic(estimator))


# ---------------------------------------------------------------------------------------------
# instruments, turnover, holding time
# ---------------------------------------------------------------------------------------------


def test_instruments_count_assets_that_ever_held_a_position() -> None:
    rng = np.random.default_rng(14)
    values = rng.choice([0.0, 0.0, 0.3, -0.4, NAN], size=(12, 5))
    values[:, 4] = np.where(np.arange(12) < 8, NAN, values[:, 4])
    book = frame(values.tolist())
    before = book.values.copy()
    result = qnstats.calc_instruments(book)
    held = np.nan_to_num(values) != 0
    expected = np.maximum.accumulate(held, axis=0).sum(axis=1)
    assert_array_equal(result.values, expected)
    assert expected[-1] > expected[0] > 0
    assert_array_equal(book.values, before)


def test_instruments_per_asset_is_a_copy_of_ones() -> None:
    book = frame([[0.3, NAN], [-0.4, 0.0], [0.0, 0.2]])
    before = book.values.copy()
    result = qnstats.calc_instruments(book, per_asset=True)
    assert result.dims == book.dims
    assert_array_equal(result.values, np.ones((3, 2)))
    assert_array_equal(book.values, before)


def turnover_market(gap: bool) -> xr.DataArray:
    base = make_data(n_assets=2, n_days=7, name=dynamic("futures"), seed=19)
    values = base.values.copy()
    if gap:
        values[1, 0, 0] = NAN
        values[6, 0, 1] = NAN
    return xr.DataArray(
        values,
        dims=base.dims,
        coords={
            "time": base.coords["time"].values,
            "field": base.coords["field"].values,
            "asset": base.coords["asset"].values,
        },
    )


TURNOVER_WEIGHTS = [
    [0.5, -0.2],
    [0.3, -0.4],
    [0.0, 0.6],
    [-0.4, 0.3],
    [-0.4, 0.0],
    [0.2, 0.5],
    [0.7, -0.1],
]
TURNOVER_EQUITY = [1.0, 1.02, 0.99, 1.05, 1.03, 1.08, 1.04]


@pytest.mark.parametrize(
    ("per_asset", "gap", "min_periods", "max_periods", "points_per_year", "window", "need"),
    [
        (False, False, 1, None, 3, 3, 1),
        (True, True, 2, 4, 3, 4, 2),
        (False, False, 1, 5, 3, 5, 1),
        (True, False, 50, None, 252, 7, 7),
        (True, True, 1, None, 4, 4, 1),
    ],
)
def test_avg_turnover_matches_independent_formula(
    per_asset: bool,
    gap: bool,
    min_periods: int,
    max_periods: int | None,
    points_per_year: int,
    window: int,
    need: int,
) -> None:
    data = turnover_market(gap)
    book = frame(TURNOVER_WEIGHTS, names=list(data.coords["asset"].values))
    book = xr.DataArray(
        book.values,
        dims=["time", "asset"],
        coords={"time": data.coords["time"].values, "asset": data.coords["asset"].values},
    )
    equity = xr.DataArray(
        np.array(TURNOVER_EQUITY), dims=["time"], coords={"time": data.coords["time"].values}
    )
    before = book.values.copy()
    result = qnstats.calc_avg_turnover(
        book,
        equity,
        data,
        max_periods=max_periods,
        min_periods=min_periods,
        per_asset=per_asset,
        points_per_year=points_per_year,
    )
    expected = reference_turnover(
        np.array(TURNOVER_WEIGHTS),
        np.array(TURNOVER_EQUITY),
        field_values(data, "open"),
        window,
        need,
        per_asset,
    )
    assert_allclose(result.values, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert_array_equal(book.values, before)
    if not gap and need == 1:
        assert np.isfinite(result.values[0]).all()


def test_avg_turnover_default_window_is_points_per_year_of_the_history() -> None:
    data = turnover_market(False)
    book = xr.DataArray(
        np.array(TURNOVER_WEIGHTS),
        dims=["time", "asset"],
        coords={"time": data.coords["time"].values, "asset": data.coords["asset"].values},
    )
    equity = xr.DataArray(
        np.array(TURNOVER_EQUITY), dims=["time"], coords={"time": data.coords["time"].values}
    )
    default = qnstats.calc_avg_turnover(book, equity, data)
    explicit = qnstats.calc_avg_turnover(book, equity, data, max_periods=251, points_per_year=251)
    assert_allclose(default.values, explicit.values, equal_nan=True)


def book_for_holding() -> xr.DataArray:
    return frame([[0, 0], [1, 0], [1, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 0]], "2020-01-01")


@pytest.mark.parametrize(("ppy", "per_day"), [(500, 24), (8766, 24), (399, 1), (252, 1)])
def test_avg_holding_time_is_divided_by_points_per_day(ppy: int, per_day: int) -> None:
    held = book_for_holding()
    before = held.values.copy()
    per_asset = qnstats.calc_avg_holding_time(held, per_asset=True, points_per_year=ppy)
    assert_allclose(per_asset.values[4:, 0], 2.0 / per_day)
    assert_allclose(per_asset.values[-1], [2.0 / per_day, 2.0 / per_day])
    total = qnstats.calc_avg_holding_time(held, points_per_year=ppy)
    assert_allclose(total.values[4:], 2.0 / per_day)
    assert_array_equal(held.values, before)


def test_avg_holding_time_of_short_histories() -> None:
    one_day = frame([[1.0, 0.0]])
    result = qnstats.calc_avg_holding_time(one_day, points_per_year=252)
    assert result.shape == (1,)
    assert np.isnan(result.values).all()
    two_days = frame([[1.0, 0.0], [1.0, 0.0]])
    assert qnstats.calc_avg_holding_time(two_days, points_per_year=252).shape == (2,)


def test_avg_holding_time_windows_and_default_points_per_year() -> None:
    held = book_for_holding()
    windowed = qnstats.calc_avg_holding_time(
        held, max_periods=2, min_periods=1, points_per_year=252
    )
    assert_allclose(
        windowed.values, [NAN, NAN, NAN, NAN, 2.0, 2.0, NAN, 2.0], equal_nan=True
    )
    default = qnstats.calc_avg_holding_time(held)
    explicit = qnstats.calc_avg_holding_time(held, points_per_year=261)
    assert_allclose(default.values, explicit.values, equal_nan=True)
    assert_allclose(default.values[-1], 2.0)


def test_holding_log_matches_hand_traced_positions() -> None:
    weights = np.array(
        [
            [0.5, 0.4, 0.0, 0.0, -0.4],
            [0.5, 0.4, 0.0, EPS, -0.4],
            [0.2, 0.8, 0.1, -0.5, -0.4],
            [-0.3, 0.2, 0.1, -0.5, -0.4],
            [-0.3, 0.2, 0.1, -0.5, -0.4],
            [0.0, NAN, -0.2, -0.5, -0.4],
            [0.0, 0.2, -0.2, -0.5, -0.4],
            [0.0, 0.2, -0.2, -0.5, -0.4],
        ]
    )
    result = qnstats.calc_holding_log_np_nb(weights)
    expected = np.zeros((8, 2, 5))

    def set_event(day: int, asset: int, cost: float, duration: float) -> None:
        expected[day, 0, asset] = cost
        expected[day, 1, asset] = duration

    set_event(3, 0, 0.3, 2.0)
    set_event(4, 0, 0.2, 3.0)
    set_event(6, 0, 0.3, 2.0)
    set_event(4, 1, 0.6, 2.0)
    set_event(6, 2, 0.1, 3.0)
    set_event(3, 3, EPS, 1.0)
    assert result.shape == (8, 2, 5)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-15)


def test_holding_log_of_wide_matrix_with_few_days() -> None:
    weights = np.array([[0.5] * 4, [0.2] * 4, [0.2] * 4])
    result = qnstats.calc_holding_log_np_nb(weights)
    assert result.shape == (3, 2, 4)
    assert_allclose(result[2, 0], 0.3)
    assert_allclose(result[2, 1], 1.0)
    assert_array_equal(result[:2], 0.0)


# ---------------------------------------------------------------------------------------------
# points per year
# ---------------------------------------------------------------------------------------------


def named_time_series(name: str, n_days: int) -> xr.DataArray:
    return xr.DataArray(
        np.zeros(n_days),
        dims=["time"],
        coords={"time": business_days(n_days)},
        name=dynamic(name),
    )


@pytest.mark.parametrize(
    ("name", "n_days", "expected"),
    [
        ("stocks", 250, 251),
        ("stocks", 251, 261),
        ("stocks", 252, 261),
        ("stocks_long", 250, 251),
        ("crypto", 250, 8766),
        ("crypto", 251, 261),
        ("cryptodaily", 250, 365),
        ("crypto_daily_long_short", 250, 365),
        ("stocks_s&p500", 250, 251),
        ("futures", 250, 251),
    ],
)
def test_points_per_year_threshold_of_251_bars(name: str, n_days: int, expected: int) -> None:
    assert qnstats.calc_avg_points_per_year(named_time_series(name, n_days)) == expected


def test_points_per_year_uses_exact_mean_step_of_timestamps() -> None:
    steps = np.where(np.arange(499) % 2 == 0, 3600.0, 3601.5)
    seconds = np.concatenate([[0.0], np.cumsum(steps)])
    times = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        np.round(seconds * 1e9).astype(np.int64), unit="ns"
    )
    data = xr.DataArray(
        np.zeros(500), dims=["time"], coords={"time": times.values}, name=dynamic("mystery")
    )
    mean_hours = float(np.diff(seconds).mean()) / 3600.0
    assert qnstats.calc_avg_points_per_year(data) == round(365.25 * 24 / mean_hours)
    assert qnstats.calc_avg_points_per_year(data) == 8764


# ---------------------------------------------------------------------------------------------
# sector distribution
# ---------------------------------------------------------------------------------------------


@dataclass
class CatalogSpy:
    calls: dict[str, dict[str, str]]


def install_catalogs(monkeypatch: pytest.MonkeyPatch) -> CatalogSpy:
    spy = CatalogSpy({})

    def dated(label: str, sector: str) -> object:
        def loader(**kwargs: str) -> list[dict[str, object]]:
            spy.calls[label] = dict(kwargs)
            return [{"id": "x1", "sector": sector}]

        return loader

    def futures() -> list[dict[str, object]]:
        spy.calls["futures"] = {}
        return [{"id": "x1", "sector": "FUT"}]

    monkeypatch.setattr(qnstats, "stocks_load_ndx_list", dated("ndx", "NDX"))
    monkeypatch.setattr(qnstats, "stocks_load_spx_list", dated("spx", "SPX"))
    monkeypatch.setattr(qnstats, "stocks_load_list", dated("stocks", "STK"))
    monkeypatch.setattr(qnstats, "futures_load_list", futures)
    return spy


SECTOR_CATALOG: list[dict[str, object]] = [
    {"id": "a", "sector": "Tech"},
    {"id": "b", "sector": None},
    {"id": "c", "sector": "n/a"},
    {"id": "d", "sector": ""},
    {"id": "e", "sector": "zebra"},
]
SECTOR_ASSETS = ["a", "b", "c", "d", "e", "f"]
SECTOR_ROWS = [
    [0.5, -0.5, 0.5, NAN, NAN, NAN],
    [0.25, 0.0, 0.0, 0.0, 0.25, 0.0],
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
]
# columns: Other, Tech, zebra, Cash
SECTOR_EXPECTED = [
    [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0],
    [0.0, 0.25, 0.25, 0.5],
    [0.5, 0.5, 0.0, 0.0],
]


def sector_book(start: str = "2020-01-06", name: Hashable | None = None) -> xr.DataArray:
    book = frame(SECTOR_ROWS, start, SECTOR_ASSETS)
    book.name = name
    return book


def test_sector_distribution_normalizes_only_oversized_books_and_treats_nan_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = install_catalogs(monkeypatch)
    monkeypatch.setattr(qnstats, "stocks_load_list", lambda **_kwargs: SECTOR_CATALOG)
    book = sector_book()
    before = book.values.copy()
    result = qnstats.calc_sector_distribution(book, kind=dynamic("stocks"))
    assert list(result.coords["sector"].values) == ["Other", "Tech", "zebra", "Cash"]
    assert_allclose(result.values, SECTOR_EXPECTED, atol=1e-12)
    assert_array_equal(result.coords["time"].values, book.coords["time"].values)
    assert_array_equal(book.values, before)
    assert spy.calls == {}


def test_sector_distribution_on_timeseries_does_not_forward_fill_characterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_catalogs(monkeypatch)
    monkeypatch.setattr(qnstats, "stocks_load_list", lambda **_kwargs: SECTOR_CATALOG)
    book = sector_book()
    grid = np.array(
        pd.bdate_range("2020-01-03", periods=7, name="time").values, dtype="datetime64[ns]"
    )
    result = qnstats.calc_sector_distribution(book, grid, dynamic("stocks"))
    assert_array_equal(result.coords["time"].values, grid)
    # the grid is pre-filled with zeros, so the ffill never fires: dates outside the book are
    # all cash (looks like a bug; the intent of `ffill` is to carry the last book forward)
    empty = [0.0, 0.0, 0.0, 1.0]
    assert_allclose(
        result.values, [empty, *SECTOR_EXPECTED, empty, empty, empty], atol=1e-12
    )


def test_sector_distribution_loader_receives_date_range_of_the_book(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = install_catalogs(monkeypatch)
    book = sector_book("2020-03-02")
    qnstats.calc_sector_distribution(book, kind=dynamic("stocks_nasdaq100"))
    assert spy.calls == {"ndx": {"min_date": "2020-03-02", "max_date": "2020-03-04"}}


@pytest.mark.parametrize(
    ("kind", "label", "sector"),
    [
        ("stocks_nasdaq100", "ndx", "NDX"),
        ("stocks_s&p500", "spx", "SPX"),
        ("stocks", "stocks", "STK"),
        ("stocks_long", "stocks", "STK"),
        ("futures", "futures", "FUT"),
    ],
)
def test_sector_distribution_chooses_catalog_by_kind(
    monkeypatch: pytest.MonkeyPatch, kind: str, label: str, sector: str
) -> None:
    spy = install_catalogs(monkeypatch)
    book = frame([[0.4, 0.2], [0.1, 0.0]], "2020-01-06", ["x1", "y1"])
    result = qnstats.calc_sector_distribution(book, kind=dynamic(kind))
    assert list(spy.calls) == [label]
    columns = [*sorted([sector, "Other"]), "Cash"]
    by_sector = {sector: [0.4, 0.1], "Other": [0.2, 0.0], "Cash": [0.4, 0.9]}
    assert list(result.coords["sector"].values) == columns
    assert_allclose(result.values, np.array([by_sector[c] for c in columns]).T, atol=1e-12)
    book.name = dynamic(kind)
    spy.calls.clear()
    by_name = qnstats.calc_sector_distribution(book)
    assert list(spy.calls) == [label]
    assert list(by_name.coords["sector"].values) == columns


@pytest.mark.parametrize(
    "kind",
    [
        "aaa",
        "zzz",
        "stocks_nasdaq101",
        "stocks_s&p501",
        "stocks_l",
        "futures2",
        "crypto",
        "",
        "future",
    ],
)
def test_sector_distribution_of_unknown_kind_is_all_other(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    spy = install_catalogs(monkeypatch)
    book = frame([[0.4, 0.2]], "2020-01-06", ["x1", "y1"])
    result = qnstats.calc_sector_distribution(book, kind=dynamic(kind))
    assert spy.calls == {}
    assert list(result.coords["sector"].values) == ["Other", "Cash"]
    assert_allclose(result.values, [[0.6, 0.4]], atol=1e-12)
    unnamed = qnstats.calc_sector_distribution(book)
    assert list(unnamed.coords["sector"].values) == ["Other", "Cash"]
