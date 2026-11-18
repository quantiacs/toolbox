from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
import pytest
import talib
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from talib._ta_lib import MA_Type

import qnt.xr_talib as xt
from qnt.data import ds, f
from qnt.tests.hl_support import FIELDS, Floats, business_days, make_data

N_DAYS = 200
# a dimension named "time" that is not the interned literal, to tell `!=` from `is not`
TIME_DIM = "".join(["ti", "me"])

Value: TypeAlias = float | int | str
Ohlc: TypeAlias = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]
Reducer: TypeAlias = Callable[..., Floats]


def trend_market() -> xr.DataArray:
    """Three assets that trend up and down in 40-bar legs, so parabolic SARs reverse and ramp up."""
    data = make_data(n_assets=3, n_days=N_DAYS, name="stocks", seed=31)
    values = data.values.copy()
    rng = np.random.default_rng(3)
    leg = 40
    wave = np.abs(((np.arange(N_DAYS) + 10) % (2 * leg)) - leg).astype(np.float64)
    for asset in range(3):
        close = 40.0 + 1.1 * wave + 5.0 * asset + rng.normal(0.0, 0.4, N_DAYS)
        open_ = close + rng.normal(0.0, 0.3, N_DAYS)
        high = np.maximum(open_, close) + rng.uniform(0.2, 1.0, N_DAYS)
        low = np.minimum(open_, close) - rng.uniform(0.2, 1.0, N_DAYS)
        for field, layer in (("open", open_), ("high", high), ("low", low), ("close", close)):
            values[:, FIELDS.index(field), asset] = layer
    return data.copy(data=values)


@pytest.fixture(scope="module")
def trend() -> xr.DataArray:
    return trend_market()


@dataclass(frozen=True)
class Default:
    name: str
    explicit: dict[str, Value]
    on_close: bool = False


DEFAULTS = [
    Default("ADOSC", {"fastperiod": 3, "slowperiod": 10}),
    Default("ADX", {"timeperiod": 14}),
    Default("ADXR", {"timeperiod": 14}),
    Default("APO", {"fastperiod": 14, "slowperiod": 26, "matype": 0}, on_close=True),
    Default("AROON", {"timeperiod": 14}),
    Default("AROONOSC", {"timeperiod": 14}),
    Default("ATR", {"timeperiod": 14}),
    Default(
        "BBANDS",
        {"timeperiod": 14, "nbdevup": 2, "nbdevdn": 2, "matype": 0},
        on_close=True,
    ),
    Default("CCI", {"timeperiod": 14}),
    Default("CMO", {"timeperiod": 14}, on_close=True),
    Default("DEMA", {"timeperiod": 30}, on_close=True),
    Default("DX", {"timeperiod": 14}),
    Default("EMA", {"timeperiod": 30}, on_close=True),
    Default("KAMA", {"timeperiod": 30}, on_close=True),
    Default("LINEARREG", {"timeperiod": 14}, on_close=True),
    Default("LINEARREG_ANGLE", {"timeperiod": 14}, on_close=True),
    Default("LINEARREG_INTERCEPT", {"timeperiod": 14}, on_close=True),
    Default("LINEARREG_SLOPE", {"timeperiod": 14}, on_close=True),
    Default("MA", {"timeperiod": 30, "matype": 0}, on_close=True),
    Default("MACD", {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, on_close=True),
    Default(
        "MACDEXT",
        {
            "fastperiod": 12,
            "fastmatype": 0,
            "slowperiod": 26,
            "slowmatype": 0,
            "signalperiod": 9,
            "signalmatype": 0,
        },
        on_close=True,
    ),
    Default("MACDFIX", {"signalperiod": 9}, on_close=True),
    Default("MAMA", {"fastlimit": 0.5, "slowlimit": 0.05}, on_close=True),
    Default("MFI", {"timeperiod": 14}),
    Default("MINUS_DI", {"timeperiod": 14}),
    Default("MINUS_DM", {"timeperiod": 14}),
    Default("MOM", {"timeperiod": 10}, on_close=True),
    Default("NATR", {"timeperiod": 14}),
    Default("OBV", {"price_field": "close"}),
    Default("PLUS_DI", {"timeperiod": 14}),
    Default("PLUS_DM", {"timeperiod": 14}),
    Default("PPO", {"fastperiod": 12, "slowperiod": 26, "matype": 0}, on_close=True),
    Default("ROC", {"timeperiod": 14}, on_close=True),
    Default("ROCP", {"timeperiod": 14}, on_close=True),
    Default("ROCR", {"timeperiod": 14}, on_close=True),
    Default("ROCR100", {"timeperiod": 14}, on_close=True),
    Default("RSI", {"timeperiod": 14}, on_close=True),
    Default("SAR", {"acceleration": 0.02, "maximum": 0.2}),
    Default(
        "SAREXT",
        {
            "startvalue": 0,
            "offsetonreverse": 0,
            "accelerationinitlong": 0.02,
            "accelerationlong": 0.02,
            "accelerationmaxlong": 0.2,
            "accelerationinitshort": 0.02,
            "accelerationshort": 0.02,
            "accelerationmaxshort": 0.2,
        },
    ),
    Default("SMA", {"timeperiod": 14}, on_close=True),
    Default("STDDEV", {"timeperiod": 14}, on_close=True),
    Default(
        "STOCH",
        {
            "fastk_period": 5,
            "slowk_period": 3,
            "slowk_matype": 0,
            "slowd_period": 3,
            "slowd_matype": 0,
        },
    ),
    Default("STOCHF", {"fastk_period": 5, "fastd_period": 3, "fastd_matype": 0}),
    Default(
        "STOCHRSI",
        {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3, "fastd_matype": 0},
        on_close=True,
    ),
    Default("T3", {"timeperiod": 5, "vfactor": 0.7}, on_close=True),
    Default("TEMA", {"timeperiod": 30}, on_close=True),
    Default("TRIMA", {"timeperiod": 30}, on_close=True),
    Default("TRIX", {"timeperiod": 30}, on_close=True),
    Default("TSF", {"timeperiod": 14}, on_close=True),
    Default("ULTOSC", {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28}),
    Default("VAR", {"timeperiod": 5}, on_close=True),
    Default("WILLR", {"timeperiod": 14}),
    Default("WMA", {"timeperiod": 30}, on_close=True),
]


@pytest.mark.parametrize("case", DEFAULTS, ids=[case.name for case in DEFAULTS])
def test_call_without_parameters_equals_call_with_the_signature_defaults(
    case: Default, trend: xr.DataArray
) -> None:
    source = trend.sel(field="close") if case.on_close else trend
    function = getattr(xt, case.name)
    default = function(source)
    explicit = function(source, **case.explicit)
    assert default.dims == explicit.dims
    assert_array_equal(default.values, explicit.values)
    assert np.isfinite(explicit.values).any()


def test_cross_series_defaults_are_5_for_beta_and_30_for_correl(trend: xr.DataArray) -> None:
    close = trend.sel(field="close")
    reference = close.sel(asset=trend.coords["asset"].values[0])
    for function, explicit in ((xt.BETA, 5), (xt.CORREL, 30)):
        default = function(close, reference)
        assert_array_equal(
            default.values, function(close, reference, explicit).values
        )
        assert np.isfinite(default.values).any()


# --- moving-average type codes ------------------------------------------------------------

MA_TYPE_NAMES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA", "KAMA", "MAMA", "T3"]


@pytest.mark.parametrize("name", MA_TYPE_NAMES)
def test_ma_type_codes_are_the_talib_codes_and_select_the_average(
    name: str, trend: xr.DataArray
) -> None:
    code = getattr(xt.MaType, name)
    assert code == int(getattr(MA_Type, name))
    close = trend.sel(field="close")
    result = xt.MA(close, 10, code)
    for asset in trend.coords["asset"].values:
        prices = close.sel(asset=asset).values
        assert_allclose(
            result.sel(asset=asset).values,
            talib.MA(prices, 10, getattr(MA_Type, name)),
            rtol=1e-12,
            equal_nan=True,
        )


# --- candlestick patterns that never fire on the random market data -----------------------
# One or two OHLC sequences per pattern (open, high, low, close), found by searching for bars
# on which TA-Lib reports the pattern.

# fmt: off
CANDLES: dict[str, tuple[Ohlc, ...]] = {
    "CDL3BLACKCROWS": (
        (
            (99.9, 99.7, 98.6, 94.2, 93.3, 91.5, 90.1, 94.5, 91.5, 92.2, 89.7, 89.7, 87.2, 86.7),
            (100.7, 99.8, 100.6, 95.5, 93.5, 92.0, 93.9, 95.1, 91.5, 92.6, 90.8, 90.4, 87.5, 86.8),
            (98.1, 99.4, 94.2, 92.1, 89.8, 89.8, 89.2, 91.5, 90.0, 89.8, 89.2, 86.4, 86.2, 85.0),
            (99.5, 99.6, 95.2, 92.5, 91.0, 90.2, 93.3, 92.2, 90.7, 89.9, 89.8, 86.5, 86.4, 85.1),
        ),
    ),
    "CDL3STARSINSOUTH": (
        (
            (100.2, 99.4, 100.0, 100.4, 99.3, 98.3, 100.6, 100.3, 101.6, 102.8, 102.9, 102.9,
                103.3, 101.8, 100.2),
            (100.7, 100.1, 100.7, 100.6, 99.8, 100.7, 100.8, 102.5, 102.8, 103.0, 103.2, 103.7,
                103.3, 101.9, 100.2),
            (99.4, 98.9, 99.5, 99.4, 97.3, 98.0, 99.8, 100.3, 101.3, 102.2, 102.7, 102.7, 96.1,
                98.9, 99.8),
            (99.5, 100.0, 100.5, 99.5, 98.0, 100.4, 100.3, 101.5, 102.4, 102.7, 103.0, 103.3,
                99.8, 99.3, 99.8),
        ),
    ),
    "CDL3WHITESOLDIERS": (
        (
            (99.2, 101.4, 102.6, 103.7, 104.2, 104.1, 104.5, 105.6, 109.3, 109.9, 109.4, 108.6,
                111.6, 113.6),
            (101.2, 102.8, 103.9, 105.4, 105.6, 105.2, 107.4, 109.2, 112.1, 110.6, 110.1, 112.3,
                114.5, 117.7),
            (98.9, 100.2, 102.6, 102.4, 104.0, 103.6, 104.3, 104.1, 109.2, 108.7, 107.0, 108.1,
                110.4, 112.8),
            (101.2, 101.7, 103.9, 104.5, 105.5, 104.7, 105.6, 108.5, 110.7, 109.1, 107.3, 112.2,
                114.5, 117.4),
        ),
    ),
    "CDLABANDONEDBABY": (
        (
            (101.0, 99.2, 98.3, 98.2, 97.0, 97.5, 98.1, 100.3, 99.6, 98.9, 97.3, 100.0, 102.3,
                101.6),
            (101.9, 100.3, 100.4, 98.6, 98.3, 97.7, 99.9, 100.9, 100.0, 99.3, 101.1, 101.9,
                103.0, 101.8),
            (98.7, 98.0, 98.0, 97.2, 96.3, 96.2, 97.1, 99.1, 99.0, 97.8, 97.1, 99.8, 102.0, 98.8),
            (98.9, 99.6, 99.6, 97.6, 97.0, 96.4, 99.3, 99.1, 99.5, 98.1, 100.1, 101.4, 102.3, 99.7),
        ),
        (
            (99.7, 100.4, 103.3, 103.0, 104.4, 108.2, 110.9, 112.3, 114.5, 117.4, 116.8, 116.9,
                114.1, 114.7),
            (101.4, 102.1, 103.9, 105.9, 106.9, 111.3, 112.2, 115.1, 117.4, 118.5, 117.8, 117.2,
                114.3, 118.3),
            (98.5, 100.3, 101.5, 101.7, 103.9, 108.0, 110.6, 112.1, 114.4, 116.9, 115.9, 114.8,
                113.5, 114.6),
            (100.8, 101.6, 102.8, 105.3, 106.5, 110.7, 111.6, 114.2, 116.4, 117.1, 116.8, 115.2,
                114.2, 117.7),
        ),
    ),
    "CDLBREAKAWAY": (
        (
            (100.0, 99.0, 101.5, 101.5, 101.1, 103.0, 103.7, 102.7, 102.5, 101.8, 99.9, 101.5,
                100.0, 91.7, 89.4, 88.4, 87.2),
            (100.1, 101.6, 102.0, 102.1, 104.0, 105.0, 103.7, 103.3, 104.4, 102.0, 102.2, 101.7,
                100.3, 92.0, 89.7, 88.7, 92.7),
            (98.2, 98.9, 100.3, 100.3, 100.6, 102.6, 101.4, 102.3, 101.9, 99.4, 99.8, 99.4,
                92.6, 88.9, 87.7, 86.6, 86.9),
            (98.9, 101.4, 101.1, 101.0, 103.0, 104.2, 102.3, 102.6, 102.0, 100.1, 101.5, 100.0,
                92.9, 89.2, 88.0, 86.9, 92.4),
        ),
    ),
    "CDLCONCEALBABYSWALL": (
        (
            (100.7, 97.8, 97.9, 95.7, 94.3, 95.6, 95.5, 97.1, 94.9, 97.1, 95.2, 93.6, 91.4, 91.2),
            (101.1, 98.7, 98.3, 96.2, 95.5, 95.9, 96.8, 97.9, 96.9, 97.5, 95.3, 93.7, 92.3, 93.0),
            (99.1, 96.8, 95.1, 93.3, 93.3, 93.0, 95.2, 94.0, 94.3, 95.0, 93.7, 92.1, 89.8, 87.1),
            (99.4, 97.3, 96.1, 93.8, 94.3, 93.3, 96.1, 94.9, 96.0, 96.1, 93.8, 92.2, 90.0, 87.6),
        ),
    ),
    "CDLCOUNTERATTACK": (
        (
            (101.5, 101.4, 102.7, 103.5, 102.8, 102.3, 104.4, 104.3, 104.2, 104.4, 105.6, 108.8,
                108.1, 110.9),
            (103.1, 102.9, 103.7, 104.6, 102.8, 104.0, 105.8, 104.6, 105.1, 106.3, 109.3, 109.3,
                110.6, 111.1),
            (101.3, 101.2, 102.4, 102.7, 102.1, 101.0, 103.3, 103.8, 102.3, 104.0, 105.6, 107.7,
                108.0, 109.4),
            (102.6, 102.7, 103.0, 103.1, 102.7, 103.9, 104.9, 103.9, 103.3, 106.1, 109.0, 108.3,
                109.8, 109.7),
        ),
        (
            (99.6, 101.6, 99.7, 97.3, 97.5, 97.7, 98.1, 98.5, 98.4, 101.1, 100.9, 102.0, 102.2,
                100.6),
            (100.7, 101.7, 100.1, 97.7, 97.6, 98.3, 98.3, 99.2, 100.4, 102.4, 101.4, 103.2,
                102.4, 101.6),
            (98.8, 99.7, 97.7, 96.2, 96.3, 97.7, 97.3, 97.7, 97.4, 100.2, 99.6, 101.5, 100.4,
                100.2),
            (100.5, 99.9, 97.9, 96.7, 97.6, 98.3, 97.9, 98.5, 99.9, 102.2, 100.5, 102.3, 101.4,
                101.3),
        ),
    ),
    "CDLDARKCLOUDCOVER": (
        (
            (100.9, 104.4, 105.2, 107.0, 107.0, 108.5, 112.0, 113.8, 118.2, 116.3, 116.7, 113.6,
                112.0, 114.4),
            (103.8, 105.0, 107.2, 107.1, 108.1, 113.0, 113.8, 117.3, 118.4, 118.3, 118.1, 113.9,
                114.3, 115.0),
            (100.9, 104.1, 104.7, 105.8, 106.9, 108.4, 111.5, 113.2, 115.9, 116.3, 113.6, 111.8,
                111.7, 111.9),
            (103.5, 104.9, 106.4, 106.8, 107.0, 112.5, 112.7, 117.1, 116.5, 117.3, 114.4, 111.9,
                114.2, 113.0),
        ),
    ),
    "CDLGAPSIDESIDEWHITE": (
        (
            (100.3, 99.6, 99.2, 99.9, 99.1, 98.7, 98.2, 100.1, 94.7, 94.1, 93.5, 92.1, 94.3, 94.4),
            (101.0, 100.0, 100.6, 101.4, 99.3, 99.1, 100.0, 100.7, 95.6, 94.2, 94.2, 93.5, 94.8,
                95.9),
            (100.1, 98.9, 99.0, 99.7, 98.0, 98.3, 97.6, 95.6, 93.6, 92.7, 91.2, 91.6, 93.8, 93.9),
            (100.9, 99.1, 100.5, 100.0, 98.4, 98.5, 99.4, 95.7, 93.8, 93.7, 91.6, 93.4, 94.5, 94.4),
        ),
        (
            (100.5, 101.0, 101.5, 102.0, 103.5, 104.2, 106.6, 109.4, 108.3, 110.0, 107.6, 108.2,
                107.3, 107.2),
            (101.6, 102.4, 103.0, 102.8, 105.3, 107.0, 111.1, 109.5, 111.0, 110.7, 109.3, 109.1,
                108.8, 109.3),
            (99.5, 100.9, 101.1, 101.9, 102.8, 103.2, 106.4, 109.0, 107.9, 108.0, 106.9, 107.6,
                106.9, 105.7),
            (100.6, 102.0, 101.4, 102.6, 104.3, 106.9, 109.3, 109.1, 110.7, 108.4, 108.1, 108.3,
                108.1, 107.9),
        ),
    ),
    "CDLHIKKAKEMOD": (
        (
            (98.0, 96.9, 96.6, 94.5, 93.3, 92.0, 90.5, 89.4, 86.4, 85.3, 83.4, 83.2, 84.5, 83.7),
            (98.6, 97.4, 97.4, 94.6, 93.5, 92.4, 91.5, 89.5, 87.6, 85.3, 83.9, 83.7, 85.6, 83.7),
            (95.6, 95.1, 92.8, 91.7, 92.4, 90.2, 88.7, 86.7, 85.5, 82.5, 82.6, 83.0, 83.2, 82.1),
            (97.0, 96.8, 93.1, 93.0, 92.7, 91.0, 88.8, 87.1, 85.7, 83.6, 83.4, 83.4, 83.7, 82.3),
        ),
        (
            (97.8, 96.1, 91.7, 91.3, 93.3, 93.0, 92.4, 91.5, 85.5, 87.8, 87.0, 87.0, 87.2, 86.9),
            (98.7, 96.5, 93.0, 94.4, 95.0, 93.5, 92.7, 92.8, 88.8, 89.1, 88.3, 87.6, 87.3, 88.3),
            (97.1, 92.8, 91.6, 90.2, 93.1, 92.0, 90.7, 84.5, 84.3, 84.8, 86.6, 86.7, 86.2, 85.8),
            (97.2, 93.0, 91.9, 93.3, 94.1, 92.8, 91.9, 85.7, 87.5, 86.9, 87.3, 87.3, 86.5, 87.9),
        ),
    ),
    "CDLIDENTICAL3CROWS": (
        (
            (99.9, 103.1, 106.6, 107.4, 104.3, 103.2, 101.4, 104.3, 103.8, 102.5, 103.0, 100.9,
                100.4, 99.7),
            (104.3, 106.0, 108.1, 108.0, 104.4, 104.6, 104.4, 105.0, 104.1, 103.7, 103.5, 101.2,
                100.9, 100.3),
            (99.8, 102.2, 105.2, 104.2, 103.3, 100.9, 101.2, 103.5, 102.3, 101.9, 101.4, 100.2,
                99.7, 98.4),
            (102.9, 105.3, 107.0, 104.8, 104.0, 101.4, 104.0, 103.9, 102.4, 103.4, 102.1, 100.4,
                99.7, 98.6),
        ),
    ),
    "CDLKICKING": (
        (
            (99.5, 101.4, 99.4, 96.8, 94.8, 95.4, 95.7, 96.9, 96.7, 98.5, 99.3, 101.5, 100.9,
                100.0),
            (102.3, 101.8, 99.9, 97.1, 96.1, 95.5, 95.9, 97.1, 98.3, 99.5, 101.6, 102.3, 102.0,
                100.0),
            (98.9, 99.8, 97.9, 94.2, 93.5, 93.7, 94.5, 96.5, 96.6, 98.1, 98.6, 100.6, 100.8, 98.2),
            (100.8, 100.1, 99.0, 94.4, 95.6, 94.1, 95.6, 96.9, 98.3, 99.4, 101.3, 101.2, 102.0,
                98.2),
        ),
        (
            (100.2, 101.5, 98.9, 97.6, 100.9, 102.3, 101.0, 101.2, 99.5, 98.3, 96.8, 96.0, 93.6,
                94.0),
            (103.5, 102.0, 99.1, 101.0, 103.1, 102.6, 101.5, 101.4, 100.2, 98.5, 97.8, 96.7,
                93.7, 96.4),
            (100.0, 98.8, 97.8, 97.2, 100.2, 100.1, 99.8, 98.5, 98.4, 96.8, 95.6, 93.8, 92.1, 93.8),
            (101.8, 99.6, 98.2, 100.1, 102.6, 101.3, 101.3, 98.6, 98.9, 96.8, 95.8, 94.3, 92.1,
                96.3),
        ),
    ),
    "CDLKICKINGBYLENGTH": (
        (
            (99.5, 101.4, 99.4, 96.8, 94.8, 95.4, 95.7, 96.9, 96.7, 98.5, 99.3, 101.5, 100.9,
                100.0),
            (102.3, 101.8, 99.9, 97.1, 96.1, 95.5, 95.9, 97.1, 98.3, 99.5, 101.6, 102.3, 102.0,
                100.0),
            (98.9, 99.8, 97.9, 94.2, 93.5, 93.7, 94.5, 96.5, 96.6, 98.1, 98.6, 100.6, 100.8, 98.2),
            (100.8, 100.1, 99.0, 94.4, 95.6, 94.1, 95.6, 96.9, 98.3, 99.4, 101.3, 101.2, 102.0,
                98.2),
        ),
        (
            (100.2, 101.5, 98.9, 97.6, 100.9, 102.3, 101.0, 101.2, 99.5, 98.3, 96.8, 96.0, 93.6,
                94.0),
            (103.5, 102.0, 99.1, 101.0, 103.1, 102.6, 101.5, 101.4, 100.2, 98.5, 97.8, 96.7,
                93.7, 96.4),
            (100.0, 98.8, 97.8, 97.2, 100.2, 100.1, 99.8, 98.5, 98.4, 96.8, 95.6, 93.8, 92.1, 93.8),
            (101.8, 99.6, 98.2, 100.1, 102.6, 101.3, 101.3, 98.6, 98.9, 96.8, 95.8, 94.3, 92.1,
                96.3),
        ),
    ),
    "CDLLADDERBOTTOM": (
        (
            (100.5, 103.9, 106.7, 102.4, 101.4, 100.0, 100.2, 100.4, 98.7, 99.4, 98.3, 99.2,
                99.8, 97.6, 93.2, 89.8, 90.7),
            (104.4, 106.8, 107.1, 103.1, 101.9, 100.6, 100.5, 100.4, 99.7, 100.1, 98.9, 100.5,
                99.9, 97.9, 93.4, 91.0, 96.0),
            (100.4, 102.5, 101.8, 101.5, 99.4, 99.0, 99.9, 99.0, 98.7, 98.6, 97.9, 98.7, 96.5,
                92.6, 89.4, 87.6, 90.4),
            (104.2, 106.7, 102.8, 101.6, 99.6, 100.1, 99.9, 99.1, 99.6, 99.2, 98.6, 99.8, 96.7,
                92.7, 89.5, 88.1, 95.7),
        ),
    ),
    "CDLMATHOLD": (
        (
            (100.0, 100.4, 98.8, 98.5, 96.3, 97.6, 94.5, 94.4, 98.4, 100.4, 102.4, 99.9, 102.0,
                112.3, 111.0, 110.2, 111.1),
            (100.3, 100.7, 99.1, 98.6, 98.5, 99.6, 95.2, 98.7, 100.3, 102.5, 102.5, 102.2,
                110.8, 112.6, 111.4, 110.4, 113.3),
            (99.8, 97.9, 98.3, 95.4, 96.0, 94.2, 94.0, 94.2, 97.9, 100.2, 99.1, 99.7, 101.7,
                110.5, 109.5, 109.5, 110.8),
            (100.2, 98.7, 98.3, 96.4, 97.9, 94.3, 94.1, 98.5, 99.9, 101.7, 99.8, 102.0, 110.5,
                110.6, 109.5, 109.5, 113.0),
        ),
    ),
    "CDLONNECK": (
        (
            (100.8, 101.6, 100.4, 101.5, 104.1, 102.7, 98.8, 100.0, 99.1, 98.7, 98.7, 95.5,
                94.5, 90.7),
            (102.0, 102.8, 102.2, 105.3, 104.9, 103.6, 101.2, 100.2, 99.3, 98.8, 99.6, 96.8,
                94.7, 92.0),
            (100.5, 100.5, 99.7, 100.9, 101.5, 99.1, 98.7, 99.3, 98.6, 96.6, 95.4, 93.3, 91.4,
                89.9),
            (101.2, 100.7, 101.4, 104.5, 101.9, 99.5, 100.0, 99.3, 99.0, 97.0, 95.4, 94.6, 91.8,
                91.4),
        ),
    ),
    "CDLPIERCING": (
        (
            (101.3, 101.2, 100.3, 99.8, 98.9, 99.7, 100.1, 97.5, 97.3, 96.8, 97.4, 98.5, 100.5,
                98.7),
            (101.8, 102.0, 100.9, 100.4, 99.6, 100.1, 100.2, 100.9, 97.3, 98.3, 98.7, 100.0,
                100.9, 101.4),
            (100.6, 100.9, 99.6, 99.4, 98.8, 99.6, 97.7, 96.7, 95.5, 96.2, 97.4, 98.1, 99.5, 98.2),
            (101.5, 101.7, 100.2, 99.9, 99.1, 99.9, 98.4, 100.3, 97.0, 97.5, 98.1, 99.6, 99.6,
                100.3),
        ),
    ),
    "CDLRISEFALL3METHODS": (
        (
            (99.6, 99.3, 97.8, 97.3, 96.1, 99.6, 99.8, 101.5, 101.4, 101.3, 98.6, 98.8, 96.4,
                104.2, 103.6, 102.6, 103.1),
            (100.5, 99.5, 98.2, 97.4, 100.5, 100.4, 102.3, 102.5, 102.5, 102.1, 99.0, 99.0,
                104.9, 104.7, 104.0, 102.7, 107.6),
            (98.6, 97.1, 96.8, 95.5, 95.9, 99.5, 99.5, 101.3, 101.3, 98.3, 98.3, 96.2, 96.1,
                103.2, 101.9, 101.7, 102.8),
            (99.7, 98.0, 97.1, 95.9, 99.9, 99.6, 101.3, 101.3, 101.5, 98.4, 98.4, 96.4, 104.6,
                103.2, 102.2, 102.0, 107.3),
        ),
    ),
    "CDLTHRUSTING": (
        (
            (99.1, 99.9, 99.3, 99.7, 99.9, 100.1, 99.8, 99.1, 96.3, 93.9, 92.8, 94.9, 93.9, 90.8),
            (99.7, 100.3, 101.2, 100.0, 100.3, 100.7, 100.3, 99.6, 96.7, 94.5, 94.2, 95.3, 93.9,
                93.0),
            (98.7, 98.8, 99.3, 99.2, 99.9, 99.9, 98.2, 96.8, 95.3, 93.2, 92.6, 94.8, 91.5, 90.6),
            (99.4, 99.3, 100.5, 99.4, 100.2, 99.9, 98.6, 96.9, 95.5, 93.3, 93.9, 94.9, 91.9, 92.6),
        ),
    ),
    "CDLTRISTAR": (
        (
            (100.0, 101.5, 103.2, 103.2, 104.8, 106.4, 106.0, 105.7, 105.9, 103.6, 103.6, 103.8,
                103.5, 106.3),
            (101.9, 103.7, 104.2, 105.8, 105.4, 107.0, 107.0, 106.5, 106.1, 105.2, 105.1, 104.0,
                104.2, 108.2),
            (99.7, 101.4, 103.1, 103.0, 103.5, 105.5, 104.9, 103.9, 103.5, 103.4, 102.5, 103.4,
                102.8, 105.8),
            (101.1, 102.7, 104.2, 105.0, 105.2, 106.2, 106.4, 106.2, 103.8, 104.0, 104.3, 103.7,
                103.6, 106.3),
        ),
        (
            (101.6, 103.1, 102.0, 105.1, 108.1, 109.3, 111.1, 113.6, 116.1, 117.6, 118.8, 122.1,
                122.2, 121.8),
            (104.2, 104.7, 105.9, 108.8, 110.7, 111.5, 116.0, 116.1, 117.4, 119.5, 122.0, 122.4,
                123.0, 121.9),
            (101.3, 102.0, 100.9, 103.8, 107.7, 108.4, 111.1, 111.5, 115.5, 116.4, 118.8, 121.6,
                121.2, 121.6),
            (103.7, 102.3, 105.0, 108.2, 110.0, 111.3, 115.0, 116.0, 116.8, 118.8, 121.7, 121.9,
                122.5, 121.8),
        ),
    ),
    "CDLUNIQUE3RIVER": (
        (
            (99.5, 101.5, 102.0, 102.2, 102.2, 101.1, 102.1, 103.7, 98.8, 97.2, 99.6, 98.7,
                97.0, 96.7),
            (102.7, 102.0, 102.0, 104.4, 103.7, 102.7, 102.6, 105.0, 100.2, 99.7, 100.8, 99.7,
                98.1, 97.2),
            (99.3, 100.7, 101.5, 100.4, 100.7, 101.0, 101.5, 98.5, 96.5, 97.0, 97.8, 96.3, 95.5,
                96.5),
            (101.1, 101.7, 101.5, 103.7, 101.5, 102.4, 102.2, 99.5, 97.4, 98.8, 98.1, 96.7,
                96.9, 97.1),
        ),
    ),
    "CDLUPSIDEGAP2CROWS": (
        (
            (98.4, 97.4, 98.2, 99.6, 97.6, 98.2, 100.0, 99.4, 97.0, 98.1, 96.7, 96.1, 99.4, 99.7),
            (99.5, 99.7, 100.6, 99.8, 98.4, 99.2, 100.6, 99.4, 97.4, 99.3, 96.9, 98.9, 99.4, 100.8),
            (96.4, 96.9, 97.6, 97.8, 97.6, 97.2, 98.9, 98.3, 96.3, 96.8, 95.7, 96.0, 98.1, 96.8),
            (97.1, 99.6, 100.0, 98.0, 98.0, 97.4, 99.0, 98.6, 97.0, 97.2, 96.3, 98.2, 99.3, 98.6),
        ),
    ),
}
# fmt: on


def candle_panel(sequences: tuple[Ohlc, ...]) -> xr.DataArray:
    length = len(sequences[0][0])
    values = np.empty((length, 4, len(sequences)))
    for asset, sequence in enumerate(sequences):
        for field, layer in enumerate(sequence):
            values[:, field, asset] = layer
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={
            "time": business_days(length),
            "field": [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
            "asset": [f"S{i}" for i in range(len(sequences))],
        },
    )


def talib_pattern(name: str, sequence: Ohlc, **parameters: float) -> Floats:
    function: Callable[..., Floats] = getattr(talib, name)
    arrays = [np.array(layer, dtype=np.float64) for layer in sequence]
    return function(*arrays, **parameters) / 100


@pytest.mark.parametrize("name", sorted(CANDLES))
def test_silent_candlestick_patterns_fire_and_are_scaled_by_one_hundred(name: str) -> None:
    sequences = CANDLES[name]
    result = getattr(xt, name)(candle_panel(sequences))
    assert set(result.dims) == {"time", "asset"}
    for asset, sequence in enumerate(sequences):
        expected = talib_pattern(name, sequence)
        assert expected.any()
        actual = result.sel(asset=f"S{asset}").values
        assert_allclose(actual, expected, atol=1e-12)
    assert np.abs(result.values).max() >= 1.0


@pytest.mark.parametrize(
    ("name", "penetration"),
    [("CDLABANDONEDBABY", 0.3), ("CDLDARKCLOUDCOVER", 0.5), ("CDLMATHOLD", 0.5)],
)
def test_candlestick_penetration_defaults(name: str, penetration: float) -> None:
    sequences = CANDLES[name]
    panel = candle_panel(sequences)
    default = getattr(xt, name)(panel).values
    explicit = getattr(xt, name)(panel, penetration=penetration).values
    assert_array_equal(default, explicit)
    for asset, sequence in enumerate(sequences):
        expected = talib_pattern(name, sequence, penetration=penetration)
        assert expected.any()
        assert_allclose(default[:, asset], expected, atol=1e-12)


# --- call machinery: dimensions, dividers, empty and one-point slices ---------------------


def talib_on_valid(function: Reducer, *series: Floats) -> Floats:
    """Applies a TA-Lib function to the bars where every input is present."""
    valid = np.all([np.isfinite(values) for values in series], axis=0)
    result = np.full(series[0].shape, np.nan)
    if valid.any():
        result[valid] = function(*(values[valid] for values in series))
    return result


def gappy_market() -> xr.DataArray:
    """Asset 1 has no data at all, asset 2 a single bar, asset 3 a gap."""
    data = make_data(n_assets=4, n_days=60, name="stocks", seed=2)
    values = data.values.copy()
    values[:, :, 1] = np.nan
    one_bar = values[30, :, 2].copy()
    values[:, :, 2] = np.nan
    values[30, :, 2] = one_bar
    values[25:28, :, 3] = np.nan
    return data.copy(data=values)


def sma_5(values: Floats) -> Floats:
    return talib.SMA(values, 5)


def correl_5(first: Floats, second: Floats) -> Floats:
    return talib.CORREL(first, second, 5)


def prices_of(data: xr.DataArray, field: str, asset: int) -> Floats:
    return data.sel(field=field, asset=data.coords["asset"].values[asset]).values


@pytest.mark.parametrize("divider", [1, 2, 0.5])
def test_multiple_series_call_divides_every_output_by_the_divider(divider: float) -> None:
    data = gappy_market()
    result = xt.multiple_series_call(
        talib.OBV, data, ds.TIME, ds.FIELD, [f.CLOSE, f.VOL], result_divider=divider
    )
    for asset in range(4):
        close, volume = prices_of(data, f.CLOSE, asset), prices_of(data, f.VOL, asset)
        expected = talib_on_valid(talib.OBV, close, volume) / divider
        actual = result.sel(asset=data.coords["asset"].values[asset]).values
        assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)
    one_bar = result.sel(asset=data.coords["asset"].values[2]).values
    assert np.isfinite(one_bar).sum() == 1
    assert_allclose(one_bar[30], prices_of(data, f.VOL, 2)[30] / divider, rtol=1e-12)


@pytest.mark.parametrize("divider", [1, 2, 0.5])
def test_single_series_call_divides_every_output_by_the_divider(divider: float) -> None:
    close = gappy_market().sel(field=f.CLOSE)
    result = xt.single_series_call(talib.SQRT, close, ds.TIME, [], result_divider=divider)
    expected = np.sqrt(close.values) / divider
    assert_allclose(result.values, expected, rtol=1e-12, equal_nan=True)
    assert (
        np.isfinite(result.sel(asset=close.coords["asset"].values[2]).values).sum() == 1
    )


def zoned(array: xr.DataArray) -> xr.DataArray:
    other = array * 1.5 + 3.0
    return xr.concat([array, other], pd.Index(["u", "v"], name="zone"))


def test_single_series_call_loops_over_every_non_time_dimension() -> None:
    close = zoned(gappy_market().sel(field=f.CLOSE))
    assert close.dims[0] == "zone"
    for source in (close, close.rename({"time": TIME_DIM})):
        result = xt.SMA(source, 5)
        assert result.dims == source.dims
        for zone in ("u", "v"):
            for asset in range(4):
                label = source.coords["asset"].values[asset]
                values = source.sel(zone=zone, asset=label).values
                expected = talib_on_valid(sma_5, values)
                actual = result.sel(zone=zone, asset=label).values
                assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_multiple_series_call_loops_over_every_non_time_non_field_dimension() -> None:
    data = zoned(gappy_market())
    result = xt.OBV(data)
    assert set(result.dims) == {"zone", "time", "asset"}
    for zone in ("u", "v"):
        for asset in range(4):
            label = data.coords["asset"].values[asset]
            close = data.sel(zone=zone, field=f.CLOSE, asset=label).values
            volume = data.sel(zone=zone, field=f.VOL, asset=label).values
            expected = talib_on_valid(talib.OBV, close, volume)
            actual = result.sel(zone=zone, asset=label).values
            assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)


def test_multiple_series_call_keeps_the_first_input_field_as_a_scalar_coordinate() -> None:
    result = xt.AD(gappy_market())
    assert str(result.coords[ds.FIELD].values) == FIELDS[0]
    assert result.dims == (ds.TIME, ds.ASSET)


def test_result_coordinates_are_ignored_without_a_result_dimension() -> None:
    data = gappy_market()
    close = data.sel(field=f.CLOSE)
    single = xt.single_series_call(talib.SMA, close, ds.TIME, [3], None, ["a", "b"])
    assert single.dims == close.dims
    assert_allclose(single.values, xt.SMA(close, 3).values, equal_nan=True)
    multiple = xt.multiple_series_call(
        talib.OBV, data, ds.TIME, ds.FIELD, [f.CLOSE, f.VOL], (), None, ["a", "b"]
    )
    assert multiple.dims == (ds.TIME, ds.ASSET)
    assert_allclose(multiple.values, xt.OBV(data).values, equal_nan=True)


def pair_market() -> tuple[xr.DataArray, xr.DataArray]:
    """First: asset 0 is empty, 1 ends at bar 29, 2 starts at bar 20, 3 is complete."""
    first = make_data(n_assets=4, n_days=60, name="a", seed=4).sel(field=f.CLOSE)
    second = make_data(n_assets=3, n_days=60, name="b", seed=5).sel(field=f.CLOSE)
    first_values = first.values.copy()
    first_values[:, 0] = np.nan
    first_values[30:, 1] = np.nan
    first_values[:20, 2] = np.nan
    second_values = second.values.copy()
    second_values[:30, 1] = np.nan
    second_values[25:27, 2] = np.nan
    renamed = xr.DataArray(
        second_values,
        dims=["time", "asset"],
        coords={"time": first.coords["time"].values, "asset": ["X0", "X1", "X2"]},
    )
    return first.copy(data=first_values), renamed


@pytest.mark.parametrize(
    ("function", "reference"), [(xt.CORREL, talib.CORREL), (xt.BETA, talib.BETA)]
)
def test_cross_series_call_covers_every_pair_of_assets(
    function: Callable[..., xr.DataArray], reference: Callable[..., Floats]
) -> None:
    first, second = pair_market()
    result = function(first, second, 5)
    assert set(result.dims) == {"time", "first", "second"}

    def with_period(x: Floats, y: Floats) -> Floats:
        return reference(x, y, 5)

    for a in first.coords["asset"].values:
        for b in second.coords["asset"].values:
            x = first.sel(asset=a).values
            y = second.sel(asset=b).values
            expected = talib_on_valid(with_period, x, y)
            actual = result.sel(first=a, second=b).values
            assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True)
    assert np.isfinite(result.sel(first="NYS:A3", second="X0").values).any()
    assert np.isnan(result.sel(first="NYS:A1", second="X1").values).all()
    assert np.isnan(result.sel(first="NYS:A0", second="X0").values).all()


def test_cross_series_call_with_one_asset_axis_only() -> None:
    first, second = pair_market()
    single = first.sel(asset="NYS:A3")
    by_second = xt.CORREL(single, second, 5)
    assert set(by_second.dims) == {"time", "second"}
    for b in second.coords["asset"].values:
        y = second.sel(asset=b).values
        expected = talib_on_valid(correl_5, single.values, y)
        actual = by_second.sel(second=b).values
        assert_allclose(actual, expected, rtol=1e-9, equal_nan=True)
    index = second.sel(asset="X0")
    by_first = xt.CORREL(first, index, 5)
    assert set(by_first.dims) == {"time", "first"}
    for a in first.coords["asset"].values:
        x = first.sel(asset=a).values
        expected = talib_on_valid(correl_5, x, index.values)
        actual = by_first.sel(first=a).values
        assert_allclose(actual, expected, rtol=1e-9, equal_nan=True)


def test_cross_series_call_loops_over_shared_extra_dimensions() -> None:
    def build(seeds: tuple[int, int], names: list[str]) -> xr.DataArray:
        layers = [
            make_data(n_assets=2, n_days=40, name="x", seed=seed).sel(field=f.CLOSE)
            for seed in seeds
        ]
        combined = xr.concat(layers, pd.Index(["u", "v"], name="zone"))
        combined.coords["asset"] = names
        return combined

    first = build((6, 7), ["P0", "P1"])
    second = build((8, 9), ["Q0", "Q1"])
    result = xt.CORREL(first, second, 6)
    assert set(result.dims) == {"zone", "time", "first", "second"}
    for zone in ("u", "v"):
        for a in ("P0", "P1"):
            for b in ("Q0", "Q1"):
                x = first.sel(zone=zone, asset=a).values
                y = second.sel(zone=zone, asset=b).values
                actual = result.sel(zone=zone, first=a, second=b).values
                assert_allclose(actual, talib.CORREL(x, y, 6), rtol=1e-9, equal_nan=True)
