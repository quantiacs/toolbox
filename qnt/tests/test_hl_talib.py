from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pytest
import talib
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from talib._ta_lib import MA_Type

import qnt.xr_talib as xt
from qnt.tests.hl_support import Floats, make_data

N_ASSETS = 4
N_DAYS = 120
GAP = 10


@dataclass(frozen=True)
class Bars:
    open: Floats
    high: Floats
    low: Floats
    close: Floats
    volume: Floats


Reference: TypeAlias = Callable[[Bars], Floats | tuple[Floats, ...]]
Indicator: TypeAlias = Callable[[xr.DataArray], xr.DataArray]


@dataclass(frozen=True)
class Case:
    name: str
    call: Indicator
    reference: Reference
    outputs: tuple[str, ...] = ()
    output_dim: str = ""
    on_close_only: bool = False


def market() -> xr.DataArray:
    data = make_data(n_assets=N_ASSETS, n_days=N_DAYS, name="stocks", seed=17)
    values = data.values.copy()
    values[:GAP, :, N_ASSETS - 1] = np.nan
    return data.copy(data=values)


@pytest.fixture(scope="module")
def data() -> xr.DataArray:
    return market()


def bars_of(data: xr.DataArray, asset: int) -> tuple[Bars, int]:
    one = data.sel(asset=data.coords["asset"].values[asset])
    start = GAP if asset == N_ASSETS - 1 else 0

    def pick(field: str) -> Floats:
        return one.sel(field=field).values[start:]

    bars = Bars(pick("open"), pick("high"), pick("low"), pick("close"), pick("vol"))
    return bars, start


CASES = [
    Case("SMA", lambda d: xt.SMA(d, 10), lambda b: talib.SMA(b.close, 10), on_close_only=True),
    Case("EMA", lambda d: xt.EMA(d, 10), lambda b: talib.EMA(b.close, 10), on_close_only=True),
    Case("WMA", lambda d: xt.WMA(d, 10), lambda b: talib.WMA(b.close, 10), on_close_only=True),
    Case("DEMA", lambda d: xt.DEMA(d, 10), lambda b: talib.DEMA(b.close, 10), on_close_only=True),
    Case("TEMA", lambda d: xt.TEMA(d, 10), lambda b: talib.TEMA(b.close, 10), on_close_only=True),
    Case(
        "TRIMA", lambda d: xt.TRIMA(d, 10), lambda b: talib.TRIMA(b.close, 10), on_close_only=True
    ),
    Case("KAMA", lambda d: xt.KAMA(d, 10), lambda b: talib.KAMA(b.close, 10), on_close_only=True),
    Case("T3", lambda d: xt.T3(d, 5, 0.7), lambda b: talib.T3(b.close, 5, 0.7), on_close_only=True),
    Case("RSI", lambda d: xt.RSI(d, 14), lambda b: talib.RSI(b.close, 14), on_close_only=True),
    Case("ROC", lambda d: xt.ROC(d, 12), lambda b: talib.ROC(b.close, 12), on_close_only=True),
    Case("MOM", lambda d: xt.MOM(d, 10), lambda b: talib.MOM(b.close, 10), on_close_only=True),
    Case("TRIX", lambda d: xt.TRIX(d, 8), lambda b: talib.TRIX(b.close, 8), on_close_only=True),
    Case("CMO", lambda d: xt.CMO(d, 14), lambda b: talib.CMO(b.close, 14), on_close_only=True),
    Case(
        "STDDEV",
        lambda d: xt.STDDEV(d, 10),
        lambda b: talib.STDDEV(b.close, 10),
        on_close_only=True,
    ),
    Case("VAR", lambda d: xt.VAR(d, 10), lambda b: talib.VAR(b.close, 10), on_close_only=True),
    Case(
        "LINEARREG",
        lambda d: xt.LINEARREG(d, 14),
        lambda b: talib.LINEARREG(b.close, 14),
        on_close_only=True,
    ),
    Case("TSF", lambda d: xt.TSF(d, 14), lambda b: talib.TSF(b.close, 14), on_close_only=True),
    Case(
        "HT_TRENDLINE", xt.HT_TRENDLINE, lambda b: talib.HT_TRENDLINE(b.close), on_close_only=True
    ),
    Case(
        "APO",
        lambda d: xt.APO(d, 5, 20, 1),
        lambda b: talib.APO(b.close, 5, 20, MA_Type.EMA),
        on_close_only=True,
    ),
    Case(
        "PPO",
        lambda d: xt.PPO(d, 5, 20, 0),
        lambda b: talib.PPO(b.close, 5, 20, MA_Type.SMA),
        on_close_only=True,
    ),
    Case("ATR", lambda d: xt.ATR(d, 14), lambda b: talib.ATR(b.high, b.low, b.close, 14)),
    Case("NATR", lambda d: xt.NATR(d, 14), lambda b: talib.NATR(b.high, b.low, b.close, 14)),
    Case("TRANGE", xt.TRANGE, lambda b: talib.TRANGE(b.high, b.low, b.close)),
    Case("ADX", lambda d: xt.ADX(d, 14), lambda b: talib.ADX(b.high, b.low, b.close, 14)),
    Case("ADXR", lambda d: xt.ADXR(d, 14), lambda b: talib.ADXR(b.high, b.low, b.close, 14)),
    Case("DX", lambda d: xt.DX(d, 14), lambda b: talib.DX(b.high, b.low, b.close, 14)),
    Case(
        "PLUS_DI", lambda d: xt.PLUS_DI(d, 14), lambda b: talib.PLUS_DI(b.high, b.low, b.close, 14)
    ),
    Case("MINUS_DM", lambda d: xt.MINUS_DM(d, 14), lambda b: talib.MINUS_DM(b.high, b.low, 14)),
    Case("CCI", lambda d: xt.CCI(d, 14), lambda b: talib.CCI(b.high, b.low, b.close, 14)),
    Case("WILLR", lambda d: xt.WILLR(d, 14), lambda b: talib.WILLR(b.high, b.low, b.close, 14)),
    Case("MFI", lambda d: xt.MFI(d, 14), lambda b: talib.MFI(b.high, b.low, b.close, b.volume, 14)),
    Case("OBV", xt.OBV, lambda b: talib.OBV(b.close, b.volume)),
    Case("AD", xt.AD, lambda b: talib.AD(b.high, b.low, b.close, b.volume)),
    Case(
        "ADOSC",
        lambda d: xt.ADOSC(d, 3, 10),
        lambda b: talib.ADOSC(b.high, b.low, b.close, b.volume, 3, 10),
    ),
    Case("ULTOSC", xt.ULTOSC, lambda b: talib.ULTOSC(b.high, b.low, b.close)),
    Case("BOP", xt.BOP, lambda b: talib.BOP(b.open, b.high, b.low, b.close)),
    Case("SAR", xt.SAR, lambda b: talib.SAR(b.high, b.low)),
    Case("TYPPRICE", xt.TYPPRICE, lambda b: talib.TYPPRICE(b.high, b.low, b.close)),
    Case("WCLPRICE", xt.WCLPRICE, lambda b: talib.WCLPRICE(b.high, b.low, b.close)),
    Case("AROONOSC", lambda d: xt.AROONOSC(d, 14), lambda b: talib.AROONOSC(b.high, b.low, 14)),
    Case("CDLDOJI", xt.CDLDOJI, lambda b: talib.CDLDOJI(b.open, b.high, b.low, b.close) / 100),
    Case(
        "CDLENGULFING",
        xt.CDLENGULFING,
        lambda b: talib.CDLENGULFING(b.open, b.high, b.low, b.close) / 100,
    ),
    Case(
        "CDLHAMMER", xt.CDLHAMMER, lambda b: talib.CDLHAMMER(b.open, b.high, b.low, b.close) / 100
    ),
    Case(
        "MACD",
        lambda d: xt.MACD(d, 12, 26, 9),
        lambda b: talib.MACD(b.close, 12, 26, 9),
        ("macd", "signal", "hist"),
        "macd",
        True,
    ),
    Case(
        "BBANDS",
        lambda d: xt.BBANDS(d, 10, 2, 2, 0),
        lambda b: talib.BBANDS(b.close, 10, 2, 2, MA_Type.SMA),
        ("upper", "middle", "lower"),
        "bbands",
        True,
    ),
    Case(
        "STOCH",
        lambda d: xt.STOCH(d, 5, 3, 0, 3, 0),
        lambda b: talib.STOCH(b.high, b.low, b.close, 5, 3, MA_Type.SMA, 3, MA_Type.SMA),
        ("slowk", "slowd"),
        "field",
    ),
    Case(
        "AROON",
        lambda d: xt.AROON(d, 14),
        lambda b: talib.AROON(b.high, b.low, 14),
        ("down", "up"),
        "field",
    ),
    Case(
        "HT_SINE",
        xt.HT_SINE,
        lambda b: talib.HT_SINE(b.close),
        ("sine", "leadsine"),
        "ht_sine",
        True,
    ),
]


EXTRA_CASES = [
    Case("HT_DCPERIOD", xt.HT_DCPERIOD, lambda b: talib.HT_DCPERIOD(b.close), on_close_only=True),
    Case("HT_DCPHASE", xt.HT_DCPHASE, lambda b: talib.HT_DCPHASE(b.close), on_close_only=True),
    Case("HT_TRENDMODE", xt.HT_TRENDMODE, lambda b: np.asarray(talib.HT_TRENDMODE(b.close), dtype=np.float64), on_close_only=True),
    Case("LINEARREG_ANGLE", lambda d: xt.LINEARREG_ANGLE(d, 14), lambda b: talib.LINEARREG_ANGLE(b.close, 14), on_close_only=True),
    Case("LINEARREG_INTERCEPT", lambda d: xt.LINEARREG_INTERCEPT(d, 14), lambda b: talib.LINEARREG_INTERCEPT(b.close, 14), on_close_only=True),
    Case("LINEARREG_SLOPE", lambda d: xt.LINEARREG_SLOPE(d, 14), lambda b: talib.LINEARREG_SLOPE(b.close, 14), on_close_only=True),
    Case("ROCR", lambda d: xt.ROCR(d, 14), lambda b: talib.ROCR(b.close, 14), on_close_only=True),
    Case("ROCR100", lambda d: xt.ROCR100(d, 14), lambda b: talib.ROCR100(b.close, 14), on_close_only=True),
    Case("ROCP", lambda d: xt.ROCP(d, 14), lambda b: talib.ROCP(b.close, 14), on_close_only=True),
    Case("MINUS_DI", lambda d: xt.MINUS_DI(d, 14), lambda b: talib.MINUS_DI(b.high, b.low, b.close, 14)),
    Case("PLUS_DM", lambda d: xt.PLUS_DM(d, 14), lambda b: talib.PLUS_DM(b.high, b.low, 14)),
    Case(
        "HT_PHASOR", xt.HT_PHASOR, lambda b: talib.HT_PHASOR(b.close),
        ("inphase", "quadrature"), "ht_phasor", True,
    ),
    Case(
        "MAMA", lambda d: xt.MAMA(d, 0.5, 0.05), lambda b: talib.MAMA(b.close, 0.5, 0.05),
        ("mama", "fama"), "mama", True,
    ),
    Case(
        "MACDFIX", lambda d: xt.MACDFIX(d, 9), lambda b: talib.MACDFIX(b.close, 9),
        ("macd", "signal", "hist"), "macd", True,
    ),
    Case(
        "MACDEXT", lambda d: xt.MACDEXT(d, 12, 1, 26, 1, 9, 1),
        lambda b: talib.MACDEXT(b.close, 12, MA_Type.EMA, 26, MA_Type.EMA, 9, MA_Type.EMA),
        ("macd", "signal", "hist"), "macd", True,
    ),
]  # fmt: skip

CDL_NAMES = sorted(name for name in dir(xt) if name.startswith("CDL") and name.isupper())


@pytest.mark.parametrize(
    "case", [*CASES, *EXTRA_CASES], ids=[c.name for c in [*CASES, *EXTRA_CASES]]
)
def test_indicator_matches_talib_per_asset(case: Case, data: xr.DataArray) -> None:
    source = data.sel(field="close") if case.on_close_only else data
    result = case.call(source)
    expected_dims = {"time", "asset"}
    if case.outputs:
        expected_dims.add(case.output_dim)
    assert set(result.dims) == expected_dims
    assert_array_equal(result.coords["time"].values, data.coords["time"].values)
    assert list(result.coords["asset"].values) == list(data.coords["asset"].values)
    if case.outputs:
        assert list(result.coords[case.output_dim].values) == list(case.outputs)

    for asset in range(N_ASSETS):
        bars, start = bars_of(data, asset)
        raw = case.reference(bars)
        parts = raw if isinstance(raw, tuple) else (raw,)
        for index, part in enumerate(parts):
            piece = result
            if case.outputs:
                piece = piece.sel({case.output_dim: case.outputs[index]})
            actual = piece.sel(asset=data.coords["asset"].values[asset]).values
            if actual.ndim != 1:
                actual = actual.T[0]
            assert np.isnan(actual[:start]).all()
            assert_allclose(actual[start:], part, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_cross_series_indicators_match_talib(data: xr.DataArray) -> None:
    close = data.sel(field="close")
    index = close.sel(asset=data.coords["asset"].values[0])
    correl = xt.CORREL(close, index, 20)
    beta = xt.BETA(close, index, 10)
    assert set(correl.dims) == {"time", "first"}
    reference = index.values
    for asset in range(N_ASSETS - 1):
        own = close.sel(asset=data.coords["asset"].values[asset]).values
        actual_correl = correl.sel(first=data.coords["asset"].values[asset]).values
        actual_beta = beta.sel(first=data.coords["asset"].values[asset]).values
        assert_allclose(actual_correl, talib.CORREL(own, reference, 20), rtol=1e-9, equal_nan=True)
        assert_allclose(actual_beta, talib.BETA(own, reference, 10), rtol=1e-9, equal_nan=True)
    late = correl.sel(first=data.coords["asset"].values[N_ASSETS - 1]).values
    assert np.isnan(late[: GAP + 19]).all()
    assert np.isfinite(late[GAP + 19 :]).all()
    assert abs(correl.sel(first=data.coords["asset"].values[0]).values[-1] - 1.0) < 1e-12


def test_default_parameters_and_wrapper_helpers(data: xr.DataArray) -> None:
    close = data.sel(field="close")
    assert_allclose(xt.SMA(close).values, xt.SMA(close, 14).values, equal_nan=True)
    # the APO docstring says fastperiod=12 but the signature default is 14
    assert_allclose(
        xt.APO(close).values, xt.APO(close, 14, 26, 0).values, equal_nan=True
    )
    assert xt.MaType.EMA == 1
    assert xt.MaType.T3 == 8
    weighted = xt.MA(close, 10, xt.MaType.WMA)
    assert_allclose(
        weighted.values, xt.WMA(close, 10).values, rtol=1e-12, equal_nan=True
    )


def test_single_series_call_preserves_extra_dimensions() -> None:
    data = make_data(n_assets=2, n_days=40, name="stocks")
    close = data.sel(field="close")
    transposed = close.transpose("asset", "time")
    straight = xt.SMA(close, 5)
    flipped = xt.SMA(transposed, 5)
    assert flipped.dims == ("asset", "time")
    assert_allclose(
        flipped.transpose("time", "asset").values, straight.values, equal_nan=True
    )
    assert np.isnan(straight.values[:4]).all()
    assert_allclose(straight.values[4], close.values[:5].mean(axis=0), rtol=1e-12)


@pytest.mark.parametrize("name", CDL_NAMES)
def test_every_candlestick_pattern_matches_talib(name: str, data: xr.DataArray) -> None:
    result = getattr(xt, name)(data)
    reference = getattr(talib, name)
    assert set(result.dims) == {"time", "asset"}
    for asset in range(N_ASSETS):
        bars, start = bars_of(data, asset)
        actual = result.sel(asset=data.coords["asset"].values[asset]).values
        expected = reference(bars.open, bars.high, bars.low, bars.close) / 100
        assert np.isnan(actual[:start]).all()
        assert_allclose(actual[start:], expected, atol=1e-12)
