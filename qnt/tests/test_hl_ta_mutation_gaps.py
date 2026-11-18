from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import talib
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from talib._ta_lib import MA_Type

import qnt.ta as qnta
import qnt.ta.ndadapter as nda
from qnt.ta.dms import adx, di
from qnt.tests.hl_support import (
    Floats,
    business_days,
    float_series,
    make_data,
    time_index,
    values_of,
)

N_DAYS = 320
NAN = np.nan
RTOL = 1e-9
# a dimension named "time" that is not the interned literal, to tell `!=` from `is not`
TIME_DIM = "".join(["ti", "me"])


@dataclass(frozen=True)
class Panel:
    """OHLCV matrices with shape (asset, time)."""

    high: Floats
    low: Floats
    close: Floats
    volume: Floats


@pytest.fixture(scope="module")
def panel() -> Panel:
    data = make_data(n_assets=3, n_days=N_DAYS, name="stocks", seed=41)

    def matrix(field: str) -> Floats:
        return data.sel(field=field).values.T.copy()

    return Panel(matrix("high"), matrix("low"), matrix("close"), matrix("vol"))


@pytest.fixture(scope="module")
def days() -> pd.DatetimeIndex:
    return business_days(N_DAYS)


def returns(prices: Floats) -> Floats:
    return np.concatenate([[NAN], prices[1:] / prices[:-1] - 1])


def valid_windows(
    valid: npt.NDArray[np.bool_], periods: int, first: int
) -> list[npt.NDArray[np.intp]]:
    """Index windows of `periods` consecutive valid positions, from position `first` on."""
    positions = np.flatnonzero(valid)
    return [positions[p - periods + 1 : p + 1] for p in range(first, len(positions))]


def rolling_covariance(x: Floats, y: Floats, periods: int) -> Floats:
    """Population covariance of the last `periods` pairs where both inputs are valid.

    The first value appears once periods + 1 valid pairs have been seen (current convention).
    """
    result = np.full(x.shape, NAN)
    pairs = np.isfinite(x) & np.isfinite(y)
    for window in valid_windows(pairs, periods, periods):
        wx, wy = x[window], y[window]
        result[window[-1]] = ((wx - wx.mean()) * (wy - wy.mean())).mean()
    return result


def rolling_std(series: Floats, periods: int) -> Floats:
    result = np.full(series.shape, NAN)
    for window in valid_windows(np.isfinite(series), periods, periods):
        result[window[-1]] = series[window].std()
    return result


def rolling_vwma(price: Floats, volume: Floats, periods: int) -> Floats:
    result = np.full(price.shape, NAN)
    pairs = np.isfinite(price) & np.isfinite(volume)
    for window in valid_windows(pairs, periods, periods - 1):
        result[window[-1]] = (price[window] * volume[window]).sum() / volume[window].sum()
    return result


def stochastic_k_reference(high: Floats, low: Floats, close: Floats, periods: int) -> Floats:
    """%K over the last `periods` bars that have all three prices; undefined for a flat range."""
    result = np.full(close.shape, NAN)
    bars = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    for window in valid_windows(bars, periods, periods - 1):
        highest, lowest = high[window].max(), low[window].min()
        if highest > lowest:
            result[window[-1]] = 100 * (close[window[-1]] - lowest) / (highest - lowest)
    return result


def pivot_reference(series: Floats, periods: int, top: bool) -> Floats:
    """1 when the value `periods` bars back is a strict extreme of the 2 * periods + 1 window."""
    size = 2 * periods + 1
    result = np.full(series.shape, NAN)
    for i in range(size - 1, len(series)):
        window = series[i - size + 1 : i + 1]
        others = np.delete(window, periods)
        extreme = (window[periods] > others).all() if top else (window[periods] < others).all()
        result[i] = float(extreme)
    return result


def wide_bars(n: int, seed: int) -> tuple[Floats, Floats, Floats, Floats]:
    """Bars with high >= 4 * low so that `a - b` and `a % b` differ for any price pair."""
    rng = np.random.default_rng(seed)
    low = rng.uniform(1.0, 10.0, n)
    high = low + rng.uniform(30.0, 90.0, n)
    close = low + rng.uniform(0.2, 0.8, n) * (high - low)
    volume = rng.uniform(1e3, 1e4, n)
    return high, low, close, volume


# --- default parameter values -------------------------------------------------------------


def test_variance_std_covariance_default_to_one_point_windows(panel: Panel) -> None:
    x, y = panel.close[0], panel.close[1]
    expected = np.zeros(N_DAYS)
    expected[0] = NAN
    assert_array_equal(qnta.variance(x), expected)
    assert_array_equal(qnta.std(x), expected)
    assert_array_equal(qnta.covariance(x, y), expected)
    assert_array_equal(qnta.variance(x), qnta.variance(x, 1))
    assert_array_equal(qnta.std(x), qnta.std(x, 1))
    assert_array_equal(qnta.covariance(x, y), qnta.covariance(x, y, 1))


def test_beta_and_correlation_default_to_a_252_point_window(panel: Panel) -> None:
    x, y = panel.close[0], panel.close[1]
    rx, ry = returns(x), returns(y)
    window_x, window_y = rx[-252:], ry[-252:]
    for default, explicit in (
        (qnta.beta(x, y), qnta.beta(x, y, 252)),
        (qnta.correlation(x, y), qnta.correlation(x, y, 252)),
    ):
        assert_array_equal(default, explicit)
        assert int(np.argmax(np.isfinite(default))) == 253
    assert_allclose(qnta.correlation(x, y)[-1], np.corrcoef(window_x, window_y)[0, 1], rtol=RTOL)
    assert_allclose(
        qnta.beta(x, y)[-1], np.cov(window_x, window_y, ddof=0)[0, 1] / np.var(window_y), rtol=RTOL
    )


def test_macd_defaults_are_12_26_9(panel: Panel) -> None:
    close = panel.close[0]
    for default, explicit in zip(qnta.macd(close), qnta.macd(close, 12, 26, 9), strict=True):
        assert_array_equal(default, explicit)
    line, signal, _ = qnta.macd(close)
    assert int(np.argmax(np.isfinite(line))) == 25
    assert int(np.argmax(np.isfinite(signal))) == 33


def test_rsi_default_is_a_14_period_wilder_average(panel: Panel) -> None:
    close = panel.close[0]
    assert_array_equal(qnta.rsi(close), qnta.rsi(close, 14))
    assert_allclose(qnta.rsi(close), talib.RSI(close, 14), rtol=RTOL, equal_nan=True)


def test_stochastic_defaults_are_5_and_3_and_3(panel: Panel) -> None:
    high, low, close = panel.high[0], panel.low[0], panel.close[0]
    fast_k, _ = talib.STOCHF(high, low, close, 5, 1, MA_Type.SMA)
    _, fast_d = talib.STOCHF(high, low, close, 5, 3, MA_Type.SMA)
    slow_k, _ = talib.STOCH(high, low, close, 5, 3, MA_Type.SMA, 1, MA_Type.SMA)
    _, slow_d = talib.STOCH(high, low, close, 5, 3, MA_Type.SMA, 3, MA_Type.SMA)
    k, d = qnta.stochastic(high, low, close)
    assert_allclose(k, fast_k, rtol=RTOL, equal_nan=True)
    assert_allclose(d, fast_d, rtol=RTOL, equal_nan=True)
    for default, explicit in zip(
        qnta.stochastic(high, low, close), qnta.stochastic(high, low, close, 5, 3), strict=True
    ):
        assert_array_equal(default, explicit)
    sk, sd = qnta.slow_stochastic(high, low, close)
    assert_allclose(sk, slow_k, rtol=RTOL, equal_nan=True)
    assert_allclose(sd, slow_d, rtol=RTOL, equal_nan=True)
    for default, explicit in zip(
        qnta.slow_stochastic(high, low, close),
        qnta.slow_stochastic(high, low, close, 5, 3, 3),
        strict=True,
    ):
        assert_array_equal(default, explicit)


@pytest.mark.xfail(strict=True, reason="pivot points have NaN in other positions")
def test_pivot_points_default_to_five_bars_each_side() -> None:
    rng = np.random.default_rng(8)
    series = rng.integers(0, 6, 120).astype(np.float64)
    top = qnta.top_pivot_points(series)
    bottom = qnta.bottom_pivot_points(series)
    assert_array_equal(top, qnta.top_pivot_points(series, 5))
    assert_array_equal(bottom, qnta.bottom_pivot_points(series, 5))
    assert_array_equal(top, pivot_reference(series, 5, top=True))
    assert_array_equal(bottom, pivot_reference(series, 5, top=False))
    assert np.nansum(top) > 0
    assert np.nansum(bottom) > 0


# --- directional movement system ----------------------------------------------------------


def quarter_ticks(values: Floats) -> Floats:
    """Quantize to 0.25 so that differences of prices are exact in binary floating point."""
    return np.round(values * 4) / 4


def dms_frames(panel: Panel) -> tuple[Floats, Floats, Floats]:
    """(time, asset) matrices with exact ties between the up-move and the down-move."""
    close = quarter_ticks(panel.close.T[:150])
    high = np.maximum(quarter_ticks(panel.high.T[:150]), close)
    low = np.minimum(quarter_ticks(panel.low.T[:150]), close)
    for i in (20, 41, 63, 88, 110, 133):
        high[i] = high[i - 1] + 0.5
        low[i] = low[i - 1] - 0.5
        close[i] = np.clip(close[i], low[i], high[i])
    return high.copy(), low.copy(), close.copy()


def wilder(values: Floats, periods: int) -> Floats:
    """Wilder smoothing seeded with the mean of the first `periods` valid points."""
    result = np.full(values.shape, NAN)
    positions = np.flatnonzero(np.isfinite(values))
    if len(positions) < periods:
        return result
    level = float(values[positions[:periods]].mean())
    result[positions[periods - 1]] = level
    for p in positions[periods:]:
        level += (values[p] - level) / periods
        result[p] = level
    return result


def dms_reference(
    high: Floats, low: Floats, close: Floats, di_periods: int, adx_periods: int, adxr_periods: int
) -> tuple[Floats, Floats, Floats, Floats]:
    """Directional movement written from the textbook rules (Wilder), one asset."""
    n = len(high)
    plus_move = np.full(n, NAN)
    minus_move = np.full(n, NAN)
    plus_move[1:] = high[1:] - high[:-1]
    minus_move[1:] = low[:-1] - low[1:]
    plus_dm = np.where(
        np.isnan(plus_move), NAN, np.where((plus_move > minus_move) & (plus_move > 0), plus_move, 0)
    )
    minus_dm = np.where(
        np.isnan(minus_move),
        NAN,
        np.where((minus_move > plus_move) & (minus_move > 0), minus_move, 0),
    )
    true_range = np.full(n, NAN)
    true_range[1:] = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    average_range = wilder(true_range, di_periods)
    plus_di = 100 * wilder(plus_dm, di_periods) / average_range
    minus_di = 100 * wilder(minus_dm, di_periods) / average_range
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder(dx, adx_periods)
    adxr = np.full(n, NAN)
    adxr[adxr_periods:] = (adx[adxr_periods:] + adx[:-adxr_periods]) / 2
    return plus_di, minus_di, adx, adxr


def frame_of(matrix: Floats, days: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=days[: len(matrix)], columns=["a", "b", "c"])


@pytest.mark.parametrize(("di_ma", "adx_ma", "adxr_periods"), [(10, 8, 5), (14, 20, 7)])
def test_dms_matches_the_textbook_formulas(
    panel: Panel, days: pd.DatetimeIndex, di_ma: int, adx_ma: int, adxr_periods: int
) -> None:
    high, low, close = dms_frames(panel)
    parts = qnta.dms(
        frame_of(high, days),
        frame_of(low, days),
        frame_of(close, days),
        di_ma,
        adx_ma,
        adxr_periods,
    )
    for column in range(3):
        expected = dms_reference(
            high[:, column], low[:, column], close[:, column], di_ma, adx_ma, adxr_periods
        )
        for actual, wanted in zip(parts, expected, strict=True):
            assert_allclose(
                values_of(actual)[:, column], wanted, rtol=RTOL, atol=RTOL, equal_nan=True
            )
        assert all(np.isfinite(wanted[-1]) for wanted in expected)


def test_dms_defaults_are_14_20_7(panel: Panel, days: pd.DatetimeIndex) -> None:
    high, low, close = dms_frames(panel)
    frames = (frame_of(high, days), frame_of(low, days), frame_of(close, days))
    for default, explicit in zip(qnta.dms(*frames), qnta.dms(*frames, 14, 20, 7), strict=True):
        assert_array_equal(values_of(default), values_of(explicit))
    expected = dms_reference(high[:, 0], low[:, 0], close[:, 0], 14, 20, 7)
    for actual, wanted in zip(qnta.dms(*frames), expected, strict=True):
        assert_allclose(values_of(actual)[:, 0], wanted, rtol=RTOL, atol=RTOL, equal_nan=True)


def test_dms_has_the_talib_lookbacks_and_converges_to_talib_values(
    panel: Panel, days: pd.DatetimeIndex
) -> None:
    """TA-Lib seeds its Wilder sums slightly differently, so only the late values must agree."""
    high, low, close = dms_frames(panel)
    parts = qnta.dms(frame_of(high, days), frame_of(low, days), frame_of(close, days), 14, 14, 13)
    h, lo, c = high[:, 0], low[:, 0], close[:, 0]
    references = (
        talib.PLUS_DI(h, lo, c, 14),
        talib.MINUS_DI(h, lo, c, 14),
        talib.ADX(h, lo, c, 14),
        talib.ADXR(h, lo, c, 14),
    )
    for part, reference in zip(parts, references, strict=True):
        actual = values_of(part)[:, 0]
        assert_array_equal(np.isnan(actual), np.isnan(reference))
        assert_allclose(actual[-60:], reference[-60:], rtol=0.0, atol=0.15)


def test_dms_series_and_dataarray_agree_with_dataframe(
    panel: Panel, days: pd.DatetimeIndex
) -> None:
    high, low, close = dms_frames(panel)
    frames = (frame_of(high, days), frame_of(low, days), frame_of(close, days))
    from_frames = qnta.dms(*frames, 10, 8, 5)

    def series_of(matrix: Floats) -> "pd.Series[float]":
        return float_series(matrix[:, 1], days[: len(matrix)])

    from_series = qnta.dms(series_of(high), series_of(low), series_of(close), 10, 8, 5)

    def array_of(matrix: Floats) -> xr.DataArray:
        return xr.DataArray(
            matrix,
            dims=["time", "asset"],
            coords={"time": days[: len(matrix)], "asset": ["a", "b", "c"]},
        )

    from_arrays = qnta.dms(array_of(high), array_of(low), array_of(close), 10, 8, 5)
    for frame_part, series_part, array_part in zip(
        from_frames, from_series, from_arrays, strict=True
    ):
        assert_allclose(values_of(series_part), values_of(frame_part)[:, 1], equal_nan=True)
        assert_allclose(array_part.values, values_of(frame_part), equal_nan=True)


# --- boundary and tie cases ---------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="pivot points have NaN in other positions")
def test_pivot_points_require_a_strict_extreme() -> None:
    plateau = np.array([0, 2, 5, 5, 1, 0, 0], dtype=np.float64)
    assert_array_equal(qnta.top_pivot_points(plateau, 2)[4:], [0, 0, 0])
    valley = np.array([5, 3, 1, 1, 4, 5, 2, 5, 5, 4], dtype=np.float64)
    assert_array_equal(qnta.bottom_pivot_points(valley, 2)[4:], [0, 0, 0, 0, 1, 0])
    rng = np.random.default_rng(12)
    series = rng.integers(0, 4, 90).astype(np.float64)
    for periods in (2, 3):
        top = qnta.top_pivot_points(series, periods)
        bottom = qnta.bottom_pivot_points(series, periods)
        assert_array_equal(top, pivot_reference(series, periods, top=True))
        assert_array_equal(bottom, pivot_reference(series, periods, top=False))
        assert np.nansum(top) > 0
        assert np.nansum(bottom) > 0


def test_stochastic_k_is_undefined_for_a_flat_window() -> None:
    high, low, close, _ = wide_bars(40, seed=5)
    high[10:18] = 50.0
    low[10:18] = 50.0
    close[10:18] = 50.0
    result = qnta.stochastic_k(high, low, close, 5)
    expected = stochastic_k_reference(high, low, close, 5)
    assert np.isnan(expected[14:18]).all()
    assert_allclose(result, expected, rtol=RTOL, equal_nan=True)


def test_stochastic_k_and_chaikin_and_true_range_on_wide_price_ranges() -> None:
    high, low, close, volume = wide_bars(60, seed=6)
    assert_allclose(
        qnta.stochastic_k(high, low, close, 7),
        stochastic_k_reference(high, low, close, 7),
        rtol=RTOL,
        equal_nan=True,
    )
    money_flow = volume * ((close - low) - (high - close)) / (high - low)
    assert_allclose(qnta.chaikin_adl(high, low, close, volume), np.cumsum(money_flow), rtol=RTOL)
    expected_range = np.full(60, NAN)
    expected_range[1:] = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    assert_allclose(qnta.tr(high, low, close), expected_range, rtol=RTOL, equal_nan=True)


@pytest.mark.parametrize("missing", ["high", "low", "close"])
def test_stochastic_k_skips_a_bar_when_only_one_price_is_missing(missing: str) -> None:
    high, low, close, _ = wide_bars(40, seed=7)
    target = {"high": high, "low": low, "close": close}[missing]
    target[[8, 21]] = NAN
    result = qnta.stochastic_k(high, low, close, 5)
    assert np.isnan(result[[8, 21]]).all()
    assert_allclose(result, stochastic_k_reference(high, low, close, 5), rtol=RTOL, equal_nan=True)
    assert np.isfinite(result[22:]).all()


@pytest.mark.parametrize("missing", ["price", "volume"])
def test_vwma_skips_a_bar_when_only_one_input_is_missing(missing: str) -> None:
    price, _, _, volume = wide_bars(40, seed=8)
    target = price if missing == "price" else volume
    target[[9, 20]] = NAN
    result = qnta.vwma(price, volume, 4)
    assert np.isnan(result[[9, 20]]).all()
    assert_allclose(result, rolling_vwma(price, volume, 4), rtol=RTOL, equal_nan=True)
    assert np.isfinite(result[21:]).all()


@pytest.mark.parametrize("missing", ["x", "y"])
def test_covariance_skips_a_pair_when_only_one_input_is_missing(missing: str) -> None:
    rng = np.random.default_rng(14)
    x = rng.normal(0.0, 1.0, 40)
    y = rng.normal(0.0, 1.0, 40)
    (x if missing == "x" else y)[[7, 23]] = NAN
    result = qnta.covariance(x, y, 5)
    assert np.isnan(result[[7, 23]]).all()
    assert_allclose(result, rolling_covariance(x, y, 5), rtol=RTOL, equal_nan=True)
    assert np.isfinite(result[24:]).all()


@pytest.mark.parametrize("periods", [3, 6])
def test_std_and_covariance_emit_nothing_before_the_window_is_old_enough(periods: int) -> None:
    rng = np.random.default_rng(15)
    x = rng.normal(0.0, 1.0, 30)
    y = rng.normal(0.0, 1.0, 30)
    std = qnta.std(x, periods)
    covariance = qnta.covariance(x, y, periods)
    assert np.isnan(std[:periods]).all()
    assert np.isnan(covariance[:periods]).all()
    assert_allclose(std, rolling_std(x, periods), rtol=RTOL, equal_nan=True)
    assert_allclose(covariance, rolling_covariance(x, y, periods), rtol=RTOL, equal_nan=True)
    assert_allclose(std[periods], np.std(x[1 : periods + 1]), rtol=RTOL)


def test_shift_by_zero_returns_the_series_unchanged() -> None:
    series = np.array([1.0, NAN, 3.0, 4.0, NAN, 6.0])
    assert_array_equal(qnta.shift(series, 0), series)
    assert_array_equal(qnta.change(series, 0), np.where(np.isnan(series), NAN, 0.0))
    wide = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert_array_equal(qnta.shift(wide, 3), [NAN, NAN, NAN, 1, 2, 3, 4])
    assert_array_equal(qnta.shift(wide, -3), [4, 5, 6, 7, NAN, NAN, NAN])


# --- ndadapter: metadata and shape come from the first argument ---------------------------


def covariance_rows(first: Floats, second: Floats, periods: int) -> Floats:
    return np.array(
        [rolling_covariance(a, b, periods) for a, b in zip(first, second, strict=False)]
    )


def test_numpy_adapter_uses_the_first_argument_shape_and_the_last_axis_as_time() -> None:
    rng = np.random.default_rng(21)
    first = rng.normal(0.0, 1.0, (2, 30))
    second = rng.normal(0.0, 1.0, (3, 30))
    result = qnta.covariance(first, second, 5)
    assert result.shape == (2, 30)
    assert_allclose(result, covariance_rows(first, second, 5), rtol=RTOL, equal_nan=True)

    cube = rng.normal(50.0, 3.0, (2, 3, 40))
    smoothed = qnta.sma(cube, 5)
    assert smoothed.shape == (2, 3, 40)
    for i in range(2):
        for j in range(3):
            assert_allclose(smoothed[i, j], talib.SMA(cube[i, j], 5), rtol=RTOL, equal_nan=True)


def test_dataframe_and_series_results_take_labels_from_the_first_argument() -> None:
    rng = np.random.default_rng(22)
    index_a = pd.date_range("2020-01-01", periods=30, name="time")
    index_b = index_a + pd.Timedelta(days=100)
    x = pd.DataFrame(rng.normal(0.0, 1.0, (30, 2)), index=index_a, columns=["a", "b"])
    y = pd.DataFrame(rng.normal(0.0, 1.0, (30, 2)), index=index_b, columns=["c", "d"])
    frame = qnta.covariance(x, y, 5)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["a", "b"]
    assert frame.index.tolist() == index_a.tolist()
    assert_allclose(
        values_of(frame).T,
        covariance_rows(values_of(x).T, values_of(y).T, 5),
        rtol=RTOL,
        equal_nan=True,
    )

    sx = float_series(values_of(x)[:, 0], index_a)
    sy = float_series(values_of(y)[:, 0], index_b)
    series = qnta.covariance(sx, sy, 5)
    assert isinstance(series, pd.Series)
    assert series.to_frame().index.tolist() == index_a.tolist()
    assert_allclose(
        values_of(series),
        rolling_covariance(values_of(sx), values_of(sy), 5),
        rtol=RTOL,
        equal_nan=True,
    )


def test_dataarray_result_takes_dims_and_coords_from_the_first_argument() -> None:
    rng = np.random.default_rng(23)
    times_a = pd.date_range("2020-01-01", periods=30, name="time")
    times_b = times_a + pd.Timedelta(days=100)
    xv = rng.normal(0.0, 1.0, (2, 30))
    yv = rng.normal(0.0, 1.0, (2, 30))
    x = xr.DataArray(xv, dims=["asset", "time"], coords={"asset": ["a", "b"], "time": times_a})
    y = xr.DataArray(yv.T, dims=["time", "asset"], coords={"asset": ["c", "d"], "time": times_b})
    result = qnta.covariance(x, y, 5)
    assert result.dims == ("asset", "time")
    assert list(result.coords["asset"].values) == ["a", "b"]
    assert time_index(result).tolist() == times_a.tolist()
    assert_allclose(result.values, covariance_rows(xv, yv, 5), rtol=RTOL, equal_nan=True)


def test_dataarray_with_a_dimension_sorted_after_time_keeps_it() -> None:
    rng = np.random.default_rng(24)
    times = pd.date_range("2020-01-01", periods=30)
    coords = {"zone": ["u", "v"], TIME_DIM: times}
    xv = rng.normal(0.0, 1.0, (2, 30))
    yv = rng.normal(0.0, 1.0, (2, 30))
    x = xr.DataArray(xv, dims=["zone", TIME_DIM], coords=coords)
    y = xr.DataArray(yv, dims=["zone", TIME_DIM], coords=coords)
    result = qnta.covariance(x, y, 5)
    assert result.dims == ("zone", "time")
    assert_allclose(result.values, covariance_rows(xv, yv, 5), rtol=RTOL, equal_nan=True)

    transposed = qnta.covariance(x.transpose(TIME_DIM, "zone"), y.transpose(TIME_DIM, "zone"), 5)
    assert transposed.dims == ("time", "zone")
    assert_allclose(transposed.transpose("zone", "time").values, result.values)


def column_total(first: Floats, second: Floats) -> Floats:
    return (first + second).sum(axis=0)


def first_rows_only(first: Floats, second: Floats) -> Floats:
    assert second.shape == first.shape
    return first.sum(axis=0)


def test_reducing_adapters_take_labels_from_the_first_argument() -> None:
    rng = np.random.default_rng(25)
    index_a = pd.date_range("2020-01-01", periods=12, name="time")
    index_b = index_a + pd.Timedelta(days=100)
    xv = rng.normal(0.0, 1.0, (12, 3))
    yv = rng.normal(0.0, 1.0, (12, 3))
    x = pd.DataFrame(xv, index=index_a, columns=["a", "b", "c"])
    y = pd.DataFrame(yv, index=index_b, columns=["d", "e", "f"])
    frame_result = nda.nd_to_1d_universal_adapter(column_total, (x, y), ())
    assert isinstance(frame_result, pd.Series)
    assert frame_result.to_frame().index.tolist() == index_a.tolist()
    assert_allclose(values_of(frame_result), (xv + yv).sum(axis=1))

    coords_x = {"zone": ["u", "v", "w"], TIME_DIM: index_a}
    coords_y = {"zone": ["x", "y", "z"], TIME_DIM: index_b}
    ax = xr.DataArray(xv.T, dims=["zone", TIME_DIM], coords=coords_x)
    ay = xr.DataArray(yv.T, dims=["zone", TIME_DIM], coords=coords_y)
    array_result = nda.nd_to_1d_universal_adapter(column_total, (ax, ay), ())
    assert isinstance(array_result, xr.DataArray)
    assert array_result.dims == ("time",)
    assert time_index(array_result).tolist() == index_a.tolist()
    assert_allclose(array_result.values, (xv + yv).sum(axis=1))


@pytest.mark.xfail(strict=True, reason="TypeError: nd_to_1d_np_adapter() is called without np_function")
def test_reducing_adapter_dispatches_on_the_type_of_the_first_argument() -> None:
    rng = np.random.default_rng(26)
    first = rng.normal(0.0, 1.0, (3, 8))
    second = pd.DataFrame(first)
    result = nda.nd_to_1d_universal_adapter(first_rows_only, (first, second), ())
    assert isinstance(result, np.ndarray)
    assert_allclose(result, first.sum(axis=0))
    with pytest.raises(AttributeError):
        nda.nd_to_1d_universal_adapter(first_rows_only, (second, first), ())


def test_advance_decline_line_on_dataarray_with_a_dimension_sorted_after_time() -> None:
    prices = np.array([[1.0, 2.0, 3.0, 2.0], [1.0, 1.0, 2.0, 3.0]])
    times = pd.date_range("2020-01-01", periods=4)
    wrapped = xr.DataArray(
        prices, dims=["zone", TIME_DIM], coords={"zone": ["u", "v"], TIME_DIM: times}
    )
    result = qnta.ad_line(wrapped)
    assert result.dims == ("time",)
    assert time_index(result).tolist() == times.tolist()
    assert_array_equal(result.values, [0, 1, 3, 3])


def test_dms_building_blocks_default_to_a_14_period_wilder_average(panel: Panel) -> None:
    plus_dm = np.abs(panel.close[0] - panel.close[1])
    minus_dm = np.abs(panel.close[0] - panel.close[2])
    average_range = panel.high[0] - panel.low[0]
    plus_di, minus_di = di(plus_dm, minus_dm, average_range)
    assert_allclose(plus_di, 100 * wilder(plus_dm, 14) / average_range, rtol=RTOL, equal_nan=True)
    assert_allclose(minus_di, 100 * wilder(minus_dm, 14) / average_range, rtol=RTOL, equal_nan=True)
    assert_allclose(adx(plus_dm), wilder(plus_dm, 14), rtol=RTOL, equal_nan=True)
