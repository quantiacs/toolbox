import numpy as np
import pandas as pd
import pytest
import talib
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.ta as qnta
from qnt.tests.hl_support import Floats, float_series, make_data, time_index, values_of

N_DAYS = 150
NAN = np.nan


class Panel:
    """OHLCV matrices with shape (asset, time), the layout the numpy indicators expect."""

    def __init__(self) -> None:
        data = make_data(n_assets=3, n_days=N_DAYS, name="stocks", seed=29)
        self.data = data

        def matrix(field: str) -> Floats:
            return data.sel(field=field).values.T.copy()

        self.open = matrix("open")
        self.high = matrix("high")
        self.low = matrix("low")
        self.close = matrix("close")
        self.volume = matrix("vol")


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel()


def each_asset(matrix: Floats) -> list[Floats]:
    return list(matrix)


def assert_same_series(actual: Floats, expected: list[Floats], rtol: float = 1e-9) -> None:
    assert_allclose(actual, np.array(expected), rtol=rtol, atol=1e-9, equal_nan=True)


def test_moving_averages_and_oscillators_match_talib(panel: Panel) -> None:
    close, high, low, volume = panel.close, panel.high, panel.low, panel.volume
    assert_same_series(qnta.sma(close, 10), [talib.SMA(x, 10) for x in close])
    assert_same_series(qnta.ema(close, 10), [talib.EMA(x, 10) for x in close])
    assert_same_series(qnta.lwma(close, 10), [talib.WMA(x, 10) for x in close])
    assert_same_series(qnta.rsi(close, 14), [talib.RSI(x, 14) for x in close])
    assert_same_series(qnta.roc(close, 12), [talib.ROC(x, 12) for x in close])
    assert_same_series(qnta.trix(close, 8), [talib.TRIX(x, 8) for x in close])
    assert_same_series(
        qnta.tr(high, low, close),
        [talib.TRANGE(h, lo, c) for h, lo, c in zip(high, low, close, strict=True)],
    )
    assert_same_series(
        qnta.atr(high, low, close, 14),
        [talib.ATR(h, lo, c, 14) for h, lo, c in zip(high, low, close, strict=True)],
    )
    assert_same_series(
        qnta.stochastic_k(high, low, close, 14),
        [talib.STOCHF(h, lo, c, 14, 1)[0] for h, lo, c in zip(high, low, close, strict=True)],
    )
    assert_same_series(
        qnta.chaikin_adl(high, low, close, volume),
        [talib.AD(h, lo, c, v) for h, lo, c, v in zip(high, low, close, volume, strict=True)],
        rtol=1e-7,
    )
    assert_allclose(qnta.wilder_ma(close, 14), qnta.ema(close, 27, 14), equal_nan=True)
    assert_same_series(qnta.std(close, 10)[:, 10:], [talib.STDDEV(x, 10)[10:] for x in close])
    assert_same_series(qnta.variance(close, 10)[:, 10:], [talib.VAR(x, 10)[10:] for x in close])


def test_ndarray_dataframe_series_and_dataarray_give_identical_results(panel: Panel) -> None:
    matrix = panel.close
    expected = qnta.ema(matrix, 12)
    frame = pd.DataFrame(matrix.T, index=time_index(panel.data))
    from_frame = qnta.ema(frame, 12)
    assert isinstance(from_frame, pd.DataFrame)
    assert_allclose(values_of(from_frame).T, expected, equal_nan=True)
    assert list(from_frame.columns) == list(frame.columns)
    assert from_frame.index.tolist() == frame.index.tolist()

    first = float_series(matrix[0], time_index(panel.data))
    from_series = qnta.ema(first, 12)
    assert isinstance(from_series, pd.Series)
    assert_allclose(values_of(from_series), expected[0], equal_nan=True)

    xr_close = panel.data.sel(field="close")
    for arranged in (xr_close, xr_close.transpose("asset", "time")):
        result = qnta.ema(arranged, 12)
        assert result.dims == arranged.dims
        assert_allclose(result.transpose("asset", "time").values, expected, equal_nan=True)


def test_nan_gaps_are_skipped_instead_of_propagated() -> None:
    series = np.array([0, 1, 2, 3, 4, NAN, 5, NAN, 6, 7], dtype=np.float64)
    assert_array_equal(
        qnta.sma(series, 3), np.array([NAN, NAN, 1, 2, 3, NAN, 4, NAN, 5, 6]), strict=False
    )
    shifted = qnta.shift(np.array([1, 2, NAN, 4, 5, NAN, 7, 9, 0], dtype=np.float64), 2)
    assert_array_equal(shifted, np.array([NAN, NAN, NAN, 1, 2, NAN, 4, 5, 7]))
    backwards = qnta.shift(np.array([1, 2, 3, 4], dtype=np.float64), -1)
    assert_array_equal(backwards, np.array([2, 3, 4, NAN]))
    changes = qnta.change(np.array([1, 2, NAN, 4, 5], dtype=np.float64), 1)
    assert_array_equal(changes, np.array([NAN, 1, NAN, 2, 1]))


def reference_ema(values: Floats, periods: int) -> Floats:
    result = np.full(values.shape, np.nan)
    level = float(values[:periods].mean())
    result[periods - 1] = level
    k = 2.0 / (periods + 1)
    for i in range(periods, len(values)):
        level += (values[i] - level) * k
        result[i] = level
    return result


def test_macd_is_difference_of_emas_with_signal_line(panel: Panel) -> None:
    close = panel.close[0]
    line, signal, hist = qnta.macd(close, 12, 26, 9)
    fast = reference_ema(close, 12)
    slow = reference_ema(close, 26)
    expected_line = fast - slow
    assert_allclose(line, expected_line, rtol=1e-9, equal_nan=True)
    valid = np.flatnonzero(np.isfinite(expected_line))
    expected_signal = np.full(close.shape, np.nan)
    expected_signal[valid[8:]] = reference_ema(expected_line[valid], 9)[8:]
    assert_allclose(signal, expected_signal, rtol=1e-9, equal_nan=True)
    assert_allclose(hist, line - signal, equal_nan=True)
    assert np.argmax(np.isfinite(line)) == 25
    assert np.argmax(np.isfinite(signal)) == 33


def test_dema_and_tema_are_nested_emas_not_the_classic_definition(panel: Panel) -> None:
    close = panel.close
    # classic DEMA is 2*EMA-EMA(EMA); here it is plain EMA(EMA), so it differs from talib
    assert_allclose(qnta.dema(close, 10), qnta.ema(qnta.ema(close, 10), 10), equal_nan=True)
    assert_allclose(
        qnta.tema(close, 10), qnta.ema(qnta.ema(qnta.ema(close, 10), 10), 10), equal_nan=True
    )
    classic = np.array([talib.DEMA(x, 10) for x in close])
    assert np.nanmax(np.abs(qnta.dema(close, 10) - classic)) > 0.5


def test_obv_is_talib_obv_without_the_initial_volume(panel: Panel) -> None:
    mine = qnta.obv(panel.close, panel.volume)
    reference = np.array([talib.OBV(c, v) for c, v in zip(panel.close, panel.volume, strict=True)])
    assert np.isnan(mine[:, 0]).all()
    assert_allclose(mine[:, 1:], reference[:, 1:] - reference[:, :1], rtol=1e-9)


def rolling_windows(values: Floats, periods: int) -> list[Floats]:
    return [values[i - periods + 1 : i + 1] for i in range(periods - 1, len(values))]


def test_variance_covariance_beta_correlation_use_a_window_that_starts_late(panel: Panel) -> None:
    x, y = panel.close[0], panel.close[1]
    periods = 20
    rx = np.concatenate([[np.nan], x[1:] / x[:-1] - 1])
    ry = np.concatenate([[np.nan], y[1:] / y[:-1] - 1])

    variance = qnta.variance(x, 10)
    assert np.isnan(variance[:10]).all()  # a full 10-point window exists already at index 9
    assert_allclose(variance[10], np.var(x[1:11]), rtol=1e-10)

    corr = qnta.correlation(x, y, periods)
    first = int(np.argmax(np.isfinite(corr)))
    assert first == periods + 1
    for index in (first, first + 7, N_DAYS - 1):
        expected = np.corrcoef(
            rx[index - periods + 1 : index + 1], ry[index - periods + 1 : index + 1]
        )[0, 1]
        assert_allclose(corr[index], expected, rtol=1e-9)

    beta = qnta.beta(x, y, periods)
    index = N_DAYS - 1
    window_x, window_y = rx[index - periods + 1 : index + 1], ry[index - periods + 1 : index + 1]
    assert_allclose(
        beta[index], np.cov(window_x, window_y, ddof=0)[0, 1] / np.var(window_y), rtol=1e-9
    )
    assert_allclose(
        qnta.covariance(rx, ry, periods)[index], np.cov(window_x, window_y, ddof=0)[0, 1], rtol=1e-9
    )


def test_dms_on_dataframes_and_unsupported_inputs(panel: Panel) -> None:
    index = time_index(panel.data)

    def frame(matrix: Floats) -> pd.DataFrame:
        return pd.DataFrame(matrix.T, index=index)

    plus_di, minus_di, adx, adxr = qnta.dms(
        frame(panel.high), frame(panel.low), frame(panel.close), 14, 14, 7
    )
    assert plus_di.shape == (N_DAYS, 3)
    first_valid = [
        int(np.argmax(np.isfinite(values_of(part)[:, 0]))) for part in (plus_di, adx, adxr)
    ]
    assert first_valid == [14, 27, 34]
    plus, minus, trend, rating = (values_of(part)[:, 0] for part in (plus_di, minus_di, adx, adxr))
    assert (np.array([plus[-1], minus[-1], trend[-1], rating[-1]]) > 0).all()
    assert_allclose(rating[-1], (trend[-1] + trend[-8]) / 2, rtol=1e-12)
    with pytest.raises(AttributeError, match="has no attribute 'where'"):
        qnta.dms(panel.high, panel.low, panel.close)


@pytest.mark.xfail(strict=True, reason="TypeError: nd_to_1d_np_adapter() is called without np_function")
def test_pivot_points_and_advance_decline() -> None:
    series = np.array([1, 2, 5, 2, 1, 2, 3, 1, 0, 4, 5, 6, 3], dtype=np.float64)
    # the first outputs read an uninitialized window (np.empty), so only a full window is checked
    top = qnta.top_pivot_points(series, 2)
    bottom = qnta.bottom_pivot_points(series, 2)
    assert_array_equal(top[4:], [1, 0, 0, 0, 1, 0, 0, 0, 0])
    assert_array_equal(bottom[4:], [0, 0, 1, 0, 0, 0, 1, 0, 0])
    assert_array_equal(qnta.pivot_points(series, 2, 2)[4:], top[4:] - bottom[4:])
    prices = np.array([[1, 2, 3, 2], [1, 1, 2, 3]], dtype=np.float64)
    assert_array_equal(qnta.ad_line(prices), [0, 1, 3, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = qnta.ad_ratio(prices)
    assert_array_equal(ratio, [NAN, np.inf, np.inf, 1.0])
    frame = pd.DataFrame(prices.T)
    assert_array_equal(values_of(qnta.ad_line(frame)), [0, 1, 3, 3])
    wrapped = xr.DataArray(prices, dims=["asset", "time"], coords={"time": [0, 1, 2, 3]})
    assert_array_equal(qnta.ad_line(wrapped).values, [0, 1, 3, 3])


def test_weighted_averages_and_composite_oscillators(panel: Panel) -> None:
    close, volume = panel.close[0], panel.volume[0]
    assert_allclose(qnta.wma(close, [3.0, 2.0, 1.0]), qnta.lwma(close, 3), equal_nan=True)
    window_price = rolling_windows(close, 5)
    window_volume = rolling_windows(volume, 5)
    expected = [
        float((p * v).sum() / v.sum()) for p, v in zip(window_price, window_volume, strict=True)
    ]
    assert_allclose(qnta.vwma(close, volume, 5)[4:], expected, rtol=1e-10)

    assert_allclose(qnta.sroc(close, 13, 21), qnta.roc(qnta.ema(close, 13), 21), equal_nan=True)
    cadl = qnta.chaikin_adl(panel.high[0], panel.low[0], close, volume)
    assert_allclose(qnta.chaikin_osc(cadl), qnta.ema(cadl, 3) - qnta.ema(cadl, 10), equal_nan=True)

    k, d = qnta.stochastic(panel.high[0], panel.low[0], close, 5, 3)
    assert_allclose(d, qnta.sma(k, 3), equal_nan=True)
    slow_k, slow_d = qnta.slow_stochastic(panel.high[0], panel.low[0], close, 5, 3, 4)
    assert_allclose(slow_k, d, equal_nan=True)
    assert_allclose(slow_d, qnta.sma(slow_k, 4), equal_nan=True)
    assert np.nanmin(k) >= 0.0
    assert np.nanmax(k) <= 100.0

    custom = qnta.rsi(close, lambda s: qnta.sma(s, 5))
    plain_changes = np.diff(close, prepend=np.nan)
    up = qnta.sma(np.maximum(plain_changes, 0), 5)
    down = qnta.sma(-np.minimum(plain_changes, 0), 5)
    assert_allclose(custom, 100 * (1 - 1 / (1 + up / down)), equal_nan=True)


def test_frame_adapter_keeps_labels(panel: Panel) -> None:
    frame = panel.data.sel(field="close").to_pandas()
    result = qnta.sma(frame, 5)
    assert result.shape == frame.shape
    assert result.index.tolist() == frame.index.tolist()
    assert list(result.columns) == list(frame.columns)
    assert_allclose(
        values_of(result)[4],
        values_of(frame)[:5].mean(axis=0),
        rtol=1e-12,
    )


def test_deprecated_integer_wma_delegates_to_lwma(
    panel: Panel, capsys: pytest.CaptureFixture[str]
) -> None:
    close = panel.close[0]
    assert_allclose(qnta.wma(close, 4), qnta.lwma(close, 4), equal_nan=True)
    assert "wma(series:ndarray, periods:int) deprecated" in capsys.readouterr().err


def test_default_periods_match_explicit_values(panel: Panel) -> None:
    close, high, low, volume = panel.close, panel.high, panel.low, panel.volume
    pairs = [
        (qnta.sma(close), qnta.sma(close, 20)),
        (qnta.ema(close), qnta.ema(close, 20)),
        (qnta.wilder_ma(close), qnta.wilder_ma(close, 14)),
        (qnta.dema(close), qnta.dema(close, 20)),
        (qnta.tema(close), qnta.tema(close, 20)),
        (qnta.lwma(close), qnta.lwma(close, 20)),
        (qnta.vwma(close, volume), qnta.vwma(close, volume, 20)),
        (qnta.roc(close), qnta.roc(close, 7)),
        (qnta.sroc(close), qnta.sroc(close, 13, 21)),
        (qnta.trix(close), qnta.trix(close, 18)),
        (qnta.atr(high, low, close), qnta.atr(high, low, close, 14)),
        (qnta.stochastic_k(high, low, close), qnta.stochastic_k(high, low, close, 14)),
        (qnta.change(close), qnta.change(close, 1)),
        (qnta.shift(close), qnta.shift(close, 1)),
    ]
    for default, explicit in pairs:
        assert_allclose(default, explicit, equal_nan=True)
