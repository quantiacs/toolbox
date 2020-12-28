import numpy as np
import numba as nb
import pandas as pd
import qnt.ta.ndadapter as nda
import typing as tp
from qnt.log import log_info, log_err


@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64), nopython=True)
def ema_np_1d(series: np.ndarray, periods: int, warm_periods: int) -> np.ndarray:
    result = np.full(series.shape, np.nan, dtype=np.double)
    k = 2 / (periods + 1)
    not_nan_cnt = 0
    value = 0
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            not_nan_cnt += 1
            if not_nan_cnt <= warm_periods:
                value += series[i] / warm_periods
            else:
                value = (series[i] - value) * k + value
            if not_nan_cnt >= warm_periods:
                result[i] = value
    return result


def ema(series: nda.NdType, periods: int = 20, warm_periods: tp.Union[int, None] = None) -> nda.NdType:
    """
    Exponential moving average
    """
    if warm_periods is None:
        warm_periods = periods
    return nda.nd_universal_adapter(ema_np_1d, (series,), (periods, warm_periods,))


def wilder_ma(series: np.ndarray, periods: int = 14, warm_periods: tp.Union[int, None] = None):
    """
    Wilder's Moving Average
    """
    if warm_periods is None:
        warm_periods = periods
    return ema(series, periods * 2 - 1, warm_periods)


def dema(series: nda.NdType, periods: int = 20, warm_periods: tp.Union[int, None] = None) -> nda.NdType:
    """
    Double Exponential Moving Average
    """
    ma = ema(series, periods, warm_periods)
    ma = ema(ma, periods, warm_periods)
    return ma


def tema(series: nda.NdType, periods: int = 20, warm_periods: tp.Union[int, None] = None) -> nda.NdType:
    """
    Triple Exponential Moving Average
    """
    ma = ema(series, periods, warm_periods)
    ma = ema(ma, periods, warm_periods)
    ma = ema(ma, periods, warm_periods)
    return ma


if __name__ == '__main__':
    d1_array = np.array([0, 1, 2, 3, 4, np.nan, 5, np.nan, 6, 7], np.double)
    d1_result = ema(d1_array, 3)
    log_info("d1_array:\n", d1_array, '\n')
    log_info('d1_result:\n', d1_result)
    log_info('---')

    date_rng = pd.date_range(start='2018-01-01', end='2018-01-10', freq='D')
    series_in = pd.Series(d1_array, date_rng)
    series_out = ema(series_in, 3)
    log_info("series_in:\n", series_in, '\n')
    log_info('series_out:\n', series_out)
    log_info('---')

    np_array = np.array([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ]
    ], np.double)
    np_result = ema(np_array, 2)
    log_info("np_array:\n", np_array, '\n')
    log_info('np_result:\n', np_result)
    log_info('---')

    date_rng = pd.date_range(start='2018-01-01', end='2018-01-04', freq='D')
    df_array = pd.DataFrame(date_rng, columns=['time']).set_index('time')
    df_array['close'] = np.array([1, 2, 3, 4], dtype=np.float)
    df_array['open'] = np.array([5, 6, 7, 8], dtype=np.float)
    df_result = ema(df_array, 2)
    log_info("df_array:\n", df_array, '\n')
    log_info('df_result:\n', df_result)
    log_info('---')

    xr_array = df_array.to_xarray().to_array("field")
    xr_result = ema(xr_array, 2)
    log_info("xr_array:\n", xr_array.to_pandas(), '\n')
    log_info('xr_result:\n', xr_result.to_pandas())
    log_info('---')

    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import EMA
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field='close')
    t1 = time.time()
    ma1 = EMA(data, 120)
    t2 = time.time()
    ma2 = ema(data, 120)
    t3 = time.time()

    log_info(
        "relative delta =", abs((ma1.fillna(0) - ma2.fillna(0)) / data).max().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )
