import numpy as np
import numba as nb
import pandas as pd
import qnt.ta.ndadapter as nda
from qnt.log import log_info, log_err

"""
Pivot Points - indicator of tops and bottoms
"""


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def bottom_pivot_points_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods * 2 + 1,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            bottom_idx = (not_nan_cnt - periods) % tail.shape[0]
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                is_bottom = True
                bottom_val = tail[bottom_idx]
                for j in range(tail.shape[0]):
                    if j != bottom_idx and tail[j] <= bottom_val:
                        is_bottom = False
                        break
                result[i] = 1 if is_bottom else 0
    return result


def bottom_pivot_points(series: nda.NdType, periods: int = 5) -> nda.NdType:
    return nda.nd_universal_adapter(bottom_pivot_points_np_1d, (series,), (periods,))


@nb.jit(nb.float64[:](nb.float64[:], nb.int8), nopython=True)
def top_pivot_points_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods * 2 + 1,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.float64)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            top_idx = (not_nan_cnt - periods) % tail.shape[0]
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                is_top = True
                top_val = tail[top_idx]
                for j in range(tail.shape[0]):
                    if j != top_idx and tail[j] >= top_val:
                        is_top = False
                        break
                result[i] = 1 if is_top else 0
    return result


def top_pivot_points(series: nda.NdType, periods: int = 5) -> nda.NdType:
    return nda.nd_universal_adapter(top_pivot_points_np_1d, (series,), (periods,))


def pivot_points(series: nda.NdType, top_periods: int, bottom_periods: int) -> nda.NdType:
    top = top_pivot_points(series, top_periods)
    bottom = bottom_pivot_points(series, bottom_periods)
    return top - bottom


if __name__ == '__main__':
    d1_array = np.array([0, 1, 2, 1, 1, np.nan, 0, np.nan, 1, -1, 0, 1], np.double)
    d1_result = pivot_points(d1_array, 2, 2)
    log_info("d1_array:\n", d1_array, '\n')
    log_info('d1_result:\n', d1_result)
    log_info('---')

    np_array = np.array([
        [
            [0, 1, 2, 1, 1, np.nan, 0, np.nan, 1, -1, 0, 1],
            [0, -1, -2, -1, -1, np.nan, 0, np.nan, -1, -1, 1, 1],
        ], [
            [0, 1, 2, 1, 1, np.nan, 0, np.nan, 1, -1, 0, 1],
            [0, -1, -2, -1, -1, np.nan, 0, np.nan, -1, -1, 1, 1],
        ]
    ], np.double)
    np_result = pivot_points(np_array, 2, 2)
    log_info("np_array:\n", np_array, '\n')
    log_info('np_result:\n', np_result)
    log_info('---')

    date_rng = pd.date_range(start='2018-01-01', end='2018-01-12', freq='D')
    df_array = pd.DataFrame(date_rng, columns=['time']).set_index('time')
    df_array['close'] = np.array([0, 1, 2, 1, 1, np.nan, 0, np.nan, 1, -1, 0, 1], dtype=np.float)
    df_array['open'] = np.array([0, -1, -2, -1, -1, np.nan, 0, np.nan, -1, -1, 1, 1], dtype=np.float)
    df_result = pivot_points(df_array, 2, 2)
    log_info("df_array:\n", df_array, '\n')
    log_info('df_result:\n', df_result)
    log_info('---')

    xr_array = df_array.to_xarray().to_array("field")
    xr_result = pivot_points(xr_array, 2)
    log_info("xr_array:\n", xr_array.to_pandas(), '\n')
    log_info('xr_result:\n', xr_result.to_pandas())
    log_info('---')

    # from qnt.data import load_data, load_assets, ds
    # from qnt.xr_talib import SMA
    # import time
    #
    # assets = load_assets()
    # ids = [i['id'] for i in assets[0:2000]]
    #
    # data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field='close')
    # t1 = time.time()
    # ma1 = SMA(data, 25)
    # t2 = time.time()
    # ma2 = sma(data, 25)
    # t3 = time.time()
    #
    # print(
    #     "relative delta =", abs((ma1.fillna(0) - ma2.fillna(0)) / data).max().values,
    #     "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    # )
