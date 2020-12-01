import numpy as np
import numba as nb
import pandas as pd
import qnt.ta.ndadapter as nda
import typing as tp
import time
import sys


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def lwma_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    w_sum = periods * (periods + 1) / 2
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % periods
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                sum = 0
                for j in range(periods):
                    w = (periods - j)
                    sum += tail[idx - j] * w
                result[i] = sum / w_sum
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True, error_model="numpy")
def wma_np_1d(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    periods = len(weights)
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    w_sum = weights.sum()
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % periods
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                sum = 0
                for j in range(periods):
                    sum += tail[idx - j] * weights[j]
                result[i] = sum / w_sum
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.int64), nopython=True, error_model="numpy")
def vwma_np_1d(price: np.ndarray, volume: np.ndarray, periods:int) -> np.ndarray:
    price_tail = np.empty((periods,), dtype=np.double)
    volume_tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(price.shape, np.nan, dtype=np.double)
    for i in range(price.shape[0]):
        if not np.isnan(price[i]) and not np.isnan(volume[i]):
            idx = not_nan_cnt % periods
            price_tail[idx] = price[i]
            volume_tail[idx] = volume[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                result[i] = (price_tail * volume_tail).sum() / volume_tail.sum()
    return result


last_alert = 0


def wma(series: nda.NdType, weights: tp.Union[tp.List[float], np.ndarray] = None) -> nda.NdType:
    """
    :param weights: weights in decreasing order. lwma(series, 3) == wma(series, [3,2,1])
    """
    global last_alert
    if (weights is None or type(weights) is int):
        if time.time() - last_alert > 60:
            last_alert = time.time()
            print("Warning! wma(series:ndarray, periods:int) deprecated. Use lwma instead of wma.", file=sys.stderr, flush=True)
        return lwma(series,weights)
    if type(weights) is list:
        weights = np.array(weights, np.float64)
    return nda.nd_universal_adapter(wma_np_1d, (series,), (weights,))


def lwma(series: nda.NdType, periods: int = 20):
    return nda.nd_universal_adapter(lwma_np_1d, (series,), (periods,))


def vwma(price: nda.NdType, volume: nda.NdType, periods: int = 20):
    return nda.nd_universal_adapter(vwma_np_1d, (price, volume), (periods,))


if __name__ == '__main__':
    print(np.divide(1., 0.))

    d1_array = np.array([0, 1, 2, 3, 4, np.nan, 5, np.nan, 6, 7], np.double)
    d1_result_lwma = lwma(d1_array, 3)
    d1_result_wma = wma(d1_array, [3, 2, 1])
    d1_result_vwma = vwma(d1_array, d1_array, 3)
    print("d1_array:\n", d1_array, '\n')
    print('d1_result_lwma:\n', d1_result_lwma)
    print('d1_result_wma:\n', d1_result_wma)
    print('d1_result_vwma:\n', d1_result_vwma)
    print('---')

    np_array = np.array([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ], [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ]
    ], np.double)
    np_result = lwma(np_array, 2)
    print("np_array:\n", np_array, '\n')
    print('np_result:\n', np_result)
    print('---')

    date_rng = pd.date_range(start='2018-01-01', end='2018-01-04', freq='D')
    df_array = pd.DataFrame(date_rng, columns=['time']).set_index('time')
    df_array['close'] = np.array([1, 2, 3, 4], dtype=np.float)
    df_array['open'] = np.array([5, 6, 7, 8], dtype=np.float)
    df_result = lwma(df_array, 2)
    print("df_array:\n", df_array, '\n')
    print('df_result:\n', df_result)
    print('---')

    xr_array = df_array.to_xarray().to_array("field")
    xr_result = lwma(xr_array, 2)
    print("xr_array:\n", xr_array.to_pandas(), '\n')
    print('xr_result:\n', xr_result.to_pandas())
    print('---')

    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import WMA
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)
    price = data.sel(field='close')
    vol = data.sel(field='vol')

    t1 = time.time()
    ma1 = WMA(price, 25)
    t2 = time.time()
    ma2 = lwma(price, 25)
    t3 = time.time()

    print(
        "relative delta =", abs((ma1.fillna(0) - ma2.fillna(0)) / data).max().values,
        "t(talib)/t(lwma) =", (t2 - t1) / (t3 - t2)
    )

    ma_lw = lwma(price, 3)
    ma_w = wma(price, [3, 2, 1])
    print("abs(ma_lw - ma_w).sum() = ", abs(ma_lw - ma_w).fillna(0).sum().values)

    ma_vw = vwma(price, vol, 3)