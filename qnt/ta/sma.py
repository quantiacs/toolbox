import numpy as np
import numba as nb
import pandas as pd
import qnt.ta.ndadapter as nda
from qnt.log import log_info, log_err


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def sma_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % periods
            tail[idx] = series[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                result[i] = tail.mean()
    return result


def sma(series: nda.NdType, periods: int = 20) -> nda.NdType:
    return nda.nd_universal_adapter(sma_np_1d, (series,), (periods,))


if __name__ == '__main__':
    d1_array = np.array([0, 1, 2, 3, 4, np.nan, 5, np.nan, 6, 7], np.double)
    d1_result = sma(d1_array, 3)
    log_info("d1_array:\n", d1_array, '\n')
    log_info('d1_result:\n', d1_result)
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
    np_result = sma(np_array, 2)
    log_info("np_array:\n", np_array, '\n')
    log_info('np_result:\n', np_result)
    log_info('---')

    date_rng = pd.date_range(start='2018-01-01', end='2018-01-04', freq='D')
    df_array = pd.DataFrame(date_rng, columns=['time']).set_index('time')
    df_array['close'] = np.array([1, 2, 3, 4], dtype=np.float)
    df_array['open'] = np.array([5, 6, 7, 8], dtype=np.float)
    df_result = sma(df_array, 2)
    log_info("df_array:\n", df_array, '\n')
    log_info('df_result:\n', df_result)
    log_info('---')

    xr_array = df_array.to_xarray().to_array("field")
    xr_result = sma(xr_array, 2)
    log_info("xr_array:\n", xr_array.to_pandas(), '\n')
    log_info('xr_result:\n', xr_result.to_pandas())
    log_info('---')

    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import SMA
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field='close')
    t1 = time.time()
    ma1 = SMA(data, 25)
    t2 = time.time()
    ma2 = sma(data, 25)
    t3 = time.time()

    log_info(
        "relative delta =", abs((ma1.fillna(0) - ma2.fillna(0)) / data).max().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )
