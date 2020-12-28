import numpy as np
import numba as nb
import qnt.ta.ndadapter as nda
from qnt.ta.sma import sma
import typing as tp
from qnt.log import log_info, log_err


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.int64), nopython=True)
def k_np_1d(high: np.ndarray, low: np.ndarray, close: np.ndarray, periods: int) -> np.ndarray:
    tail_low = np.empty((periods,), dtype=np.double)
    tail_high = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(close.shape, np.nan, dtype=np.double)
    for i in range(close.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i]):
            idx = not_nan_cnt % periods
            tail_low[idx] = low[i]
            tail_high[idx] = high[i]
            not_nan_cnt += 1
            if not_nan_cnt >= periods:
                highest = tail_high.max()
                lowest = tail_low.min()
                if highest > lowest:
                    result[i] = 100 * (close[i] - lowest) / (highest - lowest)
    return result


def stochastic_k(high: nda.NdType, low: nda.NdType, close: nda.NdType, periods: int = 14):
    return nda.nd_universal_adapter(k_np_1d, (high, low, close), (periods,))


def stochastic(
        high: nda.NdType,
        low: nda.NdType,
        close: nda.NdType,
        periods: int = 5,
        d_ma: tp.Any = 3  # lambda series: sma(series, 3)
) -> tp.Tuple[nda.NdType, nda.NdType]:
    """
    :return: (k,d)
    """
    if isinstance(d_ma, int):
        d_ma_period = d_ma
        d_ma = lambda series: sma(series, d_ma_period)

    k = stochastic_k(high, low, close, periods)
    d = d_ma(k)
    return k, d


def slow_stochastic(
        high: nda.NdType,
        low: nda.NdType,
        close: nda.NdType,
        periods: int = 5,
        k_ma: tp.Any = 3,  # lambda series: sma(series, 3)
        d_ma: tp.Any = 3  # lambda series: sma(series, 3)
) -> tp.Tuple[nda.NdType, nda.NdType]:
    """
    :return: (slow_k, slow_d)
    """
    if isinstance(d_ma, int):
        d_ma_period = d_ma
        d_ma = lambda series: sma(series, d_ma_period)

    k, d = stochastic(high, low, close, periods, k_ma)
    slow_k = d
    slow_d = d_ma(slow_k)
    return slow_k, slow_d


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import STOCH
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)
    t1 = time.time()
    stoch = STOCH(data, 5, 3, 0, 3, 0)
    t2 = time.time()
    sk, sd = slow_stochastic(
        data.sel(field='high'),
        data.sel(field='low'),
        data.sel(field='close'),
        5, 3, 3
    )
    t3 = time.time()

    log_info(
        "relative delta =", abs(stoch.sel(field='slowd').fillna(0) - sd.fillna(0)).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(stoch.coords[ds.TIME].values, stoch.sel(field='slowd', asset='NASDAQ:GOOG').values, 'r')
    plt.plot(sd.coords[ds.TIME].values, sd.sel(asset='NASDAQ:GOOG').values, 'g')
    plt.show()
