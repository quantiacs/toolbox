import numpy as np
import qnt.ta.ndadapter as nda
import numba as nb
import typing as tp
from qnt.ta.ema import ema
from qnt.log import log_info, log_err


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
def chaikin_adl_np_1d(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    cmfv = volume * ((close - low) - (high - close)) / (high - low)
    s = np.nancumsum(cmfv)
    s[np.isnan(cmfv)] = np.nan
    return s


def chaikin_adl(high: nda.NdType, low: nda.NdType, close: nda.NdType, volume: nda.NdType) -> nda.NdType:
    """
    Chaikin Accumulation Distribution Line
    """
    return nda.nd_universal_adapter(chaikin_adl_np_1d, (high, low, close, volume), ())


def chaikin_osc(cadl: nda.NdType, fast_ma: tp.Any = 3, slow_ma: tp.Any = 10):
    """
    Chaikin Accumulation Distribution Oscillator
    :return:
    """
    if isinstance(fast_ma, int):
        fast_ma_period = fast_ma
        fast_ma = lambda s: ema(s, fast_ma_period)
    if isinstance(slow_ma, int):
        slow_ma_period = slow_ma
        slow_ma = lambda s: ema(s, slow_ma_period)
    fast = fast_ma(cadl)
    slow = slow_ma(cadl)
    return fast - slow


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import ADOSC
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)

    t1 = time.time()
    cadlosc1 = ADOSC(data, 3, 10)
    t2 = time.time()
    cadl2 = chaikin_adl(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), data.sel(field='vol'))
    cadlosc2 = chaikin_osc(cadl2, 3, 10)
    t3 = time.time()
    cadl3 = chaikin_adl(
        data.sel(field='high').to_pandas(),
        data.sel(field='low').to_pandas(),
        data.sel(field='close').to_pandas(),
        data.sel(field='vol').to_pandas()
    )
    cadlosc3 = chaikin_osc(cadl3, 3, 10)
    t4 = time.time()

    log_info(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", (abs(cadlosc1 - cadlosc2) * 2 / abs(cadlosc1 + cadlosc2)).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(cadlosc1.coords[ds.TIME].values, cadlosc1.sel(asset='NASDAQ:AAPL').values, 'r')
    plt.plot(cadlosc2.coords[ds.TIME].values, cadlosc2.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
