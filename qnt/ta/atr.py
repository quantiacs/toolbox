from qnt.ta.shift import shift
from qnt.ta.ndadapter import NdType
from qnt.ta.ema import wilder_ma
import numpy as np
import typing as tp


def tr(high: NdType, low: NdType, close: NdType) -> NdType:
    prev_close = shift(close, 1)
    return np.maximum(high, prev_close) - np.minimum(low, prev_close)


def atr(high: NdType, low: NdType, close: NdType, ma: tp = 14) -> NdType:
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda s: wilder_ma(s, ma_period)
    return ma(tr(high, low, close))


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import ATR
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)

    t1 = time.time()
    atr1 = ATR(data, 14)
    t2 = time.time()
    atr2 = atr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 14)
    t3 = time.time()
    atr3 = atr(data.sel(field='high').to_pandas(), data.sel(field='low').to_pandas(),
               data.sel(field='close').to_pandas(), 14)
    t4 = time.time()

    print(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", (abs(atr1 - atr2) * 2 / (atr1 + atr2)).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(atr1.coords[ds.TIME].values, atr1.sel(asset='NASDAQ:AAPL').values - 1, 'r')
    plt.plot(atr2.coords[ds.TIME].values, atr2.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
