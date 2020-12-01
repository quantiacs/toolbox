import qnt.ta.ndadapter as nda
from qnt.ta.ema import wilder_ma
from qnt.ta.change import change
import numpy as np
import typing as tp


def rsi(
        series: nda.NdType,
        ma: tp.Any = 14  # lambda series: wilder_ma(series, 14)
) -> nda.NdType:
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda series: wilder_ma(series, ma_period)

    ch = change(series)
    up = np.maximum(ch, 0)  # positive changes
    down = -np.minimum(ch, 0)  # negative changes (inverted)

    up = ma(up)
    down = ma(down)

    rs = up / down
    rsi = 100 * (1 - 1 / (1 + rs))

    return rsi


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import RSI
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field='close')
    data_pd = data.to_pandas()

    t1 = time.time()
    rsi1 = RSI(data, 14)
    t2 = time.time()
    rsi2 = rsi(data, lambda series: wilder_ma(series, 14))
    t3 = time.time()
    rsi3 = rsi(data_pd, lambda series: wilder_ma(series, 14))
    t4 = time.time()

    print(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", abs(rsi1 - rsi2).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(rsi1.coords[ds.TIME].values, rsi1.sel(asset='NASDAQ:AAPL').values, 'r')
    plt.plot(rsi2.coords[ds.TIME].values, rsi2.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
