import qnt.ta.ndadapter as nda
from qnt.ta.ema import ema
from qnt.ta.shift import shift
import numpy as np
import typing as tp
from qnt.log import log_info, log_err


def roc(series: nda.NdType, periods: int = 7) -> nda.NdType:
    """
    Rate of change
    """
    shifted = shift(series, periods)
    return 100 * (series / shifted - 1)


def sroc(series: nda.NdType, ma: tp.Any = 13, periods: int = 21):
    """
    Smooth rate of change
    """
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda s: ema(s, ma_period)
    smooth = ma(series)
    return roc(smooth, periods)


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import ROC
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field='close')

    t1 = time.time()
    roc1 = ROC(data, 7)
    t2 = time.time()
    roc2 = roc(data, 7)
    t3 = time.time()
    roc3 = roc(data.to_pandas(), 7)
    t4 = time.time()

    log_info(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", (abs(roc1 - roc2) * 2 / (roc1 + roc2)).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(roc1.coords[ds.TIME].values, roc1.sel(asset='NASDAQ:AAPL').values, 'r')
    plt.plot(roc2.coords[ds.TIME].values, roc2.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
