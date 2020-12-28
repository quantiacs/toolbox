import numpy as np
import qnt.ta.ndadapter as nda
import numba as nb
from qnt.ta.change import change_np_1d
from qnt.log import log_info, log_err


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
def obv_np_1d(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    v = change_np_1d(close, 1)
    v = np.sign(v)
    v = v * volume
    s = np.nancumsum(v)
    s[np.isnan(v)] = np.nan
    return s


def obv(close: nda.NdType, volume: nda.NdType) -> nda.NdType:
    return nda.nd_universal_adapter(obv_np_1d, (close, volume), ())


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import OBV
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)

    t1 = time.time()
    obv1 = OBV(data)
    t2 = time.time()
    obv2 = obv(data.sel(field='close'), data.sel(field='vol'))
    t3 = time.time()
    obv3 = obv(data.sel(field='close').to_pandas(), data.sel(field='vol').to_pandas())
    t4 = time.time()

    log_info(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", (abs(obv1 - obv2) * 2 / (obv1 + obv2)).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(obv1.coords[ds.TIME].values, obv1.sel(asset='NASDAQ:AAPL').values, 'r')
    plt.plot(obv2.coords[ds.TIME].values, obv2.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
