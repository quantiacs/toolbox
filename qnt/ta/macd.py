from qnt.ta.ndadapter import NdType
from qnt.ta.ema import ema
import typing as tp
from qnt.log import log_info, log_err


def macd(series: NdType, fast_ma: tp.Any = 12, slow_ma: tp.Any = 26, signal_ma: tp.Any = 9) \
        -> tp.Tuple[NdType, NdType, NdType]:
    """
    MACD
    :return: (macd_line, signal_line, histogram)
    """
    if isinstance(fast_ma, int):
        fast_ma_period = fast_ma
        fast_ma = lambda s: ema(s, fast_ma_period)
    if isinstance(slow_ma, int):
        slow_ma_period = slow_ma
        slow_ma = lambda s: ema(s, slow_ma_period)
    if isinstance(signal_ma, int):
        signal_ma_period = signal_ma
        signal_ma = lambda s: ema(s, signal_ma_period)
    fast = fast_ma(series)
    slow = slow_ma(series)
    macd_line = fast - slow
    signal_line = signal_ma(macd_line)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import MACD
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True).sel(field="close")

    t1 = time.time()
    macd1 = MACD(data, 12, 26, 9)
    t2 = time.time()
    macd2_line, macd2_signal, macd2_hist = macd(data, 12, 26, 9)
    t3 = time.time()
    macd3 = macd(data.to_pandas(), 12, 26, 9)
    t4 = time.time()

    log_info(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", (abs((macd1.sel(macd='hist') - macd2_hist) * 2 / macd2_hist)).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(macd1.coords[ds.TIME].values, macd1.sel(macd='hist', asset='NASDAQ:AAPL').values, 'r')
    plt.plot(macd2_hist.coords[ds.TIME].values, macd2_hist.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
