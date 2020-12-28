from qnt.ta.ndadapter import NdType
from qnt.ta.ema import wilder_ma
from qnt.ta.shift import shift
from qnt.ta.atr import atr
import numpy as np
import typing as tp
from qnt.log import log_info, log_err

"""
Directional Movement System
"""


def m(high: NdType, low: NdType) -> tp.Tuple[NdType, NdType]:
    plus_m = high - shift(high, 1)
    minus_m = shift(low, 1) - low
    return plus_m, minus_m


def dm(plus_m: NdType, minus_m: NdType) -> tp.Tuple[NdType, NdType]:
    plus_dm = plus_m.where(np.logical_or(np.isnan(plus_m), np.logical_and(plus_m > minus_m, plus_m > 0)), 0)
    minus_dm = minus_m.where(np.logical_or(np.isnan(minus_m), np.logical_and(minus_m > plus_m, minus_m > 0)), 0)
    return plus_dm, minus_dm


def di(plus_dm: NdType, minus_dm: NdType, atr: NdType, ma: tp.Any = 14) -> tp.Tuple[NdType, NdType]:
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda series: wilder_ma(series, ma_period)

    plus_di = 100 * ma(plus_dm) / atr  # round ?
    minus_di = 100 * ma(minus_dm) / atr  # round ?
    return plus_di, minus_di


def dx(plus_di: NdType, minus_di: NdType) -> NdType:
    return 100 * abs(plus_di - minus_di) / (plus_di + minus_di)


def adx(dx: NdType, ma: tp.Any = 14) -> NdType:
    if isinstance(ma, int):
        ma_period = ma
        ma = lambda series: wilder_ma(series, ma_period)
    return ma(dx)


def adxr(adx: NdType, periods: int) -> NdType:
    prev_adx = shift(adx, periods)
    return (adx + prev_adx) / 2


def dms(
        high: NdType, low: NdType, close: NdType,
        di_ma: tp.Any = 14, adx_ma: tp.Any = 20, adxr_periods: int = 7
) -> tp.Tuple[NdType, NdType, NdType, NdType]:
    """
    :return: (plus_di, minus_di, adx, adxr)
    """
    _plus_m, _minus_m = m(high, low)
    _plus_dm, _minus_dm = dm(_plus_m, _minus_m)
    del _plus_m, _minus_m
    _atr = atr(high, low, close, di_ma)
    _plus_di, _minus_di = di(_plus_dm, _minus_dm, _atr, di_ma)
    del _plus_dm, _minus_dm, _atr
    _dx = dx(_plus_di, _minus_di)
    _adx = adx(_dx, adx_ma)
    del _dx
    _adxr = adxr(_adx, adxr_periods)
    return _plus_di, _minus_di, _adx, _adxr


if __name__ == '__main__':
    from qnt.data import load_data, load_assets, ds
    from qnt.xr_talib import ADXR, ADX, PLUS_DI, PLUS_DM
    import time

    assets = load_assets()
    ids = [i['id'] for i in assets[0:2000]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)
    data_pd = (
        data.sel(field="high").to_pandas(),
        data.sel(field="low").to_pandas(),
        data.sel(field="close").to_pandas()
    )

    t1 = time.time()

    adx1 = ADX(data, 14)

    t2 = time.time()

    _plus_di, _minus_di, _adx, _adxr = dms(
        data.sel(field="high"), data.sel(field="low"), data.sel(field="close"),
        14, 14, 14
    )

    t3 = time.time()
    (pd_res) = dms(
        data_pd[0], data_pd[1], data_pd[2],
        14, 14, 14
    )
    t4 = time.time()

    log_info(
        t2 - t1, t3 - t2, t4 - t3,
        "relative delta =", abs(adx1 - _adx).mean().values,
        "t(talib)/t(this) =", (t2 - t1) / (t3 - t2)
    )

    import matplotlib.pyplot as plt

    plt.plot(adx1.coords[ds.TIME].values, adx1.sel(asset='NASDAQ:AAPL').values, 'r')
    plt.plot(_adx.coords[ds.TIME].values, _adx.sel(asset='NASDAQ:AAPL').values, 'g')
    plt.show()
