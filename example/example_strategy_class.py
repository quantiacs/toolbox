import xarray as xr
import numpy as np
import pandas as pd
import datetime

from qnt.data import ds, f, load_data, write_output, load_assets
from qnt.stepper import test_strategy
from qnt.stats import *

print(xr.DataArray(np.full([0, 10], np.nan), dims=[ds.TIME, ds.FIELD], coords={ds.TIME: [], ds.FIELD: [
    stf.EQUITY, stf.RELATIVE_RETURN, stf.VOLATILITY,
    stf.UNDERWATER, stf.MAX_DRAWDOWN, stf.SHARPE_RATIO,
    stf.MEAN_RETURN, stf.BIAS, stf.INSTRUMENTS, stf.AVG_TURNOVER
]}))

class SimpleStrategy:
    init_data_length = 14  # optional - data length for init
    need_fix_output = True

    def init(self, data):
        print("init")

    def step(self, data):
        assets = data.sel(**{ds.FIELD: f.OPEN})
        assets = assets.where(data.sel(**{ds.FIELD: f.IS_LIQUID}) > 0)  # liquidity check
        assets = assets.isel(**{ds.TIME: 0})
        assets = assets.dropna(ds.ASSET)
        assets = assets.coords[ds.ASSET]
        pct = 1. / max(len(assets), 1)
        return xr.DataArray(
            np.full([len(assets)], pct, dtype=np.float64),
            dims = [ds.ASSET],
            coords = {ds.ASSET:assets}
        )


assets = load_assets(min_date='2018-01-01')
print(len(assets))
ids = ["NASDAQ:AAPL", "NASDAQ:AMZN", "NASDAQ:FB", "NASDAQ:GOOG"]
data = load_data(assets=ids, min_date='2015-01-01', dims=(ds.TIME, ds.ASSET, ds.FIELD))

output = test_strategy(data, SimpleStrategy())

write_output(output)

RRo = calc_relative_return(data, output)

print(RRo.mean())

RRn = calc_relative_return(data, output)

print(RRn.mean())
