import qnt.data as qndata
import qnt.stats as qnstats
import qnt.xr_talib as qnxrtalib

import xarray as xr
import pandas as pd
from qnt.stepper import test_strategy

import xarray.ufuncs as xrf

data = qndata.load_data(min_date="2010-01-01", max_date=None, dims=("time", "field", "asset"), forward_order=True)

wma = qnxrtalib.WMA(data.sel(field='close'), 120)
sroc = qnxrtalib.ROCP(wma, 60)
stoch = qnxrtalib.STOCH(data, 8, 3, 3)
k = stoch.sel(field='slowk')
d = stoch.sel(field='slowd')

data_ext = xr.concat([wma, sroc, k, d], pd.Index(['wma', 'sroc', 'k', 'd'], name='field'))
data_ext = xr.concat([data, data_ext], 'field')

weights = data.isel(time=0, field=0)
weights[:] = 0


def step(data):
    latest = data.isel(time=-1)

    is_liquid = latest.sel(field="is_liquid")
    sroc = latest.sel(field='sroc')
    k = latest.sel(field='k')
    d = latest.sel(field='d')

    need_open = xrf.logical_and(sroc > 0.05, xrf.logical_and(k < 31, d < 31))
    need_close = xrf.logical_or(xrf.logical_or(sroc < -0.05, is_liquid == 0), xrf.logical_and(k > 92, d > 92))

    global weights
    weights.loc[need_open] = 1
    weights.loc[need_close] = 0

    return (weights / weights.sum('asset')).fillna(0)


output = test_strategy(data_ext, step=step, init_data_length=200)

stat = qnstats.calc_stat(data, output, max_periods=252 * 3)

print(stat.to_pandas())

qndata.write_output(output)

qnstats.check_exposure(output)
