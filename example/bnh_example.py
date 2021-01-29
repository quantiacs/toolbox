import qnt.data as qndata
import qnt.stats as qnstats
import pandas as pd
import xarray as xr
import numpy as np
import qnt.forward_looking as qnfl
import time
from qnt.neutralization import neutralize
import datetime as dt
import qnt.exposure as qne

# data = qndata.crypto_load_data(assets=['foo'], max_date='2020-01-01')
# print(data)
# exit(0)

# data = qndata.index_load_data(assets=['foo'], max_date='2020-01-01')
# print(data)
# exit(0)

# data = qndata.futures_load_data(assets=['foo'], max_date='2020-01-01')
# print(data)
# exit(0)

# data = qndata.load_data( max_date='2020-11-01', tail=1)
# print(data)
# exit(0)

# data = qndata.index_major_load_data()
# print(data)
# exit(0)

assets = qndata.load_assets()

data = qndata.futures.load_data(
    # assets=[a['id'] for a in assets[-150:]],
    max_date='2020-03-01',
    tail=4*365,
    forward_order=True,
    dims=("time", "field", "asset"))


output = xr.ones_like(data.sel(field=qndata.f.CLOSE))
#output = qndata.sort_and_crop_output(output)
#output = neutralize(output, assets, 'industry')

# output *= 1

output.loc[{"time":slice('2017-01-01','2019-01-01')}] = np.nan
print(output.asset.values.tolist())
# output.loc[{"time":slice('2018-01-01','2019-01-01'), "asset": "NASDAQ:MSFT"}] = 1
output = output.dropna('time', 'all')
#
# print("----")
# print("First check.")
#
# qndata.check_output(output, data)
#
# print("----")
# print("Fix output.")
#
# output = qndata.clean_output(output, data)
#
# print("----")
# print("Second check.")
#
# qndata.check_output(output, data)
#
# print(output.to_pandas())
# print(output[0, 0].item())
#
#
#

#align test
import qnt.output as qnout
import qnt.stats as qns

data = qndata.futures_load_data(min_date='2005-12-10')
print("---\n", data.time[0].values)

output = xr.ones_like(data.sel(field=qndata.f.CLOSE))

qnout.check(output, data)

output_slice = output.sel(time=slice('2015-12-01',None))
rr = qns.calc_relative_return(data, output_slice)
sr = qns.calc_sharpe_ratio_annualized(rr)
print(rr.time[0].values, sr.isel(time=-1).values)


output_slice = qnout.align(output_slice, data.time, '2009-01-01')
rr = qns.calc_relative_return(data, output_slice)
sr = qns.calc_sharpe_ratio_annualized(rr)
print(rr.time[0].values, sr.isel(time=-1).values)
