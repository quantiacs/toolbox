import qnt.data as qndata
import qnt.stats as qnstats
import qnt.xr_talib as qnxrtalib

import xarray as xr
import pandas as pd
from qnt.stepper import test_strategy

import xarray.ufuncs as xrf
import datetime as dt

data = qndata.cryptofutures_load_data(tail=dt.timedelta(days=10*365), max_date=None, dims=("time", "field", "asset"),
                                       forward_order=True)

print(data.sel(asset='BTC').to_pandas())

print(qnstats.calc_avg_points_per_year(data))

# exit(0)

print(data.sel(field='close').to_pandas())
#
output = data.sel(field='close', asset=['BTC'])
output[:] = 1

#
# output *= 1
#
# print(output.to_pandas())
# print(output[0, 0].item())
#
# print(qnstats.calc_slippage(data).to_pandas()[13:])
#
stat2 = qnstats.calc_stat(data, output, slippage_factor=0.05)

print(stat2.sel(field=[qnstats.stf.AVG_HOLDINGTIME, qnstats.stf.MEAN_RETURN, qnstats.stf.SHARPE_RATIO,
                       qnstats.stf.EQUITY]).to_pandas())

s = qnstats.calc_stat(data, output, per_asset=True).sel(asset='BTC')

print(s.sel(field=[qnstats.stf.AVG_HOLDINGTIME, qnstats.stf.MEAN_RETURN, qnstats.stf.SHARPE_RATIO,
                       qnstats.stf.EQUITY]).to_pandas())


#
# qndata.write_output(output)
