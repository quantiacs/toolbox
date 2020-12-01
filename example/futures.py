import qnt.data as qndata
import qnt.stats as qns
import time
import datetime as dt
import xarray as xr


ds = qndata.load_futures_list()

print(ds)


ds = qndata.load_futures_data(forward_order=True)

o = xr.ones_like(ds.sel(field='open'))

s = qns.calc_stat(ds, o, per_asset=True)

print(s.sel(field='sharpe_ratio').to_pandas())


