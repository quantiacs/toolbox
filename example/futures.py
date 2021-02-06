import qnt.data as qndata
import qnt.stats as qns
import time
import datetime as dt
import xarray as xr
import qnt.output as qnout

ds = qndata.load_futures_list()

print(ds)


ds = qndata.load_futures_data(forward_order=True)

o = xr.ones_like(ds.sel(field='open'))

o.loc[{'time':'2021-01-18', 'asset':'F_ES'}] = 0

o = qnout.clean(o, ds)

print(ds.sel(asset=['F_BC', 'F_ES', 'F_GC'], field='close', time=slice('2021-01-15', '2021-01-19')).to_pandas())

print(o.sel(asset=['F_BC', 'F_ES', 'F_GC'], time=slice('2021-01-15', '2021-01-25')).to_pandas())

s = qns.calc_stat(ds, o, per_asset=True)

print(s.sel(field='sharpe_ratio', asset=['F_BC', 'F_ES', 'F_GC']).to_pandas())

se = qns.calc_sector_distribution(o)

print(se.to_pandas().tail())


