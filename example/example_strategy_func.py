import xarray as xr
import numpy as np
import pandas as pd
import datetime

from qnt.data import ds, f, load_data, write_output, load_assets
from qnt.stepper import test_strategy
from qnt.stats import *

def init(data):
    log_info("init")

def step(data):
    assets = data.sel(**{ds.FIELD: f.OPEN})
    assets = assets.where(data.sel(**{ds.FIELD: f.IS_LIQUID}) > 0)  # liquidity check
    assets = assets.isel(**{ds.TIME: 0})
    assets = assets.dropna(ds.ASSET)
    assets = assets.coords[ds.ASSET]
    pct = 1. / max(len(assets), 1)
    return xr.DataArray(
        np.full([len(assets)], pct, dtype = np.float64),
        dims = [ds.ASSET],
        coords = {ds.ASSET:assets}
    )


init_data_length = 14  # data length for init

assets = load_assets()
log_info(len(assets))
ids = [i['id'] for i in assets]
log_info(ids)
data = load_data(assets=ids, min_date='2015-01-01', dims=(ds.TIME, ds.ASSET, ds.FIELD))

# print(data.sel(asset='AMEX:AIRI').dropna('time').to_pandas().to_csv())

output = test_strategy(data, init=init, step=step, init_data_length=init_data_length)

write_output(output)

# output = output.loc[:'2019-01-05']
# print(len(find_missed_dates(output, data)))

pd.set_option('display.max_colwidth', -1)
stat1 = calc_stat(data, output)
stat2 = calc_stat(data, output[::-1])
log_info(xr.concat([stat1.sel(field='sharpe_ratio'), stat2.sel(field='sharpe_ratio')],
                   pd.Index(['f', 'b'], name='d')).to_pandas())

stat1.to_pandas().to_csv("stat.csv")

sector_distr = calc_sector_distribution(output, stat1.coords[ds.TIME])

# print(sector_distr.to_pandas())

print_correlation(output, data)
