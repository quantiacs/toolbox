# Optimizer
# Place it to a separate file. Optimization is a very time-demanding task.
# If you need more details, see the corresponding notebook on the site.
# Your strategy have a time limit.

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import json

import xarray as xr
import numpy as np

import qnt.data as qndata          # data loading and manipulation
import qnt.stats as qnstats        # key statistics
import qnt.ta as qnta              # technical analysis indicators
import qnt.log as qnlog            # log configuration
import qnt.optimizer as qno        # optimizer

data = qndata.futures_load_data(min_date='2005-01-01')


def strategy_long(data, asset=None, ma_period=150):
    # filter by asset, we need it for optimization
    if asset is not None:
        data = data.sel(asset=[asset])

    close = data.sel(field='close')

    ma = qnta.lwma(close, ma_period)
    ma_roc = qnta.roc(ma, 1)

    # define signals
    buy_signal = ma_roc > 0
    buy_stop_signal = ma_roc < 0

    # transform signals to positions
    position = xr.where(buy_signal, 1, np.nan)
    position = xr.where(buy_stop_signal, 0, position)
    position = position.ffill('time').fillna(0)

    # clean the output (not necessary)
    # with qnlog.Settings(info=False,err=False): # suppress logging
    #     position = qnout.clean(position, data)
    return position


# initial performance
output = strategy_long(data)
stats = qnstats.calc_stat(data, output.sel(time=slice('2006-01-01',None)))
print(stats.to_pandas().tail())

# full range scan
optimize_result = qno.optimize_strategy(
    data,
    strategy_long,
    qno.full_range_args_generator(
        # argument values for scan, range or list are ok
        ma_period=range(10, 200, 10),
        asset=data.asset.values.tolist()
    ),
    workers=1  # you can set more workers on your local PC to speed up
)

# search suitable parameters and save to `config.json`

def asset_weight(result, asset, cnt):
    asset_iterations = [i for i in result['iterations'] if i['args']['asset'] == asset]
    asset_iterations.sort(key=lambda i: -i['result']['sharpe_ratio'])
    # weight is a sum of the three best iterations
    return sum(i['result']['sharpe_ratio'] for i in asset_iterations[:cnt])


def get_best_parameters_for_asset(result, asset, cnt):
    asset_iterations = [i for i in result['iterations'] if i['args']['asset'] == asset]
    asset_iterations.sort(key=lambda i: -i['result']['sharpe_ratio'])
    return [i['args'] for i in asset_iterations[:cnt]]


def find_best_parameters(result, asset_cnt, parameters_cnt):
    assets = data.asset.values.tolist()
    assets.sort(key=lambda a: -asset_weight(result, a, parameters_cnt))
    assets = assets[:asset_cnt]
    params = []
    for a in assets:
        params += get_best_parameters_for_asset(result, a, parameters_cnt)
    return params


config = find_best_parameters(optimize_result, 15, 3)

json.dump(config, open('config.json', 'w'), indent=2)
