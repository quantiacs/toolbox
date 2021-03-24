# Optimized strategy
# uses results from optimizer

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import json

import xarray as xr
import numpy as np

import qnt.data as qndata
import qnt.stats as qnstats
import qnt.graph as qngraph
import qnt.ta as qnta
import qnt.backtester as qnbk
import qnt.output as qnout


def strategy_long(data, asset=None, ma_period=150):
    # filter by asset, we will need it for further optimization
    if asset is not None:
        data = data.sel(asset=[asset])

    close = data.sel(field='close')

    ma = qnta.lwma(close, ma_period)
    ma_roc = qnta.roc(ma, 1)

    # define signal
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


def bag_strategy(data, config):
    results = []
    for c in config:
        results.append(strategy_long(data, **c))
    #align and join results
    results = xr.align(*results, join='outer')
    results = [r.fillna(0) for r in results]
    output = sum(results)/len(results)
    return output


config = json.load(open('config.json', 'r'))

# single-pass
data = qndata.futures_load_data(min_date='2005-01-01')

output = bag_strategy(data, config)
output = qnout.clean(output, data)

stats = qnstats.calc_stat(data, output.sel(time=slice('2006-01-01',None)))
print(stats.to_pandas().tail())
# qngraph.make_major_plots(stats)  # works in juoyter

qnout.check(output, data)
qnout.write(output)

# # multi-pass
# # It may look slow, but it is ok. The evaluator will run only one iteration per day.
# qnbk.backtest(
#     competition_type='futures',
#     lookback_period=365,
#     strategy=lambda d: bag_strategy(d, config),
#     # strategy=strategy_long,
#     start_date='2006-01-01'
# )



