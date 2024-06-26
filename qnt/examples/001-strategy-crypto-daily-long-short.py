# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import xarray as xr

import qnt.stats as qnstats
import qnt.data as qndata
import qnt.output as qnout
import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.graph as qngraph


def strategy(data):
    # calc weights:
    close = data.sel(field="close")
    is_liquid = data.sel(field='is_liquid')
    ma_slow = qnta.lwma(close, 50)
    ma_fast = qnta.lwma(close, 10)
    return xr.where(ma_fast > ma_slow, 1, -1) * is_liquid


# SINGLE-PASS
# ---
# This is fast implementation, but it can easily become looking forward (common problem).
# Use this approach for research and optimization. And use multi-pass to detect looking forward.
data = qndata.cryptodaily.load_data(min_date="2013-04-01")  # load data

output = strategy(data)
output = qnout.clean(output, data, 'crypto_daily_long_short') # fix common errors

qnout.check(output, data, 'crypto_daily_long_short') # check that weights are correct:
qnout.write(output) # write results, necessary for submission:

stats = qnstats.calc_stat(data, output.sel(time=slice("2014-01-01", None))) # calc stats
print(stats.to_pandas().tail())
# qngraph.make_major_plots(stats) # works in jupyter
# ---


# # MULTI-PASS
# # ---
# Use this approach to make sure that your strategy is not looking forward.
# weights = qnbt.backtest(
#     competition_type='crypto_daily_long_short',  # Crypto Daily Long contest
#     lookback_period=365,  # lookback in calendar days
#     start_date="2014-01-01",
#     strategy=strategy,
#     analyze=True,
#     build_plots=True
# )
# # ---
