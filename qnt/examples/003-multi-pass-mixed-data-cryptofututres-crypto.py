"""
Cryptofutures & Crypto

This example shows how to use multiple various datasets with qnt.backtester.backtest
You have to implement a data load function and a window function.
"""

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development


import xarray as xr
import numpy as np

import qnt.data as qndata
import qnt.ta as qnta
import qnt.backtester as qnbt


def load_data(period):
    futures = qndata.futures_load_data(tail=period)
    crypto = qndata.cryptofutures_load_data(tail=period)
    return {"futures": futures, "crypto": crypto}, futures.time.values


def window(data, max_date: np.datetime64, lookback_period: int):
    min_date = max_date - np.timedelta64(lookback_period, 'D')
    return {
        "futures": data['futures'].sel(time=slice(min_date, max_date)),
        "crypto": data['crypto'].sel(time=slice(min_date, max_date)),
    }


def strategy(data):
    close = data['futures'].sel(field='close')
    close_prev = data['futures'].sel(field='close').shift(time=1)
    close_change = (close - close_prev)/close_prev

    close_crypto = data['crypto'].sel(field='close')
    close_crypto_prev = data['crypto'].sel(field='close').shift(time=1)
    close_change_crypto = (close_crypto - close_crypto_prev)/close_crypto_prev

    sma200 = qnta.sma(close_change, 20).fillna(0).mean('asset').isel(time=-1)
    sma200_crypto = qnta.sma(close_change_crypto, 20).isel(time=-1)
    return xr.where(sma200 < sma200_crypto, 1, -1)


qnbt.backtest(
    competition_type="cryptofutures",
    load_data=load_data,
    lookback_period=1 * 365,
    start_date='2014-01-01',
    strategy=strategy,
    window=window
)