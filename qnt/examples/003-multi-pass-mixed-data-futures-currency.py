"""
Futures & Currency

This example shows how to use multiple various datasets with qnt.backtester.backtest

You have to implement a data load function and a window function.
"""

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.backtester as qnbt
import qnt.data as qndata


def load_data(period):
    futures = qndata.futures_load_data(assets=['F_AE'], tail=period, dims=("time","field","asset"))
    currency = qndata.imf_load_currency_data(assets=['EUR'], tail=period).isel(asset=0)
    return dict(currency=currency, futures=futures), futures.time.values


def window(data, max_date: np.datetime64, lookback_period: int):
    min_date = max_date - np.timedelta64(lookback_period, 'D')
    return dict(
        futures=data['futures'].sel(time=slice(min_date, max_date)),
        currency=data['currency'].sel(time=slice(min_date, max_date))
    )


def strategy(data):
    close = data['futures'].sel(field="close")
    currency = data['currency']
    ma1 = qnta.lwma(currency,10)
    ma2 = qnta.lwma(currency,50)
    ma3 = qnta.lwma(currency,250)
    if ma1.isel(time=-1) > ma2.isel(time=-1) and ma2.isel(time=-1) < ma3.isel(time=-1):
        return xr.ones_like(close.isel(time=-1))
    else:
        return xr.zeros_like(close.isel(time=-1))


weights = qnbt.backtest(
    competition_type="futures",
    load_data=load_data,
    window=window,
    lookback_period=365,
    start_date="2006-01-01",
    strategy=strategy,
    analyze=True,
    build_plots=True
)