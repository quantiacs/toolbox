"""
Futures & Commodity

This example shows how to use multiple various datasets with qnt.backtester.backtest

You have to implement a data load function and a window function.
"""

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import xarray as xr
import numpy as np

import qnt.backtester as qnbt
import qnt.data as qndata


# print(qnt.data.imf_load_commodity_list()) # displays the commodity list


def load_data(period):
    futures = qndata.futures_load_data(assets=['F_GC'], tail=period, dims=("time","field","asset"))
    commodity = qndata.imf_load_commodity_data(assets=['PGOLD'], tail=period).isel(asset=0)
    return dict(commodity=commodity, futures=futures), futures.time.values


def window(data, max_date: np.datetime64, lookback_period: int):
    min_date = max_date - np.timedelta64(lookback_period, 'D')
    return dict(
        futures=data['futures'].sel(time=slice(min_date, max_date)),
        commodity=data['commodity'].sel(time=slice(min_date, max_date))
    )


def strategy(data):
    close = data['futures'].sel(field="close")
    commodity = data['commodity']
    if commodity.isel(time=-1) > commodity.isel(time=-2) and close.isel(time=-1) > close.isel(time=-20):
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