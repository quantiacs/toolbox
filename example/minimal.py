import xarray as xr

import qnt.ta as qnta
import qnt.data as qndata
import qnt.backtester as qnbk


def load_data(period):
    data = qndata.cryptofutures.load_data(tail=period)
    return data


def strategy(data):
    close = data.sel(field='close')
    sma200 = qnta.sma(close, 200).isel(time=-1)
    sma20 = qnta.sma(close, 20).isel(time=-1)
    return xr.where(sma200 < sma20, 1, -1)


qnbk.backtest(
    competition_type="cryptofutures",
    load_data=load_data,
    lookback_period=365,
    start_date='2014-01-01',
    strategy=strategy,
    build_plots=False
)