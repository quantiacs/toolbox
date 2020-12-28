import time
import pandas as pd
import xarray as xr
import numpy as np

from .data import load_data, f, ds, write_output
from .stats import calc_non_liquid
from qnt.log import log_info, log_err

class SimpleStrategy:
    init_data_length = 0 # optional - data length for init

    def init(self, data):
        """
        optional

        called before testing,
        use it for learning or indicators warming

        :param data: xarray
        :return:
        """
        pass

    def step(self, data):
        """
        process one step of strategy test

        :param data: xarray, dims : field, time, asset
            available fields:
                open, close, high, low, volume, vol, divs, split - src raw data
                open_sp, close_sp, low_sp, high_sp, vol, divs_sp - data with elliminated splits
        :param portfolio: xarray with last portfolio ASSET -> fraction
        :return: xarray with next portfolio ticker -> fraction
        """
        assets = data.sel[f.OPEN][0].dropna(ds.ASSET).coords[ds.ASSET]
        pct = 1./len(assets)
        return xr.DataArray(
            np.full([len(assets)], pct, dtype = np.float64),
            dims = [ds.ASSET],
            coords = {ds.ASSET:assets}
        )


class StrategyAdapter:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def test_strategy(data, strategy=None, **kwargs):
    """
    :param data: xarray with historical data
    :param strategy: startegy class with an algorithm to run on daily basis
    :param kwargs: if strategy is None, arguments to build a strategy class
    :return: portfolio weights set for every day
    """
    log_info("Testing started...")

    if strategy is None:
        strategy = StrategyAdapter(**kwargs)

    init_data_length = 0
    if(hasattr(strategy, "init_data_length")):
        init_data_length = strategy.init_data_length

    ts = data.coords[ds.TIME]
    forward_order = ts[0] < ts[-1]
    point_count = len(ts) - init_data_length

    current_portfolio = xr.DataArray(
        np.zeros([len(data.coords[ds.ASSET])], dtype=np.float64),
        dims=[ds.ASSET],
        coords={ds.ASSET: data.coords[ds.ASSET]}
    )

    if hasattr(strategy, 'init'):
        strategy.init(None if init_data_length <= 0 else data.isel(**{ds.TIME: slice(point_count)}))

    portfolio_history = xr.DataArray(
        np.zeros([point_count, len(data.coords[ds.ASSET])], dtype=np.float64),
        dims = [ds.TIME, ds.ASSET],
        coords = {
            ds.TIME: data.coords[ds.TIME][init_data_length:(init_data_length + point_count):] if forward_order else
            data.coords[ds.TIME][0:point_count:][::-1],
            ds.ASSET : data.coords[ds.ASSET]
        }
    )
    start_time = time.time()
    last_time = time.time()

    for step in range(point_count):
        piece = data.isel(**{ds.TIME: slice(0, init_data_length + step + 1)}) if forward_order else data.isel(
            **{ds.TIME: slice(point_count - 1 - step, None)})
        # normalize portfolio to the start of day
        next_portfolio = strategy.step(piece)
        if ds.TIME in next_portfolio.dims:
            next_portfolio = next_portfolio.sel({ds.TIME: next_portfolio.coords[ds.TIME].max().values})
        next_portfolio.loc[np.logical_not(np.isfinite(next_portfolio))] = 0

        sum = abs(next_portfolio).sum()
        if sum > 1:
            next_portfolio = next_portfolio / sum

        portfolio_history[step].loc[next_portfolio.coords[ds.ASSET]] = next_portfolio

        if time.time() - last_time > 5:
            last_time = time.time()
            log_info("Testing progress: " +
                     str(step + 1) + "/" + str(point_count) + " " +
                     str(round(last_time - start_time)) + "s")

    last_time = time.time()

    log_info("Testing complete " + str(last_time - start_time) + "s")

    non_liquid = calc_non_liquid(data, portfolio_history)
    if len(non_liquid.coords[ds.TIME]) > 0:
        log_info("WARNING: Strategy trades non-liquid assets.")

    return portfolio_history


calc_step_by_step = test_strategy


# for testing purpose
if __name__ == "__main__":
    #print( os.path.abspath(os.curdir))
    data = load_data(assets=["NYSE:ATGE", "NYSE:AET"], min_date='2017-01-01', dims=(ds.TIME, ds.ASSET, ds.FIELD))
    output = test_strategy(data, SimpleStrategy())
    write_output(output)
    t = time.time()

    log_info(output.to_pandas())
