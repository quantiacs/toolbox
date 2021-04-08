"""
This example shows how to save the state between iterations.

The evaluator will execute all iterations sequentially, so it may be slower than without state.
"""

# import os
# os.environ['API_KEY'] = "{your_api_key_here}"  # you may need it for local development

import xarray as xr
import numpy as np

import qnt.backtester as qnbt


def strategy(data, state):
    close = data.sel(field="close")
    last_close = close.ffill('time').isel(time=-1)

    # state may be null, so define a default value
    if state is None:
        state = {
            "ma": last_close,
            "ma_prev": last_close,
            "output": xr.zeros_like(last_close)
        }

    ma_prev = state['ma']
    ma_prev_prev = state['ma_prev']
    output_prev = state['output']

    ma = ma_prev.where(np.isfinite(ma_prev), last_close) * 0.97 + last_close * 0.03

    buy_signal = np.logical_and(ma > ma_prev, ma_prev > ma_prev_prev)
    stop_signal = ma < ma_prev_prev

    output = xr.where(buy_signal, 1, output_prev)
    output = xr.where(stop_signal, 0, output)

    next_state = {
        "ma": ma,
        "ma_prev": ma_prev,
        "output": output
    }
    return output, next_state


weights, state = qnbt.backtest(
    competition_type="futures",  # Futures contest
    lookback_period=365,  # lookback in calendar days
    start_date="2006-01-01",
    strategy=strategy,
    analyze=True,
    build_plots=True,
    collect_all_states=False # if it is False, then the function returns the last state, otherwise - all states
)
# ---
