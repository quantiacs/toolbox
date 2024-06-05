import numpy as np
import xarray as xr
import qnt.data as qndata
import qnt.ta as qnta

#ATR-based take profit
def take_profit(data, weights, state, threshold, long = True):
    atr14 = qnta.atr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 14).isel(time=-1)
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    last_pos = weights.isel(time=-1).values
    prev_pos = weights.isel(time=-2).values
    last_bar = data.isel(time=-1)
        
    open_prev_long = state['open_price_long']
    open_prev_short = state['open_price_short']

    if long == True:
        # Update open price for long positions
        open_price_long = xr.where(last_pos > 0, xr.where(prev_pos <= 0, last_bar.sel(field='open'), open_prev_long), np.nan)
        # Determine if positions should be kept based on take profit threshold
        keep_pos = xr.where(np.isnan(open_price_long), 1, (keep_pos * (last_bar.sel(field='close') < open_price_long + threshold * atr14)))
        state['open_price_long'] = open_price_long
        return keep_pos
    else:
        # Update open price for short positions
        open_price_short = xr.where(last_pos < 0, xr.where(prev_pos >= 0, last_bar.sel(field='open'), open_prev_short), np.nan)
        # Determine if positions should be kept based on take profit threshold
        keep_pos = xr.where(np.isnan(open_price_short), 1, (keep_pos * (last_bar.sel(field='close') > open_price_short - threshold * atr14)))
        state['open_price_short'] = open_price_short
        return keep_pos

#ATR-based stop loss
def stop_loss(data, weights, state, threshold, long = True):
    atr14 = qnta.atr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 14).isel(time=-1)
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    last_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)
    last_bar = data.isel(time=-1)
    
    open_prev_long = state['open_price_long']
    open_prev_short = state['open_price_short']
    
    if long == True:
        # Update open price for long positions
        open_price_long = xr.where(last_pos > 0, xr.where(prev_pos <= 0, last_bar.sel(field='open'), open_prev_long), np.nan)
        # Determine if positions should be kept based on stop loss threshold
        keep_pos = xr.where(np.isnan(open_price_long), 1, (keep_pos * (last_bar.sel(field='close') > open_price_long - threshold * atr14)))
        state['open_price_long'] = open_price_long
        return keep_pos
    else:
        # Update open price for short positions
        open_price_short = xr.where(last_pos < 0, xr.where(prev_pos >= 0, last_bar.sel(field='open'), open_prev_short), np.nan)
        # Determine if positions should be kept based on stop loss threshold
        keep_pos = xr.where(np.isnan(open_price_short), 1, (keep_pos * (last_bar.sel(field='close') < open_price_short + threshold * atr14)))
        state['open_price_short'] = open_price_short
        return keep_pos

#Percentage-based take profit
def take_profit_percentage(data, weights, state, threshold, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    last_pos = weights.isel(time=-1).values
    prev_pos = weights.isel(time=-2).values
    last_bar = data.isel(time=-1)
        
    open_prev_long = state['open_price_long']
    open_prev_short = state['open_price_short']

    if long == True:
        # Update open price for long positions
        open_price_long = xr.where(last_pos > 0, xr.where(prev_pos <= 0, last_bar.sel(field='open'), open_prev_long), np.nan)
        # Determine if positions should be kept based on take profit threshold
        keep_pos = xr.where(np.isnan(open_price_long), 1, (keep_pos * (last_bar.sel(field='close') < open_price_long * (1+threshold/100))))
        state['open_price_long'] = open_price_long
        return keep_pos
    else:
        # Update open price for short positions
        open_price_short = xr.where(last_pos < 0, xr.where(prev_pos >= 0, last_bar.sel(field='open'), open_prev_short), np.nan)
        # Determine if positions should be kept based on take profit threshold
        keep_pos = xr.where(np.isnan(open_price_short), 1, (keep_pos * (last_bar.sel(field='close') > open_price_short * (1-threshold/100))))
        state['open_price_short'] = open_price_short
        return keep_pos

#Percentage-based stop loss
def stop_loss_percentage(data, weights, state, threshold, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    last_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)
    last_bar = data.isel(time=-1)
    
    open_prev_long = state['open_price_long']
    open_prev_short = state['open_price_short']
    
    if long == True:
        # Update open price for long positions
        open_price_long = xr.where(last_pos > 0, xr.where(prev_pos <= 0, last_bar.sel(field='open'), open_prev_long), np.nan)
        # Determine if positions should be kept based on stop loss threshold
        keep_pos = xr.where(np.isnan(open_price_long), 1, (keep_pos * (last_bar.sel(field='close') > open_price_long * (1-threshold/100))))
        state['open_price_long'] = open_price_long
        return keep_pos
    else:
        # Update open price for short positions
        open_price_short = xr.where(last_pos < 0, xr.where(prev_pos >= 0, last_bar.sel(field='open'), open_prev_short), np.nan)
        # Determine if positions should be kept based on stop loss threshold
        keep_pos = xr.where(np.isnan(open_price_short), 1, (keep_pos * (last_bar.sel(field='close') < open_price_short * (1+threshold/100))))
        state['open_price_short'] = open_price_short
        return keep_pos    
    
#Exit after a certain number of days
def day_counter(data, weights, state, threshold, long = True):
    atr14 = qnta.atr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 14).isel(time=-1)
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    last_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)
    last_bar = data.isel(time=-1)

    counter_prev_long = state['counter_long']
    counter_prev_short = state['counter_short']

    if long == True:
        # Update counter for long positions
        counter_long = xr.where(last_pos > 0, counter_prev_long+1, 0)
        keep_pos = (keep_pos * (counter_long < threshold)).astype(int)
        counter_long = xr.where(counter_long > threshold, 0, counter_long)
        state['counter_long'] = counter_long
        return keep_pos
    else:
        # Update counter for short positions
        counter_short = xr.where(last_pos < 0, counter_prev_short+1, 0)
        keep_pos = (keep_pos * (counter_short < threshold)).astype(int)
        counter_short = xr.where(counter_short > threshold, 0, counter_short)
        state['counter_short'] = counter_short
        return keep_pos