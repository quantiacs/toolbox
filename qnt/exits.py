import numpy as np
import xarray as xr
import qnt.data as qndata
import qnt.ta as qnta

#ATR-based take profit
def take_profit(data, weights, state, last_atr, threshold, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    curr_pos = weights.isel(time=-1).values
    prev_pos = weights.isel(time=-2).values
    last_open = data.isel(time=-1).sel(field='open')
    last_close = data.isel(time=-1).sel(field='close')
    open_prev = state['open_price']
    
    open_price = xr.where(curr_pos == 0, np.nan, last_open)
    open_price = xr.where(curr_pos * prev_pos <= 0, open_price, open_prev)
    
    if long:
        stop_condition = xr.where(curr_pos > 0, last_close < open_price + threshold * last_atr, 1)
    else:
        stop_condition = xr.where(curr_pos < 0, last_close > open_price - threshold * last_atr, 1)  
        
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)       
    state['open_price'] = open_price
    return keep_pos

#ATR-based stop loss
def stop_loss(data, weights, state, last_atr, threshold, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    curr_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)
    last_open = data.isel(time=-1).sel(field='open')
    last_close = data.isel(time=-1).sel(field='close')
    open_prev = state['open_price']
    
    open_price = xr.where(curr_pos == 0, np.nan, last_open)
    open_price = xr.where(curr_pos * prev_pos <= 0, open_price, open_prev)

    if long:
        stop_condition = xr.where(curr_pos > 0, last_close > open_price - threshold * last_atr, 1)
    else:
        stop_condition = xr.where(curr_pos < 0, last_close < open_price + threshold * last_atr, 1)
        
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)
    state['open_price'] = open_price
    return keep_pos

#Percentage-based take profit
def take_profit_percentage(data, weights, state, threshold, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    curr_pos = weights.isel(time=-1).values
    prev_pos = weights.isel(time=-2).values
    last_open = data.isel(time=-1).sel(field='open')
    last_close = data.isel(time=-1).sel(field='close')
    open_prev = state['open_price']
    
    open_price = xr.where(curr_pos == 0, np.nan, last_open)
    open_price = xr.where(curr_pos * prev_pos <= 0, open_price, open_prev)
    
    if long:
        stop_condition = xr.where(curr_pos > 0, last_close < open_price * (1+threshold/100), 1)
    else:
        stop_condition = xr.where(curr_pos < 0, last_close > open_price * (1-threshold/100), 1)  
        
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)       
    state['open_price'] = open_price
    return keep_pos

#Percentage-based stop loss
def stop_loss_percentage(data, weights, state, threshold, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    curr_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)
    last_open = data.isel(time=-1).sel(field='open')
    last_close = data.isel(time=-1).sel(field='close')
    open_prev = state['open_price']
    
    open_price = xr.where(curr_pos == 0, np.nan, last_open)
    open_price = xr.where(curr_pos * prev_pos <= 0, open_price, open_prev)

    if long:
        stop_condition = xr.where(curr_pos > 0, last_close > open_price * (1-threshold/100), 1)
    else:
        stop_condition = xr.where(curr_pos < 0, last_close < open_price * (1+threshold/100), 1)
        
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)
    state['open_price'] = open_price
    return keep_pos
    
#Exit after a certain number of days
def day_counter(data, weights, state, days, long = True):
    keep_pos = [1] * len(data.isel(time=-1).asset) # Initialize positions to keep as 1 (hold)
    curr_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)

    counter_prev = state['counter']
    
    reset_or_increase = xr.where(counter_prev > days, 0, counter_prev+1)
        
    if long:
        counter = xr.where(curr_pos > 0, reset_or_increase, counter_prev) 
        keep_pos = xr.where(curr_pos > 0, counter < days, 1)
    else:
        counter = xr.where(curr_pos < 0, reset_or_increase, counter_prev)        
        keep_pos = xr.where(curr_pos < 0, counter < days, 1)
        
    counter = xr.where(curr_pos != 0, counter, 0)    
    counter = xr.where(curr_pos * prev_pos < 0, 1, counter)
    state['counter'] = counter
    return keep_pos
