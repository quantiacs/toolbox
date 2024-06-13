import numpy as np
import xarray as xr
import qnt.data as qndata
import qnt.ta as qnta

#Update open price when position is changed
def update_open_price(data, weights, state):
    last_open = data.isel(time=-1).sel(field='open')
    curr_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)
    open_prev = state['open_price']
    open_price = xr.where(curr_pos == 0, np.nan, last_open)
    open_price = xr.where(curr_pos * prev_pos <= 0, open_price, open_prev)
    state['open_price'] = open_price
    return open_price

#ATR-based take profit for long positions
def take_profit_long_atr(data, weights, open_price, last_atr, atr_amount):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos > 0, last_close < open_price + atr_amount * last_atr, 1)
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)       
    return keep_pos

#ATR-based take profit for short positions
def take_profit_short_atr(data, weights, open_price, last_atr, atr_amount):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos < 0, last_close > open_price - atr_amount * last_atr, 1) 
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)     
    return keep_pos     

#ATR-based stop loss for long positions
def stop_loss_long_atr(data, weights, open_price, last_atr, atr_amount):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos > 0, last_close > open_price - atr_amount * last_atr, 1)
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)       
    return keep_pos

#ATR-based stop loss for short positions
def stop_loss_short_atr(data, weights, open_price, last_atr, atr_amount):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos < 0, last_close < open_price + atr_amount * last_atr, 1) 
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)     
    return keep_pos

#Percentage-based take profit for long positions
def take_profit_long_percentage(data, weights, open_price, percent):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos > 0, last_close < open_price * (1+percent/100), 1)
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)       
    return keep_pos

#Percentage-based take profit for short positions
def take_profit_short_percentage(data, weights, open_price, percent):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos < 0, last_close > open_price * (1-percent/100), 1) 
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)     
    return keep_pos     

#Percentage-based stop loss for long positions
def stop_loss_long_percentage(data, weights, open_price, percent):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos > 0, last_close > open_price * (1-percent/100), 1)
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)       
    return keep_pos

#Percentage-based stop loss for short positions
def stop_loss_short_percentage(data, weights, open_price, percent):
    curr_pos = weights.isel(time=-1)
    last_close = data.isel(time=-1).sel(field='close')
    stop_condition = xr.where(curr_pos < 0, last_close < open_price * (1+percent/100), 1) 
    keep_pos = xr.where(np.isnan(open_price), 1, stop_condition)     
    return keep_pos

#Exit long positions after a certain period 
def max_hold_long(weights, state, max_period):
    curr_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)

    holding_time_prev = state['holding_time']
    
    reset_or_increase = xr.where(holding_time_prev >= max_period, 0, holding_time_prev+1)
        
    holding_time = xr.where(curr_pos > 0, reset_or_increase, holding_time_prev) 

    holding_time = xr.where(curr_pos != 0, holding_time, 0)    
    holding_time = xr.where(curr_pos * prev_pos < 0, 1, holding_time)
    keep_pos = xr.where(curr_pos > 0, holding_time < max_period, 1)
    state['holding_time'] = holding_time
    return keep_pos

#Exit short positions after a certain period
def max_hold_short(weights, state, max_period):
    curr_pos = weights.isel(time=-1)
    prev_pos = weights.isel(time=-2)

    holding_time_prev = state['holding_time']
    
    reset_or_increase = xr.where(holding_time_prev >= max_period, 0, holding_time_prev+1)
    holding_time = xr.where(curr_pos < 0, reset_or_increase, holding_time_prev)        
        
    holding_time = xr.where(curr_pos != 0, holding_time, 0)    
    holding_time = xr.where(curr_pos * prev_pos < 0, 1, holding_time)
    keep_pos = xr.where(curr_pos < 0, holding_time < max_period, 1)
    state['holding_time'] = holding_time
    return keep_pos
