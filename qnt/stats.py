from .data import f, ds, stocks_load_list, get_env, futures_load_list, stocks_load_ndx_list
from .data.common import track_event
from .output import normalize as output_normalize
from qnt.log import log_info, log_err
import xarray as xr
import numpy as np
import pandas as pd
import gzip, base64, json
from urllib import request
from tabulate import tabulate
import numba
import sys, os
from qnt.output import normalize

EPS = 10 ** -7


def calc_slippage(data, period_days=14, fract=None, points_per_year=None):
    """
    :param data: xarray with historical data
    :param period_days: period for atr
    :param fract: slippage factor
    :return: xarray with slippage
    """
    if fract is None:
        fract = get_default_slippage(data)

    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(data)

    time_series = np.sort(data.coords[ds.TIME])
    data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET).loc[[f.CLOSE, f.HIGH, f.LOW], time_series, :]

    points_per_day = calc_points_per_day(points_per_year)
    daily_period = min(points_per_day, len(data.time))

    cl = data.loc[f.CLOSE].shift({ds.TIME: daily_period})
    hi = data.loc[f.HIGH].rolling({ds.TIME: daily_period}).max()
    lo = data.loc[f.LOW].rolling({ds.TIME: daily_period}).min()
    d1 = hi - lo
    d2 = abs(hi - cl)
    d3 = abs(cl - lo)
    dd = xr.concat([d1, d2, d3], dim='d').max(dim='d', skipna=False)

    atr_period = min(len(dd.time), period_days * points_per_day)

    dd = dd.rolling({ds.TIME: atr_period}, min_periods=atr_period).mean(skipna=False).ffill(ds.TIME)
    return dd * fract


def calc_relative_return(data, portfolio_history,
                         slippage_factor=None, roll_slippage_factor=None,
                         per_asset=False, points_per_year=None):
    if slippage_factor is None:
        slippage_factor = get_default_slippage(data)

    if roll_slippage_factor is None:
        roll_slippage_factor = get_default_slippage(data)

    target_weights = portfolio_history.shift(**{ds.TIME: 1})  # shift
    target_weights[{ds.TIME: 0}] = 0
    min_time = target_weights.coords[ds.TIME].min()

    slippage = calc_slippage(data, 14, slippage_factor, points_per_year=points_per_year)
    roll_slippage = calc_slippage(data, 14, roll_slippage_factor, points_per_year=points_per_year)

    data, target_weights, slippage, roll_slippage = arrange_data(data, target_weights, per_asset, slippage,
                                                                 roll_slippage)

    # adjust weights according to price changes close->open : W_open = W_close_prev * OPEN / CLOSE_prev
    prev_close = data.loc[f.CLOSE].shift(**{ds.TIME: 1}).ffill(ds.TIME)
    target_weights = target_weights * data.loc[f.OPEN] / prev_close
    target_weights = output_normalize(target_weights, per_asset)

    W = target_weights
    D = data

    OPEN = D.loc[f.OPEN].ffill(ds.TIME).fillna(0)
    CLOSE = D.loc[f.CLOSE].ffill(ds.TIME).fillna(0)
    DIVS = D.loc[f.DIVS].fillna(0) if f.DIVS in D.coords[ds.FIELD] else xr.full_like(D.loc[f.CLOSE], 0)
    ROLL = D.loc[f.ROLL].fillna(0) if f.ROLL in D.coords[ds.FIELD] else None
    ROLL_SLIPPAGE = roll_slippage.where(ROLL != 0).fillna(0) if ROLL is not None else None

    # boolean matrix when assets available for trading
    UNLOCKED = np.logical_and(np.isfinite(D.loc[f.OPEN].values), np.isfinite(D.loc[f.CLOSE].values))
    UNLOCKED = np.logical_and(np.isfinite(W.values), UNLOCKED)
    UNLOCKED = np.logical_and(np.isfinite(slippage.values), UNLOCKED)
    UNLOCKED = np.logical_and(OPEN > EPS, UNLOCKED)

    if per_asset:
        RR = W.copy(True)
        RR[:] = calc_relative_return_np_per_asset(W.values, UNLOCKED.values, OPEN.values, CLOSE.values, slippage.values,
                                                  DIVS.values,
                                                  ROLL.values if ROLL is not None else None,
                                                  ROLL_SLIPPAGE.values if ROLL_SLIPPAGE is not None else None
                                                  )
        return RR.loc[min_time:]
    else:
        RR = xr.DataArray(
            np.full([len(W.coords[ds.TIME])], np.nan, np.double),
            dims=[ds.TIME],
            coords={ds.TIME: W.coords[ds.TIME]}
        )
        res = calc_relative_return_np(W.values, UNLOCKED.values, OPEN.values, CLOSE.values, slippage.values,
                                      DIVS.values,
                                      ROLL.values if ROLL is not None else None,
                                      ROLL_SLIPPAGE.values if ROLL_SLIPPAGE is not None else None
                                      )
        RR[:] = res
        return RR.loc[min_time:]


@numba.njit
def calc_relative_return_np_per_asset(WEIGHT, UNLOCKED, OPEN, CLOSE, SLIPPAGE, DIVS, ROLL, ROLL_SLIPPAGE):
    N = np.zeros(WEIGHT.shape)  # shares count

    equity_before_buy = np.zeros(WEIGHT.shape)
    equity_after_buy = np.zeros(WEIGHT.shape)
    equity_tonight = np.zeros(WEIGHT.shape)

    for t in range(0, WEIGHT.shape[0]):
        unlocked = UNLOCKED[t]  # available for trading

        if t == 0:
            equity_before_buy[0] = 1
            N[0] = 0
        else:
            N[t] = N[t - 1]
            equity_before_buy[t] = equity_after_buy[t - 1] + (OPEN[t] - OPEN[t - 1] + DIVS[t]) * N[t]

        N[t][unlocked] = equity_before_buy[t][unlocked] * WEIGHT[t][unlocked] / OPEN[t][unlocked]
        dN = N[t]
        if t > 0:
            dN = dN - N[t - 1]
        S = SLIPPAGE[t] * np.abs(dN)  # slippage for this step
        equity_after_buy[t] = equity_before_buy[t] - S

        if ROLL is not None and t > 0:
            pN = np.where(np.sign(N[t]) == np.sign(N[t - 1]), np.minimum(np.abs(N[t]), np.abs(N[t - 1])), 0)
            R = np.sign(N[t]) * pN * ROLL[t] + pN * ROLL_SLIPPAGE[t]
            equity_after_buy[t] -= R

        equity_tonight[t] = equity_after_buy[t] + (CLOSE[t] - OPEN[t]) * N[t]

        locked = np.logical_not(unlocked)
        if t == 0:
            equity_before_buy[0][locked] = 1
            equity_after_buy[0][locked] = 1
            equity_tonight[0][locked] = 1
            N[0][locked] = 0
        else:
            N[t][locked] = N[t - 1][locked]
            equity_after_buy[t][locked] = equity_after_buy[t - 1][locked]
            equity_before_buy[t][locked] = equity_before_buy[t - 1][locked]
            equity_tonight[t][locked] = equity_tonight[t - 1][locked]

    E = equity_tonight
    # Ep = np.roll(E, 1, axis=0)
    Ep = E.copy()
    for i in range(1, Ep.shape[0]):
        Ep[i] = E[i - 1]
    Ep[0] = 1
    RR = E / Ep - 1
    RR = np.where(np.isfinite(RR), RR, 0)
    return RR


@numba.njit
def calc_relative_return_np(WEIGHT, UNLOCKED, OPEN, CLOSE, SLIPPAGE, DIVS, ROLL, ROLL_SLIPPAGE):
    N = np.zeros(WEIGHT.shape)  # shares count

    equity_before_buy = np.zeros(WEIGHT.shape[0])
    equity_operable_before_buy = np.zeros(WEIGHT.shape[0])
    equity_after_buy = np.zeros(WEIGHT.shape[0])
    equity_tonight = np.zeros(WEIGHT.shape[0])

    for t in range(WEIGHT.shape[0]):
        unlocked = UNLOCKED[t]  # available for trading
        locked = np.logical_not(unlocked)

        if t == 0:
            equity_before_buy[0] = 1
            N[0] = 0
        else:
            N[t] = N[t - 1]
            equity_before_buy[t] = equity_after_buy[t - 1] + np.nansum((OPEN[t] - OPEN[t - 1] + DIVS[t]) * N[t])

        w_sum = np.nansum(np.abs(WEIGHT[t]))
        w_free_cash = max(1, w_sum) - w_sum
        w_unlocked = np.nansum(np.abs(WEIGHT[t][unlocked]))
        w_operable = w_unlocked + w_free_cash

        equity_operable_before_buy[t] = equity_before_buy[t] - np.nansum(OPEN[t][locked] * np.abs(N[t][locked]))

        if w_operable < EPS:
            equity_after_buy[t] = equity_before_buy[t]
        else:
            N[t][unlocked] = equity_operable_before_buy[t] * WEIGHT[t][unlocked] / (w_operable * OPEN[t][unlocked])
            dN = N[t][unlocked]
            if t > 0:
                dN = dN - N[t - 1][unlocked]
            S = np.nansum(SLIPPAGE[t][unlocked] * np.abs(dN))  # slippage for this step
            equity_after_buy[t] = equity_before_buy[t] - S

        if ROLL is not None and t > 0:
            pN = np.where(np.sign(N[t]) == np.sign(N[t - 1]), np.minimum(np.abs(N[t]), np.abs(N[t - 1])), 0)
            R = np.sign(N[t]) * pN * ROLL[t] + pN * ROLL_SLIPPAGE[t]
            equity_after_buy[t] -= np.nansum(R)

        equity_tonight[t] = equity_after_buy[t] + np.nansum((CLOSE[t] - OPEN[t]) * N[t])
        # Update the number of shares for locked assets
        N[t][locked] = 0

    E = equity_tonight
    Ep = np.roll(E, 1)
    Ep[0] = 1
    RR = E / Ep - 1
    RR = np.where(np.isfinite(RR), RR, 0)
    return RR


def arrange_data(data, target_weights, per_asset, *additional_series):
    """
    arranges data for proper calculations
    :param per_asset:
    :param data:
    :param target_weights:
    :param additional_series:
    :return:
    """
    data = data.dropna(ds.ASSET, 'all').dropna(ds.TIME, 'all')
    target_weights, data = xr.align(target_weights, data, join='right')
    ra1 = []
    for a in additional_series:
        a, _ = xr.align(a, data, join='right')
        ra1.append(a)

    time_series = data.coords[ds.TIME]
    time_series = np.sort(time_series)

    assets = data.coords[ds.ASSET].values
    assets = np.sort(assets)

    adjusted_data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    adjusted_data = adjusted_data.loc[:, time_series, assets]

    target_weights = target_weights.transpose(ds.TIME, ds.ASSET)
    target_weights = target_weights.loc[time_series, assets]
    target_weights = normalize(target_weights, per_asset)

    ra2 = []
    for a in list(ra1):
        a = a.sel(time=time_series)
        if ds.ASSET in a.dims:
            a = a.sel(asset=assets)
        ra2.append(a)

    target_weights = target_weights.drop(ds.FIELD, errors='ignore')

    return (adjusted_data, target_weights, *ra2)


def calc_equity(relative_return):
    """
    :param relative_return: daily return
    :return: daily portfolio equity
    """
    return (relative_return + 1).cumprod(ds.TIME)


def calc_volatility(relative_return, max_periods=None, min_periods=2):
    """
    :param relative_return: daily return
    :param max_periods: maximal number of days
    :param min_periods: minimal number of days
    :return: portfolio volatility
    """
    if max_periods is None:
        max_periods = len(relative_return.time)
    max_periods = min(max_periods, len(relative_return.coords[ds.TIME]))
    min_periods = min(min_periods, max_periods)
    return relative_return.rolling({ds.TIME: max_periods}, min_periods=min_periods).std()


def calc_volatility_annualized(relative_return, max_periods=None, min_periods=2, points_per_year=None):
    """
    :param relative_return: daily return
    :param min_periods: minimal number of days
    :return: annualized volatility
    """
    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(relative_return)
    if max_periods is None:
        max_periods = len(relative_return.time)
    return calc_volatility(relative_return, max_periods, min_periods) * pow(
        points_per_year, 1. / 2)


def calc_underwater(equity):
    """
    :param equity: daily portfolio equity
    :return: daily underwater
    """
    mx = equity.rolling({ds.TIME: len(equity)}, min_periods=1).max()
    return equity / mx - 1


def calc_max_drawdown(underwater):
    """
    :param underwater: daily underwater
    :return: daily maximum drawdown
    """
    return (underwater).rolling({ds.TIME: len(underwater)}, min_periods=1).min()


def calc_sharpe_ratio_annualized(relative_return, max_periods=None, min_periods=2, points_per_year=None):
    """
    :param relative_return: daily return
    :param max_periods: maximal number of days
    :param min_periods: minimal number of days
    :return: annualized Sharpe ratio
    """
    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(relative_return)
    if max_periods is None:
        max_periods = len(relative_return.time)
    m = calc_mean_return_annualized(relative_return, max_periods, min_periods, points_per_year=points_per_year)
    v = calc_volatility_annualized(relative_return, max_periods, min_periods, points_per_year=points_per_year)
    sr = m / v
    return sr


def calc_mean_return(relative_return, max_periods=None, min_periods=1, points_per_year=None):
    """
    :param relative_return: daily return
    :param max_periods: maximal number of days
    :param min_periods: minimal number of days
    :return: daily mean return
    """
    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(relative_return)
    if max_periods is None:
        max_periods = len(relative_return.coords[ds.TIME])
    max_periods = min(max_periods, len(relative_return.coords[ds.TIME]))
    min_periods = min(min_periods, max_periods)
    return np.exp(
        np.log(relative_return + 1).rolling({ds.TIME: max_periods},
                                            min_periods=min_periods).mean(skipna=True)) - 1


def calc_mean_return_annualized(relative_return, max_periods=None, min_periods=1, points_per_year=None):
    """
    :param relative_return: daily return
    :param min_periods: minimal number of days
    :return: annualized mean return
    """
    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(relative_return)

    return np.power(calc_mean_return(relative_return, max_periods, min_periods, points_per_year=points_per_year) + 1,
                    points_per_year) - 1


def calc_bias(portfolio_history, per_asset=False):
    """
    :param per_asset:
    :param portfolio_history: portfolio weights set for every day
    :return: daily portfolio bias
    """
    if per_asset:
        return portfolio_history
    ph = portfolio_history
    sum = ph.sum(ds.ASSET)
    abs_sum = abs(ph).sum(ds.ASSET)
    res = sum / abs_sum
    res = res.where(np.isfinite(res)).fillna(0)
    return res


def calc_instruments(portfolio_history, per_asset=False):
    """
    :param per_asset:
    :param portfolio_history: portfolio weights set for every day
    :return: daily portfolio instrument count
    """
    if per_asset:
        I = portfolio_history.copy(True)
        I[:] = 1
        return I
    ph = portfolio_history.copy().fillna(0)
    ic = ph.where(ph == 0).fillna(1)
    ic = ic.cumsum(ds.TIME)
    ic = ic.where(ic == 0).fillna(1)
    ic = ic.sum(ds.ASSET)
    return ic


def calc_avg_turnover(portfolio_history, equity, data, max_periods=None, min_periods=1, per_asset=False,
                      points_per_year=None):
    '''
    Calculates average capital turnover, all args must be adjusted
    :param portfolio_history: history of portfolio changes
    :param equity: equity of changes
    :param data:
    :param max_periods:
    :param min_periods:
    :param per_asset:
    :return:
    '''
    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(portfolio_history)
    if max_periods is None:
        max_periods = points_per_year

    W = portfolio_history.transpose(ds.TIME, ds.ASSET)
    W = W.shift({ds.TIME: 1})
    W[0] = 0

    Wp = W.shift({ds.TIME: 1})
    Wp[0] = 0

    OPEN = data.transpose(ds.TIME, ds.FIELD, ds.ASSET).loc[W.coords[ds.TIME], f.OPEN, W.coords[ds.ASSET]]
    OPENp = OPEN.shift({ds.TIME: 1})
    OPENp[0] = OPEN[0]

    E = equity

    Ep = E.shift({ds.TIME: 1})
    Ep[0] = 1

    turnover = abs(W - Wp * Ep * OPEN / (OPENp * E))
    if not per_asset:
        turnover = turnover.sum(ds.ASSET)
    max_periods = min(max_periods, len(turnover.coords[ds.TIME]))
    min_periods = min(min_periods, len(turnover.coords[ds.TIME]))
    turnover = turnover.rolling({ds.TIME: max_periods}, min_periods=min_periods).mean()
    try:
        turnover = turnover.drop(ds.FIELD)
    except ValueError:
        pass
    return turnover


def calc_avg_holding_time(portfolio_history,
                          max_periods=None, min_periods=1, per_asset=False, points_per_year=None):
    '''
    Calculates holding time.
    :param portfolio_history:
    :param max_periods:
    :param min_periods:
    :param per_asset:
    :param points_per_year:
    :return:
    '''
    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(portfolio_history)
    if max_periods is None:
        max_periods = len(portfolio_history.time)

    ph = portfolio_history.copy(True)

    try:
        ph[-2] = 0  # avoids NaN for buy-and-hold
    except:
        pass

    log = calc_holding_log_np_nb(ph.values)  # , equity.values, data.sel(field='open').values)

    log = xr.DataArray(log, dims=[ds.TIME, ds.FIELD, ds.ASSET], coords={
        ds.TIME: portfolio_history.time,
        ds.FIELD: ['cost', 'duration'],
        ds.ASSET: portfolio_history.asset
    })

    if not per_asset:
        log2d = log.isel(asset=0).copy(True)
        log2d.loc[{ds.FIELD: 'cost'}] = log.sel(field='cost').sum(ds.ASSET)
        log2d.loc[{ds.FIELD: 'duration'}] = (log.sel(field='cost') * log.sel(field='duration')).sum(ds.ASSET) / \
                                            log2d.sel(field='cost')
        log = log2d

        try:
            log = log.drop(ds.ASSET)
        except ValueError:
            pass

    max_periods = min(max_periods, len(log.coords[ds.TIME]))
    min_periods = min(min_periods, len(log.coords[ds.TIME]))

    res = (log.sel(field='cost') * log.sel(field='duration')) \
              .rolling({ds.TIME: max_periods}, min_periods=min_periods).sum() / \
          log.sel(field='cost') \
              .rolling({ds.TIME: max_periods}, min_periods=min_periods).sum()

    try:
        res = res.drop(ds.FIELD)
    except ValueError:
        pass

    points_per_day = calc_points_per_day(points_per_year)

    return res / points_per_day


@numba.jit
def calc_holding_log_np_nb(weights: np.ndarray) -> np.ndarray:  # , equity: np.ndarray, open: np.ndarray) -> np.ndarray:
    prev_pos = np.zeros(weights.shape[1])
    holding_time = np.zeros(weights.shape[1])  # position holding time
    holding_log = np.zeros(weights.shape[0] * 2 * weights.shape[1])  # time, field (position_cost, holding_time), asset
    holding_log = holding_log.reshape(weights.shape[0], 2, weights.shape[1])

    for t in range(1, weights.shape[0]):
        holding_time[:] += 1
        for a in range(weights.shape[1]):
            # price = open[t][a]
            # if not np.isfinite(price):
            #     continue
            pos = weights[t - 1][a]  # * equity[t] / price
            ppos = prev_pos[a]
            if not np.isfinite(pos):
                continue
            dpos = pos - ppos
            if abs(dpos) < EPS:
                continue
            if ppos > 0 > dpos or ppos < 0 < dpos:  # opposite change direction
                if abs(dpos) > abs(ppos):
                    holding_log[t][0][a] = abs(ppos)  # * price
                    holding_log[t][1][a] = holding_time[a]
                    holding_time[a] = 0
                else:
                    holding_log[t][0][a] = abs(dpos)  # * price
                    holding_log[t][1][a] = holding_time[a]
            elif pos != 0:
                holding_time[a] = holding_time[a] * abs(ppos) / abs(pos)
            prev_pos[a] = pos
    return holding_log


def calc_non_liquid(data, portfolio_history):
    (adj_data, adj_ph) = arrange_data(data, portfolio_history, False)
    if f.IS_LIQUID in adj_data.coords[ds.FIELD]:
        non_liquid = adj_ph.where(adj_data.loc[f.IS_LIQUID] == 0)
    else:
        non_liquid = xr.full_like(adj_data.loc[f.CLOSE], np.nan)
    non_liquid = non_liquid.where(abs(non_liquid) > 0)
    non_liquid = non_liquid.dropna(ds.ASSET, 'all')
    non_liquid = non_liquid.dropna(ds.TIME, 'all')
    return non_liquid


def find_missed_dates(output, data):
    out_ts = np.sort(output.coords[ds.TIME].values)

    min_out_ts = min(out_ts)

    data_ts = data.where(data.time >= min_out_ts)
    if f.IS_LIQUID in data.coords[ds.FIELD]:
        data_ts = data_ts.where(data.sel({ds.FIELD: f.IS_LIQUID}) > 0)
    else:
        data_ts = data_ts.where(data.sel({ds.FIELD: f.CLOSE}) > 0)
    data_ts = data_ts.dropna(ds.TIME, 'all').coords[ds.TIME]
    data_ts = np.sort(data_ts.values)
    missed = np.setdiff1d(data_ts, out_ts)
    return missed


def calc_avg_points_per_year(data: xr.DataArray):
    if len(data.time) < 251:
        if data.name == 'crypto':
            return round(365.25 * 24)
        if data.name in ['cryptofutures', 'crypto_futures', 'crypto_daily', 'cryptodaily',
                         'crypto_daily_long', 'crypto_daily_long_short']:
            return 365
        if data.name in ['stocks', 'stocks_long', 'futures', 'stocks_nasdaq100']:
            return 251
    t = np.sort(data.coords[ds.TIME].values)
    tp = np.roll(t, 1)
    dh = (t[1:] - tp[1:]).mean().item() / (10 ** 9) / 60 / 60  # avg diff in hours
    return round(365.25 * 24 / dh)


def get_default_is_period(data):
    isp = get_default_is_period_for_type(data.name)
    if isp is not None:
        return isp
    points_per_year = calc_avg_points_per_year(data)
    return (points_per_year * 5)


def get_default_is_period_for_type(name):
    if name == 'stocks_nasdaq100':
        return int(get_env('IS_STOCKS_NASDAQ100', '3528', True))
    if name == 'stocks' or name == 'stocks_long':
        return int(get_env('IS_STOCKS', '1512', True))
    if name == 'futures':
        return int(get_env('IS_FUTURES', '3528', True))
    if name == 'cryptofutures' or name == 'crypto_futures':
        return int(get_env('IS_CRYPTOFUTURES', '1764', True))
    if name == 'cryptodaily' or name == 'crypto_daily' or name == 'crypto_daily_long' or name == 'crypto_daily_long_short':
        return int(get_env('IS_CRYPTOFUTURES', '1764', True))
    if name == 'crypto':
        return int(get_env('IS_CRYPTO', '60000', True))
    return None


def get_default_is_start_date_for_type(name):
    if name == 'stocks_nasdaq100':
        return get_env('SD_STOCKS_NASDAQ100', '2006-01-01', True)
    if name == 'stocks' or name == 'stocks_long':
        return get_env('SD_STOCKS', '2015-01-01', True)
    if name == 'futures':
        return get_env('SD_FUTURES', '2006-01-01', True)
    if name == 'cryptofutures' or name == 'crypto_futures':
        return get_env('SD_CRYPTOFUTURES', '2014-01-01', True)
    if name == 'cryptodaily' or name == 'crypto_daily' or name == 'crypto_daily_long' or name == 'crypto_daily_long_short':
        return get_env('SD_CRYPTODAILY', '2014-01-01', True)
    if name == 'crypto':
        return get_env('SD_CRYPTO', '2014-01-01', True)
    return None


def get_default_slippage(data):
    if data.name == 'stocks_nasdaq100':
        return float(get_env('SL_STOCKS_NASDAQ100', '0.05', True))
    if data.name == 'stocks':
        return float(get_env('SL_STOCKS', '0.05', True))
    if data.name == 'futures':
        return float(get_env('SL_FUTURES', '0.04', True))
    if data.name == 'cryptofutures' or data.name == 'crypto_futures':
        return float(get_env('SL_CRYPTOFUTURES', '0.04', True))
    if data.name == 'cryptodaily' or data.name == 'crypto_daily' or data.name == 'crypto_daily_long' or data.name == 'crypto_daily_long_short':
        return float(get_env('SL_CRYPTODAILY', '0.04', True))
    if data.name == 'crypto':
        return float(get_env('SL_CRYPTO', '0.05', True))
    return 0.05


def calc_points_per_day(days_per_year):
    if days_per_year < 400:
        return 1
    else:
        return 24


class StatFields:
    RELATIVE_RETURN = "relative_return"
    EQUITY = "equity"
    VOLATILITY = "volatility"
    UNDERWATER = "underwater"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    MEAN_RETURN = "mean_return"
    BIAS = "bias"
    INSTRUMENTS = "instruments"
    AVG_TURNOVER = "avg_turnover"
    AVG_HOLDINGTIME = 'avg_holding_time'


stf = StatFields


def calc_stat(data, portfolio_history,
              slippage_factor=None, roll_slippage_factor=None,
              min_periods=1, max_periods=None,
              per_asset=False, points_per_year=None):
    """
    :param data: xarray with historical data, data must be split adjusted
    :param portfolio_history: portfolio weights set for every day
    :param slippage_factor: slippage
    :param roll_slippage_factor: slippage for contract roll
    :param min_periods: minimal number of days
    :param max_periods: max number of days for rolling
    :param per_asset: calculate stats per asset
    :return: xarray with all statistics
    """
    track_event("CALC_STAT")

    if points_per_year is None:
        points_per_year = calc_avg_points_per_year(data)

    if max_periods is None:
        max_periods = len(data.time)

    if slippage_factor is None:
        slippage_factor = get_default_slippage(data)

    if roll_slippage_factor is None:
        roll_slippage_factor = get_default_slippage(data)

    missed_dates = find_missed_dates(portfolio_history, data)
    if len(missed_dates) > 0:
        log_err("WARNING: some dates are missed in the portfolio_history")

    portfolio_history = output_normalize(portfolio_history, per_asset)

    non_liquid = calc_non_liquid(data, portfolio_history)
    if len(non_liquid.coords[ds.TIME]) > 0:
        log_err("WARNING: Strategy trades non-liquid assets.")

    RR = calc_relative_return(data, portfolio_history, slippage_factor, roll_slippage_factor, per_asset,
                              points_per_year)

    E = calc_equity(RR)
    V = calc_volatility_annualized(RR, max_periods=max_periods, min_periods=min_periods,
                                   points_per_year=points_per_year)
    U = calc_underwater(E)
    DD = calc_max_drawdown(U)
    SR = calc_sharpe_ratio_annualized(RR, max_periods=max_periods, min_periods=min_periods,
                                      points_per_year=points_per_year)
    MR = calc_mean_return_annualized(RR, max_periods=max_periods, min_periods=min_periods,
                                     points_per_year=points_per_year)
    adj_data, adj_ph = arrange_data(data, portfolio_history, per_asset)
    B = calc_bias(adj_ph, per_asset)
    I = calc_instruments(adj_ph, per_asset)
    T = calc_avg_turnover(adj_ph, E, adj_data,
                          min_periods=min_periods,
                          max_periods=max_periods,
                          per_asset=per_asset,
                          points_per_year=points_per_year
                          )

    HT = calc_avg_holding_time(adj_ph,  # E, adj_data,
                               min_periods=min_periods, max_periods=max_periods,
                               per_asset=per_asset,
                               points_per_year=points_per_year)

    stat = xr.concat([
        E, RR, V,
        U, DD, SR,
        MR, B, I, T, HT
    ], pd.Index([
        stf.EQUITY, stf.RELATIVE_RETURN, stf.VOLATILITY,
        stf.UNDERWATER, stf.MAX_DRAWDOWN, stf.SHARPE_RATIO,
        stf.MEAN_RETURN, stf.BIAS, stf.INSTRUMENTS, stf.AVG_TURNOVER, stf.AVG_HOLDINGTIME
    ], name=ds.FIELD))

    dims = [ds.TIME, ds.FIELD]
    if per_asset:
        dims.append(ds.ASSET)
    return stat.transpose(*dims)


def calc_sector_distribution(portfolio_history, timeseries=None, kind=None):
    """
    :param portfolio_history: portfolio weights set for every day
    :param timeseries: time range
    :param kind: 'stocks' or 'futures'
    :return: sector distribution
    """
    ph = abs(portfolio_history.transpose(ds.TIME, ds.ASSET)).fillna(0)
    s = ph.sum(ds.ASSET)
    s[s < 1] = 1
    ph = ph / s

    if timeseries is not None:  # arrange portfolio to timeseries
        _ph = xr.DataArray(np.full([len(timeseries), len(ph.coords[ds.ASSET])], 0, dtype=np.float64),
                           dims=[ds.TIME, ds.ASSET],
                           coords={
                               ds.TIME: timeseries,
                               ds.ASSET: ph.coords[ds.ASSET]
                           })
        intersection = np.intersect1d(timeseries, ph.coords[ds.TIME], True)
        _ph.loc[intersection] = ph.loc[intersection]
        ph = _ph.ffill(ds.TIME).fillna(0)

    max_date = str(portfolio_history.coords[ds.TIME].max().values)[0:10]
    min_date = str(portfolio_history.coords[ds.TIME].min().values)[0:10]

    if kind is None:
        kind = portfolio_history.name

    if kind == 'stocks_nasdaq100':
        assets = stocks_load_ndx_list(min_date=min_date, max_date=max_date)
    elif kind == 'stocks' or kind == 'stocks_long':
        assets = stocks_load_list(min_date=min_date, max_date=max_date)
    elif kind == 'futures':
        assets = futures_load_list()
    else:
        assets = []

    assets = dict((a['id'], a) for a in assets)

    sectors = []

    SECTOR_FIELD = 'sector'

    for aid in portfolio_history.coords[ds.ASSET].values:
        sector = "Other"
        if aid in assets:
            asset = assets[aid]
            s = asset[SECTOR_FIELD]
            if s is not None and s != 'n/a' and s != '':
                sector = s
        sectors.append(sector)

    uq_sectors = sorted(list(set(sectors)))
    sectors = np.array(sectors)

    CASH_SECTOR = 'Cash'
    sector_distr = xr.DataArray(
        np.full([len(ph.coords[ds.TIME]), len(uq_sectors) + 1], 0, dtype=np.float64),
        dims=[ds.TIME, SECTOR_FIELD],
        coords={
            ds.TIME: ph.coords[ds.TIME],
            SECTOR_FIELD: uq_sectors + [CASH_SECTOR]
        }
    )

    for sector in uq_sectors:
        sum_by_sector = ph.loc[:, sectors == sector].sum(ds.ASSET)
        sector_distr.loc[:, sector] = sum_by_sector

    sector_distr.loc[:, CASH_SECTOR] = 1 - ph.sum(ds.ASSET)

    return sector_distr


def check_correlation(portfolio_history, data, print_stack_trace=True):
    """ Checks correlation for current output. """
    track_event("CHECK_CORRELATION")
    portfolio_history = output_normalize(portfolio_history)
    rr = calc_relative_return(data, portfolio_history)

    try:
        cr_list = calc_correlation(rr, False)
    except:
        import logging
        if print_stack_trace:
            logging.exception("Correlation check failed.")
        else:
            log_err("Correlation check failed.")
        return

    log_info()

    if len(cr_list) == 0:
        log_info("Ok. This strategy does not correlate with other strategies.")
        return

    log_err("WARNING! This strategy correlates with other strategies and will be rejected.")
    log_err("Modify the strategy to produce the different output.")
    log_info("The number of systems with a larger Sharpe ratio and correlation larger than 0.9:", len(cr_list))
    log_info("The max correlation value (with systems with a larger Sharpe ratio):",
             max([i['cofactor'] for i in cr_list]))
    my_cr = [i for i in cr_list if i['my']]

    log_info("Current sharpe ratio(3y):",
             calc_sharpe_ratio_annualized(rr, calc_avg_points_per_year(data) * 3)[-1].values.item())

    log_info()

    if len(my_cr) > 0:
        log_info("My correlated submissions:\n")
        headers = ['Name', "Coefficient", "Sharpe ratio"]
        rows = []

        for i in my_cr:
            rows.append([i['name'], i['cofactor'], i['sharpe_ratio']])

        log_info(tabulate(rows, headers))

    ex_cr = [i for i in cr_list if i['template']]
    if len(ex_cr) > 0:
        log_info("Correlated examples:\n")
        headers = ['Name', "Coefficient", "Sharpe ratio"]
        rows = []

        for i in ex_cr:
            rows.append([i['name'], i['cofactor'], i['sharpe_ratio']])

        log_info(tabulate(rows, headers))


print_correlation = check_correlation


def calc_correlation(relative_returns, suppress_exception=True):
    try:
        if "SUBMISSION_ID" in os.environ and os.environ["SUBMISSION_ID"] != "":
            log_info("correlation check disabled")
            return []

        ENGINE_CORRELATION_URL = get_env("ENGINE_CORRELATION_URL",
                                         "https://quantiacs.io/referee/submission/forCorrelation")
        STATAN_CORRELATION_URL = get_env("STATAN_CORRELATION_URL", "https://quantiacs.io/statan/correlation")
        PARTICIPANT_ID = get_env("PARTICIPANT_ID", "0")

        with request.urlopen(ENGINE_CORRELATION_URL + "?participantId=" + PARTICIPANT_ID) as response:
            submissions = response.read()
            submissions = json.loads(submissions)
            submission_ids = [s['id'] for s in submissions]

        rr = relative_returns.to_netcdf(compute=True)
        rr = gzip.compress(rr)
        rr = base64.b64encode(rr)
        rr = rr.decode()

        cofactors = []

        chunks = [submission_ids[x:x + 50] for x in range(0, len(submission_ids), 50)]

        for c in chunks:
            r = {"relative_returns": rr, "submission_ids": c}
            r = json.dumps(r)
            r = r.encode()
            with request.urlopen(STATAN_CORRELATION_URL, r) as response:
                cs = response.read()
                cs = json.loads(cs)
                cofactors = cofactors + cs

        result = []
        for c in cofactors:
            sub = next((s for s in submissions if str(c['id']) == str(s['id'])))
            sub['cofactor'] = c['cofactor']
            sub['sharpe_ratio'] = c['sharpe_ratio']
            result.append(sub)
        return result
    except Exception as e:
        log_err("WARNING! Can't calculate correlation.")
        if suppress_exception:
            import logging
            logging.exception("network error")
            return []
        else:
            raise e


def check_exposure(portfolio_history,
                   soft_limit=0.05, hard_limit=0.1,
                   days_tolerance=0.02, excess_tolerance=0.02,
                   avg_period=252, check_period=252 * 5
                   ):
    """
    Checks exposure according to the submission filters.
    :param portfolio_history: output DataArray
    :param soft_limit: soft limit for exposure
    :param hard_limit: hard limit for exposure
    :param days_tolerance: the number of days when exposure may be in range 0.05..0.1
    :param excess_tolerance: max allowed average excess
    :param avg_period: period for the ratio calculation
    :param check_period: period for checking
    :return:
    """
    portfolio_history = portfolio_history.loc[{ds.TIME: np.sort(portfolio_history.coords[ds.TIME])}]

    exposure = calc_exposure(portfolio_history)
    max_exposure = exposure.max(ds.ASSET)

    max_exposure_over_limit = max_exposure.where(max_exposure > soft_limit).dropna(ds.TIME)
    if len(max_exposure_over_limit) > 0:
        max_exposure_asset = exposure.sel({ds.TIME: max_exposure_over_limit.coords[ds.TIME]}).idxmax(ds.ASSET)
        log_info("Positions with max exposure over the limit:")
        pos = xr.concat([max_exposure_over_limit, max_exposure_asset], pd.Index(['exposure', 'asset'], name='field'))
        log_info(pos.to_pandas().T)

    periods = min(avg_period, len(portfolio_history.coords[ds.TIME]))

    bad_days = xr.where(max_exposure > soft_limit, 1.0, 0.0)
    bad_days_proportion = bad_days[-check_period:].rolling(dim={ds.TIME: periods}).mean()
    days_ok = xr.where(bad_days_proportion > days_tolerance, 1, 0).sum().values == 0

    excess = exposure - soft_limit
    excess = excess.where(excess > 0, 0).sum(ds.ASSET)
    excess = excess[-check_period:].rolling(dim={ds.TIME: periods}).mean()
    excess_ok = xr.where(excess > excess_tolerance, 1, 0).sum().values == 0

    hard_limit_ok = xr.where(max_exposure > hard_limit, 1, 0).sum().values == 0

    if hard_limit_ok and (days_ok or excess_ok):
        log_info("Ok. The exposure check succeed.")
        return True
    else:
        log_err("WARNING! The exposure check failed.")
        log_info("Hard limit check: ", 'Ok.' if hard_limit_ok else 'Failed.')
        log_info("Days check: ", 'Ok.' if days_ok else 'Failed.')
        log_info("Excess check:", 'Ok.' if excess_ok else 'Failed.')
        return False


def calc_exposure(portfolio_history):
    """
    Calculates exposure per position (range: 0..1)
    :param portfolio_history:
    :return:
    """
    sum = abs(portfolio_history).sum(ds.ASSET)
    sum = sum.where(sum > EPS, 1)  # prevents div by zero
    return abs(portfolio_history) / sum
