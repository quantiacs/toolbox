import numpy as np
import xarray as xr
import pandas as pd
import gzip
from qnt.log import log_info, log_err


def normalize(output, per_asset=False):
    from qnt.data.common import ds
    output = output.where(np.isfinite(output)).fillna(0)
    if ds.TIME in output.dims:
        output = output.transpose(ds.TIME, ds.ASSET)
        output = output.loc[np.sort(output.coords[ds.TIME].values), np.sort(output.coords[ds.ASSET].values)]
    if per_asset:
        output = xr.where(output > 1, 1, output)
        output = xr.where(output < -1, -1, output)
    else:
        s = abs(output).sum(ds.ASSET)
        if ds.TIME in output.dims:
            s[s < 1] = 1
        else:
            s = 1 if s < 1 else s
        output = output / s
    try:
        output = output.drop(ds.FIELD)
    except ValueError:
        pass
    return output


def clean(output, data, kind=None):
    """
    Checks the output and fix common errors:
        - liquidity
        - missed dates
        - exposure
        - normalization
    :param output:
    :param data:
    :param kind:
    :return:
    """
    import qnt.stats as qns
    import qnt.exposure as qne
    from qnt.data.common import ds, f

    if kind is None:
        kind = data.name

    output = output.drop(ds.FIELD, errors='ignore')

    single_day = ds.TIME not in output.dims
    if single_day:
        output = output.drop(ds.TIME, errors='ignore')
        output = xr.concat([output], pd.Index([data.coords[ds.TIME].values.max()], name=ds.TIME))

    if kind == "stocks" or kind == "stocks_long":
        log_info("Check liquidity...")
        non_liquid = qns.calc_non_liquid(data, output)
        if len(non_liquid.coords[ds.TIME]) > 0:
            log_info("ERROR! Strategy trades non-liquid assets.")
            log_info("Fix liquidity...")
            output = output.where(data.sel(field=f.IS_LIQUID).fillna(0) > 0).fillna(0)
        log_info("Ok.")

    if not single_day:
        log_info("Check missed dates...")
        missed_dates = qns.find_missed_dates(output, data)
        if len(missed_dates) > 0:
            log_info("WARNING! Output contain missed dates.")
            log_info("Adding missed dates and set zero...")
            add = xr.concat([output.isel(time=-1)] * len(missed_dates), pd.DatetimeIndex(missed_dates, name="time"))
            add = xr.full_like(add, np.nan)
            output = xr.concat([output, add], dim='time')
            output = output.fillna(0)
            if kind == "stocks" or kind == "stocks_long":
                output = output.where(data.sel(field='is_liquid') > 0)
            output = output.dropna('asset', 'all').dropna('time', 'all').fillna(0)
            output = normalize(output)
        else:
            log_info("Ok.")

    if kind == 'stocks_long':
        log_info("Check positive positions...")
        neg = output.where(output < 0).dropna(ds.TIME, 'all')
        if len(neg.time) > 0:
            log_info("Output contains negative positions. Clean...")
            output = output.where(output >= 0).fillna(0)
        else:
            log_info("Ok.")

    if kind == "stocks" or kind == "stocks_long":
        log_info("Check exposure...")
        if not qns.check_exposure(output):
            log_info("Cut big positions...")
            output = qne.cut_big_positions(output)
            log_info("Check exposure...")
            if not qns.check_exposure(output):
                log_info("Drop bad days...")
                output = qne.drop_bad_days(output)

    if kind == "crypto":
        log_info("Check BTC...")
        if output.where(output != 0).dropna("asset", "all").coords[ds.ASSET].values.tolist() != ['BTC']:
            log_info("ERROR! Output contains not only BTC.")
            log_info("Fixing...")
            output=output.sel(asset=['BTC'])
        else:
            log_info("Ok.")

    log_info("Normalization...")
    output = normalize(output)
    log_info("Done.")

    return output


def check(output, data, kind=None):
    """
    This function checks your output and warn you if it contains errors.
    :return:
    """
    import qnt.stats as qns
    from qnt.data.common import ds, f, get_env

    if kind is None:
        kind = data.name

    single_day = ds.TIME not in output.dims
    if single_day:
        output = xr.concat([output], pd.Index([data.coords[ds.TIME].values.max()], name=ds.TIME))

    try:
        if kind == "stocks" or kind == "stocks_long":
            log_info("Check liquidity...")
            non_liquid = qns.calc_non_liquid(data, output)
            if len(non_liquid.coords[ds.TIME]) > 0:
                log_err("ERROR! Strategy trades non-liquid assets.")
            else:
                log_info("Ok.")

        if not single_day:
            log_info("Check missed dates...")
            missed_dates = qns.find_missed_dates(output, data)
            if len(missed_dates) > 0:
                log_err("ERROR! Some dates were missed.")
            else:
                log_info("Ok.")

        if kind == "stocks" or kind == "stocks_long":
            log_info("Check exposure...")
            qns.check_exposure(output)

        if kind == "crypto":
            log_info("Check BTC...")
            if output.where(output != 0).dropna("asset", "all").coords[ds.ASSET].values.tolist() != ['BTC']:
                log_err("ERROR! Output contains not only BTC.")
            else:
                log_info("Ok.")

        if not single_day:
            # if kind == 'crypto' or kind == 'cryptofutures' or kind == 'crypto_futures':
            #     log_info("Check holding time...")
            #     ht = qns.calc_avg_holding_time(output)
            #     ht = ht.isel(time=-1).values
            #     if ht < 4:
            #         log_err("ERROR! The holding time is too low.", ht, "<", 4)
            #     else:
            #         log_info("Ok.")
            #
            # if kind == 'stocks_long':
            #     log_info("Check holding time...")
            #     ht = qns.calc_avg_holding_time(output)
            #     ht = ht.isel(time=-1).values
            #     if ht < 15:
            #         log_err("ERROR! The holding time is too low.", ht, "<", 15)
            #     else:
            #         log_info("Ok.")

            if kind == 'stocks_long':
                log_info("Check positive positions...")
                neg = output.where(output < 0).dropna(ds.TIME, 'all')
                if len(neg.time) > 0:
                    log_err("ERROR! Output contains negative positions")
                else:
                    log_info("Ok.")

            rr = qns.calc_relative_return(data, output)
            sr = qns.calc_sharpe_ratio_annualized(rr, max_periods=qns.get_default_is_period_for_type(kind))
            sr = sr.isel(time=-1).values
            log_info("Check sharpe ratio.")
            if sr < 1:
                log_err("ERROR! The sharpe ratio is too low.", sr, '<', 1)
            else:
                log_info("Ok.")

            log_info("Check correlation.")
            qns.check_correlation(output, data, False)

    except Exception as e:
        log_err(e)


def write(output):
    """
    writes output in the file for submission
    :param output: xarray with daily weights
    """
    import qnt.data.id_translation as idt
    from qnt.data.common import ds, get_env
    output = output.copy()
    output.coords[ds.ASSET] = [idt.translate_user_id_to_server_id(id) for id in output.coords[ds.ASSET].values]
    output = normalize(output)
    data = output.to_netcdf(compute=True)
    data = gzip.compress(data)
    path = get_env("OUTPUT_PATH", "fractions.nc.gz")
    log_info("Write output: " + path)
    with open(path, 'wb') as out:
        out.write(data)