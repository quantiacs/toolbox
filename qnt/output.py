import numpy as np
import xarray as xr
import pandas as pd
import gzip

from qnt.log import log_info, log_err, Settings as LogSettings


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


def clean(output, data, kind=None, debug=True):
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
    from qnt.data.common import ds, f, track_event

    if kind is None:
        kind = data.name

    output = output.drop(ds.FIELD, errors='ignore')

    with LogSettings(err2info=True):
        log_info("Output cleaning...")

        single_day = ds.TIME not in output.dims

        if not single_day:
            track_event("OUTPUT_CLEAN")

        log_info("fix uniq")
        if not single_day:
            # uniq time fix
            val,idx = np.unique(output.time, return_index=True)
            output = output.isel(time=idx)
        # uniq asset fix
        val,idx = np.unique(output.asset, return_index=True)
        output = output.isel(asset=idx)

        if single_day:
            output = output.drop(ds.TIME, errors='ignore')
            output = xr.concat([output], pd.Index([data.coords[ds.TIME].values.max()], name=ds.TIME))
        else:
            log_info("ffill if the current price is None...")
            output = output.fillna(0)
            output = output.where(np.isfinite(data.sel(field='close')))
            output = output.ffill('time')
            output = output.fillna(0)

        if kind == "stocks" or kind == "stocks_long":
            log_info("Check liquidity...")
            non_liquid = qns.calc_non_liquid(data, output)
            if len(non_liquid.coords[ds.TIME]) > 0:
                log_info("WARNING! Strategy trades non-liquid assets.")
                log_info("Fix liquidity...")
                is_liquid = data.sel(field=f.IS_LIQUID)
                is_liquid = xr.align(is_liquid, output, join='right')[0]
                output = xr.where(is_liquid == 0, 0, output)
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
                log_info("WARNING! Output contains negative positions. Clean...")
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
                log_info("WARNING! Output contains not only BTC.")
                log_info("Fixing...")
                output=output.sel(asset=['BTC'])
            else:
                log_info("Ok.")

        log_info("Normalization...")
        output = normalize(output)
        log_info("Output cleaning is complete.")

    return output


def check(output, data, kind=None):
    """
    This function checks your output and warn you if it contains errors.
    :return:
    """
    import qnt.stats as qns
    from qnt.data.common import ds, f, get_env, track_event

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
                log_err("Multiply the output by data.sel(field='is_liquid') or use qnt.output.clean")
            else:
                log_info("Ok.")

        if not single_day:
            log_info("Check missed dates...")
            missed_dates = qns.find_missed_dates(output, data)
            if len(missed_dates) > 0:
                log_err("ERROR! Some dates were missed)")
                log_err("Your strategy dropped some days, your strategy should produce a continuous series.")
            else:
                log_info("Ok.")
            track_event("OUTPUT_CHECK")

        if kind == "stocks" or kind == "stocks_long":
            log_info("Check exposure...")
            if not qns.check_exposure(output):
                log_err("Use more assets or/and use qnt.output.clean")

        if kind == "crypto":
            log_info("Check BTC...")
            if output.where(output != 0).dropna("asset", "all").coords[ds.ASSET].values.tolist() != ['BTC']:
                log_err("ERROR! Output contains not only BTC.\n")
                log_err("Remove the other assets from the output or use qnt.output.clean")
            else:
                log_info("Ok.")

        if not single_day:
            if abs(output).sum() == 0:
                log_err("ERROR! Output is empty. All positions are zero.")
            else:
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
                        log_err("ERROR! Output contains negative positions.")
                        log_err("Drop all negative positions.")
                    else:
                        log_info("Ok.")

                log_info("Check the sharpe ratio...")

                sr = calc_sharpe_ratio_for_check(data, output, kind, True)
                log_info("Sharpe Ratio =", sr)

                if sr < 1:
                    log_err("ERROR! The Sharpe Ratio is too low.", sr, '<', 1,)
                    log_err("Improve the strategy and make sure that the in-sample Sharpe Ratio more than 1.")
                else:
                    log_info("Ok.")

                log_info("Check correlation.")
                qns.check_correlation(output, data, False)
    except Exception as e:
        log_err(e)


def calc_sharpe_ratio_for_check(data, output, kind=None, check_dates=True):
    """
    Calculates sharpe ratio for check according to the rules
    :param data:
    :param output:
    :param kind: competition type
    :param check_dates: do you need to check the sharpe ratio dates?
    :return:
    """
    import qnt.stats as qns

    if kind is None:
        kind = data.name

    start_date = qns.get_default_is_start_date_for_type(kind)
    sdd = pd.Timestamp(start_date)
    osd = pd.Timestamp(output.where(abs(output).sum('asset') > 0).dropna('time', 'all').time.min().values)
    dsd = pd.Timestamp(data.time.min().values)
    if check_dates:
        if (dsd - sdd).days > 10:
            log_err("WARNING! There are not enough points in the data")
            log_err("The first point(" + str(dsd.date()) + ") should be earlier than " + str(sdd.date()))
            log_err("Load data more historical data.")
        else:
            if len(data.sel(time=slice(None, sdd)).time) < 15:
                log_err("WARNING! There are not enough points in the data for the slippage calculation.")
                log_err("Add 15 extra data points to the data head (load data more historical data).")
        if (osd - sdd).days > 7:
            log_err("WARNING! There are not enough points in the output.")
            log_err("The output series should start from " + str(sdd.date()) + " or earlier instead of " + str(osd.date()))
    sd = max(sdd, dsd)
    sd = sd.to_pydatetime()
    fd = pd.Timestamp(data.time.max().values).to_pydatetime()
    log_info("Period: " + str(sd.date()) + " - " + str(fd.date()))
    output_slice = align(output, data.time, sd, fd)
    rr = qns.calc_relative_return(data, output_slice)
    sr = qns.calc_sharpe_ratio_annualized(rr)
    sr = sr.isel(time=-1).values
    return sr


def write(output):
    """
    writes output in the file for submission
    :param output: xarray with daily weights
    """
    import qnt.data.id_translation as idt
    from qnt.data.common import ds, get_env, track_event
    output = output.copy()
    output.coords[ds.ASSET] = [idt.translate_user_id_to_server_id(id) for id in output.coords[ds.ASSET].values]
    output = normalize(output)
    data = output.to_netcdf(compute=True)
    data = gzip.compress(data)
    path = get_env("OUTPUT_PATH", "fractions.nc.gz")
    log_info("Write output: " + path)
    with open(path, 'wb') as out:
        out.write(data)
    track_event("OUTPUT_WRITE")


def align(output, time_coord, start=None, end=None):
    """
    Normalizes, aligns the output with the data and cut the piece.
    It is necessary for precise Sharpe ratio calculation.
    :param output: the output array
    :param time_coord: the time coord for align
    :param start: start date
    :param end: end date
    :return: aligned and cut output
    """
    res = normalize(output)
    res = xr.align(res, time_coord, join='right')[0]
    res = res.fillna(0)
    if start is not None:
        res = res.sel(time=slice(start, None))
    if end is not None:
        res = res.sel(time=slice(None, end))
    return res
