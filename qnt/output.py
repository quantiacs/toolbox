import numpy as np
import xarray as xr
import pandas as pd
import gzip
import sys


def normalize(output, per_asset=False):
    from qnt.data.common import ds
    output = output.where(np.isfinite(output)).where(output != 0).dropna(ds.ASSET, 'all').fillna(0)
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

    if f.IS_LIQUID in data.coords[ds.FIELD]:
        print("Check liquidity...")
        non_liquid = qns.calc_non_liquid(data, output)
        if len(non_liquid.coords[ds.TIME]) > 0:
            print("ERROR! Strategy trades non-liquid assets.")
            print("Fix liquidity...")
            output = output.where(data.sel(field=f.IS_LIQUID).fillna(0) > 0).fillna(0)
        print("Ok.")

    print("Check missed dates...")
    missed_dates = qns.find_missed_dates(output, data)
    if len(missed_dates) > 0:
        print("WARNING! Output contain missed dates.")
        print("Adding missed dates and ffill...")
        add = xr.concat([output.isel(time=-1)] * len(missed_dates), pd.DatetimeIndex(missed_dates, name="time"))
        output = xr.concat([output, add], dim='time')
        output = normalize(output)
        output = output.ffill('time')
        if kind == "stocks":
            output = output.where(data.sel(field='is_liquid') > 0)
        output = output.dropna('asset', 'all').dropna('time', 'all').fillna(0)
        output = normalize(output)
    else:
        print("Ok.")

    if kind == "stocks":
        print("Check exposure...")
        if not qns.check_exposure(output):
            print("Cut big positions...")
            output = qne.cut_big_positions(output)
            print("Check exposure...")
            if not qns.check_exposure(output):
                print("Drop bad days...")
                output = qne.drop_bad_days(output)

    if kind == "crypto":
        print("Check BTC...")
        if output.where(output != 0).dropna("asset", "all").coords[ds.ASSET].values.tolist() != ['BTC']:
            print("ERROR! Output contains not only BTC.")
            print("Fixing...")
            output=output.sel(asset=['BTC'])
        else:
            print("Ok.")

    print("Normalization...")
    output = normalize(output)
    print("Done.")

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

    try:
        if f.IS_LIQUID in data.coords[ds.FIELD]:
            print("Check liquidity...")
            non_liquid = qns.calc_non_liquid(data, output)
            if len(non_liquid.coords[ds.TIME]) > 0:
                print("ERROR! Strategy trades non-liquid assets.", file=sys.stderr, flush=True)
            else:
                print("Ok.")

        print("Check missed dates...")
        missed_dates = qns.find_missed_dates(output, data)
        if len(missed_dates) > 0:
            print("ERROR! Some dates were missed.", file=sys.stderr, flush=True)
        else:
            print("Ok.")

        if kind == "stocks":
            print("Check exposure...")
            qns.check_exposure(output)

        if kind == "crypto":
            print("Check BTC...")
            if output.where(output != 0).dropna("asset", "all").coords[ds.ASSET].values.tolist() != ['BTC']:
                print("ERROR! Output contains not only BTC.", file=sys.stderr, flush=True)
            else:
                print("Ok.")
            print("Check holding time...")
            ht = qns.calc_avg_holding_time(output)
            ht = ht.isel(time=-1).values
            if ht < 4:
                print("ERROR! The holding time is too low.", ht, "<", 4, file=sys.stderr, flush=True)
            else:
                print("Ok.")

        if kind == "cryptofutures":
            print("Check holding time...")
            ht = qns.calc_avg_holding_time(output)
            ht = ht.isel(time=-1).values
            if ht < 4:
                print("ERROR! The holding time is too low.", ht, "<", 4, file=sys.stderr, flush=True)
            else:
                print("Ok.")

        rr = qns.calc_relative_return(data, output)
        sr = qns.calc_sharpe_ratio_annualized(rr)
        sr = sr.isel(time=-1).values
        print("Check sharpe ratio.")
        if sr < 1:
            print("ERROR! The sharpe ratio is too low.", sr, '<', 1, file=sys.stderr, flush=True)
        else:
            print("Ok.")

        print("Check correlation.")
        qns.check_correlation(output, data, False)

    except Exception as e:
        print(e, file=sys.stderr, flush=True)


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
    print("write output: " + path)
    with open(path, 'wb') as out:
        out.write(data)