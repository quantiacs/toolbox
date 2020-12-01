import typing as tp
import datetime
import os
import logging
from urllib.parse import urljoin
import sys
import urllib.request
import time
import json
import math
import xarray as xr
import numpy as np
import pandas as pd
import re

MAX_DATE_LIMIT: tp.Union[datetime.date, None] = None
MAX_DATETIME_LIMIT: tp.Union[datetime.datetime, None] = None

DEFAULT_TAIL = 6 * 365


class Fields:
    OPEN = "open"
    LOW = "low"
    HIGH = "high"
    CLOSE = "close"
    VOL = "vol"
    DIVS = "divs"  # only for stocks
    SPLIT = "split"  # only for stocks
    SPLIT_CUMPROD = "split_cumprod"  # only for stocks
    IS_LIQUID = "is_liquid"  # only for stocks
    OPEN_INTEREST = "oi"  # only for futures
    ROLL = "roll"  # only for futures


f = Fields


class Dimensions:
    TIME = 'time'
    FIELD = 'field'
    ASSET = 'asset'


ds = Dimensions

TIMEOUT = 60
RETRY_DELAY = 1


def get_env(key, def_val, silent=False):
    if key in os.environ:
        return os.environ[key]
    else:
        if not silent:
            print("WARNING: env is not set " + key)
        return def_val


BASE_URL = get_env('DATA_BASE_URL', 'http://127.0.0.1:8000/')


def request_with_retry(uri, data):
    url = urljoin(BASE_URL, uri)
    retries = sys.maxsize if "SUBMISSION_ID" in os.environ else 5
    for r in range(0, retries):
        try:
            with urllib.request.urlopen(url, data, timeout=TIMEOUT) as response:
                response_body = response.read()
                return response_body
        except KeyboardInterrupt:
            raise
        except:
            logging.exception("download error " + uri)
            time.sleep(RETRY_DELAY)
    raise Exception("can't download " + uri)


def parse_date(dt: tp.Union[None, str, datetime.datetime, datetime.date]) -> datetime.date:
    if dt is None:
        try:
            return parse_date_and_hour(dt).date()
        except:
            return datetime.date.today()
    if isinstance(dt, np.datetime64):
        return pd.Timestamp(dt).date()
    if isinstance(dt, str):
        return datetime.datetime.strptime(dt + "Z+00:00", "%Y-%m-%dZ%z").date()
    if isinstance(dt, datetime.datetime):
        dt = datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc)  # rm timezone
        return dt.date()
    if isinstance(dt, datetime.date):
        return dt
    raise Exception("invalid date " + str(type(dt)))


def parse_tail(tail: tp.Union[datetime.timedelta, int]):
    return tail if type(tail) == datetime.timedelta else datetime.timedelta(days=tail)


def parse_date_and_hour(dt: tp.Union[None, str, datetime.datetime, datetime.date]) -> datetime.datetime:
    if dt is None:
        try:
            dt = BASE_URL.split("/")[-2]
            return parse_date_and_hour(dt)
        except:
            return datetime.datetime.now(tz=datetime.timezone.utc)
    if isinstance(dt, np.datetime64):
        return pd.Timestamp(dt)
    if isinstance(dt, datetime.date):
        return datetime.datetime(dt.year, dt.month, dt.day, tzinfo=datetime.timezone.utc)
    if isinstance(dt, datetime.datetime):
        dt = datetime.datetime.fromtimestamp(dt.timestamp(), tz=datetime.timezone.utc)  # rm timezone
        dt = dt.isoformat()
    if isinstance(dt, str):
        dt = dt.split(":")[0]
        if 'T' in dt:
            return datetime.datetime.strptime(dt + "Z+00:00", "%Y-%m-%dT%HZ%z")
        else:
            return datetime.datetime.strptime(dt + "Z+00:00", "%Y-%m-%dZ%z")
    raise Exception("invalid date " + str(type(dt)))


def datetime_to_hours_str(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H")


# TODO Strange stuff, need to check usage

def from_xarray_3d_to_dict_of_pandas_df(xarray_data):
    assets_names = xarray_data.coords[ds.ASSET].values
    pandas_df_dict = {}
    for asset_name in assets_names:
        pandas_df_dict[asset_name] = xarray_data.loc[:, :, asset_name].to_pandas()

    return pandas_df_dict


def from_dict_to_xarray_1d(weights):
    weights_assets_list = [key for key in weights]
    weights_values_list = [weights[key] for key in weights]

    return xr.DataArray(weights_values_list, dims=[ds.ASSET], coords={ds.ASSET: weights_assets_list})


def filter_liquids_xarray_assets_dataarray(assets_xarray_dataarray):
    liquid_xarray_assets_dataarray = assets_xarray_dataarray \
        .where(assets_xarray_dataarray.loc[:, 'is_liquid', :] == 1) \
        .dropna(ds.TIME, 'all').dropna(ds.ASSET, 'all')

    return liquid_xarray_assets_dataarray


def check_weights_xarray_dataarray_for_nonliquids(xarray_weights_dataarray, xarray_assets_dataarray):
    non_liquid_weights = xarray_weights_dataarray.where(xarray_assets_dataarray[0].loc['is_liquid', :] == 0)
    non_liquid_weights = non_liquid_weights.where(non_liquid_weights != 0)
    non_liquid_weights = non_liquid_weights.dropna(ds.ASSET)
    if len(non_liquid_weights) > 0:
        raise Exception(non_liquid_weights.coords[ds.ASSET].values)


def exclude_weights_xarray_dataarray_from_nonliquids(weights_xarray_dataarray, assets_xarray_dataarray):
    liquid_weights_xarray_dataarray = weights_xarray_dataarray \
        .where(assets_xarray_dataarray[0].loc['is_liquid', :] == 1) \
        .dropna(ds.ASSET, 'all')

    return liquid_weights_xarray_dataarray


def parse_max_datetime_from_url(url):
    r = re.compile("^.+/(\\d{4}-\\d{2}-\\d{2})/{0,1}$")
    m = r.match(url)
    if m is not None:
        return parse_date_and_hour(m.group(1))
    r = re.compile("^.+/(\\d{4}-\\d{2}-\\d{2})T\\d{2}/{0,1}$")
    m = r.match(url)
    if m is not None:
        return parse_date_and_hour(m.group(1))
    return None


def deprecated_wrap(origin):
    import sys, traceback
    stack = traceback.extract_stack(limit=2)
    deprecated_name = stack[-2][3].split("=")[0].strip()

    try:
        f = sys._getframe(1)
        deprecated_name = f.f_locals['__name__'] + '.' + deprecated_name
    except:
        pass

    def wrap(*args, **kwargs):
        print('WARNING: ' + deprecated_name + ' deprecated, use ' + origin.__module__ + '.' + origin.__name__,
              file=sys.stderr, flush=True)
        return origin(*args, **kwargs)

    return wrap


if MAX_DATE_LIMIT is None:
    MAX_DATETIME_LIMIT = parse_max_datetime_from_url(BASE_URL)
    MAX_DATE_LIMIT = None if MAX_DATETIME_LIMIT is None else MAX_DATETIME_LIMIT.date()

if __name__ == '__main__':
    print(parse_max_datetime_from_url('http://hl.datarelay:7070/last/2020-10-07T10/'))
    print(parse_max_datetime_from_url('http://hl.datarelay:7070/last/2016-10-28/'))
    # t = parse_max_datetime_from_url('http://hl.datarelay:7070/last/2020-10-07T10/')
    # print(datetime.datetime.combine(t.date(), datetime.time.min))
