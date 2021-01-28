import typing as tp
import datetime
import os
import logging
from urllib.parse import urljoin
import urllib.error
import sys
import urllib.request
import time
import json
import math
import gzip
import xarray as xr
import numpy as np
import pandas as pd
import re
from qnt.log import log_info, log_err
import pickle, hashlib
import io
import progressbar

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
            log_err("NOTICE: The environment variable " + key + " was not specified. The default value is '" + def_val + "'")
        return def_val


ACCESS_KEY = get_env('API_KEY', '')
BASE_URL = get_env('DATA_BASE_URL', 'https://data-api.quantiacs.io/')

def request_with_retry(uri, data):
    url = urljoin(BASE_URL, uri)
    cached = cache_get(uri, data)
    if cached is not None:
        return cached
    retries = sys.maxsize if "SUBMISSION_ID" in os.environ else 5
    for r in range(0, retries):
        try:
            req = urllib.request.Request(url, data, headers={'Accept-Encoding': 'gzip', "X-Api-Key": api_key})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                length = response.getheader('content-length')
                if length:
                    length = int(length)
                    blocksize = max(4096, length//100)
                else:
                    blocksize = 4096
                    length = None
                buf = io.BytesIO()
                size = 0
                sys.stdout.flush()
                with progressbar.ProgressBar(max_value=length, poll_interval=1) as p:
                    while True:
                        buf1 = response.read(blocksize)
                        if not buf1:
                            break
                        buf.write(buf1)
                        size += len(buf1)
                        p.update(size)
                sys.stderr.flush()
                response_body = buf.getvalue()
                if response.getheader('Content-Encoding') == 'gzip':
                    response_body = gzip.decompress(response_body)
                cache_put(response_body, uri, data)
                return response_body
        except KeyboardInterrupt:
            raise
        except:
            logging.exception("download error: " + uri)
            time.sleep(RETRY_DELAY)
    raise Exception("can't download " + uri)


def parse_date(dt: tp.Union[None, str, datetime.datetime, datetime.date]) -> datetime.date:
    if dt is None:
        res = datetime.date.today()
    else:
        res = pd.Timestamp(dt).date()
    if MAX_DATE_LIMIT is not None:
        if res is not None:
            res = min(MAX_DATE_LIMIT, res)
        else:
            res = MAX_DATE_LIMIT
    return res


def parse_tail(tail: tp.Union[datetime.timedelta, int]):
    return tail if type(tail) == datetime.timedelta else datetime.timedelta(days=tail)


def parse_date_and_hour(dt: tp.Union[None, str, datetime.datetime, datetime.date]) -> datetime.datetime:
    print(dt)
    if dt is None:
        res = datetime.datetime.now()
    else:
        res = pd.Timestamp(dt).to_pydatetime()
    if MAX_DATETIME_LIMIT is not None:
        if res is not None:
            res = min(MAX_DATETIME_LIMIT, res)
        else:
            res = MAX_DATETIME_LIMIT
    return res


def datetime_to_hours_str(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H")


def parse_max_datetime_from_url(url):
    r = re.compile("^.+/(\\d{4}-\\d{2}-\\d{2}T\\d{2})/{0,1}$")
    m = r.match(url)
    if m is not None:
        return parse_date_and_hour(m.group(1))
    r = re.compile("^.+/(\\d{4}-\\d{2}-\\d{2})/{0,1}$")
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
        log_err('WARNING: ' + deprecated_name + ' deprecated, use ' + origin.__module__ + '.' + origin.__name__)
        return origin(*args, **kwargs)

    return wrap


CACHE_RETENTION = datetime.timedelta(days=float(get_env('CACHE_RETENTION', '7')))
CACHE_DIR = get_env('CACHE_DIR', 'data-cache')

def cache_get(*args):
    crop_cache()
    p = pickle.dumps(args)
    key = hashlib.sha1(p).hexdigest()
    value_fn = os.path.join(CACHE_DIR, key + ".value.pickle.gz")
    args_fn = os.path.join(CACHE_DIR, key + ".args.pickle.gz")
    if os.path.exists(value_fn) and os.path.exists(args_fn):
        try:
            old_args = pickle.load(gzip.open(args_fn, 'rb'))
            if old_args == args:
                old_data = pickle.load(gzip.open(value_fn, 'rb'))
                return old_data
        except Exception as e:
            log_err("Cache read problem:", e)
    return None


def cache_put(value, *args):
    if CACHE_RETENTION.total_seconds() == 0:
        return
    p = pickle.dumps(args)
    key = hashlib.sha1(p).hexdigest()
    value_fn = os.path.join(CACHE_DIR, key + ".value.pickle.gz")
    args_fn = os.path.join(CACHE_DIR, key + ".args.pickle.gz")
    pickle.dump(args, gzip.open(args_fn, 'wb', compresslevel=5))
    pickle.dump(value, gzip.open(value_fn, 'wb', compresslevel=5))


def crop_cache():
    global cache_min_mod_time
    now = datetime.datetime.now()
    if cache_min_mod_time is not None and datetime.datetime.now() - cache_min_mod_time < CACHE_RETENTION:
        return
    cache_min_mod_time = None
    for fn in os.listdir(CACHE_DIR):
        full_name = os.path.join(CACHE_DIR, fn)
        if not os.path.isfile(full_name):
            continue
        m_time = os.path.getmtime(full_name)
        m_time = datetime.datetime.fromtimestamp(m_time)
        if now - m_time > CACHE_RETENTION:
            os.remove(full_name)
        else:
            if cache_min_mod_time is None or cache_min_mod_time > m_time:
                cache_min_mod_time = m_time


cache_min_mod_time = None
os.makedirs(CACHE_DIR, exist_ok=True)


if MAX_DATE_LIMIT is None:
    MAX_DATETIME_LIMIT = parse_max_datetime_from_url(BASE_URL)
    MAX_DATE_LIMIT = None if MAX_DATETIME_LIMIT is None else MAX_DATETIME_LIMIT.date()

api_key = os.environ.get("API_KEY", '').strip()
tracking_host = os.environ.get("TRACKING_HOST", "https://quantiacs.io")
if api_key != 'default':
    if api_key == '':
        log_err("Please, specify the API_KEY.")
        log_err("See: https://quantiacs.io/documentation/en/user_guide/local_development.html")
        exit(1)
    else:
        url = tracking_host + "/auth/system/account/accountByKey?apiKey=" + api_key
        print(url)
        try:
            resp = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log_err("Wrong API_KEY.")
                log_err("See: https://quantiacs.io/documentation/en/user_guide/local_development.html")
                exit(1)
sent_events = set()


def track_event(event):
    if os.environ.get("SUBMISSION_ID", '') != '':
        return
    if event in sent_events:
        return
    sent_events.add(event)
    import threading
    url = tracking_host + '/engine/tracklib?apiKey=' + api_key + '&event=' + event
    if 'STRATEGY_ID' in os.environ:
        url = url + '&strategyId=' + os.environ.get('STRATEGY_ID', '')
    t = threading.Thread(target=get_url_silent, args=(url,))
    t.start()


def get_url_silent(url):
    try:
        urllib.request.urlopen(url)
    except:
        pass


if __name__ == '__main__':
    log_info(parse_max_datetime_from_url('http://hl.datarelay:7070/last/2020-10-07T10/'))
    log_info(parse_max_datetime_from_url('http://hl.datarelay:7070/last/2016-10-28/'))
    # t = parse_max_datetime_from_url('http://hl.datarelay:7070/last/2020-10-07T10/')
    # print(datetime.datetime.combine(t.date(), datetime.time.min))


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
# ///
