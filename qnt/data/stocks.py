import qnt.data.id_translation as idt
import sys

from qnt.log import log_info, log_err
from qnt.data.common import *


def load_list(
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = 4 * 365,
        stocks_type:tp.Union[str, int] = ''
):
    """
    :return: list of dicts with info for all tickers
    """
    track_event("DATA_STOCKS_LIST")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    if min_date > max_date:
        raise Exception("min_date must be less than or equal to max_date")

    uri = "assets?min_date=" + str(min_date) + "&max_date=" + str(max_date) + "&type=" + stocks_type
    js = request_with_retry(uri, None)
    js = js.decode()
    tickers = json.loads(js)
    if tickers is None:
        return []

    tickers.sort(key=lambda a: str(a.get('last_point', '0000-00-00')) + "_" + a['id'], reverse=True)
    setup_ids()
    for t in tickers:
        t['id'] = idt.translate_asset_to_user_id(t)
        t.pop('last_point', None)
    tickers.sort(key=lambda a: a['id'])

    return tickers


load_assets = deprecated_wrap(load_list)


def load_ndx_list(min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = 4 * 365):
    return load_list(min_date, max_date, tail, stocks_type='NASDAQ100')


def load_data(
        assets: tp.List[tp.Union[dict,str]] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        stocks_type: tp.Union[str, int] = ''
) -> xr.DataArray:
    """
    :param assets: list of ticker names to load
    :param min_date: first date in data
    :param max_date: last date of data
    :param dims: tuple with ds.FIELD, ds.TIME, ds.ASSET in the specified order
    :param forward_order: boolean, set true if you need the forward order of dates, otherwise the order is backward
    :param tail: datetime.timedelta, tail size of data. min_date = max_date - tail
    :return: xarray DataArray with historical data for selected assets
    """
    t = time.time()
    data = load_origin_data(assets=assets, min_date=min_date, max_date=max_date, tail=tail, stocks_type=stocks_type)
    log_info("Data loaded " + str(round(time.time() - t)) + "s")
    if stocks_type == '':
        data = adjust_by_splits(data, False)
    data = data.transpose(*dims)
    if forward_order:
        data = data.sel(**{ds.TIME: slice(None, None, -1)})
    data.name = "stocks_nasdaq100" if stocks_type.lower() in ["nasdaq100", "ndx100"] else "stocks"
    return data


def load_ndx_data(assets: tp.List[tp.Union[dict,str]] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
) -> xr.DataArray:
    return load_data(assets, min_date, max_date, dims, forward_order, tail, stocks_type='NDX100')


def adjust_by_splits(data, make_copy=True):
    """
    :param data: xarray
    :param make_copy: if True the initial data isn't changed
    :return: xarray with data adjusted by splits
    """
    if make_copy:
        data = data.copy()
    dims = data.dims
    data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    data.loc[f.OPEN] = data.loc[f.OPEN] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.LOW] = data.loc[f.LOW] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.HIGH] = data.loc[f.HIGH] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.CLOSE] = data.loc[f.CLOSE] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.VOL] = data.loc[f.VOL] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.DIVS] = data.loc[f.DIVS] * data.loc[f.SPLIT_CUMPROD]
    return data.transpose(*dims)


def restore_origin_data(data, make_copy=True):
    """
    :param data: xarray
    :param make_copy: if True the initial data isn't changed
    :return: xarray with origin data
    """
    if make_copy:
        data = data.copy()
    dims = data.dims
    data = data.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    data.loc[f.OPEN] = data.loc[f.OPEN] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.LOW] = data.loc[f.LOW] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.HIGH] = data.loc[f.HIGH] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.CLOSE] = data.loc[f.CLOSE] / data.loc[f.SPLIT_CUMPROD]
    data.loc[f.VOL] = data.loc[f.VOL] * data.loc[f.SPLIT_CUMPROD]
    data.loc[f.DIVS] = data.loc[f.DIVS] / data.loc[f.SPLIT_CUMPROD]
    return data.transpose(*dims)


BATCH_LIMIT = 300000


def load_origin_data(assets=None, min_date=None, max_date=None,
                     tail: tp.Union[datetime.timedelta, float, int] = 4 * 365, stocks_type=''):
    track_event("DATA_STOCKS_SERIES")
    setup_ids()

    if assets is None:
        assets_array = load_list(min_date=min_date, max_date=max_date, tail=tail, stocks_type=stocks_type)
        assets_arg = [a['id'] for a in assets_array]
    else:
        if not idt.is_asset_ids_cache_exist():
            load_list(min_date="2005-01-01", stocks_type=stocks_type)

        assets = [a['id'] if type(a) == dict else a for a in assets]
        assets_arg = assets

    assets_arg = [idt.translate_user_id_to_server_id(id) for id in assets_arg]

    assets_arg = list(set(assets_arg))  # rm duplicates

    # load data from server
    if max_date is None and "LAST_DATA_PATH" in os.environ:
        whole_data_file_flag_name = get_env("LAST_DATA_PATH", "last_data.txt")
        with open(whole_data_file_flag_name, "w") as text_file:
            text_file.write("last")

    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    # print(str(max_date))

    if min_date > max_date:
        raise Exception("min_date must be less than or equal to max_date")

    start_time = time.time()

    days = (max_date - min_date).days + 1
    chunk_asset_count = math.floor(BATCH_LIMIT / days)

    chunks = []
    assets_arg.sort()

    for offset in range(0, len(assets_arg), chunk_asset_count):
        chunk_assets = assets_arg[offset:(offset + chunk_asset_count)]
        chunk = load_origin_data_chunk(chunk_assets, min_date.isoformat(), max_date.isoformat(), stocks_type)
        if chunk is not None:
            chunks.append(chunk)
        log_info(
            "fetched chunk "
            + str(round(offset / chunk_asset_count + 1)) + "/"
            + str(math.ceil(len(assets_arg) / chunk_asset_count)) + " "
            + str(round(time.time() - start_time)) + "s"
        )

    fields = [f.OPEN, f.LOW, f.HIGH, f.CLOSE, f.VOL, f.DIVS, f.SPLIT_CUMPROD, f.IS_LIQUID] if stocks_type.lower() == 'ndx100' \
        else [f.OPEN, f.LOW, f.HIGH, f.CLOSE, f.VOL, f.DIVS, f.SPLIT, f.SPLIT_CUMPROD, f.IS_LIQUID]
    if len(chunks) == 0:
        whole = xr.DataArray(
            [[[np.nan]]]*len(fields),
            dims=[ds.FIELD, ds.TIME, ds.ASSET],
            coords={
                ds.FIELD: fields,
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['ignore']
            }
        )[:,1:,1:]
    else:
        whole = xr.concat(chunks, ds.ASSET)

    whole.coords[ds.ASSET] = [idt.translate_server_id_to_user_id(id) for id in whole.coords[ds.ASSET].values]

    if assets is not None:
        assets = sorted(assets)
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        whole = whole.broadcast_like(assets)

    whole = whole.transpose(ds.FIELD, ds.TIME, ds.ASSET)
    whole = whole.loc[
        fields,
        np.sort(whole.coords[ds.TIME])[::-1],
        np.sort(whole.coords[ds.ASSET])
    ]

    return whole.dropna(ds.TIME, 'all')


def load_origin_data_chunk(assets, min_date, max_date, stocks_type):  # min_date and max_date - iso date str
    params = {
        'assets': assets,
        'min_date': min_date,
        'max_date': max_date,
        'type': stocks_type
    }
    params = json.dumps(params)
    raw = request_with_retry("data", params.encode())
    if len(raw) == 0:
        return None
    arr = xr.open_dataarray(raw, cache=False, decode_times=True)
    arr = arr.compute()
    return arr


FIRST = True


def setup_ids():
    global FIRST
    if idt.USE_ID_TRANSLATION and FIRST:
        js = request_with_retry('assets', None)
        js = js.decode()
        tickers = json.loads(js)
        idt.USE_ID_TRANSLATION = next((i for i in tickers if i.get('symbol') is not None), None) is not None
        FIRST = False


if __name__ == '__main__':
    # import qnt.id_translation
    # qnt.id_translation.USE_ID_TRANSLATION = False
    assets = load_list()
    log_info(len(assets))
    ids = [i['id'] for i in assets]
    log_info(ids)
    data = load_data(min_date='1998-11-09', assets=ids[-2000:])
    log_info(data.sel(field='close').transpose().to_pandas())


