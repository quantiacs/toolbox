"""

There are currency and commodity data. (origin: https://imf.org)

"""

from qnt.data.common import *


def load_currency_list():
    """
    Loads currency list (origin: https://imf.org)
    :return:
    """
    track_event("DATA_IMF_CURRENCY_LIST")
    uri = "imf.org/currency/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)
    return idx


def load_currency_data(
        assets: tp.Union[None, tp.List[tp.Union[str,dict]]] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        dims: tp.Tuple[str, str, str] = (ds.TIME, ds.ASSET),
        forward_order: bool = True,
):
    """
    Loads currency timeseries (origin: https://imf.org)

    :param assets:
    :param min_date:
    :param max_date:
    :param tail:
    :param dims:
    :param forward_order:
    :return:
    """
    track_event("DATA_IMF_CURRENCY_DATA")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "imf.org/currency/data"
    raw = request_with_retry(uri, None)
    if raw is None:
        arr = xr.DataArray(
            [[np.nan]],
            dims=[ds.TIME, ds.ASSET],
            coords={
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['']
            }
        )[:,1:]
    else:
        arr = xr.open_dataarray(raw, cache=False, decode_times=True)
        arr = arr.compute()

    arr = arr.sel(time=slice(max_date,min_date))
    if assets is not None:
        arr = arr.broadcast_like(xr.DataArray(assets, dims='asset', coords={'asset':assets}))
        arr = arr.sel(asset=assets)

    arr = arr.sortby(ds.TIME, ascending=forward_order)

    arr = arr.dropna(ds.TIME, how='all')

    arr.name = "imf_currency"
    return arr.transpose(*dims)


def load_commodity_list():
    """
    Loads commodity list (origin: https://imf.org)
    :return:
    """
    track_event("DATA_IMF_COMMODITY_LIST")
    uri = "imf.org/commodity/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)
    return idx


def load_commodity_data(
        assets: tp.Union[None, tp.List[tp.Union[str,dict]]] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        dims: tp.Tuple[str, str, str] = (ds.TIME, ds.ASSET),
        forward_order: bool = True,
):
    """
    Loads commodity timeseries (origin: https://imf.org)
    :param assets:
    :param min_date:
    :param max_date:
    :param tail:
    :param dims:
    :param forward_order:
    :return:
    """
    track_event("DATA_IMF_COMMODITY_DATA")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "imf.org/commodity/data"
    raw = request_with_retry(uri, None)
    if raw is None:
        arr = xr.DataArray(
            [[np.nan]],
            dims=[ds.TIME, ds.ASSET],
            coords={
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['']
            }
        )[:,1:]
    else:
        arr = xr.open_dataarray(raw, cache=False, decode_times=True)
        arr = arr.compute()

    arr = arr.sel(time=slice(max_date,min_date))
    if assets is not None:
        arr = arr.broadcast_like(xr.DataArray(assets, dims='asset', coords={'asset':assets}))
        arr = arr.sel(asset=assets)

    arr = arr.sortby(ds.TIME, ascending=forward_order)

    arr = arr.dropna(ds.TIME, how='all')

    arr.name = "imf_commodity"
    return arr.transpose(*dims)


if __name__ == '__main__':
    cl = load_currency_list()
    print('currency list', json.dumps(cl, indent=1))
    cd = load_currency_data(tail=60, assets=['EUR'])
    print('currency data', cd.to_pandas())

    cl = load_commodity_list()
    print('commodity list', json.dumps(cl, indent=1))
    cd = load_commodity_data(tail=600, assets=['PSOYB'])
    print('commodity data', cd.to_pandas())