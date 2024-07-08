from qnt.data.common import *


def major_load_list():
    """
    Loads major indexes list.
    :return:
    """
    track_event("DATA_INDEX_MAJOR_META")
    uri = "major-idx/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)
    return idx


def major_load_data(
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
):
    """
    Loads major indexes data
    :param min_date:
    :param max_date:
    :param tail:
    :param dims:
    :param forward_order:
    :return:
    """
    track_event("DATA_INDEX_MAJOR_SERIES")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "major-idx/data?min_date=" + str(min_date) + "&max_date=" + str(max_date)
    raw = request_with_retry(uri, None)
    if raw is None:
        arr = xr.DataArray(
            [[[np.nan]*3]]*5,
            dims=[ds.FIELD, ds.TIME, ds.ASSET],
            coords={
                ds.FIELD: [
                  f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOL
                ],
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['US30', 'US500', 'USTEC']
            }
        )[:,1:,:]
    else:
        arr = xr.open_dataarray(raw, cache=False, decode_times=True)
        arr = arr.compute()

    arr = arr.sortby(ds.TIME, ascending=forward_order)

    arr = arr.dropna(ds.TIME, how='all')

    arr.name = "major_indexes"
    return arr.transpose(*dims)


def load_list(
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
) -> list:
    """
    Loads index list
    :return:
    """
    track_event("DATA_INDEX_META")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    if min_date > max_date:
        raise Exception("min_date must be less than or equal to max_date")

    # print(str(max_date))

    uri = "idx/list?min_date=" + str(min_date) + "&max_date=" + str(max_date)
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)

    idx.sort(key=lambda a: a['id'])

    return idx


def load_data(
        assets: tp.Union[None, tp.List[tp.Union[str,dict]]] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        dims: tp.Tuple[str, str] = (ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL,
) -> tp.Union[None, xr.DataArray]:
    """
    Loads index time series.
    :param assets:
    :param min_date:
    :param max_date:
    :param dims:
    :param forward_order:
    :param tail:
    :return:
    """
    track_event("DATA_INDEX_SERIES")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    if assets is not None:
        assets = [a['id'] if type(a) == dict else a for a in assets]

    if assets is None:
        assets_array = load_list(min_date, max_date)
        assets_arg = [i['id'] for i in assets_array]
    else:
        assets_arg = assets

    params = {"ids": assets_arg, "min_date": min_date.isoformat(), "max_date": max_date.isoformat()}
    params = json.dumps(params)
    params = params.encode()
    raw = request_with_retry("idx/data", params)

    if raw is None or len(raw) < 1:
        arr = xr.DataArray(
            [[np.nan]],
            dims=[ds.TIME, ds.ASSET],
            coords={
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['ignore']
            }
        )[1:,1:]
    else:
        arr = xr.open_dataarray(raw, cache=False, decode_times=True)
        arr = arr.compute()

    arr = arr.sortby(ds.TIME, ascending=forward_order)

    if assets is not None:
        assets = list(set(assets))
        assets = sorted(assets)
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        arr = arr.broadcast_like(assets).sel(asset=assets)

    arr = arr.dropna(ds.TIME, how='all')

    arr.name = "indexes"
    return arr.transpose(*dims)