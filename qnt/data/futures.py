from qnt.data.common import *
import xarray as  xr
import json


def load_list() -> xr.DataArray:
    track_event("DATA_FUTURES_META")
    uri = "futures/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_data(
        assets: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        max_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
) -> tp.Union[None, xr.DataArray]:
    track_event("DATA_FUTURES_SERIES")
    if max_date is None and "LAST_DATA_PATH" in os.environ:
        whole_data_file_flag_name = get_env("LAST_DATA_PATH", "last_data.txt")
        with open(whole_data_file_flag_name, "w") as text_file:
            text_file.write("last")

    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "futures/data"
    raw = request_with_retry(uri, None)

    if raw is None or len(raw) < 1:
        fields = [f.OPEN, f.LOW, f.HIGH, f.CLOSE, f.VOL, f.OPEN_INTEREST, f.ROLL]
        arr = xr.DataArray(
            [[[np.nan] * len(assets)]] * len(fields),
            dims=[ds.FIELD, ds.TIME, ds.ASSET],
            coords={
                ds.FIELD: fields,
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['ignore']
            }
        )[1:,1:]
    else:
        arr = xr.open_dataarray(raw, cache=True, decode_times=True)
        arr = arr.compute()

    arr = arr.sel(time=slice(max_date.isoformat(), min_date.isoformat(), 1))

    if assets is not None:
        assets = [a['id'] if type(a) == dict else a for a in assets]

    if assets is not None:
        assets = sorted(list(set(assets)))
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        arr = arr.broadcast_like(assets)
        arr = arr.sel(asset=assets).dropna(ds.TIME, 'all')

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    arr.name = "futures"
    return arr.transpose(*dims)
