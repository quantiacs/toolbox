from qnt.data.common import *
import qnt.data.common as qdc

def load_data(
        assets: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date, datetime.datetime] = None,
        max_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
) -> tp.Union[None, xr.DataArray]:
    track_event("DATA_CRYPTOFUTURES")
    if max_date is None and "LAST_DATA_PATH" in os.environ:
        whole_data_file_flag_name = get_env("LAST_DATA_PATH", "last_data.txt")
        with open(whole_data_file_flag_name, "w") as text_file:
            text_file.write("last")

    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "cryptofutures?min_date=" + min_date.isoformat() + "&max_date=" + max_date.isoformat()
    raw = request_with_retry(uri, None)
    try:
        arr = xr.open_dataarray(raw, cache=True, decode_times=True)
        arr = arr.compute()
    except:
        arr = xr.DataArray(
            [[[np.nan]*5]],
            dims=[ds.TIME, ds.ASSET, ds.FIELD],
            coords={
                ds.TIME: pd.DatetimeIndex([max_date]),
                ds.ASSET: ['ignore'],
                ds.FIELD: [f.OPEN, f.LOW, f.HIGH, f.CLOSE, f.VOL]
            }
        )[1:,1:,:]

    if assets is not None:
        assets = list(set(assets))
        assets = sorted(assets)
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        arr = arr.broadcast_like(assets).sel(asset=assets)

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    arr = arr.dropna(ds.TIME, 'all')

    arr.name = "cryptofutures"
    return arr.transpose(*dims)
