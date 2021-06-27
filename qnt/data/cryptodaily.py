from qnt.data.common import *

def load_data(
        assets: tp.Union[None, tp.List[str]] = None,
        min_date: tp.Union[str, datetime.date, datetime.datetime] = None,
        max_date: tp.Union[str, datetime.date, datetime.datetime, None] = None,
        dims: tp.Tuple[str, str, str] = (ds.FIELD, ds.TIME, ds.ASSET),
        forward_order: bool = True,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
) -> tp.Union[None, xr.DataArray]:
    track_event("DATA_CRYPTO")
    if max_date is None and "LAST_DATA_PATH" in os.environ:
        whole_data_file_flag_name = get_env("LAST_DATA_PATH", "last_data.txt")
        with open(whole_data_file_flag_name, "w") as text_file:
            text_file.write("last")

    max_date = parse_date_and_hour(max_date)

    if min_date is not None:
        min_date = parse_date_and_hour(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "cryptodaily/data?min_date=" + datetime_to_hours_str(min_date) + "&max_date=" + datetime_to_hours_str(max_date)
    raw = request_with_retry(uri, None)
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

    if assets is not None:
        assets = list(set(assets))
        assets = sorted(assets)
        assets = xr.DataArray(assets, dims=[ds.ASSET], coords={ds.ASSET:assets})
        arr = arr.broadcast_like(assets).sel(asset=assets)

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    arr = arr.dropna(ds.TIME, 'all')

    arr.name = "cryptodaily"
    return arr.transpose(*dims)


BATCH_LIMIT = 300000
