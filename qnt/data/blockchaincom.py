"""

There are metrics from https://blockchain.com

"""

from qnt.data.common import *


def load_list():
    """
    Loads currency list (origin: https://imf.org)
    :return:
    """
    track_event("DATA_BLOCKCHAINCOM_LIST")
    uri = "blockchain.com/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    idx = json.loads(js)
    return idx


def load_data(
        id: str,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        forward_order: bool = True,
):
    """
    Loads metric timeseries (origin: https://blockchain.com)

    :param id:
    :param min_date:
    :param max_date:
    :param tail:
    :param forward_order:
    :return:
    """
    track_event("DATA_BLOCKCHAINCOM_DATA")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)

    uri = "blockchain.com/data?id=" + str(id)
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

    if forward_order:
        arr = arr.sel(**{ds.TIME: slice(None, None, -1)})

    arr = arr.dropna(ds.TIME, 'all')

    arr.name = "blockchaincom_metric"
    return arr


if __name__ == '__main__':
    cl = load_list()
    print('list', json.dumps(cl, indent=1))
    cd = load_data(tail=60, id=cl[0]['id'])
    print('data', cd.to_pandas())
