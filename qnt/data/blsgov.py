from qnt.data.common import *


def load_db_list():
    """
    :return: list of DBs from bls.gov
    """
    track_event("DATA_BLSGOV_META")
    uri = "bls.gov/db/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_db_meta(db_id:str):
    """
    :return: list of DBs from bls.gov
    """

    # print(str(max_date))
    track_event("DATA_BLSGOV_META")
    uri = "bls.gov/db/meta?id=" + db_id
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_series_list(db_id:str):
    """
    :return: generator of series
    """
    track_event("DATA_BLSGOV_META")
    uri = "bls.gov/series/list?id=" + db_id + "&last_series="
    last_series = ''
    while True:
        js = request_with_retry(uri + last_series, None)
        js = js.decode()
        js = json.loads(js)
        if len(js) == 0:
            return
        for s in js:
            last_series = s['id']
            yield s


def load_series_data(
        series_id:str,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
):
    track_event("DATA_BLSGOV")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)
    uri = "bls.gov/series/data?id=" + series_id + "&min_date=" + min_date.isoformat() + "&max_date=" + max_date.isoformat()
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_series_aspect(
        series_id:str,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, int, float] = DEFAULT_TAIL
):
    track_event("DATA_BLSGOV")
    max_date = parse_date(max_date)

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)
    uri = "bls.gov/series/aspect?id=" + series_id + "&min_date=" + min_date.isoformat() + "&max_date=" + max_date.isoformat()
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)
