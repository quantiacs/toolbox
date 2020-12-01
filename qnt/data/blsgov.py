from qnt.data.common import *


def load_db_list():
    """
    :return: list of DBs from bls.gov
    """
    uri = "bls.gov/db/list"
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_db_meta(db_id:str):
    """
    :return: list of DBs from bls.gov
    """

    # print(str(max_date))

    uri = "bls.gov/db/meta?id=" + db_id
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)


def load_series_list(db_id:str):
    """
    :return: generator of series
    """
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
    max_date = parse_date(max_date)
    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

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
    max_date = parse_date(max_date)
    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - parse_tail(tail)
    uri = "bls.gov/series/aspect?id=" + series_id + "&min_date=" + min_date.isoformat() + "&max_date=" + max_date.isoformat()
    js = request_with_retry(uri, None)
    js = js.decode()
    return json.loads(js)
