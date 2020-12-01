from qnt.data.common import *
import itertools


def load_forms(
        ciks: tp.Union[None, tp.List[str]] = None,
        types: tp.Union[None, tp.List[str]] = None,
        facts: tp.Union[None, tp.List[str]] = None,
        skip_segment: bool = False,
        min_date: tp.Union[str, datetime.date] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = None
) -> tp.Generator[dict, None, None]:
    """
    Load SEC Forms (Fundamental data)
    :param ciks: list of cik (you can get cik from asset id)
    :param types: list of form types: ['10-K', '10-Q', '10-K/A', '10-Q/A']
    :param facts: list of facts for extraction, for example: ['us-gaap:Goodwill']
    :param skip_segment: skip facts with segment
    :param min_date: min form date
    :param max_date: max form date
    :param tail: datetime.timedelta, tail size of data. min_date = max_date - tail
    :return: generator
    """
    max_date = parse_date(max_date)
    if MAX_DATE_LIMIT is not None:
        if max_date is not None:
            max_date = min(MAX_DATE_LIMIT, max_date)
        else:
            max_date = MAX_DATE_LIMIT

    if min_date is not None:
        min_date = parse_date(min_date)
    else:
        min_date = max_date - tail

    params = {
        'ciks': list(set(ciks)) if ciks is not None else None,
        'types':  list(set(types)) if types is not None else None,
        'facts': list(set(facts)) if facts is not None else None,
        'skip_segment': skip_segment,
        'min_date': min_date.isoformat(),
        'max_date': max_date.isoformat()
    }
    go = True
    while go:
        params_js = json.dumps(params)
        raw = request_with_retry("sec.gov/forms", params_js.encode())
        js = raw.decode()
        forms = json.loads(js)
        for f in forms:
            yield f
        go = len(forms) > 0
        params['offset'] = params.get('offset', 0) + len(forms)


def load_facts(
        ciks: tp.List[str],
        facts: tp.List[str],
        types: tp.Union[None, tp.List[str]] = None,
        skip_segment: bool = False,
        period: tp.Union[str, None] = None, # 'A', 'S', 'Q'
        columns: tp.Union[tp.List[str], None] = None,
        min_date: tp.Union[str, datetime.date, None] = None,
        max_date: tp.Union[str, datetime.date, None] = None,
        tail: tp.Union[datetime.timedelta, float, int] = DEFAULT_TAIL,
        group_by_cik: bool = False
) -> tp.Generator[dict, None, None]:
    """
    Load SEC Forms (Fundamental data)
    :param ciks: list of cik (you can get cik from asset id)
    :param types: list of form types: ['10-K', '10-Q', '10-K/A', '10-Q/A']
    :param facts: list of facts for extraction, for example: ['us-gaap:Goodwill']
    :param skip_segment: skip facts with segment
    :param period: fact periods ('Q', 'A' or 'S')
    :param columns: list of columns to load: ['fact_name','unit_type','unit','segment','period_type','period','period_length','report_type','report_url','report_date']
    :param min_date: min form date
    :param max_date: max form date
    :param tail: datetime.timedelta, tail size of data. min_date = max_date - tail
    :return: generator
    """
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

    params = {
        'ciks': list(set(ciks)),
        'types': list(set(types)) if types is not None else None,
        'facts': list(set(facts)),
        'skip_segment': skip_segment,
        'columns': list(set(columns)) if columns is not None else None,
        'period': period,
        'min_date': min_date.isoformat(),
        'max_date': max_date.isoformat()
    }

    max_batch_size = min(50, SECGOV_BATCH_SIZE//len(facts))
    print("load secgov facts...")
    t = time.time()
    for offset in range(0,len(ciks), max_batch_size):
        batch_ciks = []
        if offset + max_batch_size > len(ciks):
            batch_ciks = ciks[offset:]
        else:
            batch_ciks = ciks[offset:(offset+max_batch_size)]
        params['ciks'] = batch_ciks
        params_js = json.dumps(params)
        raw = request_with_retry("sec.gov/facts", params_js.encode())
        js = raw.decode()
        facts = json.loads(js)
        if group_by_cik:
            facts = sorted(facts, key=lambda k: k['cik'])
            groups = itertools.groupby(facts, key=lambda f:f['cik'])
            for g in groups:
                yield (g[0], list(g[1]))
        else:
            for f in facts:
                yield f
        print("fetched chunk", (offset//max_batch_size + 1), '/', math.ceil(len(ciks)/max_batch_size), math.ceil(time.time()-t), 's')

    print("facts loaded.")


SECGOV_BATCH_SIZE = 2000
