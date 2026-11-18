import datetime
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.data as qndata
import qnt.data.common as qndc
import qnt.data.id_translation as idt
import qnt.data.stocks as qnstocks
from qnt.tests.hl_server import (
    CATALOG,
    DAYS,
    STOCK_FIELDS,
    FakeServer,
    Query,
    cube,
    netcdf_payload,
    stock_server,
)

AssetInfo = dict[str, object]
LISTING_IDS = ["NAS:BBB", "NYS:AAA", "NYS:AAA~1"]
EMPTY_SELECTION_ERROR = "min_date must be less than or equal to max_date"
INDEX_FIELDS = [name for name in STOCK_FIELDS if name != "split"]
ASSETS_TEN = [f"a{i:02d}" for i in range(10)]


class Lister(Protocol):
    def __call__(self, *, max_date: str) -> list[AssetInfo]: ...


class IndexLoader(Protocol):
    def __call__(
        self, *, assets: list[str], min_date: str, max_date: str, forward_order: bool = True
    ) -> xr.DataArray: ...


class Recorder(Protocol):
    def __call__(self, **kwargs: object) -> None: ...


class Clock:
    def __init__(self, readings: list[float]) -> None:
        self.readings = iter(readings)

    def time(self) -> float:
        return next(self.readings)


def echo_server(server: FakeServer) -> None:
    """Serves the catalog and, for any requested assets, two days of random data."""
    server.serve_json("assets", CATALOG)

    def data(body: bytes | None, _query: Query) -> bytes:
        assert body is not None
        params = json.loads(body)
        return netcdf_payload(cube(STOCK_FIELDS, params["assets"], DAYS[:2]))

    server.route("data", data)


def data_requests(server: FakeServer) -> list[dict[str, object]]:
    return [
        json.loads(body)
        for call, body in zip(server.calls, server.bodies, strict=True)
        if call == "data" and body is not None
    ]


def captured_log(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []

    def spy(*args: object) -> None:
        messages.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(qnstocks, "log_info", spy)
    return messages


def recorder(seen: list[str], label: str) -> Recorder:
    def record(**_kwargs: object) -> None:
        seen.append(label)

    return record


def pristine_stocks_module() -> ModuleType:
    assert qndc.__file__ is not None
    spec = importlib.util.spec_from_file_location(
        "qnt_data_stocks_pristine", Path(qndc.__file__).with_name("stocks.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LISTERS: list[tuple[Lister, str]] = [
    (qndata.stocks_load_list, ""),
    (qndata.stocks_load_ndx_list, "NASDAQ100"),
    (qndata.stocks_load_spx_list, "SPX500"),
]


@pytest.mark.parametrize(("lister", "stocks_type"), LISTERS, ids=["list", "ndx_list", "spx_list"])
def test_listing_default_tail_is_four_years_of_365_days(
    server: FakeServer, lister: Lister, stocks_type: str
) -> None:
    stock_server(server)
    lister(max_date="2021-02-01")
    assert f"assets?min_date=2017-02-02&max_date=2021-02-01&type={stocks_type}" in server.calls


def test_origin_data_default_tail_is_four_years_of_365_days(server: FakeServer) -> None:
    stock_server(server)
    qnstocks.load_origin_data(max_date="2021-02-05")
    assert "assets?min_date=2017-02-06&max_date=2021-02-05&type=" in server.calls
    assert [request["min_date"] for request in data_requests(server)] == ["2017-02-06"]


TAILS: list[int | float | datetime.timedelta] = [10, 10.9, datetime.timedelta(days=10)]


@pytest.mark.parametrize("tail", TAILS, ids=["int", "float", "timedelta"])
def test_listing_tail_is_subtracted_from_max_date(
    server: FakeServer, tail: int | float | datetime.timedelta
) -> None:
    stock_server(server)
    qndata.stocks_load_list(max_date="2021-02-05", tail=tail)
    assert "assets?min_date=2021-01-26&max_date=2021-02-05&type=" in server.calls


@pytest.mark.parametrize("tail", TAILS, ids=["int", "float", "timedelta"])
def test_origin_data_tail_is_subtracted_from_max_date(
    server: FakeServer, tail: int | float | datetime.timedelta
) -> None:
    stock_server(server)
    qnstocks.load_origin_data(max_date="2021-02-05", tail=tail)
    assert "assets?min_date=2021-01-26&max_date=2021-02-05&type=" in server.calls
    assert [request["min_date"] for request in data_requests(server)] == ["2021-01-26"]


def test_listing_accepts_equal_dates(server: FakeServer) -> None:
    stock_server(server)
    listing = qndata.stocks_load_list(min_date="2021-01-06", max_date="2021-01-06")
    assert [a["id"] for a in listing] == LISTING_IDS
    assert "assets?min_date=2021-01-06&max_date=2021-01-06&type=" in server.calls


def test_origin_data_rejects_reversed_dates_and_accepts_equal_dates(server: FakeServer) -> None:
    stock_server(server)
    with pytest.raises(Exception, match=EMPTY_SELECTION_ERROR):
        qnstocks.load_origin_data(assets=["NAS:BBB"], min_date="2021-01-08", max_date="2021-01-06")
    result = qnstocks.load_origin_data(
        assets=["NAS:BBB"], min_date="2021-01-06", max_date="2021-01-06"
    )
    assert_array_equal(result.coords["time"].values, np.array(["2021-01-06"], dtype="datetime64[ns]"))
    assert [request["min_date"] for request in data_requests(server)][-1] == "2021-01-06"


def test_origin_data_comes_back_newest_first_with_sorted_assets_and_fields(
    server: FakeServer,
) -> None:
    stock_server(server)
    result = qnstocks.load_origin_data(
        assets=["NYS:AAA", "NAS:BBB"], min_date="2021-01-06", max_date="2021-01-08"
    )
    expected = np.array(["2021-01-08", "2021-01-07", "2021-01-06"], dtype="datetime64[ns]")
    assert_array_equal(result.coords["time"].values, expected)
    assert list(result.coords["asset"].values) == ["NAS:BBB", "NYS:AAA"]
    assert list(result.coords["field"].values) == STOCK_FIELDS


INDEX_LOADERS: list[tuple[IndexLoader, str, str]] = [
    (qndata.stocks_load_ndx_data, "NDX100", "stocks_nasdaq100"),
    (qndata.stocks_load_spx_data, "SPX500", "stocks_s&p500"),
]


@pytest.mark.parametrize(("loader", "server_type", "name"), INDEX_LOADERS, ids=["ndx", "spx"])
def test_index_loaders_default_to_forward_order_and_keep_origin_values(
    server: FakeServer, loader: IndexLoader, server_type: str, name: str
) -> None:
    source = stock_server(server)
    assets, first, last = ["NYS:AAA"], "2021-01-06", "2021-01-12"
    default = loader(assets=assets, min_date=first, max_date=last)
    explicit = loader(assets=assets, min_date=first, max_date=last, forward_order=True)
    backward = loader(assets=assets, min_date=first, max_date=last, forward_order=False)

    days = np.array(
        ["2021-01-06", "2021-01-07", "2021-01-08", "2021-01-11", "2021-01-12"],
        dtype="datetime64[ns]",
    )
    assert_array_equal(default.coords["time"].values, days)
    assert_array_equal(explicit.coords["time"].values, days)
    assert_array_equal(backward.coords["time"].values, days[::-1])
    assert default.name == name
    assert default.dims == ("field", "time", "asset")
    assert list(default.coords["field"].values) == INDEX_FIELDS
    assert data_requests(server)[0]["type"] == server_type

    raw = source.sel(asset="s1", time=days)
    assert (raw.sel(field="split_cumprod").values == 0.5).all()
    for field in ("open", "close", "vol"):
        assert_allclose(
            default.sel(field=field, asset="NYS:AAA").values,
            raw.sel(field=field).values,
        )


STOCKS_TYPE_NAMES = [
    ("", "stocks"),
    ("NDX100", "stocks_nasdaq100"),
    ("ndx100", "stocks_nasdaq100"),
    ("NASDAQ100", "stocks_nasdaq100"),
    ("SPX500", "stocks_s&p500"),
    ("spx500", "stocks_s&p500"),
    ("Spx500", "stocks_s&p500"),
    ("a", "stocks"),
    ("zzz", "stocks"),
]


@pytest.mark.parametrize(("stocks_type", "name"), STOCKS_TYPE_NAMES)
def test_data_name_follows_stocks_type(server: FakeServer, stocks_type: str, name: str) -> None:
    stock_server(server)
    result = qnstocks.load_data(
        assets=["NAS:BBB"], min_date="2021-01-06", max_date="2021-01-08", stocks_type=stocks_type
    )
    assert result.name == name


def test_load_data_adjusts_its_own_sorted_copy_in_place(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    stock_server(server)
    flags: list[bool] = []
    original = qnstocks.adjust_by_splits

    def spy(data: xr.DataArray, make_copy: bool = True) -> xr.DataArray:
        flags.append(make_copy)
        return original(data, make_copy)

    monkeypatch.setattr(qnstocks, "adjust_by_splits", spy)
    qnstocks.load_data(assets=["NAS:BBB"], min_date="2021-01-06", max_date="2021-01-08")
    assert flags == [False]


def test_load_data_logs_elapsed_seconds(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    stock_server(server)
    messages = captured_log(monkeypatch)
    monkeypatch.setattr(qnstocks, "time", Clock([3.0, 7.0, 30.0, 100.0]))
    qnstocks.load_data(assets=["NAS:BBB"], min_date="2021-01-06", max_date="2021-01-08")
    assert messages == ["fetched chunk 1/1 23s", "Data loaded 97s"]


def test_origin_data_is_fetched_in_chunks_sized_by_batch_limit_and_days(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    echo_server(server)
    messages = captured_log(monkeypatch)
    monkeypatch.setattr(qnstocks, "BATCH_LIMIT", 12)
    monkeypatch.setattr(qnstocks, "time", Clock([7.0, 30.0, 45.0, 52.0]))
    result = qnstocks.load_origin_data(
        assets=ASSETS_TEN, min_date="2021-01-06", max_date="2021-01-08"
    )
    assert [request["assets"] for request in data_requests(server)] == [
        ASSETS_TEN[:4],
        ASSETS_TEN[4:8],
        ASSETS_TEN[8:],
    ]
    assert messages == [
        "fetched chunk 1/3 23s",
        "fetched chunk 2/3 38s",
        "fetched chunk 3/3 45s",
    ]
    assert list(result.coords["asset"].values) == ASSETS_TEN


def test_batch_limit_and_first_flag_defaults() -> None:
    pristine = pristine_stocks_module()
    assert vars(pristine)["BATCH_LIMIT"] == 300000
    assert vars(pristine)["FIRST"] is True


@pytest.mark.parametrize(
    ("cache_exists", "translated", "loads"),
    [(True, True, 0), (False, True, 1), (True, False, 1), (False, False, 1)],
)
def test_catalog_is_loaded_unless_cache_and_translations_both_exist(
    server: FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    cache_exists: bool,
    translated: bool,
    loads: int,
) -> None:
    echo_server(server)
    monkeypatch.setattr(idt, "is_asset_ids_cache_exist", lambda: cache_exists)
    monkeypatch.setattr(idt, "has_asset_id_translations", lambda _assets: translated)
    seen: list[dict[str, object]] = []

    def spy(**kwargs: object) -> list[AssetInfo]:
        seen.append(kwargs)
        return []

    monkeypatch.setattr(qnstocks, "load_list", spy)
    qnstocks.load_origin_data(assets=["a00"], min_date="2021-01-06", max_date="2021-01-08")
    assert seen == [{"min_date": "2005-01-01", "stocks_type": ""}] * loads


def test_last_data_flag_is_written_only_when_max_date_is_missing(
    server: FakeServer, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stock_server(server)
    flag = sandbox / "flag.txt"
    monkeypatch.setenv("LAST_DATA_PATH", str(flag))
    qnstocks.load_origin_data(assets=["NAS:BBB"], min_date="2021-01-06", max_date="2021-01-08")
    assert not flag.exists()
    qnstocks.load_origin_data(assets=["NAS:BBB"], min_date="2021-01-06")
    assert flag.read_text() == "last"


def test_last_data_flag_is_not_written_without_environment_variable(
    server: FakeServer, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stock_server(server)
    monkeypatch.delenv("LAST_DATA_PATH", raising=False)
    qnstocks.load_origin_data(assets=["NAS:BBB"], min_date="2021-01-06")
    assert not (sandbox / "last_data.txt").exists()


@pytest.mark.parametrize(
    ("use_translation", "first", "requests", "first_after"),
    [
        (True, True, 1, False),
        (False, True, 0, True),
        (True, False, 0, False),
        (False, False, 0, False),
    ],
)
def test_setup_ids_fetches_catalog_once_while_translation_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    use_translation: bool,
    first: bool,
    requests: int,
    first_after: bool,
) -> None:
    uris: list[str] = []

    def fake_request(uri: str, _data: bytes | None) -> bytes:
        uris.append(uri)
        return json.dumps(CATALOG).encode()

    monkeypatch.setattr(qnstocks, "request_with_retry", fake_request)
    monkeypatch.setattr(idt, "USE_ID_TRANSLATION", use_translation)
    monkeypatch.setattr(qnstocks, "FIRST", first)
    qnstocks.setup_ids()
    assert uris == ["assets"] * requests
    assert qnstocks.FIRST is first_after
    qnstocks.setup_ids()
    assert uris == ["assets"] * requests


@pytest.mark.parametrize(
    ("catalog", "enabled"),
    [([{"id": "x", "symbol": "XXX"}], True), ([{"id": "x"}, {"id": "y", "symbol": None}], False)],
)
def test_setup_ids_disables_translation_for_catalog_without_symbols(
    monkeypatch: pytest.MonkeyPatch, catalog: list[dict[str, str | None]], enabled: bool
) -> None:
    monkeypatch.setattr(
        qnstocks, "request_with_retry", lambda _uri, _data: json.dumps(catalog).encode()
    )
    monkeypatch.setattr(idt, "USE_ID_TRANSLATION", True)
    monkeypatch.setattr(qnstocks, "FIRST", True)
    qnstocks.setup_ids()
    assert idt.USE_ID_TRANSLATION is enabled


DISPATCH = [
    ("stocks", "stocks_load_data"),
    ("stocks_long", "stocks_load_data"),
    ("stocks_s&p500", "stocks_load_spx_data"),
    ("stocks_nasdaq100", "stocks_load_ndx_data"),
    ("futures", "futures_load_data"),
    ("crypto", "crypto_load_data"),
    ("crypto_futures", "cryptofutures_load_data"),
    ("cryptofutures", "cryptofutures_load_data"),
    ("crypto_daily", "cryptodaily_load_data"),
    ("cryptodaily", "cryptodaily_load_data"),
    ("crypto_daily_long", "cryptodaily_load_data"),
    ("crypto_daily_long_short", "cryptodaily_load_data"),
]


def install_recorders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    for label in sorted({loader for _type, loader in DISPATCH}):
        monkeypatch.setattr(qndata, label, recorder(seen, label))
    return seen


@pytest.mark.parametrize(("data_type", "loader"), DISPATCH)
def test_load_data_by_type_routes_every_type_to_its_loader(
    monkeypatch: pytest.MonkeyPatch, data_type: str, loader: str
) -> None:
    seen = install_recorders(monkeypatch)
    qndata.load_data_by_type("".join(list(data_type)))
    assert seen == [loader]


@pytest.mark.parametrize(
    "data_type",
    [
        "",
        "a",
        "bonds",
        "crypto_",
        "futures_",
        "stocks_",
        "stocks!",
        "stocksz",
        "zzz",
        "cryptodaily_",
    ],
)
def test_load_data_by_type_rejects_unknown_types(
    monkeypatch: pytest.MonkeyPatch, data_type: str
) -> None:
    seen = install_recorders(monkeypatch)
    with pytest.raises(Exception, match=r"Wrong data_type\."):
        qndata.load_data_by_type("".join(list(data_type)))
    assert seen == []
