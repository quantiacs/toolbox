import datetime
import json
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.data.blockchaincom as qnblockchain
import qnt.data.blsgov as qnbls
import qnt.data.common as qndc
import qnt.data.crypto as qncrypto
import qnt.data.cryptodaily as qncryptodaily
import qnt.data.cryptofutures as qncryptofutures
import qnt.data.futures as qnfutures
import qnt.data.id_translation as idt
import qnt.data.imf as qnimf
import qnt.data.index as qnindex
from qnt.tests.hl_server import DAYS, FakeServer, Query, cube, netcdf_payload, plane

MAX_DAY = "2021-02-12"
TAIL_DAYS = 10
TAIL_START = "2021-02-02"
GIVEN_START = "2021-02-05"
MAX_HOUR = "2021-02-12T05"
TAIL_START_HOUR = "2021-02-02T05"
GIVEN_START_HOUR = "2021-02-05T03"
FUTURES_FIELDS = ["open", "low", "high", "close", "vol", "oi", "roll"]
CRYPTO_FIELDS = ["open", "low", "high", "close", "vol"]
HOURS = pd.date_range("2021-02-01", periods=240, freq="h", name="time")
TRANSLATION_FILE = "id-translation.csv"
WINDOW_START = "2021-01-04"


def sizes_of(arr: xr.DataArray) -> dict[str, int]:
    return {str(name): int(size) for name, size in arr.sizes.items()}


def day_of(stamp: np.datetime64) -> str:
    return str(stamp)[:10]


def assert_ascending(arr: xr.DataArray) -> None:
    assert (np.diff(arr.coords["time"].values) > np.timedelta64(0)).all()


def assert_descending(arr: xr.DataArray) -> None:
    assert (np.diff(arr.coords["time"].values) < np.timedelta64(0)).all()


def stocks_plane() -> bytes:
    return netcdf_payload(plane(["AAA", "BBB"], DAYS, seed=5))


def test_translation_collisions_get_numbered_suffixes(sandbox: Path) -> None:
    (sandbox / TRANSLATION_FILE).write_text("server_id,user_id\ns1,NYS:AAA\ns2,NYS:AAA~1\n")
    assert idt.get_or_create_translation("s3", "NYS:AAA") == "NYS:AAA~2"
    assert idt.get_or_create_translation("s4", "NYS:AAA") == "NYS:AAA~3"
    lines = (sandbox / TRANSLATION_FILE).read_text().splitlines()
    assert lines[-2:] == ["s3,NYS:AAA~2", "s4,NYS:AAA~3"]


@pytest.mark.parametrize(
    ("submission", "created"),
    [(None, True), ("", True), ("77", False)],
    ids=["unset", "empty", "set"],
)
def test_translation_is_created_unless_submission_id_is_non_empty(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, submission: str | None, created: bool
) -> None:
    if submission is not None:
        monkeypatch.setenv("SUBMISSION_ID", submission)
    assert idt.get_or_create_translation("s1", "NYS:AAA") == ("NYS:AAA" if created else "s1")
    assert (sandbox / TRANSLATION_FILE).exists() is created
    if created:
        assert (sandbox / TRANSLATION_FILE).read_text() == "server_id,user_id\ns1,NYS:AAA\n"


def test_translation_table_is_read_once_and_cached(sandbox: Path) -> None:
    table = sandbox / TRANSLATION_FILE
    table.write_text("server_id,user_id\ns1,u1\n")
    idt.load_id_translation_table()
    server_to_user = idt.server_id_to_user_id
    user_to_server = idt.user_id_to_server_id
    assert server_to_user == {"s1": "u1"}
    assert user_to_server == {"u1": "s1"}

    table.write_text("server_id,user_id\ns1,changed\ns2,u2\n")
    assert idt.translate_server_id_to_user_id("s1") == "u1"
    assert idt.translate_server_id_to_user_id("s2") == "s2"
    assert idt.translate_user_id_to_server_id("u2") == "u2"
    idt.load_id_translation_table()
    assert idt.server_id_to_user_id is server_to_user
    assert idt.user_id_to_server_id is user_to_server


@pytest.mark.parametrize("server_side_set", [False], ids=["user-set"])
def test_half_initialised_translation_tables_are_reloaded(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, server_side_set: bool
) -> None:
    (sandbox / TRANSLATION_FILE).write_text("server_id,user_id\ns1,u1\n")
    stale = {"stale": "x"}
    monkeypatch.setattr(idt, "server_id_to_user_id", stale if server_side_set else None)
    monkeypatch.setattr(idt, "user_id_to_server_id", None if server_side_set else stale)
    assert idt.translate_server_id_to_user_id("s1") == "u1"
    assert idt.translate_user_id_to_server_id("u1") == "s1"
    assert idt.translate_server_id_to_user_id("stale") == "stale"
    assert idt.translate_user_id_to_server_id("stale") == "stale"


@pytest.mark.parametrize(
    ("server_side", "user_side", "expected"),
    [({}, {}, True), ({}, None, False), (None, {}, False), (None, None, False)],
    ids=["both", "only-server", "only-user", "none"],
)
def test_translation_cache_exists_only_when_both_tables_are_loaded(
    monkeypatch: pytest.MonkeyPatch,
    server_side: dict[str, str] | None,
    user_side: dict[str, str] | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(idt, "server_id_to_user_id", server_side)
    monkeypatch.setattr(idt, "user_id_to_server_id", user_side)
    assert idt.is_asset_ids_cache_exist() is expected


def test_translation_table_skips_rows_wider_than_two_columns(sandbox: Path) -> None:
    (sandbox / TRANSLATION_FILE).write_text("server_id,user_id\ns1,u1\ns2,u2,extra\ns3,u3\n")
    assert idt.translate_server_id_to_user_id("s1") == "u1"
    assert idt.translate_server_id_to_user_id("s2") == "s2"
    assert idt.translate_user_id_to_server_id("u2") == "u2"
    assert idt.translate_server_id_to_user_id("s3") == "u3"


def test_translation_table_accepts_rows_shorter_than_the_header(sandbox: Path) -> None:
    (sandbox / TRANSLATION_FILE).write_text("server_id,user_id,note\ns1,u1\n")
    assert idt.translate_server_id_to_user_id("s1") == "u1"
    assert idt.translate_user_id_to_server_id("u1") == "s1"


def test_blsgov_series_list_follows_pages_until_an_empty_page(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(qndc, "CACHE_RETENTION", datetime.timedelta(0))
    pages: dict[str, list[dict[str, str]]] = {
        "": [{"id": "s1"}, {"id": "s2"}],
        "s2": [{"id": "s3"}],
        "s3": [],
    }

    def page(_body: bytes | None, query: Query) -> bytes:
        if len(server.calls) > 6:
            raise AssertionError("series list does not stop")
        return json.dumps(pages[query.get("last_series", [""])[0]]).encode()

    server.route("bls.gov/series/list", page)
    listed = [s["id"] for s in qnbls.load_series_list("CU")]
    assert listed == ["s1", "s2", "s3"]
    assert server.calls == [
        "bls.gov/series/list?id=CU&last_series=",
        "bls.gov/series/list?id=CU&last_series=s2",
        "bls.gov/series/list?id=CU&last_series=s3",
    ]


def test_blsgov_series_data_and_aspect_window(server: FakeServer) -> None:
    server.serve_json("bls.gov/series/data", [{"value": "1"}])
    server.serve_json("bls.gov/series/aspect", {"aspect": "x"})
    assert qnbls.load_series_data("S1", max_date=MAX_DAY, tail=TAIL_DAYS) == [{"value": "1"}]
    assert qnbls.load_series_aspect("S1", max_date=MAX_DAY, tail=TAIL_DAYS) == {"aspect": "x"}
    qnbls.load_series_data("S1", min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    qnbls.load_series_aspect("S1", min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    assert server.calls == [
        f"bls.gov/series/data?id=S1&min_date={TAIL_START}&max_date={MAX_DAY}",
        f"bls.gov/series/aspect?id=S1&min_date={TAIL_START}&max_date={MAX_DAY}",
        f"bls.gov/series/data?id=S1&min_date={GIVEN_START}&max_date={MAX_DAY}",
        f"bls.gov/series/aspect?id=S1&min_date={GIVEN_START}&max_date={MAX_DAY}",
    ]


def test_index_requests_start_tail_days_before_max_date(server: FakeServer) -> None:
    server.serve_bytes("major-idx/data", netcdf_payload(cube(["close"], ["WORLD"], DAYS, seed=8)))
    qnindex.major_load_data(max_date=MAX_DAY, tail=TAIL_DAYS)
    qnindex.major_load_data(min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    assert server.calls == [
        f"major-idx/data?min_date={TAIL_START}&max_date={MAX_DAY}",
        f"major-idx/data?min_date={GIVEN_START}&max_date={MAX_DAY}",
    ]


def test_index_list_requests_start_tail_days_before_max_date(server: FakeServer) -> None:
    server.serve_json("idx/list", [{"id": "B"}, {"id": "A"}])
    assert qnindex.load_list(max_date=MAX_DAY, tail=TAIL_DAYS) == [{"id": "A"}, {"id": "B"}]
    qnindex.load_list(min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    assert server.calls == [
        f"idx/list?min_date={TAIL_START}&max_date={MAX_DAY}",
        f"idx/list?min_date={GIVEN_START}&max_date={MAX_DAY}",
    ]


def test_index_list_accepts_equal_bounds_and_rejects_reversed_ones(server: FakeServer) -> None:
    server.serve_json("idx/list", [{"id": "A"}])
    assert qnindex.load_list(min_date=MAX_DAY, max_date=MAX_DAY) == [{"id": "A"}]
    with pytest.raises(Exception, match="min_date must be less than or equal to max_date"):
        qnindex.load_list(min_date="2021-02-13", max_date=MAX_DAY)
    assert server.calls == [f"idx/list?min_date={MAX_DAY}&max_date={MAX_DAY}"]


def test_index_data_requests_start_tail_days_before_max_date(server: FakeServer) -> None:
    server.serve_json("idx/list", [{"id": "BBB"}])
    server.serve_bytes("idx/data", stocks_plane())
    qnindex.load_data(assets=["AAA"], max_date=MAX_DAY, tail=TAIL_DAYS)
    qnindex.load_data(assets=["AAA"], min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    qnindex.load_data(max_date=MAX_DAY, tail=TAIL_DAYS)
    expected = {"ids": ["AAA"], "min_date": TAIL_START, "max_date": MAX_DAY}
    assert [json.loads(body or b"{}") for body in server.bodies[:2]] == [
        expected,
        {**expected, "min_date": GIVEN_START},
    ]
    assert server.calls[2:] == [
        f"idx/list?min_date={TAIL_START}&max_date={MAX_DAY}",
        "idx/data",
    ]
    assert json.loads(server.bodies[3] or b"{}") == {**expected, "ids": ["BBB"]}


def test_index_weights_request_starts_tail_days_before_max_date(server: FakeServer) -> None:
    server.serve_bytes("idx-weights/data", stocks_plane())
    qnindex.load_weights("SPX", max_date=MAX_DAY, tail=TAIL_DAYS)
    qnindex.load_weights("SPX", min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    first, second = (json.loads(body or b"{}") for body in server.bodies)
    assert first == {"id": "SPX", "min_date": TAIL_START, "max_date": MAX_DAY}
    assert second == {"id": "SPX", "min_date": GIVEN_START, "max_date": MAX_DAY}


def test_index_weights_accept_equal_bounds_and_reject_reversed_ones(server: FakeServer) -> None:
    server.serve_bytes("idx-weights/data", stocks_plane())
    equal = qnindex.load_weights("SPX", min_date=MAX_DAY, max_date=MAX_DAY)
    assert equal.name == "index_weights"
    assert len(server.calls) == 1
    with pytest.raises(Exception, match="min_date must be less than or equal to max_date"):
        qnindex.load_weights("SPX", min_date="2021-02-13", max_date=MAX_DAY)
    assert len(server.calls) == 1


def test_cryptofutures_request_starts_tail_days_before_max_date(server: FakeServer) -> None:
    server.serve_bytes("cryptofutures", netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], DAYS)))
    qncryptofutures.load_data(max_date=MAX_DAY, tail=TAIL_DAYS)
    qncryptofutures.load_data(min_date=GIVEN_START, max_date=MAX_DAY, tail=TAIL_DAYS)
    assert server.calls == [
        f"cryptofutures?min_date={TAIL_START}&max_date={MAX_DAY}",
        f"cryptofutures?min_date={GIVEN_START}&max_date={MAX_DAY}",
    ]


def test_hourly_crypto_requests_start_tail_days_before_max_date(server: FakeServer) -> None:
    payload = netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], HOURS))
    server.serve_bytes("crypto", payload)
    server.serve_bytes("cryptodaily/data", payload)
    qncrypto.load_data(max_date=MAX_HOUR, tail=TAIL_DAYS)
    qncrypto.load_data(min_date=GIVEN_START_HOUR, max_date=MAX_HOUR, tail=TAIL_DAYS)
    qncryptodaily.load_data(max_date=MAX_HOUR, tail=TAIL_DAYS)
    qncryptodaily.load_data(min_date=GIVEN_START_HOUR, max_date=MAX_HOUR, tail=TAIL_DAYS)
    assert server.calls == [
        f"crypto?min_date={TAIL_START_HOUR}&max_date={MAX_HOUR}",
        f"crypto?min_date={GIVEN_START_HOUR}&max_date={MAX_HOUR}",
        f"cryptodaily/data?min_date={TAIL_START_HOUR}&max_date={MAX_HOUR}",
        f"cryptodaily/data?min_date={GIVEN_START_HOUR}&max_date={MAX_HOUR}",
    ]


def test_futures_window_starts_tail_days_before_max_date(server: FakeServer) -> None:
    server.serve_bytes(
        "futures/data", netcdf_payload(cube(FUTURES_FIELDS, ["ES", "NQ"], DAYS, seed=2))
    )
    result = qnfutures.load_data(max_date=MAX_DAY, tail=TAIL_DAYS)
    assert result is not None
    assert day_of(result.coords["time"].values[0]) == TAIL_START
    assert day_of(result.coords["time"].values[-1]) == MAX_DAY
    assert result.sizes["time"] == 9


def assert_imf_window(result: xr.DataArray, name: str) -> None:
    assert result.name == name
    assert day_of(result.coords["time"].values[0]) == TAIL_START
    assert day_of(result.coords["time"].values[-1]) == MAX_DAY
    assert result.sizes["time"] == 9


def test_imf_windows_start_tail_days_before_max_date(server: FakeServer) -> None:
    descending = plane(["EUR", "JPY"], DAYS, seed=7).sortby("time", ascending=False)
    server.serve_bytes("imf.org/currency/data", netcdf_payload(descending))
    server.serve_bytes("imf.org/commodity/data", netcdf_payload(descending))
    assert_imf_window(qnimf.load_currency_data(max_date=MAX_DAY, tail=TAIL_DAYS), "imf_currency")
    assert_imf_window(qnimf.load_commodity_data(max_date=MAX_DAY, tail=TAIL_DAYS), "imf_commodity")


def test_blockchaincom_window_starts_tail_days_before_max_date(server: FakeServer) -> None:
    metric = xr.DataArray(np.arange(30, dtype=np.float64), dims=["time"], coords={"time": DAYS})
    server.serve_bytes("blockchain.com/data", netcdf_payload(metric))
    result = qnblockchain.load_data("hash-rate", max_date=MAX_DAY, tail=TAIL_DAYS)
    assert_allclose(result.values, np.arange(21, 30, dtype=np.float64))


@dataclass(frozen=True)
class OrderCase:
    prepare: Callable[[FakeServer], None]
    default: Callable[[], xr.DataArray | None]
    explicit: Callable[[bool], xr.DataArray | None]


def serve_major(server: FakeServer) -> None:
    server.serve_bytes("major-idx/data", netcdf_payload(cube(["close"], ["WORLD"], DAYS, seed=8)))


def serve_index(server: FakeServer) -> None:
    server.serve_bytes("idx/data", stocks_plane())


def serve_weights(server: FakeServer) -> None:
    server.serve_bytes("idx-weights/data", stocks_plane())


def serve_cryptodaily(server: FakeServer) -> None:
    server.serve_bytes("cryptodaily/data", netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], HOURS)))


def serve_cryptofutures(server: FakeServer) -> None:
    server.serve_bytes("cryptofutures", netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], DAYS)))


def serve_commodity(server: FakeServer) -> None:
    descending = plane(["EUR", "JPY"], DAYS, seed=7).sortby("time", ascending=False)
    server.serve_bytes("imf.org/commodity/data", netcdf_payload(descending))


ORDER_CASES = {
    "major": OrderCase(
        serve_major,
        lambda: qnindex.major_load_data(min_date=WINDOW_START, max_date=MAX_DAY),
        lambda order: qnindex.major_load_data(
            min_date=WINDOW_START, max_date=MAX_DAY, forward_order=order
        ),
    ),
    "index": OrderCase(
        serve_index,
        lambda: qnindex.load_data(["AAA"], min_date=WINDOW_START, max_date=MAX_DAY),
        lambda order: qnindex.load_data(
            ["AAA"], min_date=WINDOW_START, max_date=MAX_DAY, forward_order=order
        ),
    ),
    "weights": OrderCase(
        serve_weights,
        lambda: qnindex.load_weights("SPX", min_date=WINDOW_START, max_date=MAX_DAY),
        lambda order: qnindex.load_weights(
            "SPX", min_date=WINDOW_START, max_date=MAX_DAY, forward_order=order
        ),
    ),
    "cryptodaily": OrderCase(
        serve_cryptodaily,
        lambda: qncryptodaily.load_data(min_date="2021-02-01T00", max_date="2021-02-10T23"),
        lambda order: qncryptodaily.load_data(
            min_date="2021-02-01T00", max_date="2021-02-10T23", forward_order=order
        ),
    ),
    "cryptofutures": OrderCase(
        serve_cryptofutures,
        lambda: qncryptofutures.load_data(min_date=WINDOW_START, max_date=MAX_DAY),
        lambda order: qncryptofutures.load_data(
            min_date=WINDOW_START, max_date=MAX_DAY, forward_order=order
        ),
    ),
    "commodity": OrderCase(
        serve_commodity,
        lambda: qnimf.load_commodity_data(min_date=WINDOW_START, max_date=MAX_DAY),
        lambda order: qnimf.load_commodity_data(
            min_date=WINDOW_START, max_date=MAX_DAY, forward_order=order
        ),
    ),
}


ORDER_ROWS = {
    "major": 30,
    "index": 30,
    "weights": 30,
    "cryptodaily": 240,
    "cryptofutures": 30,
    "commodity": 30,
}


@pytest.mark.parametrize("name", list(ORDER_CASES))
def test_loaders_default_to_forward_order(server: FakeServer, name: str) -> None:
    case = ORDER_CASES[name]
    case.prepare(server)
    default = case.default()
    forward = case.explicit(True)
    backward = case.explicit(False)
    assert default is not None
    assert forward is not None
    assert backward is not None
    assert default.sizes["time"] == ORDER_ROWS[name]
    assert_ascending(default)
    assert_array_equal(default.coords["time"].values, forward.coords["time"].values)
    assert_descending(backward)
    assert_array_equal(backward.coords["time"].values, forward.coords["time"].values[::-1])


@dataclass(frozen=True)
class EmptyCase:
    route: str
    call: Callable[[], xr.DataArray | None]
    sizes: dict[str, int]
    listing: bool = False


EMPTY_CASES = {
    "index": EmptyCase(
        "idx/data",
        lambda: qnindex.load_data(min_date=WINDOW_START, max_date=MAX_DAY),
        {"time": 0, "asset": 0},
        listing=True,
    ),
    "weights": EmptyCase(
        "idx-weights/data",
        lambda: qnindex.load_weights("SPX", min_date=WINDOW_START, max_date=MAX_DAY),
        {"time": 0, "asset": 0},
    ),
    "futures": EmptyCase(
        "futures/data",
        lambda: qnfutures.load_data(min_date=WINDOW_START, max_date=MAX_DAY),
        {"field": 6, "time": 0, "asset": 1},
    ),
    "crypto": EmptyCase(
        "crypto",
        lambda: qncrypto.load_data(min_date="2021-02-01T00", max_date="2021-02-10T23"),
        {"time": 0, "asset": 0},
    ),
    "cryptodaily": EmptyCase(
        "cryptodaily/data",
        lambda: qncryptodaily.load_data(min_date="2021-02-01T00", max_date="2021-02-10T23"),
        {"time": 0, "asset": 0},
    ),
}


def keep_axes(self: xr.DataArray, *_dims: Hashable) -> xr.DataArray:
    return self


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=pytest.mark.xfail(strict=True, reason="TypeError: object of type 'NoneType' has no len() on an empty futures response"),
        )
        if name == "futures"
        else case
        for name, case in EMPTY_CASES.items()
    ],
    ids=list(EMPTY_CASES),
)
def test_empty_response_gives_an_empty_array_without_its_first_row_and_column(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch, case: EmptyCase
) -> None:
    monkeypatch.setattr(xr.DataArray, "transpose", keep_axes)
    if case.listing:
        server.serve_json("idx/list", [{"id": "AAA"}])
    server.serve_bytes(case.route, b"")
    result = case.call()
    assert result is not None
    assert sizes_of(result) == case.sizes


@pytest.mark.parametrize("case", EMPTY_CASES.values(), ids=list(EMPTY_CASES))
def test_single_byte_response_is_parsed_not_treated_as_empty(
    server: FakeServer, case: EmptyCase
) -> None:
    if case.listing:
        server.serve_json("idx/list", [{"id": "AAA"}])
    server.serve_bytes(case.route, b"x")
    with pytest.raises(ValueError, match="did not find a match in any of xarray's"):
        case.call()


@pytest.mark.xfail(strict=True, reason="TypeError: object of type 'NoneType' has no len() on an empty futures response")
def test_futures_empty_response_drops_only_the_first_field(server: FakeServer) -> None:
    server.serve_bytes("futures/data", b"")
    result = qnfutures.load_data(min_date=WINDOW_START, max_date=MAX_DAY)
    assert result is not None
    assert list(result.coords["field"].values) == ["low", "high", "close", "vol", "oi", "roll"]
    assert list(result.coords["asset"].values) == ["ignore"]


def test_hourly_loaders_empty_response_with_default_dims_characterization(
    server: FakeServer,
) -> None:
    """Input: empty body, default dims. Expected (by the signature): empty (field, time, asset)
    array. Actual: the placeholder has no field dimension, so transpose raises ValueError (Dimensions {field} do not exist)."""
    server.serve_bytes("crypto", b"")
    server.serve_bytes("cryptodaily/data", b"")
    with pytest.raises(ValueError, match=r"Dimensions {.field.} do not exist"):
        qncrypto.load_data(min_date="2021-02-01T00", max_date="2021-02-10T23")
    with pytest.raises(ValueError, match=r"Dimensions {.field.} do not exist"):
        qncryptodaily.load_data(min_date="2021-02-01T00", max_date="2021-02-10T23")


@dataclass(frozen=True)
class FlagCase:
    route: str
    payload: bytes
    given_max: str
    call: Callable[[str | None], xr.DataArray | None]


FLAG_CASES = {
    "futures": FlagCase(
        "futures/data",
        netcdf_payload(cube(FUTURES_FIELDS, ["ES"], DAYS, seed=2)),
        MAX_DAY,
        lambda latest: qnfutures.load_data(min_date=WINDOW_START, max_date=latest),
    ),
    "crypto": FlagCase(
        "crypto",
        netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], HOURS)),
        MAX_HOUR,
        lambda latest: qncrypto.load_data(min_date="2021-02-01T00", max_date=latest),
    ),
    "cryptodaily": FlagCase(
        "cryptodaily/data",
        netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], HOURS)),
        MAX_HOUR,
        lambda latest: qncryptodaily.load_data(min_date="2021-02-01T00", max_date=latest),
    ),
    "cryptofutures": FlagCase(
        "cryptofutures",
        netcdf_payload(cube(CRYPTO_FIELDS, ["BTC"], DAYS)),
        MAX_DAY,
        lambda latest: qncryptofutures.load_data(min_date=WINDOW_START, max_date=latest),
    ),
}


@pytest.mark.parametrize(
    ("max_given", "flag_set", "written"),
    [(False, True, True), (True, True, False), (False, False, False), (True, False, False)],
    ids=["open-end-flagged", "explicit-end", "open-end-unflagged", "explicit-end-unflagged"],
)
@pytest.mark.parametrize("case", FLAG_CASES.values(), ids=list(FLAG_CASES))
def test_last_data_flag_is_written_only_for_open_ended_flagged_requests(
    server: FakeServer,
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: FlagCase,
    max_given: bool,
    flag_set: bool,
    written: bool,
) -> None:
    flag = sandbox / "flag.txt"
    if flag_set:
        monkeypatch.setenv("LAST_DATA_PATH", str(flag))
    else:
        monkeypatch.delenv("LAST_DATA_PATH", raising=False)
    server.serve_bytes(case.route, case.payload)
    case.call(case.given_max if max_given else None)
    assert flag.exists() is written
    assert not (sandbox / "last_data.txt").exists()
    if written:
        assert flag.read_text() == "last"


def test_batch_limits_of_hourly_crypto_modules() -> None:
    assert qncrypto.BATCH_LIMIT == 300000
    assert qncryptodaily.BATCH_LIMIT == 300000
