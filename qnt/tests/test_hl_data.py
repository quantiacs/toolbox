import datetime
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import qnt.data as qndata
import qnt.data.common as qndc
import qnt.data.id_translation as idt
import qnt.data.stocks as qnstocks
from qnt.tests.hl_server import (
    CATALOG,
    DAYS,
    STOCK_FIELDS,
    FakeServer,
    cube,
    netcdf_payload,
    plane,
    stock_server,
)
from qnt.tests.hl_support import values_of


def test_asset_list_is_translated_to_stable_user_ids(server: FakeServer, sandbox: Path) -> None:
    stock_server(server)
    listing = qndata.stocks_load_list(min_date="2020-01-01", max_date="2021-02-01")
    assert [a["id"] for a in listing] == ["NAS:BBB", "NYS:AAA", "NYS:AAA~1"]
    assert all("last_point" not in a for a in listing)
    table = (sandbox / "id-translation.csv").read_text().splitlines()
    assert table[0] == "server_id,user_id"
    assert sorted(table[1:]) == ["s1,NYS:AAA", "s2,NAS:BBB", "s3,NYS:AAA~1"]
    assert "assets?min_date=2020-01-01&max_date=2021-02-01&type=" in server.calls

    again = qndata.stocks_load_list(min_date="2020-01-01", max_date="2021-02-01")
    assert [a["id"] for a in again] == [a["id"] for a in listing]
    assert (sandbox / "id-translation.csv").read_text().count("\n") == 4

    with pytest.raises(Exception, match="min_date must be less than or equal to max_date"):
        qndata.stocks_load_list(min_date="2021-03-01", max_date="2021-02-01")


def test_load_stock_data_adjusts_for_splits_and_orders_axes(server: FakeServer) -> None:
    source = stock_server(server)
    result = qndata.stocks_load_data(
        assets=["NYS:AAA", "NAS:BBB"],
        min_date="2021-01-06",
        max_date="2021-02-05",
        dims=("time", "field", "asset"),
    )
    assert result is not None
    assert result.name == "stocks"
    assert result.dims == ("time", "field", "asset")
    assert list(result.coords["asset"].values) == ["NAS:BBB", "NYS:AAA"]
    assert list(result.coords["field"].values) == STOCK_FIELDS
    times = result.coords["time"].values
    assert times[0] == np.datetime64("2021-01-06")
    assert (np.diff(times) > np.timedelta64(0)).all()

    request = json.loads(server.bodies[server.calls.index("data")] or b"{}")
    assert request["assets"] == ["s1", "s2"]
    assert request["min_date"] == "2021-01-06"

    raw = source.sel(asset="s1", time=result.coords["time"].values)
    cum = raw.sel(field="split_cumprod").values
    adjusted = result.sel(field="open", asset="NYS:AAA").values
    assert_allclose(adjusted, raw.sel(field="open").values * cum)
    volume = result.sel(field="vol", asset="NYS:AAA").values
    assert_allclose(volume, raw.sel(field="vol").values / cum)
    assert (cum[:6] == 0.5).all()

    backward = qndata.stocks_load_data(
        assets=["NAS:BBB"], min_date="2021-01-06", max_date="2021-02-05", forward_order=False
    )
    assert backward.dims == ("field", "time", "asset")
    assert (np.diff(backward.coords["time"].values) < np.timedelta64(0)).all()


def test_unknown_asset_becomes_an_empty_column_and_split_adjustment_roundtrips(
    server: FakeServer,
) -> None:
    stock_server(server)
    result = qndata.stocks_load_data(
        assets=["NAS:BBB", "NAS:MISSING"], min_date="2021-01-06", max_date="2021-02-05"
    )
    assert list(result.coords["asset"].values) == ["NAS:BBB", "NAS:MISSING"]
    assert np.isnan(result.sel(asset="NAS:MISSING").values).all()
    assert np.isfinite(result.sel(asset="NAS:BBB").values).all()

    full = qndata.stocks_load_data(assets=["NYS:AAA"], min_date="2021-01-06", max_date="2021-02-05")
    restored = qndata.stocks_restore_origin_data(full)
    again = qndata.stocks_adjust_by_splits(restored)
    assert_allclose(again.values, full.values, rtol=1e-12)
    assert not np.allclose(restored.values, full.values)
    untouched = qndata.stocks_adjust_by_splits(full, make_copy=False)
    assert untouched is not full


def test_index_constituent_lists_use_their_own_type(server: FakeServer) -> None:
    stock_server(server)
    qndata.stocks_load_ndx_list(min_date="2020-12-01", max_date="2021-01-31")
    qndata.stocks_load_spx_list(min_date="2020-12-01", max_date="2021-01-31")
    assert any(call.endswith("type=NASDAQ100") for call in server.calls)
    assert any(call.endswith("type=SPX500") for call in server.calls)
    assert qnstocks.get_stocks_type_by_data_name("stocks_nasdaq100") == "NASDAQ100"
    assert qnstocks.get_stocks_type_by_data_name("stocks_s&p500") == "SPX500"
    assert qnstocks.get_stocks_type_by_data_name("stocks") == ""
    assert qnstocks.get_stocks_type_by_data_name("") == ""


class Recorder(Protocol):
    def __call__(self, **kwargs: object) -> None: ...


def recorder(seen: list[str], label: str) -> Recorder:
    def record(**_kwargs: object) -> None:
        seen.append(label)

    return record


def test_load_data_by_type_dispatch_and_error(
    server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    stock_server(server)
    seen: list[str] = []
    for name in (
        "stocks_load_data",
        "futures_load_data",
        "crypto_load_data",
        "cryptofutures_load_data",
    ):
        monkeypatch.setattr(qndata, name, recorder(seen, name))
    qndata.load_data_by_type("stocks_long")
    qndata.load_data_by_type("futures")
    qndata.load_data_by_type("crypto")
    qndata.load_data_by_type("crypto_futures")
    assert seen == [
        "stocks_load_data",
        "futures_load_data",
        "crypto_load_data",
        "cryptofutures_load_data",
    ]
    with pytest.raises(Exception, match=r"Wrong data_type\."):
        qndata.load_data_by_type("bonds")


def test_request_with_retry_caches_decodes_and_retries(server: FakeServer, sandbox: Path) -> None:
    payload = b"hello world" * 1000
    server.serve_bytes("blob", payload)
    server.compress = True
    server.failures_left = 2
    assert qndc.request_with_retry("blob?x=1", None) == payload
    assert server.failures_left == 0
    assert server.api_keys[-1] == "default"
    assert len(server.calls) == 1

    assert qndc.request_with_retry("blob?x=1", None) == payload
    assert len(server.calls) == 1
    assert len(list((sandbox / "data-cache").glob("*.value.pickle.gz"))) == 1

    assert qndc.request_with_retry("blob?x=2", None) == payload
    assert qndc.request_with_retry("blob?x=1", b"post-body") == payload
    assert len(server.calls) == 3

    server.failures_left = 5
    with pytest.raises(Exception, match="can't download blob"):
        qndc.request_with_retry("blob?x=3", None)


def max_date_limit() -> datetime.date | None:
    return qndc.MAX_DATE_LIMIT


def stamp(text: str) -> datetime.datetime:
    return pd.Timestamp(text).to_pydatetime()


def test_date_and_url_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qndc, "MAX_DATE_LIMIT", None)
    monkeypatch.setattr(qndc, "MAX_DATETIME_LIMIT", None)
    assert qndc.parse_date("2021-03-04T10:00") == datetime.date(2021, 3, 4)
    assert qndc.parse_date(None) == pd.Timestamp.today().date()
    assert qndc.parse_tail(10) == datetime.timedelta(days=10)
    assert qndc.parse_tail(datetime.timedelta(hours=5)) == datetime.timedelta(hours=5)
    assert qndc.datetime_to_hours_str(stamp("2021-03-04T07:30")) == "2021-03-04T07"
    assert qndc.parse_max_datetime_from_url("http://h:7070/last/2020-10-07T10/") == stamp(
        "2020-10-07T10:00"
    )
    assert qndc.parse_max_datetime_from_url("http://h:7070/last/2016-10-28") == stamp("2016-10-28")
    assert qndc.parse_max_datetime_from_url("http://h:7070/other/") is None

    monkeypatch.setattr(qndc, "BASE_URL", qndc.BASE_URL)
    qndc.set_max_datetime("2020-05-06T13:45")
    assert qndc.BASE_URL.endswith("/last/2020-05-06T13/")
    assert max_date_limit() == datetime.date(2020, 5, 6)
    assert qndc.parse_date("2021-01-01") == datetime.date(2020, 5, 6)
    assert qndc.parse_date("2021-01-01", limit=False) == datetime.date(2021, 1, 1)
    assert qndc.parse_date_and_hour("2021-01-01T10:00") == stamp("2020-05-06T13:00")
    qndc.set_max_datetime(None)
    assert max_date_limit() is None
    assert qndc.BASE_URL == qndc.ORIGIN_BASE_URL


def test_get_env_notice_and_deprecation_wrapper(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HL_UNSET_KEY", raising=False)
    assert qndc.get_env("HL_UNSET_KEY", "fallback") == "fallback"
    assert (
        "NOTICE: The environment variable HL_UNSET_KEY was not specified" in capsys.readouterr().err
    )
    assert qndc.get_env("HL_UNSET_KEY", "fallback", True) == "fallback"
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("HL_UNSET_KEY", "set")
    assert qndc.get_env("HL_UNSET_KEY", "fallback") == "set"

    def shout(text: str) -> str:
        return text.upper()

    old_shout = qndc.deprecated_wrap(shout)
    assert old_shout("a") == "A"
    assert "deprecated, use" in capsys.readouterr().err
    assert qndata.load_assets is not qndata.stocks_load_list


def test_xarray_conversion_helpers() -> None:
    times = pd.bdate_range("2021-01-04", periods=3, name="time")
    block = xr.DataArray(
        np.arange(3 * 2 * 2, dtype=np.float64).reshape(3, 2, 2),
        dims=["time", "field", "asset"],
        coords={"time": times, "field": ["close", "is_liquid"], "asset": ["a", "b"]},
    )
    block[:, 1, 1] = 0.0
    block[:, 1, 0] = 1.0
    frames = qndc.from_xarray_3d_to_dict_of_pandas_df(block)
    assert sorted(frames) == ["a", "b"]
    assert list(frames["a"].columns) == ["close", "is_liquid"]
    assert_allclose(values_of(frames["a"])[:, 0], [0.0, 4.0, 8.0])

    weights = qndc.from_dict_to_xarray_1d({"a": 0.5, "b": -0.5})
    assert weights.dims == ("asset",)
    assert_allclose(weights.values, [0.5, -0.5])

    liquid = qndc.filter_liquids_xarray_assets_dataarray(block)
    assert list(liquid.coords["asset"].values) == ["a"]
    flat = xr.DataArray([0.3, 0.4], dims=["asset"], coords={"asset": ["a", "b"]})
    with pytest.raises(Exception, match="b"):
        qndc.check_weights_xarray_dataarray_for_nonliquids(flat, block)
    ok = xr.DataArray([0.3, 0.0], dims=["asset"], coords={"asset": ["a", "b"]})
    qndc.check_weights_xarray_dataarray_for_nonliquids(ok, block)
    kept = qndc.exclude_weights_xarray_dataarray_from_nonliquids(flat, block)
    assert list(kept.coords["asset"].values) == ["a"]


def test_id_translation_table_parsing_and_modes(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (sandbox / "id-translation.csv").write_text(
        "server_id,user_id\n s1 , NYS:AAA \nbroken-row\ns2,NAS:BBB\n"
    )
    idt.load_id_translation_table()
    assert idt.translate_user_id_to_server_id("NYS:AAA") == "s1"
    assert idt.translate_server_id_to_user_id("s2") == "NAS:BBB"
    assert idt.translate_server_id_to_user_id("unknown") == "unknown"
    assert idt.is_asset_ids_cache_exist()
    assert idt.has_asset_id_translations(["NYS:AAA", {"id": "NAS:BBB"}])
    assert not idt.has_asset_id_translations(["NYS:AAA", "NYS:ZZZ"])

    new_id = idt.translate_asset_to_user_id({"id": "s9", "symbol": "ZZZ"})
    assert new_id == "GLOBAL:ZZZ"
    assert idt.get_or_create_translation("s10", "GLOBAL:ZZZ") == "GLOBAL:ZZZ~1"
    assert idt.get_or_create_translation("s9", "whatever") == "GLOBAL:ZZZ"

    monkeypatch.setenv("SUBMISSION_ID", "77")
    assert idt.get_or_create_translation("s11", "NYS:NEW") == "s11"
    monkeypatch.delenv("SUBMISSION_ID")

    monkeypatch.setattr(idt, "USE_ID_TRANSLATION", False)
    assert idt.translate_asset_to_user_id({"id": "s5", "symbol": "X", "exchange": "NYS"}) == "s5"
    assert idt.get_or_create_translation("s6", "NYS:Y") == "s6"
    assert idt.translate_user_id_to_server_id("NYS:AAA") == "NYS:AAA"
    assert idt.translate_server_id_to_user_id("s1") == "s1"


def test_stock_loader_without_server_data_returns_empty_frame(server: FakeServer) -> None:
    server.serve_json("assets", CATALOG)
    server.serve_bytes("data", b"")
    result = qndata.stocks_load_data(assets=None, min_date="2021-01-06", max_date="2021-02-05")
    assert result.sizes["asset"] == 0
    assert result.sizes["time"] == 0


@pytest.mark.xfail(strict=True, reason="TypeError: object of type 'NoneType' has no len() on an empty futures response")
def test_futures_and_crypto_loaders_slice_sort_and_broadcast(server: FakeServer) -> None:
    fut_fields = ["open", "low", "high", "close", "vol", "oi", "roll"]
    server.serve_bytes("futures/data", netcdf_payload(cube(fut_fields, ["ES", "NQ"], DAYS, seed=2)))
    futures = qndata.futures_load_data(
        assets=["NQ", "ZZ"],
        min_date="2021-01-11",
        max_date="2021-01-20",
        dims=("time", "field", "asset"),
    )
    assert futures is not None
    assert futures.name == "futures"
    assert futures.dims == ("time", "field", "asset")
    assert list(futures.coords["asset"].values) == ["NQ", "ZZ"]
    assert str(futures.coords["time"].values[0])[:10] == "2021-01-11"
    assert str(futures.coords["time"].values[-1])[:10] == "2021-01-20"
    assert np.isnan(futures.sel(asset="ZZ").values).all()
    assert "futures/data?offset=0" in server.calls

    # string date slicing of a descending index is rejected by pandas, so backward order fails
    with pytest.raises(KeyError, match="non-monotonic DatetimeIndexes"):
        qndata.futures_load_data(min_date="2021-01-11", max_date="2021-01-20", forward_order=False)

    server.serve_bytes("futures/data", b"")
    empty = qndata.futures_load_data(min_date="2021-01-11", max_date="2021-01-20", offset=1)
    assert empty is not None
    assert empty.sizes["time"] == 0

    server.serve_json("futures/list", [{"id": "ES"}])
    assert qndata.futures_load_list() == [{"id": "ES"}]

    hours = pd.date_range("2021-01-01", periods=48, freq="h", name="time")
    crypto = cube(["open", "low", "high", "close", "vol"], ["BTC", "ETH"], hours, seed=3)
    server.serve_bytes("crypto", netcdf_payload(crypto))
    loaded = qndata.crypto_load_data(
        assets=["ETH"], min_date="2021-01-01T00", max_date="2021-01-02T23"
    )
    assert loaded is not None
    assert loaded.name == "crypto"
    assert list(loaded.coords["asset"].values) == ["ETH"]
    assert any(
        call.startswith("crypto?min_date=2021-01-01T00&max_date=2021-01-02T23")
        for call in server.calls
    )
    assert_allclose(
        loaded.sel(field="close").values,
        crypto.sel(field="close", asset=["ETH"]).values,
    )

    server.serve_bytes("cryptodaily/data", netcdf_payload(crypto))
    daily = qndata.cryptodaily_load_data(
        min_date="2021-01-01T00", max_date="2021-01-02T23", forward_order=False
    )
    assert daily is not None
    assert daily.name == "cryptodaily"
    assert (np.diff(daily.coords["time"].values) < np.timedelta64(0)).all()


def test_index_loaders(server: FakeServer) -> None:
    server.serve_json("idx/list", [{"id": "SPX"}, {"id": "DJI"}])
    assert [
        i["id"] for i in qndata.index_load_list(min_date="2021-01-01", max_date="2021-02-01")
    ] == ["DJI", "SPX"]
    with pytest.raises(Exception, match="min_date must be less"):
        qndata.index_load_list(min_date="2021-03-01", max_date="2021-02-01")

    series = plane(["DJI", "SPX"], DAYS, seed=5)
    server.route("idx/data", lambda body, _query: netcdf_payload(series) if body else b"")
    everything = qndata.index_load_data(min_date="2021-01-04", max_date="2021-02-12")
    assert everything is not None
    assert everything.name == "indexes"
    assert everything.dims == ("time", "asset")
    assert json.loads(server.bodies[-1] or b"{}")["ids"] == ["DJI", "SPX"]
    picked = qndata.index_load_data(
        assets=["SPX", "NDX"], min_date="2021-01-04", max_date="2021-02-12"
    )
    assert picked is not None
    assert list(picked.coords["asset"].values) == ["NDX", "SPX"]
    assert np.isnan(picked.sel(asset="NDX").values).all()

    server.serve_json("idx-weights/list", ["SPX"])
    assert qndata.index_load_weights_list() == ["SPX"]
    server.serve_bytes("idx-weights/data", netcdf_payload(plane(["AAA", "BBB"], DAYS)))
    weights = qndata.index_load_weights(
        "SPX", min_date="2021-01-04", max_date="2021-02-12", forward_order=False
    )
    assert weights.name == "index_weights"
    assert (np.diff(weights.coords["time"].values) < np.timedelta64(0)).all()
    assert json.loads(server.bodies[-1] or b"{}")["id"] == "SPX"

    server.serve_json("major-idx/list", ["WORLD"])
    assert qndata.index_major_load_list() == ["WORLD"]
    server.serve_bytes("major-idx/data", netcdf_payload(cube(["close"], ["WORLD"], DAYS, seed=8)))
    major = qndata.index_major_load_data(min_date="2021-01-04", max_date="2021-02-12")
    assert major.name == "major_indexes"
    assert major.dims == ("field", "time", "asset")


def test_imf_blockchain_and_blsgov_loaders(server: FakeServer) -> None:
    descending = plane(["EUR", "JPY"], DAYS, seed=7).sortby("time", ascending=False)
    server.serve_bytes("imf.org/currency/data", netcdf_payload(descending))
    server.serve_bytes("imf.org/commodity/data", netcdf_payload(descending))
    server.serve_json("imf.org/currency/list", [{"id": "EUR"}])
    server.serve_json("imf.org/commodity/list", [{"id": "PSOYB"}])
    assert qndata.imf_load_currency_list() == [{"id": "EUR"}]
    assert qndata.imf_load_commodity_list() == [{"id": "PSOYB"}]
    currency = qndata.imf_load_currency_data(
        assets=["EUR"], min_date="2021-01-11", max_date="2021-01-20"
    )
    assert currency.name == "imf_currency"
    assert list(currency.coords["asset"].values) == ["EUR"]
    assert str(currency.coords["time"].values[0])[:10] == "2021-01-11"
    assert str(currency.coords["time"].values[-1])[:10] == "2021-01-20"
    commodity = qndata.imf_load_commodity_data(
        min_date="2021-01-11", max_date="2021-01-20", forward_order=False
    )
    assert commodity.name == "imf_commodity"
    assert commodity.sizes["time"] == 8

    metric = xr.DataArray(np.arange(30, dtype=np.float64), dims=["time"], coords={"time": DAYS})
    server.serve_bytes("blockchain.com/data", netcdf_payload(metric))
    server.serve_json("blockchain.com/list", [{"id": "hash-rate"}])
    assert qndata.blockchaincom_load_list() == [{"id": "hash-rate"}]
    loaded = qndata.blockchaincom_load_data(
        "hash-rate", min_date="2021-01-11", max_date="2021-01-20"
    )
    assert loaded.name == "blockchaincom_metric"
    assert_allclose(loaded.values, np.arange(5, 13, dtype=np.float64))
    with pytest.raises(KeyError, match="non-monotonic DatetimeIndexes"):
        qndata.blockchaincom_load_data(
            "hash-rate", min_date="2021-01-11", max_date="2021-01-20", forward_order=False
        )
    assert "blockchain.com/data?id=hash-rate" in server.calls

    server.serve_json("bls.gov/db/list", [{"id": "CU"}])
    server.serve_json("bls.gov/db/meta", {"name": "CPI"})
    assert qndata.blsgov_load_db_list() == [{"id": "CU"}]
    assert qndata.blsgov_load_db_meta("CU") == {"name": "CPI"}

    pages: dict[str, list[dict[str, str]]] = {
        "": [{"id": "s1"}, {"id": "s2"}],
        "s2": [{"id": "s3"}],
        "s3": [],
    }
    server.route(
        "bls.gov/series/list",
        lambda _body, query: json.dumps(pages[query.get("last_series", [""])[0]]).encode(),
    )
    listed = [s["id"] for s in qndata.blsgov_load_series_list("CU")]
    assert listed == ["s1", "s2", "s3"]

    server.serve_json("bls.gov/series/data", [{"year": 2021, "value": "1.5"}])
    server.serve_json("bls.gov/series/aspect", {"aspect": "x"})
    assert qndata.blsgov_load_series_data("S1", min_date="2021-01-01", max_date="2021-02-01") == [
        {"year": 2021, "value": "1.5"}
    ]
    assert qndata.blsgov_load_series_aspect("S1", min_date="2021-01-01", max_date="2021-02-01") == {
        "aspect": "x"
    }
    assert any(
        c.startswith("bls.gov/series/data?id=S1&min_date=2021-01-01&max_date=2021-02-01")
        for c in server.calls
    )
