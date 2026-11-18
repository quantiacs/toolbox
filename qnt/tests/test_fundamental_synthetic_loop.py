from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from qnt.data import secgov_fundamental as sf

AssetInfo: TypeAlias = dict[str, str]
SecFact: TypeAlias = dict[str, Any]
IndicatorSpec: TypeAlias = dict[str, Any]
Point: TypeAlias = tuple[float | None, dt.datetime]

FACT_A = "us-gaap:Assets"
FACT_B = "us-gaap:StockholdersEquity"
TIMES = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
ASSETS = ["AAA", "BBB"]


def asset(asset_id: str, cik: str | None = None) -> AssetInfo:
    info: AssetInfo = {"id": asset_id, "symbol": asset_id}
    if cik is not None:
        info["cik"] = cik
    return info


def make_market(times: list[str], assets: list[str], name: str | None = None) -> xr.DataArray:
    data = (
        np.arange(len(times) * 1 * len(assets), dtype=np.float64).reshape(
            len(times), 1, len(assets)
        )
        + 100
    )
    return xr.DataArray(
        data,
        dims=("time", "field", "asset"),
        coords={"time": pd.DatetimeIndex(times), "field": ["close"], "asset": assets},
        name=name,
    )


def identity_builder(facts: list[str]) -> dict[str, IndicatorSpec]:
    def build(data: xr.DataArray) -> xr.DataArray:
        return data.sel(field=facts[0])

    return {"ind": {"facts": facts, "build": build}}


@dataclass
class Scenario:
    points: dict[tuple[str, str], list[tuple[str, float]]] = field(default_factory=dict)
    cik_order: list[str] = field(default_factory=list)
    assets: list[AssetInfo] = field(default_factory=list)
    load_facts_calls: list[tuple[list[str], list[str], dict[str, object]]] = field(
        default_factory=list
    )
    load_list_calls: list[tuple[object, object, str]] = field(default_factory=list)


def install(
    monkeypatch: pytest.MonkeyPatch, sc: Scenario, facts: list[str]
) -> Callable[[list[Point]], list[Point]]:
    keys: dict[float, tuple[str, str]] = {}
    for cik in sc.cik_order:
        for fact in facts:
            keys[float(len(keys) + 1)] = (cik, fact)

    def fake_list(
        min_date: object = None, max_date: object = None, tail: object = None, stocks_type: str = ""
    ) -> list[AssetInfo]:
        sc.load_list_calls.append((min_date, max_date, stocks_type))
        return sc.assets

    def fake_facts(
        ciks: list[str], fact_names: list[str], **kwargs: object
    ) -> Iterator[tuple[str, list[SecFact]]]:
        sc.load_facts_calls.append((ciks, fact_names, kwargs))
        for cik in sc.cik_order:
            reports: list[SecFact] = []
            for key, (k_cik, k_fact) in keys.items():
                if k_cik == cik and k_fact in fact_names:
                    reports.append(
                        {
                            "cik": cik,
                            "report_id": "r1",
                            "report_type": "10-K",
                            "report_date": "2020-01-03",
                            "fact_name": k_fact,
                            "period": "2020-01-01",
                            "period_length": None,
                            "value": key,
                            "segment": None,
                        }
                    )
            yield cik, reports

    def strategy(series: list[Point]) -> list[Point]:
        key = series[0][0]
        assert key is not None
        cik, fact = keys[key]
        return [(v, dt.datetime.fromisoformat(d)) for d, v in sc.points[(cik, fact)]]

    monkeypatch.setattr(sf, "load_list", fake_list)
    monkeypatch.setattr(sf, "load_facts", fake_facts)
    return strategy


def run(
    monkeypatch: pytest.MonkeyPatch,
    points: dict[tuple[str, str], list[tuple[str, float]]],
    cik_asset: dict[str, str],
    times: list[str] = TIMES,
    market_assets: list[str] = ASSETS,
    facts: list[str] | None = None,
    cik_order: list[str] | None = None,
) -> tuple[xr.DataArray, Scenario]:
    facts = facts or [FACT_A]
    sc = Scenario(
        points=points,
        cik_order=cik_order or list(cik_asset),
        assets=[asset(a, c) for c, a in cik_asset.items()],
    )
    strategy = install(monkeypatch, sc, facts)
    result = sf.get_computed_facts(
        make_market(times, market_assets),
        ["ind"],
        "ltm",
        identity_builder(facts),
        build_instant_strategy=strategy,
    )
    return result, sc


def values(result: xr.DataArray, asset: str = "AAA", fact: str = FACT_A) -> list[float]:
    return [float(v) for v in result.sel(asset=asset, field=fact).values]


NAN = float("nan")


def assert_values(actual: list[float], expected: list[float]) -> None:
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_date_between_labels_goes_to_next_label(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = run(monkeypatch, {("c1", FACT_A): [("2020-01-04", 7.0)]}, {"c1": "AAA"})
    assert_values(values(result), [NAN, NAN, 7.0, NAN])


def test_date_exactly_on_label(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = run(monkeypatch, {("c1", FACT_A): [("2020-01-03", 7.0)]}, {"c1": "AAA"})
    assert_values(values(result), [NAN, 7.0, NAN, NAN])


def test_date_before_first_label_goes_to_first(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = run(monkeypatch, {("c1", FACT_A): [("2019-06-01", 7.0)]}, {"c1": "AAA"})
    assert_values(values(result), [7.0, NAN, NAN, NAN])


def test_date_exactly_on_last_label_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = run(monkeypatch, {("c1", FACT_A): [("2020-01-07", 7.0)]}, {"c1": "AAA"})
    assert_values(values(result), [NAN, NAN, NAN, 7.0])


def test_date_after_last_label_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = run(monkeypatch, {("c1", FACT_A): [("2020-01-08", 7.0)]}, {"c1": "AAA"})
    assert_values(values(result), [NAN] * 4)


def test_single_time_point_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {("c1", FACT_A): [("2020-01-05", 1.0), ("2020-01-06", 2.0)]}
    result, _ = run(monkeypatch, points, {"c1": "AAA"}, times=["2020-01-06"])
    assert_values(values(result), [2.0])
    result, _ = run(
        monkeypatch, {("c1", FACT_A): [("2020-01-07", 1.0)]}, {"c1": "AAA"}, times=["2020-01-06"]
    )
    assert_values(values(result), [NAN])


def test_out_of_range_date_does_not_stop_following_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {("c1", FACT_A): [("2021-01-01", 9.0), ("2020-01-03", 7.0)]}
    result, _ = run(monkeypatch, points, {"c1": "AAA"})
    assert_values(values(result), [NAN, 7.0, NAN, NAN])


def test_duplicate_cell_last_written_wins_ascending(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {("c1", FACT_A): [("2020-01-04", 1.0), ("2020-01-05", 2.0)]}
    result, _ = run(monkeypatch, points, {"c1": "AAA"})
    assert_values(values(result), [NAN, NAN, 2.0, NAN])


def test_duplicate_cell_last_written_wins_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {("c1", FACT_A): [("2020-01-05", 2.0), ("2020-01-04", 1.0)]}
    result, _ = run(monkeypatch, points, {"c1": "AAA"})
    assert_values(values(result), [NAN, NAN, 1.0, NAN])


def test_duplicate_cell_with_nan_last_overwrites(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {("c1", FACT_A): [("2020-01-04", 1.0), ("2020-01-05", NAN)]}
    result, _ = run(monkeypatch, points, {"c1": "AAA"})
    assert_values(values(result), [NAN] * 4)


def test_duplicate_cell_across_ciks_of_same_asset_later_cik_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = {("c1", FACT_A): [("2020-01-03", 1.0)], ("c2", FACT_A): [("2020-01-03", 2.0)]}
    result, _ = run(monkeypatch, points, {"c1": "AAA", "c2": "AAA"})
    assert_values(values(result), [NAN, 2.0, NAN, NAN])


def test_values_land_in_own_asset_and_field(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {
        ("c1", FACT_A): [("2020-01-03", 1.0)],
        ("c1", FACT_B): [("2020-01-06", 2.0)],
        ("c2", FACT_A): [("2020-01-07", 3.0)],
        ("c2", FACT_B): [("2020-01-02", 4.0)],
    }
    result, _ = run(monkeypatch, points, {"c1": "AAA", "c2": "BBB"}, facts=[FACT_A, FACT_B])
    assert_values(values(result, "AAA", FACT_A), [NAN, 1.0, NAN, NAN])
    assert_values(values(result, "AAA", FACT_B), [NAN, NAN, 2.0, NAN])
    assert_values(values(result, "BBB", FACT_A), [NAN, NAN, NAN, 3.0])
    assert_values(values(result, "BBB", FACT_B), [4.0, NAN, NAN, NAN])


def test_unknown_cik_is_skipped_and_following_cik_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sc = Scenario(
        points={("zz", FACT_A): [("2020-01-03", 5.0)], ("c1", FACT_A): [("2020-01-06", 7.0)]},
        cik_order=["zz", "c1"],
        assets=[asset("AAA", "c1")],
    )
    strategy = install(monkeypatch, sc, [FACT_A])
    result = sf.get_computed_facts(
        make_market(TIMES, ASSETS),
        ["ind"],
        "ltm",
        identity_builder([FACT_A]),
        build_instant_strategy=strategy,
    )
    assert_values(values(result), [NAN, NAN, 7.0, NAN])
    assert_values(values(result, "BBB"), [NAN] * 4)


def test_cik_asset_missing_in_axis_raises_only_when_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(KeyError):
        run(monkeypatch, {("c1", FACT_A): [("2020-01-03", 1.0)]}, {"c1": "ZZZ"})
    result, _ = run(monkeypatch, {("c1", FACT_A): [("2021-01-03", 1.0)]}, {"c1": "ZZZ"})
    assert_values(values(result), [NAN] * 4)


def test_asset_without_cik_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = Scenario(points={}, cik_order=[], assets=[asset("AAA")])
    install(monkeypatch, sc, [FACT_A])
    with pytest.raises(KeyError, match="cik"):
        sf.get_computed_facts(
            make_market(TIMES, ASSETS), ["ind"], "ltm", identity_builder([FACT_A])
        )


def test_empty_series_leaves_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = run(monkeypatch, {("c1", FACT_A): []}, {"c1": "AAA"})
    assert_values(values(result), [NAN] * 4)


def test_default_builders_used_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = Scenario(
        points={("c1", FACT_A): [("2020-01-03", 5.0)]},
        cik_order=["c1"],
        assets=[asset("AAA", "c1")],
    )
    strategy = install(monkeypatch, sc, [FACT_A])
    result = sf.get_computed_facts(
        make_market(TIMES, ASSETS), ["assets"], "ltm", None, build_instant_strategy=strategy
    )
    assert list(result.field.values) == [FACT_A]
    assert_values(values(result), [NAN, 5.0, NAN, NAN])


def test_default_indicator_names_use_all_indicators(monkeypatch: pytest.MonkeyPatch) -> None:
    sc = Scenario(points={}, cik_order=[], assets=[])
    install(monkeypatch, sc, [FACT_A])
    result = sf.get_computed_facts(make_market(TIMES, ASSETS))
    expected = {fact for spec in sf.GLOBAL_INDICATORS.values() for fact in spec["facts"]}
    assert set(result.field.values) == expected


def test_invalid_time_period_raises() -> None:
    with pytest.raises(ValueError, match="time_period"):
        sf.get_computed_facts(make_market(TIMES, ASSETS), ["assets"], "bad")


def test_input_market_not_modified(monkeypatch: pytest.MonkeyPatch) -> None:
    market = make_market(TIMES, ASSETS)
    before = market.copy(deep=True)
    sc = Scenario(
        points={("c1", FACT_A): [("2020-01-03", 5.0)]},
        cik_order=["c1"],
        assets=[asset("AAA", "c1")],
    )
    strategy = install(monkeypatch, sc, [FACT_A])
    sf.get_computed_facts(
        market, ["ind"], "ltm", identity_builder([FACT_A]), build_instant_strategy=strategy
    )
    xr.testing.assert_identical(market, before)


def test_unsorted_time_axis_behaviour_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    times = ["2020-01-06", "2020-01-02", "2020-01-03"]
    with pytest.raises(KeyError):
        run(monkeypatch, {("c1", FACT_A): [("2020-01-04", 1.0)]}, {"c1": "AAA"}, times=times)


def test_unique_axes_emit_no_duplicate_warning(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    run(monkeypatch, {("c1", FACT_A): [("2020-01-03", 7.0)]}, {"c1": "AAA"})
    assert not [w for w in recwarn if "duplicate" in str(w.message)]
