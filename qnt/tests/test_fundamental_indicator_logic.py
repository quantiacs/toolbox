from __future__ import annotations

import datetime as dt
import math
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from qnt.data import secgov_fundamental as sf, secgov_indicators as si
from qnt.data.common import DEFAULT_TAIL
from qnt.data.secgov_indicators import (
    IndicatorBuilder,
    IndicatorUtils,
    InstantIndicatorBuilder,
    PeriodIndicatorBuilder,
    append_period_variants,
    load_indicator_dicts,
    load_indicators,
    standard_indicator_builders,
)
from qnt.tests.test_fundamental_synthetic_loop import TIMES, asset, identity_builder, make_market

AssetInfo: TypeAlias = dict[str, str]
SecFact: TypeAlias = dict[str, Any]
IndicatorSpec: TypeAlias = dict[str, Any]
SeriesDict: TypeAlias = dict[str, float | None]

NAN = float("nan")
GAP = dt.timedelta(days=380)
FACT_X = "us-gaap:RevenueX"
FACT_Y = "us-gaap:RevenueY"
DEP = "us-gaap:Depreciation"


def tail_key(date: str) -> str:
    return (dt.datetime.fromisoformat(date) + GAP).strftime("%Y-%m-%d")


def same_value(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return left == right or (math.isnan(left) and math.isnan(right))


def assert_series(actual: SeriesDict, expected: SeriesDict) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert same_value(actual[key], value), (key, actual[key], value)


def sec_fact(
    name: str,
    period: str | list[str] | None,
    length: float | None,
    value: float | None,
    report_type: str,
    report_date: str,
    report_id: str = "r1",
    segment: object = None,
) -> SecFact:
    return {
        "cik": "c1",
        "report_id": report_id,
        "report_type": report_type,
        "report_date": report_date,
        "fact_name": name,
        "period": period,
        "period_length": length,
        "value": value,
        "segment": segment,
    }


def dep(
    period: list[str], length: float | None, value: float | None, report_type: str, report_date: str
) -> SecFact:
    return sec_fact(DEP, period, length, value, report_type, report_date)


class HasFloatValues(Protocol):
    @property
    def values(self) -> npt.NDArray[np.float64]: ...


def float_values(arr: object) -> npt.NDArray[np.float64]:
    return cast("HasFloatValues", arr).values


def period_builder(periods: str, use_report_date: bool = True) -> PeriodIndicatorBuilder:
    return PeriodIndicatorBuilder("x", [FACT_X], use_report_date, periods)


# build_series_fill_gaps


def test_fill_gaps_of_empty_series_is_empty() -> None:
    assert IndicatorUtils.build_series_fill_gaps([]) == []


# get_filtered and get_restored_by_segment


def lagged(name: str, lag: int, value: float) -> SecFact:
    end = dt.date(2020, 1, 1)
    report = (end + dt.timedelta(days=lag)).isoformat()
    return sec_fact(name, end.isoformat(), 90, value, "10-Q", report)


def test_get_filtered_drops_facts_reported_740_days_or_more_after_the_period_end() -> None:
    facts = [lagged(FACT_X, lag, float(lag)) for lag in (-5, 0, 739, 740, 741)]
    facts.append(lagged(FACT_Y, 0, 99.0))
    kept = IndicatorUtils.get_filtered(facts, [FACT_X])
    assert [fact["value"] for fact in kept] == [-5.0, 0.0, 739.0]


def test_get_filtered_accepts_a_custom_window() -> None:
    facts = [lagged(FACT_X, lag, float(lag)) for lag in (5, 9, 10, 11)]
    kept = IndicatorUtils.get_filtered(facts, [FACT_X], count_days_for_remove_old_period=10)
    assert [fact["value"] for fact in kept] == [5.0, 9.0]


def test_get_filtered_skips_facts_without_period_and_measures_from_the_period_end() -> None:
    facts = [
        sec_fact(FACT_X, None, 90, 1.0, "10-Q", "2020-04-01"),
        sec_fact(FACT_X, ["2015-01-01", "2020-03-31"], 90, 2.0, "10-Q", "2020-04-01"),
        sec_fact(FACT_X, ["2017-01-01", "2017-03-31"], 90, 3.0, "10-Q", "2020-04-01"),
        sec_fact(FACT_X, "2020-03-31", 90, 4.0, "10-Q", "2020-04-01"),
    ]
    kept = IndicatorUtils.get_filtered(facts, [FACT_X])
    assert [fact["value"] for fact in kept] == [2.0, 4.0]


def test_segment_facts_are_summed_by_period_type_and_length() -> None:
    first_quarter = ["2020-01-01", "2020-03-31"]
    second_quarter = ["2020-04-01", "2020-06-30"]
    facts = [
        sec_fact(FACT_X, first_quarter, 90, None, "10-Q", "2020-05-01", segment="S0"),
        sec_fact(FACT_X, first_quarter, 90, 3.0, "10-Q", "2020-05-01", segment="S1"),
        sec_fact(FACT_X, second_quarter, 91, 7.0, "10-Q", "2020-05-01", segment="S1"),
        sec_fact(FACT_X, first_quarter, 90, 5.0, "10-Q", "2020-05-01", segment="S2"),
        sec_fact(FACT_X, first_quarter, 90, 100.0, "10-K", "2020-05-01", segment="S1"),
        sec_fact(FACT_X, first_quarter, 90, None, "10-Q", "2020-05-01", segment="S3"),
        sec_fact(FACT_X, first_quarter, 90, 9.0, "10-Q", "2020-05-01", segment="S3"),
    ]
    restored = IndicatorUtils.get_restored_by_segment(facts)
    assert [fact["value"] for fact in restored] == [17.0, 7.0, 100.0]


def test_facts_without_segment_win_over_segment_facts() -> None:
    period = ["2020-01-01", "2020-03-31"]
    with_segment = sec_fact(FACT_X, period, 90, 3.0, "10-Q", "2020-05-01", segment="S1")
    plain = sec_fact(FACT_X, period, 90, 8.0, "10-Q", "2020-05-01")
    assert IndicatorUtils.get_restored_by_segment([with_segment, plain]) == [plain]


# get_ltm_amortization


def amortization_story() -> list[SecFact]:
    return [
        dep(["2019-01-01", "2019-03-31"], 89, 6.0, "10-Q", "2019-05-10"),
        dep(["2019-01-01", "2019-12-31"], 364, 100.0, "10-K", "2020-02-20"),
        sec_fact(FACT_Y, ["2019-01-01", "2019-12-31"], 364, 999.0, "10-K", "2020-02-20"),
        dep(["2020-01-01", "2020-03-31"], 90, 7.0, "S-1", "2020-04-10"),
        dep(["2020-01-01", "2020-03-31"], 90, 4.0, "10-Q", "2020-05-10"),
        dep(["2018-07-01", "2018-09-30"], 91, 1.0, "10-Q", "2020-05-10"),
        dep(["2019-01-01", "2019-03-31"], 89, 2.0, "10-Q", "2020-05-10"),
        dep(["2019-01-01", "2019-06-30"], 180, 30.0, "10-Q", "2020-08-10"),
        dep(["2020-01-01", "2020-06-30"], 181, 41.0, "10-Q", "2020-08-10"),
        dep(["2019-01-01", "2019-09-30"], 272, 60.0, "10-Q/A", "2020-11-10"),
        dep(["2020-01-01", "2020-09-30"], 273, 75.0, "10-Q/A", "2020-11-10"),
        dep(["2019-12-01", "2020-11-30"], 364, 50.0, "10-Q", "2020-12-01"),
        dep(["2020-01-01", "2020-12-31"], 365, 120.0, "10-K", "2021-02-20"),
        dep(["2021-01-01", "2021-03-31"], 90, 9.0, "10-Q", "2021-05-10"),
        dep(["2020-01-01", "2020-06-30"], 181, 3.0, "10-Q", "2021-08-10"),
        dep(["2021-01-01", "2021-06-30"], 181, 8.0, "10-Q", "2021-08-10"),
        dep(["2020-01-01", "2020-06-30"], 181, 3.0, "10-Q", "2021-11-10"),
        dep(["2021-07-01", "2021-09-30"], 90, 8.0, "10-Q", "2021-11-10"),
        dep(["2021-07-01", "2021-09-30"], 90, 2.0, "10-Q", "2022-11-10"),
        dep(["2022-01-01", "2022-09-30"], 273, 5.0, "10-Q", "2022-11-10"),
    ]


def test_amortization_is_last_year_plus_current_minus_previous_year_to_date() -> None:
    expected: SeriesDict = {
        "2020-02-20": 100.0,
        "2020-05-10": 100.0 + 4.0 - 2.0,
        "2020-08-10": 100.0 + 41.0 - 30.0,
        "2020-11-10": 100.0 + 75.0 - 60.0,
        "2021-02-20": 120.0,
        "2021-05-10": 9.0,
        "2021-08-10": 0.0 + 8.0 - 3.0,
        "2021-11-10": NAN,
        "2022-11-10": NAN,
    }
    assert_series(IndicatorUtils.get_ltm_amortization(DEP, amortization_story()), expected)


@pytest.mark.parametrize(
    ("report_type", "length", "recognised"),
    [
        ("10-K", 341, True),
        ("10-K", 364, True),
        ("10-K", 379, True),
        ("10-K", 340, False),
        ("10-K", 380, False),
        ("10-K/A", 364, True),
        ("8-K", 364, True),
        ("10-Q", 364, False),
        ("10-Q/A", 364, False),
        ("S-1", 364, False),
    ],
)


def test_amortization_recognises_a_year_by_report_type_and_length(
    report_type: str, length: float, recognised: bool
) -> None:
    facts = [dep(["2019-01-01", "2019-12-31"], length, 50.0, report_type, "2020-02-20")]
    expected: SeriesDict = {"2020-02-20": 50.0} if recognised else {}
    assert_series(IndicatorUtils.get_ltm_amortization(DEP, facts), expected)


@pytest.mark.parametrize(("value", "length"), [(None, 364), (50.0, None)])
def test_amortization_does_not_take_a_year_without_value_or_length(
    value: float | None, length: float | None
) -> None:
    facts = [dep(["2019-01-01", "2019-12-31"], length, value, "10-K", "2020-02-20")]
    assert IndicatorUtils.get_ltm_amortization(DEP, facts) == {}


@pytest.mark.parametrize("lag", [739, 740])
def test_amortization_drops_a_year_reported_740_days_after_its_end(lag: int) -> None:
    report_date = (dt.date(2018, 1, 1) + dt.timedelta(days=lag)).isoformat()
    facts = [dep(["2017-01-01", "2018-01-01"], 364, 50.0, "10-K", report_date)]
    expected: SeriesDict = {report_date: 50.0} if lag == 739 else {}
    assert IndicatorUtils.get_ltm_amortization(DEP, facts) == expected


def year_then(*quarter_facts: SecFact) -> list[SecFact]:
    year = dep(["2019-01-01", "2019-12-31"], 364, 100.0, "10-K", "2020-02-20")
    return [year, *quarter_facts]


def year_to_date_pair(length: float, report_type: str) -> list[SecFact]:
    return year_then(
        dep(["2019-01-01", "2019-03-31"], length, 3.0, report_type, "2020-05-10"),
        dep(["2020-01-01", "2020-03-31"], length, 8.0, report_type, "2020-05-10"),
    )


@pytest.mark.parametrize("length", [76, 100, 119, 151, 180, 209, 241, 270, 309])
def test_amortization_accepts_year_to_date_pairs_inside_the_quarter_ranges(length: float) -> None:
    result = IndicatorUtils.get_ltm_amortization(DEP, year_to_date_pair(length, "10-Q"))
    assert_series(result, {"2020-02-20": 100.0, "2020-05-10": 105.0})


@pytest.mark.parametrize("length", [30, 74, 75, 120, 130, 149, 150, 210, 220, 239, 240, 310, 330])
def test_amortization_rejects_year_to_date_pairs_outside_the_quarter_ranges(length: float) -> None:
    result = IndicatorUtils.get_ltm_amortization(DEP, year_to_date_pair(length, "10-Q"))
    assert_series(result, {"2020-02-20": 100.0, "2020-05-10": NAN})


@pytest.mark.parametrize(
    ("report_type", "accepted"),
    [("10-Q", True), ("10-Q/A", True), ("10-K", False), ("10-K/A", False), ("8-K", False)],
)


def test_amortization_pairs_need_a_quarterly_report_type(report_type: str, accepted: bool) -> None:
    result = IndicatorUtils.get_ltm_amortization(DEP, year_to_date_pair(100, report_type))
    assert_series(result, {"2020-02-20": 100.0, "2020-05-10": 105.0 if accepted else NAN})


@pytest.mark.parametrize(
    ("length", "year_long"),
    [
        (200, False),
        (339, False),
        (340, False),
        (341, True),
        (364, True),
        (379, True),
        (380, False),
        (500, False),
    ],
)


def test_amortization_leaves_year_long_facts_out_of_the_quarter_facts(
    length: float, year_long: bool
) -> None:
    facts = year_then(
        dep(["2019-01-01", "2019-12-31"], length, 3.0, "10-Q", "2020-05-10"),
        dep(["2020-01-01", "2020-03-31"], 90, 8.0, "10-Q", "2020-05-10"),
    )
    result = IndicatorUtils.get_ltm_amortization(DEP, facts)
    assert_series(result, {"2020-02-20": 100.0, "2020-05-10": 8.0 if year_long else NAN})


@pytest.mark.parametrize(("older_value", "older_length"), [(None, 90), (4.0, None)])
def test_amortization_ignores_facts_without_value_or_length_among_quarter_facts(
    older_value: float | None, older_length: float | None
) -> None:
    facts = year_then(
        dep(["2019-01-01", "2019-03-31"], older_length, older_value, "10-Q", "2020-05-10"),
        dep(["2020-01-01", "2020-03-31"], 90, 8.0, "10-Q", "2020-05-10"),
    )
    result = IndicatorUtils.get_ltm_amortization(DEP, facts)
    assert_series(result, {"2020-02-20": 100.0, "2020-05-10": 8.0})


# PeriodIndicatorBuilder


def test_annual_series_takes_only_year_long_facts() -> None:
    period = ["2019-01-01", "2019-12-31"]
    facts = [
        sec_fact(FACT_X, period, 339, 339.0, "10-K", "2020-01-01"),
        sec_fact(FACT_X, period, 340, 340.0, "10-K", "2020-02-01"),
        sec_fact(FACT_X, period, 341, 341.0, "10-K", "2020-03-01"),
        sec_fact(FACT_X, period, 379, 379.0, "10-K", "2020-04-01"),
        sec_fact(FACT_X, period, 380, 380.0, "10-K", "2020-05-01"),
        sec_fact(FACT_X, period, 500, 500.0, "10-K", "2020-07-01"),
        sec_fact(FACT_X, period, None, 1.0, "10-K", "2020-06-01"),
    ]
    assert period_builder("af").build_series_dict(facts) == {
        "2020-03-01": 341.0,
        "2020-04-01": 379.0,
    }


def test_annual_series_takes_the_latest_year_of_a_report() -> None:
    facts = [
        sec_fact(FACT_X, ["2019-01-01", "2019-12-31"], 364, 1.0, "10-K", "2021-02-20"),
        sec_fact(FACT_X, ["2020-01-01", "2020-12-31"], 365, 2.0, "10-K", "2021-02-20"),
    ]
    assert period_builder("af").build_series_dict(facts) == {"2021-02-20": 2.0}


# InstantIndicatorBuilder


@pytest.mark.parametrize("use_report_date", [True, False])
def test_first_listed_fact_has_priority_inside_a_report(use_report_date: bool) -> None:
    period = ["2020-01-01", "2020-12-31"]
    only_c = sec_fact("x:C", period, None, 33.0, "10-K", "2021-02-20")
    only_b = sec_fact("x:B", period, None, 22.0, "10-K", "2021-02-20")
    builder = InstantIndicatorBuilder("i", ["x:A", "x:B", "x:C"], use_report_date)
    key = "2021-02-20" if use_report_date else "2020-12-31"
    assert builder.build_series_dict([only_c, only_b]) == {key: 22.0, tail_key(key): 0}


def test_instant_series_by_period_groups_facts_by_the_period_end() -> None:
    older_start = sec_fact("x:A", ["2020-01-01", "2020-12-31"], None, 1.0, "10-K", "2021-02-20")
    newer_start = sec_fact("x:A", ["2020-06-01", "2020-12-31"], None, 2.0, "10-K", "2021-02-20")
    builder = InstantIndicatorBuilder("i", ["x:A"], False)
    expected = {"2020-12-31": 2.0, tail_key("2020-12-31"): 0}
    assert builder.build_series_dict([older_start, newer_start]) == expected


def test_instant_series_takes_the_latest_period_of_a_report() -> None:
    current = sec_fact("x:A", "2020-12-31", None, 8.0, "10-K", "2021-02-20")
    prior = sec_fact("x:A", "2019-12-31", None, 5.0, "10-K", "2021-02-20")
    builder = InstantIndicatorBuilder("i", ["x:A"], True)
    expected = {"2021-02-20": 8.0, tail_key("2021-02-20"): 0}
    assert builder.build_series_dict([prior, current]) == expected


# standard builders


def test_standard_builders_are_keyed_by_the_report_date() -> None:
    assert all(builder.use_report_date for builder in standard_indicator_builders)


def test_period_variants_exist_for_period_indicators_only() -> None:
    by_alias = {builder.alias: builder for builder in standard_indicator_builders}
    for alias in ("income", "capex", "cashflow_op"):
        base = by_alias[alias]
        for suffix, periods in (("_ltm", "ltm"), ("_af", "af"), ("_qf", "qf")):
            variant = by_alias[alias + suffix]
            assert isinstance(variant, PeriodIndicatorBuilder)
            assert (variant.periods, variant.facts) == (periods, base.facts)
    assert "assets_ltm" not in by_alias
    assert "assets_af" not in by_alias


def test_append_period_variants_keeps_the_list_and_adds_three_variants() -> None:
    base = PeriodIndicatorBuilder("rev", [FACT_X], True, "ltm")
    existing = [PeriodIndicatorBuilder("old", [FACT_Y], True, "af")]
    result = append_period_variants(existing, base)
    assert [(builder.alias, builder.periods) for builder in result] == [
        ("old", "af"),
        ("rev_ltm", "ltm"),
        ("rev_af", "af"),
        ("rev_qf", "qf"),
    ]
    assert result[0] is existing[0]
    assert all(builder.facts == [FACT_X] for builder in result[1:])


# load_indicator_dicts


@dataclass
class FactsCall:
    ciks: list[str]
    facts: list[str]
    kwargs: dict[str, object]


def patch_load_facts(
    monkeypatch: pytest.MonkeyPatch, groups: list[tuple[str, list[SecFact]]]
) -> list[FactsCall]:
    calls: list[FactsCall] = []

    def fake(
        ciks: list[str], facts: list[str], **kwargs: object
    ) -> Iterator[tuple[str, list[SecFact]]]:
        calls.append(FactsCall(ciks, facts, kwargs))
        yield from groups

    monkeypatch.setattr(si, "load_facts", fake)
    return calls


def test_indicator_dicts_use_custom_builders_first_and_selected_standard_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets_fact = sec_fact("us-gaap:Assets", "2019-12-31", None, 5.0, "10-K", "2020-02-20")
    calls = patch_load_facts(monkeypatch, [("c1", [assets_fact])])
    custom = InstantIndicatorBuilder("mine", ["x:Mine"], True)
    result = list(
        load_indicator_dicts(
            ["c1"], ["assets", "income"], [custom], "2020-01-01", "2020-12-31", tail=100
        )
    )
    expected = {
        "mine": {},
        "assets": {"2020-02-20": 5.0, tail_key("2020-02-20"): 0},
        "income": {},
    }
    assert result == [("c1", expected)]
    assert list(result[0][1]) == ["mine", "assets", "income"]
    assert len(calls) == 1
    assert calls[0].ciks == ["c1"]
    assert sorted(calls[0].facts) == ["us-gaap:Assets", "us-gaap:NetIncomeLoss", "x:Mine"]
    kwargs = calls[0].kwargs
    assert kwargs["min_date"] == "2020-01-01"
    assert kwargs["max_date"] == "2020-12-31"
    assert kwargs["tail"] == 100
    assert kwargs["skip_segment"] is True
    assert kwargs["group_by_cik"] is True


def test_indicator_dicts_without_a_selection_use_every_standard_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_load_facts(monkeypatch, [("c1", [])])
    result = list(load_indicator_dicts(["c1"]))
    assert list(result[0][1]) == list(
        dict.fromkeys(builder.alias for builder in standard_indicator_builders)
    )
    assert set(calls[0].facts) == {
        fact for builder in standard_indicator_builders for fact in builder.facts
    }
    assert calls[0].kwargs["tail"] == DEFAULT_TAIL
    assert calls[0].kwargs["min_date"] is None


def test_indicator_dicts_with_an_empty_selection_use_no_standard_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_load_facts(monkeypatch, [("c1", [])])
    assert list(load_indicator_dicts(["c1"], [])) == [("c1", {})]
    assert calls[0].facts == []


def test_indicator_dicts_share_annual_series_with_the_latest_year_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = [
        sec_fact(FACT_X, ["2019-01-01", "2019-12-31"], 364, 1.0, "10-K", "2021-02-20"),
        sec_fact(FACT_X, ["2020-01-01", "2020-12-31"], 365, 2.0, "10-K", "2021-02-20"),
    ]
    patch_load_facts(monkeypatch, [("c1", facts)])
    builders: list[IndicatorBuilder] = [PeriodIndicatorBuilder("fy", [FACT_X], True, "af")]
    assert list(load_indicator_dicts(["c1"], [], builders)) == [("c1", {"fy": {"2021-02-20": 2.0}})]


# load_indicators


@dataclass
class DictsCall:
    ciks: list[str]
    standard_indicators: Collection[str] | None
    builders: Iterable[IndicatorBuilder] | None
    min_date: str | dt.date | None
    max_date: str | dt.date | None


def patch_indicator_dicts(
    monkeypatch: pytest.MonkeyPatch, groups: list[tuple[str, dict[str, SeriesDict]]]
) -> list[DictsCall]:
    calls: list[DictsCall] = []

    def fake(
        ciks: list[str],
        standard_indicators: Collection[str] | None = None,
        builders: Iterable[IndicatorBuilder] | None = None,
        min_date: str | dt.date | None = None,
        max_date: str | dt.date | None = None,
    ) -> Iterator[tuple[str, dict[str, SeriesDict]]]:
        calls.append(DictsCall(ciks, standard_indicators, builders, min_date, max_date))
        yield from groups

    monkeypatch.setattr(si, "load_indicator_dicts", fake)
    return calls


LOAD_TIMES = ["2020-01-03", "2020-01-06", "2020-01-09"]


def cell(result: xr.DataArray, asset_id: str, field: str) -> list[float]:
    return [float(value) for value in float_values(result.sel(asset=asset_id, field=field))]


def assert_cell(result: xr.DataArray, asset_id: str, field: str, expected: list[float]) -> None:
    np.testing.assert_array_equal(np.array(cell(result, asset_id, field)), np.array(expected))


def test_load_indicators_builds_the_cube_in_cik_order_and_fills_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_indicator_dicts(
        monkeypatch,
        [
            (
                "c2",
                {"f1": {"2020-01-04": 5.0}, "f2": {"2020-01-02": 7.0, "2020-01-08": 9.0}, "f3": {}},
            ),
            (
                "c1",
                {"f1": {"2020-01-01": 1.0, "2020-01-10": 4.0}, "f2": {}, "f3": {"2020-01-06": 3.0}},
            ),
        ],
    )
    assets = [asset("AAA", "c1"), asset("BBB", "c2"), asset("NOC")]
    result = load_indicators(assets, make_market(LOAD_TIMES, ["AAA"]).time)
    assert result is not None
    assert result.dims == ("time", "field", "asset")
    assert result.name == "secgov_indicators"
    assert list(result.field.values) == ["f1", "f2", "f3"]
    assert list(result.asset.values) == ["BBB", "AAA"]
    assert [str(t)[:10] for t in result.time.values] == LOAD_TIMES
    assert_cell(result, "BBB", "f1", [NAN, 5.0, 5.0])
    assert_cell(result, "BBB", "f2", [7.0, 7.0, 9.0])
    assert_cell(result, "BBB", "f3", [NAN, NAN, NAN])
    assert_cell(result, "AAA", "f1", [1.0, 1.0, 1.0])
    assert_cell(result, "AAA", "f2", [NAN, NAN, NAN])
    assert_cell(result, "AAA", "f3", [NAN, 3.0, 3.0])
    assert len(calls) == 1
    assert calls[0].ciks == ["c1", "c2"]
    assert calls[0].standard_indicators is None
    assert calls[0].builders is None
    assert calls[0].min_date == dt.date(2020, 1, 3) - dt.timedelta(days=730)
    assert calls[0].max_date == dt.date(2020, 1, 9)


@pytest.mark.parametrize(
    ("assets", "expected_ciks"),
    [([asset("AAA", "c1")], ["c1"]), ([asset("NOC")], [])],
    ids=["no-dicts", "no-cik-assets"],
)


def test_load_indicators_returns_none_without_data_to_fill(
    monkeypatch: pytest.MonkeyPatch, assets: list[AssetInfo], expected_ciks: list[str]
) -> None:
    calls = patch_indicator_dicts(monkeypatch, [])
    assert load_indicators(assets, make_market(LOAD_TIMES, ["AAA"]).time) is None
    assert calls[0].ciks == expected_ciks


@pytest.mark.parametrize(("offset", "days"), [(10, 10), (dt.timedelta(days=5), 5)])
def test_load_indicators_single_asset_with_custom_offset_and_fill(
    monkeypatch: pytest.MonkeyPatch, offset: dt.timedelta | int, days: int
) -> None:
    calls = patch_indicator_dicts(
        monkeypatch, [("c1", {"f1": {"2020-01-03": 1.0, "2020-01-04": 2.0}})]
    )
    time_coord = make_market(["2020-01-03", "2020-01-06"], ["AAA"]).time
    assets = [asset("AAA", "c1")]
    seen: list[list[str]] = []

    def no_fill(arr: xr.DataArray) -> xr.DataArray:
        seen.append([str(moment)[:10] for moment in np.array(arr.coords["time"].values, dtype="datetime64[ns]")])
        return arr

    unfilled = load_indicators(
        assets, time_coord, ["f"], [], start_date_offset=offset, fill_strategy=no_fill
    )
    assert unfilled is not None
    assert_cell(unfilled, "AAA", "f1", [1.0, NAN])
    assert seen == [["2020-01-03", "2020-01-04", "2020-01-06"]]
    assert calls[0].standard_indicators == ["f"]
    assert calls[0].builders == []
    assert calls[0].min_date == dt.date(2020, 1, 3) - dt.timedelta(days=days)
    filled = load_indicators(assets, time_coord)
    assert filled is not None
    assert_cell(filled, "AAA", "f1", [1.0, 2.0])


# secgov_fundamental.load_indicators_for


def patch_computed_facts(
    monkeypatch: pytest.MonkeyPatch, fields: list[str], value: float
) -> list[tuple[list[str] | None, str, dict[str, IndicatorSpec] | None]]:
    calls: list[tuple[list[str] | None, str, dict[str, IndicatorSpec] | None]] = []

    def fake(
        data: xr.DataArray,
        names: list[str] | None = None,
        period: str = "ltm",
        builders: dict[str, IndicatorSpec] | None = None,
    ) -> xr.DataArray:
        calls.append((names, period, builders))
        computed = sf.initialize_fundamental_data(data, fields)
        key = {"field": fields[0], "time": pd.Timestamp("2020-01-03"), "asset": "AAA"}
        computed.loc[key] = value
        return computed

    monkeypatch.setattr(sf, "get_computed_facts", fake)
    return calls


def test_load_indicators_for_builds_the_requested_indicators_and_fills_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = patch_computed_facts(monkeypatch, [FACT_X], 5.0)
    builders = identity_builder([FACT_X])
    result = sf.load_indicators_for(make_market(TIMES, ["AAA", "BBB"]), ["ind"], "qf", builders)
    assert calls == [(["ind"], "qf", builders)]
    assert result.dims == ("time", "field", "asset")
    assert result.name == "secgov_indicators"
    assert list(result.field.values) == ["ind"]
    assert_cell(result, "AAA", "ind", [NAN, 5.0, 5.0, 5.0])
    assert_cell(result, "BBB", "ind", [NAN] * 4)


def test_load_indicators_for_defaults_to_all_indicators(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = sorted({fact for spec in sf.GLOBAL_INDICATORS.values() for fact in spec["facts"]})
    calls = patch_computed_facts(monkeypatch, facts, 5.0)
    result = sf.load_indicators_for(make_market(TIMES, ["AAA", "BBB"]))
    assert len(calls) == 1
    assert calls[0][0] == sf.get_all_indicator_names()
    assert calls[0][2] == sf.GLOBAL_INDICATORS
    assert list(result.field.values) == sf.get_all_indicator_names()
