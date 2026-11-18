from __future__ import annotations

import warnings
from typing import Any, TypeAlias

import numpy as np
import pytest
import xarray as xr

from qnt.data import secgov_fundamental as sf
from qnt.data.secgov_indicators import PeriodIndicatorBuilder
from qnt.tests.test_fundamental_synthetic_loop import (
    FACT_A,
    TIMES,
    Scenario,
    asset,
    identity_builder,
    install,
    make_market,
    values,
)

SecFact: TypeAlias = dict[str, Any]
NAN = float("nan")


def mk_fact(
    name: str,
    period: tuple[str, str],
    length: float,
    value: float | None,
    report_type: str,
    report_date: str,
    report_id: str,
) -> SecFact:
    return {
        "cik": "c1",
        "report_id": report_id,
        "report_type": report_type,
        "report_date": report_date,
        "fact_name": name,
        "period": [period[0], period[1]],
        "period_length": length,
        "value": value,
        "segment": None,
    }


FACT_X = "us-gaap:RevenueX"


def recovered_quarter(last_q_end: str) -> float | None:
    facts = [
        mk_fact(FACT_X, ("2020-01-01", "2020-03-31"), 90, 10.0, "10-Q", "2020-05-01", "r1"),
        mk_fact(FACT_X, ("2020-01-01", "2020-12-31"), 365, 400.0, "10-K", "2021-02-15", "r2"),
        mk_fact(FACT_X, ("2020-10-01", last_q_end), 90, 100.0, "10-K", "2021-02-15", "r2"),
    ]
    result = PeriodIndicatorBuilder("x", [FACT_X], True, "qf").build_series_dict(facts)
    assert result["2020-05-01"] == 10.0
    return result["2021-02-15"]


COMPUTED_RECOVERY = 400.0 - (400.0 - 30.0) / 3


@pytest.mark.parametrize(
    ("last_q_end", "expected"),
    [
        ("2020-12-31", 100.0),
        ("2020-12-26", 100.0),
        ("2020-12-27", 100.0),
        ("2021-01-05", 100.0),
        ("2021-01-04", 100.0),
        ("2020-12-25", COMPUTED_RECOVERY),
        ("2020-12-24", COMPUTED_RECOVERY),
        ("2021-01-06", COMPUTED_RECOVERY),
        ("2021-01-07", COMPUTED_RECOVERY),
        ("2021-01-20", COMPUTED_RECOVERY),
        ("2020-11-30", COMPUTED_RECOVERY),
    ],
)


def test_recovered_quarter_tolerance_is_five_days(last_q_end: str, expected: float) -> None:
    assert recovered_quarter(last_q_end) == pytest.approx(expected)


def run_direct(
    monkeypatch: pytest.MonkeyPatch,
    times: list[str],
    assets: list[str],
    points: list[tuple[str, float]],
    name: str | None = None,
) -> tuple[xr.DataArray, Scenario]:
    sc = Scenario(
        points={("c1", FACT_A): points}, cik_order=["c1"], assets=[asset(assets[0], "c1")]
    )
    strategy = install(monkeypatch, sc, [FACT_A])
    result = sf.get_computed_facts(
        make_market(times, assets, name),
        ["ind"],
        "ltm",
        identity_builder([FACT_A]),
        build_instant_strategy=strategy,
    )
    return result, sc


def test_unsorted_time_axis_with_duplicates_does_not_warn_about_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = ["2020-01-02", "2020-01-06", "2020-01-06", "2020-01-03"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(KeyError):
            run_direct(monkeypatch, times, ["AAA", "BBB"], [("2020-01-04", 1.0)])
    assert not [w for w in caught if "duplicate" in str(w.message)]


def test_unsorted_time_axis_without_points_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    times = ["2020-01-02", "2020-01-06", "2020-01-03"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, _ = run_direct(monkeypatch, times, ["AAA", "BBB"], [])
    assert not caught
    assert np.isnan(result.values).all()


def test_many_unique_assets_do_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    assets = [f"A{i}" for i in range(300)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, _ = run_direct(monkeypatch, TIMES, assets, [("2020-01-03", 7.0)])
    assert not [w for w in caught if "duplicate" in str(w.message)]
    np.testing.assert_array_equal(np.array(values(result, "A0")), np.array([NAN, 7.0, NAN, NAN]))


@pytest.mark.parametrize(
    ("name", "stocks_type"),
    [
        ("nasdaq100_close", "NASDAQ100"),
        ("my_s&p500_data", "SPX500"),
        ("something", ""),
        (None, ""),
    ],
)


def test_stocks_type_is_taken_from_the_data_name(
    monkeypatch: pytest.MonkeyPatch, name: str | None, stocks_type: str
) -> None:
    _, sc = run_direct(monkeypatch, TIMES, ["AAA", "BBB"], [("2020-01-03", 7.0)], name=name)
    assert [call[2] for call in sc.load_list_calls] == [stocks_type]
