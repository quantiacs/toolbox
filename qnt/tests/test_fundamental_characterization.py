import datetime as dt
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import pytest
import xarray as xr

from qnt.data import secgov_fundamental as sf
from qnt.data.secgov_indicators import (
    IndicatorUtils,
    InstantIndicatorBuilder,
    PeriodIndicatorBuilder,
)
from qnt.tests.test_fundamental_indicator_logic import (
    DEP,
    FACT_X,
    dep,
    float_values,
    sec_fact,
)
from qnt.tests.test_fundamental_synthetic_builders import facts as fact_cube
from qnt.tests.test_fundamental_synthetic_loop import (
    ASSETS,
    FACT_A,
    NAN,
    TIMES,
    Scenario,
    assert_values,
    asset,
    identity_builder,
    install,
    make_market,
    values,
)

SecFact: TypeAlias = dict[str, Any]
SeriesDict: TypeAlias = dict[str, float | None]
Items: TypeAlias = list[tuple[str, float | None]]

YEAR = ["2020-01-01", "2020-12-31"]
THREE_YEAR_REPORT_DATES = ["2021-02-20", "2021-08-20", "2022-02-20"]
BORDER_GAP_START = "2021-02-20"
BORDER_GAP_END = "2022-03-07"
AFTER_BORDER_GAP_END = "2023-03-22"
AFTER_THREE_YEARS = "2023-03-07"

QUARTER_REPORT_DATES = ["2020-05-08", "2020-08-07", "2020-11-06", "2021-02-05", "2021-05-07"]
QUARTER_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]
AFTER_QUARTERS = "2022-05-22"


def items(series: SeriesDict) -> Items:
    return list(series.items())


def annual_facts(report_dates: list[str], year_values: list[float]) -> list[SecFact]:
    return [
        sec_fact(FACT_X, YEAR, 365, value, "10-K", report_date)
        for report_date, value in zip(report_dates, year_values, strict=True)
    ]


def quarter_facts() -> list[SecFact]:
    return [
        sec_fact(
            FACT_X, [f"2020-0{index + 1}-01", f"2020-0{index + 1}-28"], 90, value, "10-Q", date
        )
        for index, (date, value) in enumerate(
            zip(QUARTER_REPORT_DATES, QUARTER_VALUES, strict=True)
        )
    ]


def instant_facts(report_dates: list[str], year_values: list[float]) -> list[SecFact]:
    return [
        sec_fact(FACT_X, "2020-12-31", None, value, "10-K", report_date)
        for report_date, value in zip(report_dates, year_values, strict=True)
    ]


# Key order of SeriesDict


def test_ltm_series_keys_are_in_reverse_order() -> None:
    builder = PeriodIndicatorBuilder("x", [FACT_X], True, "ltm")
    series = builder.build_series_dict(quarter_facts())
    assert items(series) == [
        (AFTER_QUARTERS, 0),
        ("2021-05-07", 30.0),
        ("2021-02-05", 15.0),
    ]


def test_ltm_series_keys_from_annual_values_are_in_reverse_order() -> None:
    builder = PeriodIndicatorBuilder("x", [FACT_X], True, "ltm")
    series = builder.build_series_dict(
        annual_facts([THREE_YEAR_REPORT_DATES[1], *THREE_YEAR_REPORT_DATES[::2]], [2.0, 1.0, 3.0])
    )
    assert items(series) == [
        (AFTER_THREE_YEARS, 0),
        ("2022-02-20", 3.0),
        ("2021-08-20", 2.0),
        ("2021-02-20", 1.0),
    ]


def test_qf_series_keys_are_in_reverse_order() -> None:
    builder = PeriodIndicatorBuilder("x", [FACT_X], True, "qf")
    series = builder.build_series_dict(quarter_facts()[::-1])
    assert items(series) == [
        ("2021-05-07", 16.0),
        ("2021-02-05", 8.0),
        ("2020-11-06", 4.0),
        ("2020-08-07", 2.0),
        ("2020-05-08", 1.0),
    ]


def test_instant_series_keys_are_in_forward_order() -> None:
    builder = InstantIndicatorBuilder("x", [FACT_X], True)
    series = builder.build_series_dict(
        instant_facts([THREE_YEAR_REPORT_DATES[1], *THREE_YEAR_REPORT_DATES[::2]], [2.0, 1.0, 3.0])
    )
    assert items(series) == [
        ("2021-02-20", 1.0),
        ("2021-08-20", 2.0),
        ("2022-02-20", 3.0),
        (AFTER_THREE_YEARS, 0),
    ]


def test_amortization_series_keys_are_in_reverse_order() -> None:
    all_facts = [
        dep(["2019-01-01", "2019-12-31"], 364, 100.0, "10-K", "2020-02-20"),
        dep(["2020-01-01", "2020-12-31"], 365, 120.0, "10-K", "2021-02-20"),
        dep(["2021-01-01", "2021-12-31"], 365, 140.0, "10-K", "2022-02-20"),
    ]
    series = IndicatorUtils.get_ltm_amortization(DEP, all_facts)
    assert items(series) == [("2022-02-20", 140.0), ("2021-02-20", 120.0), ("2020-02-20", 100.0)]


# Collision on exactly BORDER_MAX_GAP_DAYS


def test_ltm_zero_marker_overwrites_value_at_exact_border_gap_current_behavior() -> None:
    assert IndicatorUtils.BORDER_MAX_GAP_DAYS == 380
    builder = PeriodIndicatorBuilder("x", [FACT_X], True, "ltm")
    series = builder.build_series_dict(
        annual_facts([BORDER_GAP_START, BORDER_GAP_END], [5.0, 7.0]),
    )
    assert (
        dt.date.fromisoformat(BORDER_GAP_END) - dt.date.fromisoformat(BORDER_GAP_START)
    ).days == 380
    assert items(series) == [
        (AFTER_BORDER_GAP_END, 0),
        (BORDER_GAP_END, 0),
        (BORDER_GAP_START, 5.0),
    ]


def test_instant_value_wins_over_zero_marker_at_exact_border_gap_current_behavior() -> None:
    builder = InstantIndicatorBuilder("x", [FACT_X], True)
    series = builder.build_series_dict(
        instant_facts([BORDER_GAP_START, BORDER_GAP_END], [5.0, 7.0])
    )
    assert items(series) == [
        (BORDER_GAP_START, 5.0),
        (BORDER_GAP_END, 7.0),
        (AFTER_BORDER_GAP_END, 0),
    ]


# Segment accumulation mutates the input facts


def test_segment_accumulation_changes_the_value_of_the_first_input_fact_current_behavior() -> None:
    first = sec_fact(FACT_X, YEAR, 365, 30.0, "10-K", "2021-02-20", segment="S1")
    second = sec_fact(FACT_X, YEAR, 365, 12.0, "10-K", "2021-02-20", segment="S2")
    restored = IndicatorUtils.get_restored_by_segment([first, second])
    assert len(restored) == 1
    assert restored[0] is first
    assert first["value"] == 42.0
    assert second["value"] == 12.0


def test_ltm_annual_series_sees_accumulated_segment_value_current_behavior() -> None:
    first = sec_fact(FACT_X, YEAR, 365, 30.0, "10-K", "2021-02-20", segment="S1")
    second = sec_fact(FACT_X, YEAR, 365, 12.0, "10-K", "2021-02-20", segment="S2")
    series = PeriodIndicatorBuilder("x", [FACT_X], True, "ltm").build_series_dict([first, second])
    assert items(series) == [("2022-03-07", 0), ("2021-02-20", 42.0)]
    assert first["value"] == 42.0
    assert second["value"] == 12.0


def test_annual_series_alone_does_not_accumulate_segments_current_behavior() -> None:
    first = sec_fact(FACT_X, YEAR, 365, 30.0, "10-K", "2021-02-20", segment="S1")
    second = sec_fact(FACT_X, YEAR, 365, 12.0, "10-K", "2021-02-20", segment="S2")
    series = PeriodIndicatorBuilder("x", [FACT_X], True, "af").build_series_dict([first, second])
    assert items(series) == [("2021-02-20", 30.0)]
    assert first["value"] == 30.0
    assert second["value"] == 12.0


# Single-fact builders


SINGLE_FACT_BUILDERS: list[tuple[Callable[[xr.DataArray], xr.DataArray], str]] = [
    (sf.build_losses_on_extinguishment_of_debt, "us-gaap:GainsLossesOnExtinguishmentOfDebt"),
    (sf.build_shares, "dei:EntityCommonStockSharesOutstanding"),
    (sf.build_assets, "us-gaap:Assets"),
    (sf.build_revenues, "us-gaap:Revenues"),
    (
        sf.build_income_before_income_taxes,
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ),
    (sf.build_operating_income, "us-gaap:OperatingIncomeLoss"),
    (sf.build_income_interest, "us-gaap:InvestmentIncomeInterest"),
    (sf.build_interest_expense, "us-gaap:InterestExpense"),
    (sf.build_interest_expense_debt, "us-gaap:InterestExpenseDebt"),
    (
        sf.build_interest_expense_capital_lease,
        "us-gaap:InterestExpenseLesseeAssetsUnderCapitalLease",
    ),
    (sf.build_interest_income_expense_net, "us-gaap:InterestIncomeExpenseNet"),
    (sf.build_other_nonoperating_income_expense, "us-gaap:OtherNonoperatingIncomeExpense"),
    (sf.build_nonoperating_income_expense, "us-gaap:NonoperatingIncomeExpense"),
    (sf.build_cash_and_cash_equivalents, "us-gaap:CashAndCashEquivalentsAtCarryingValue"),
    (sf.build_short_term_investments, "us-gaap:ShortTermInvestments"),
    (sf.build_net_income, "us-gaap:NetIncomeLoss"),
]


@pytest.mark.parametrize(
    ("builder", "fact_name"),
    SINGLE_FACT_BUILDERS,
    ids=[fact_name for _, fact_name in SINGLE_FACT_BUILDERS],
)


def test_single_fact_builder_returns_its_own_fact(
    builder: Callable[[xr.DataArray], xr.DataArray], fact_name: str
) -> None:
    columns = {
        name: [1000.0 * (index + 1) + step for step in (1.0, 2.0, 3.0)]
        for index, (_, name) in enumerate(SINGLE_FACT_BUILDERS)
    }
    result = builder(fact_cube(**columns))
    assert result.dims == ("time",)
    np.testing.assert_array_equal(float_values(result), np.array(columns[fact_name]))


# get_computed_facts


def test_computed_facts_keep_market_shape_name_and_attrs_without_forward_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = make_market(TIMES, ASSETS, name="stocks")
    market.attrs = {"source": "synthetic"}
    scenario = Scenario(
        points={("c1", FACT_A): [("2020-01-03", 7.0)]},
        cik_order=["c1"],
        assets=[asset("AAA", "c1")],
    )
    strategy = install(monkeypatch, scenario, [FACT_A])
    result = sf.get_computed_facts(
        market, ["ind"], "ltm", identity_builder([FACT_A]), build_instant_strategy=strategy
    )
    assert result.dims == ("field", "time", "asset")
    assert result.name == market.name
    assert result.attrs == market.attrs
    assert list(result.field.values) == [FACT_A]
    np.testing.assert_array_equal(result.time.values, market.time.values)
    np.testing.assert_array_equal(result.asset.values, market.asset.values)
    assert_values(values(result), [NAN, 7.0, NAN, NAN])
    assert_values(values(result, "BBB"), [NAN] * 4)
