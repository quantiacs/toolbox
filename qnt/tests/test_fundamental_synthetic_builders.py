from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from qnt.data import secgov_fundamental as sf

NAN = float("nan")
P = "us-gaap:"
ALL_FIELDS = sorted(
    {fact for spec in sf.GLOBAL_INDICATORS.values() for fact in spec["facts"]}
    | {"close", "dei:EntityCommonStockSharesOutstanding"}
)
IBT_FACT = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"
EQUITY_FULL = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"


def facts(**columns: list[float]) -> xr.DataArray:
    n = len(next(iter(columns.values())))
    data = np.full((n, len(ALL_FIELDS)), np.nan)
    for key, column in columns.items():
        name = key if ":" in key or key == "close" else P + key
        data[:, ALL_FIELDS.index(name)] = column
    return xr.DataArray(
        data,
        dims=("time", "field"),
        coords={"time": pd.date_range("2020-01-01", periods=n), "field": ALL_FIELDS},
    )


def check(result: xr.DataArray, expected: list[float]) -> None:
    np.testing.assert_allclose(result.values, np.array(expected), rtol=1e-12, equal_nan=True)


def test_equity_prefers_full_then_simple() -> None:
    f = facts(**{EQUITY_FULL: [11, NAN, NAN], "StockholdersEquity": [7, 5, NAN]})
    check(sf.build_equity(f), [11, 5, NAN])


def test_debt_primary_sum_and_secondary_fallback() -> None:
    nan = NAN
    f = facts(
        LongTermDebtAndCapitalLeaseObligationsCurrent=[2, 0, nan, 2, nan, -6, 2, 0, 2, 2],
        LongTermDebtAndCapitalLeaseObligations=[4, 0, nan, -2, 4, 2, 4, 0, 4, 4],
        ShortTermBorrowings=[8, 8, 8, 8, 8, 8, nan, nan, 8, nan],
        CommercialPaper=[16, 16, 16, 16, 16, 16, nan, nan, nan, 16],
        FinanceLeaseLiabilityCurrent=[32, 32, 32, 32, 32, 32, 32, nan, 32, 32],
        FinanceLeaseLiabilityNoncurrent=[64] * 10,
        LongTermDebtCurrent=[128] * 10,
        LongTermDebtNoncurrent=[256] * 10,
        CapitalLeaseObligationsCurrent=[512] * 10,
        CapitalLeaseObligationsNoncurrent=[1024] * 10,
        OperatingLeaseLiabilityCurrent=[2048] * 10,
        OperatingLeaseLiabilityNoncurrent=[4096] * 10,
    )
    check(sf.build_debt(f), [30, 8184, 8184, 8184, 28, 20, 6, 8128, 14, 22])


def test_debt_without_any_fact_is_zero() -> None:
    f = facts(LongTermDebtAndCapitalLeaseObligationsCurrent=[NAN, 0])
    check(sf.build_debt(f), [0, 0])


def test_cash_full_sums_parts_and_treats_nan_as_zero() -> None:
    nan = NAN
    f = facts(
        CashAndCashEquivalentsAtCarryingValue=[2, nan, 2, 2, 2, nan],
        ShortTermInvestments=[4, 4, nan, 4, 4, nan],
        AvailableForSaleSecuritiesCurrent=[8, 8, 8, nan, 8, nan],
        MarketableSecuritiesCurrent=[16, 16, 16, 16, nan, nan],
    )
    check(sf.build_cash_and_cash_equivalents_full(f), [30, 28, 26, 22, 14, 0])


def test_net_debt_is_debt_minus_cash() -> None:
    nan = NAN
    f = facts(
        LongTermDebtAndCapitalLeaseObligationsCurrent=[64, nan, 64],
        LongTermDebtAndCapitalLeaseObligations=[32, nan, 32],
        CashAndCashEquivalentsAtCarryingValue=[2, 2, nan],
        ShortTermInvestments=[4, nan, nan],
        AvailableForSaleSecuritiesCurrent=[8, nan, nan],
        MarketableSecuritiesCurrent=[16, nan, nan],
    )
    check(sf.build_net_debt(f), [66, -2, 96])


def test_eps_prefers_diluted() -> None:
    f = facts(EarningsPerShareDiluted=[3, NAN, NAN], EarningsPerShare=[5, 7, NAN])
    check(sf.build_eps(f), [3, 7, NAN])


def test_liabilities_fallback_total_minus_equity() -> None:
    nan = NAN
    f = facts(
        Liabilities=[8, nan, nan, nan, nan],
        LiabilitiesAndStockholdersEquity=[100, 100, 100, nan, 100],
        **{EQUITY_FULL: [40, 40, nan, nan, nan]},
        StockholdersEquity=[9, 9, 24, nan, nan],
    )
    check(sf.build_liabilities(f), [8, 60, 76, nan, nan])


def test_income_before_taxes_fallback_net_income_plus_tax() -> None:
    nan = NAN
    f = facts(
        **{IBT_FACT: [50, nan, nan, nan]},
        NetIncomeLoss=[11, 11, nan, 11],
        IncomeTaxExpenseBenefit=[3, 3, 3, nan],
    )
    check(sf.build_income_before_taxes(f), [50, 14, nan, nan])


def test_interest_net_is_operating_income_minus_income_before_taxes() -> None:
    nan = NAN
    f = facts(
        OperatingIncomeLoss=[70, 70, nan],
        **{IBT_FACT: [30, nan, 30]},
        NetIncomeLoss=[1, 20, 1],
        IncomeTaxExpenseBenefit=[1, 6, 1],
    )
    check(sf.build_interest_net(f), [40, 44, nan])


def test_depreciation_priority_chain_and_restore() -> None:
    nan = NAN
    f = facts(
        DepreciationAmortizationAndAccretionNet=[2, nan, nan, nan, nan, nan],
        DepreciationAndAmortization=[4, 4, nan, nan, nan, nan],
        DepreciationDepletionAndAmortization=[8, 8, 8, nan, nan, nan],
        Depreciation=[16, 16, 16, 16, nan, nan],
        AmortizationOfIntangibleAssets=[1, 1, 1, 1, 32, 64],
    )
    check(sf.build_depreciation_and_amortization(f), [2, 4, 8, 16, 48, 80])


@pytest.mark.parametrize(
    ("depreciation", "amortization", "expected"),
    [
        ([0, NAN], [2, 3], [0, NAN]),
        ([-4, NAN], [1, 3], [-4, NAN]),
        ([NAN, NAN], [5, 5], [NAN, NAN]),
        ([0.5, NAN], [1, 3], [0.5, 3.5]),
        ([NAN, 16, NAN], [2, 4, 8], [NAN, 16, 24]),
        ([16, NAN], [NAN, NAN], [16, NAN]),
    ],
)
def test_depreciation_restore_thresholds(
    depreciation: list[float], amortization: list[float], expected: list[float]
) -> None:
    f = facts(Depreciation=depreciation, AmortizationOfIntangibleAssets=amortization)
    check(sf.build_depreciation_and_amortization(f), expected)


def test_ebitda_use_income_before_taxes() -> None:
    nan = NAN
    f = facts(
        **{IBT_FACT: [100, 100, 100, 100, 100, 100, nan, 100]},
        NetIncomeLoss=[1, 1, 1, 1, 1, 1, 20, 1],
        IncomeTaxExpenseBenefit=[1, 1, 1, 1, 1, 1, 6, 1],
        DepreciationAmortizationAndAccretionNet=[8, 8, 8, 8, 8, nan, 8, 8],
        InterestIncomeExpenseNet=[4, 0, nan, nan, -4, 4, 4, 0],
        InterestExpense=[2, 2, 2, nan, 2, 2, 2, nan],
        InvestmentIncomeInterest=[32, 32, 32, 32, 32, 32, 32, 16],
    )
    check(sf.build_ebitda_use_income_before_taxes(f), [104, 106, 106, 108, 112, 96, 30, 108])


def test_ebitda_use_operating_income() -> None:
    nan = NAN
    f = facts(
        OperatingIncomeLoss=[2, 2, 2, 2, nan, 2, 2, 2, 2, nan],
        DepreciationAmortizationAndAccretionNet=[4, 4, 4, 4, 4, nan, 4, 4, 4, nan],
        NonoperatingIncomeExpense=[8, 8, 8, 8, 8, 8, nan, 8, 8, nan],
        GainsLossesOnExtinguishmentOfDebt=[16, 16, 16, 16, 16, 16, 16, nan, 16, nan],
        InterestExpense=[32, 0, nan, nan, 32, 32, 32, 32, -32, nan],
        InvestmentIncomeInterest=[64, 64, 64, nan, 64, 64, 64, 64, 64, nan],
    )
    check(sf.build_ebitda_use_operating_income(f), [62, 94, 94, 30, 60, 58, 54, 46, -2, 0])


def test_ebitda_simple_is_operating_income_plus_depreciation() -> None:
    nan = NAN
    f = facts(
        OperatingIncomeLoss=[10, nan, 10],
        DepreciationAmortizationAndAccretionNet=[4, 4, nan],
    )
    check(sf.build_ebitda_simple(f), [14, nan, nan])


def ratio_facts() -> xr.DataArray:
    nan = NAN
    shares: dict[str, list[float]] = {"dei:EntityCommonStockSharesOutstanding": [10, 10, 10]}
    return facts(
        close=[3.5, 3.5, 2.0],
        **shares,
        LongTermDebtAndCapitalLeaseObligationsCurrent=[4, 4, 1],
        LongTermDebtAndCapitalLeaseObligations=[6, 6, 1],
        CashAndCashEquivalentsAtCarryingValue=[5, 5, nan],
        OperatingIncomeLoss=[10, 0, -9],
        DepreciationAndAmortization=[3, 0, 2],
        NetIncomeLoss=[4, 0, -6],
        StockholdersEquity=[8, 0, 3],
        Revenues=[16, 0, 5],
        Liabilities=[110, 50, 21],
    )


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (sf.build_market_capitalization, [35, 35, 20]),
        (sf.build_ev, [40, 40, 22]),
        (sf.build_ebitda_simple, [13, 0, -7]),
        (sf.build_ev_divide_by_ebitda, [40 / 13, NAN, -22 / 7]),
        (sf.build_liabilities_divide_by_ebitda, [110 / 13, NAN, -3]),
        (sf.build_net_debt_divide_by_ebitda, [5 / 13, NAN, -2 / 7]),
        (sf.build_p_divide_by_e, [35 / 4, NAN, -20 / 6]),
        (sf.build_p_divide_by_bv, [35 / 8, NAN, 20 / 3]),
        (sf.build_p_divide_by_s, [35 / 16, NAN, 4]),
        (sf.build_ev_divide_by_s, [2.5, NAN, 22 / 5]),
        (sf.build_roe, [0.5, NAN, -2]),
    ],
)
def test_ratio_builders(
    builder: Callable[[xr.DataArray], xr.DataArray], expected: list[float]
) -> None:
    check(builder(ratio_facts()), expected)


def test_annual_indicator_names_are_those_with_only_annual_facts() -> None:
    names = sf.get_annual_indicator_names()
    expected = [
        name
        for name, spec in sf.GLOBAL_INDICATORS.items()
        if all(fact in sf.GLOBAL_ANNUAL_US_GAAPS for fact in spec["facts"])
    ]
    assert names
    assert names == expected
    assert len(names) < len(sf.GLOBAL_INDICATORS)
