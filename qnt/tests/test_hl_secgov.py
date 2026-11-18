import json
import math
from dataclasses import dataclass
from typing import TypeAlias, TypedDict, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import qnt.data.secgov_fundamental as qnfund
import qnt.data.secgov_indicators as qnind
from qnt.tests.hl_server import FakeServer, Query
from qnt.tests.hl_support import business_days, make_data, renamed_assets

pytestmark = pytest.mark.usefixtures("quiet_tracking")

CIK = "0000001"
Fact: TypeAlias = dict[str, object]
AssetInfo: TypeAlias = dict[str, object]

QUARTERS = [
    ("10-Q", "2019-05-10", "2019-01-01", "2019-03-31", 89, 100.0),
    ("10-Q", "2019-08-09", "2019-04-01", "2019-06-30", 90, 110.0),
    ("10-Q", "2019-08-09", "2019-01-01", "2019-06-30", 180, 210.0),
    ("10-Q", "2019-11-08", "2019-07-01", "2019-09-30", 91, 120.0),
    ("10-Q", "2019-11-08", "2019-01-01", "2019-09-30", 272, 330.0),
    ("10-K", "2020-02-14", "2019-01-01", "2019-12-31", 364, 470.0),
    ("10-Q", "2020-05-08", "2020-01-01", "2020-03-31", 90, 130.0),
    ("10-Q", "2020-08-07", "2020-04-01", "2020-06-30", 90, 150.0),
    ("10-Q", "2020-08-07", "2020-01-01", "2020-06-30", 181, 280.0),
    ("10-Q", "2020-11-06", "2020-07-01", "2020-09-30", 91, 160.0),
    ("10-Q", "2020-11-06", "2020-01-01", "2020-09-30", 273, 440.0),
    ("10-K", "2021-02-12", "2020-01-01", "2020-12-31", 365, 620.0),
]
BALANCE_SHEETS = [
    ("10-K", "2020-02-14", "2019-12-31", 1000.0),
    ("10-Q", "2020-05-08", "2020-03-31", 1050.0),
    ("10-K", "2021-02-12", "2020-12-31", 1200.0),
]
EXPECTED_LTM = {
    "2020-02-14": 470.0,
    "2020-05-08": 500.0,
    "2020-08-07": 540.0,
    "2020-11-06": 580.0,
    "2021-02-12": 620.0,
}
EXPECTED_QUARTERS = {
    "2019-08-09": 110.0,
    "2019-11-08": 120.0,
    "2020-02-14": 140.0,
    "2020-05-08": 130.0,
    "2020-08-07": 150.0,
    "2020-11-06": 160.0,
    "2021-02-12": 180.0,
}


def flow_facts(name: str, scale: float = 1.0) -> list[Fact]:
    return [
        {
            "cik": CIK,
            "report_id": f"{name}-{index}",
            "report_type": report_type,
            "report_date": report_date,
            "fact_name": name,
            "period": [start, end],
            "period_length": length,
            "value": value * scale,
        }
        for index, (report_type, report_date, start, end, length, value) in enumerate(QUARTERS)
    ]


def stock_facts(name: str, scale: float = 1.0) -> list[Fact]:
    return [
        {
            "cik": CIK,
            "report_id": f"{name}-{index}",
            "report_type": report_type,
            "report_date": report_date,
            "fact_name": name,
            "period": period,
            "period_length": None,
            "value": value * scale,
        }
        for index, (report_type, report_date, period, value) in enumerate(BALANCE_SHEETS)
    ]


class FactsRequest(TypedDict):
    ciks: list[str]
    facts: list[str]
    skip_segment: bool


@dataclass
class Library:
    facts: list[Fact]
    requests: list[FactsRequest]


@pytest.fixture
def library(server: FakeServer) -> Library:
    facts = [
        *flow_facts("us-gaap:Revenues"),
        *flow_facts("us-gaap:NetIncomeLoss", 0.1),
        *stock_facts("us-gaap:Assets"),
        *stock_facts("us-gaap:StockholdersEquity", 0.5),
        *stock_facts("dei:EntityCommonStockSharesOutstanding", 0.01),
    ]
    store = Library(facts, [])
    server.serve_json(
        "assets",
        [{"id": "s1", "symbol": "AAA", "exchange": "NYS", "cik": CIK, "last_point": "2021-06-30"}],
    )

    def serve_facts(body: bytes | None, _query: Query) -> bytes:
        assert body is not None
        params = cast("FactsRequest", json.loads(body))
        store.requests.append(params)
        chosen = [
            f
            for f in store.facts
            if f["cik"] in params["ciks"] and f["fact_name"] in params["facts"]
        ]
        return json.dumps(chosen).encode()

    server.route("sec.gov/facts", serve_facts)
    return store


def revenue_assets() -> list[AssetInfo]:
    return [{"id": "NYS:AAA", "symbol": "AAA", "exchange": "NYS", "cik": CIK}]


def calendar() -> xr.DataArray:
    days = pd.bdate_range("2019-06-03", "2021-06-30", name="time")
    stamps = np.asarray(days, dtype="datetime64[ns]")
    return xr.DataArray(stamps, dims=["time"], coords={"time": days})


def series_at(arr: xr.DataArray, field: str) -> "pd.Series[float]":
    return arr.sel(field=field, asset="NYS:AAA").to_pandas()


def test_quarterly_annual_and_ltm_indicators_from_reports(library: Library) -> None:
    result = qnind.load_indicators(
        revenue_assets(),
        calendar(),
        standard_indicators=["total_revenue", "total_revenue_qf", "total_revenue_af", "assets"],
    )
    assert result is not None
    assert result.name == "secgov_indicators"
    assert result.dims == ("time", "field", "asset")
    assert list(result.coords["asset"].values) == ["NYS:AAA"]
    assert sorted(result.coords["field"].values) == sorted(
        ["total_revenue", "total_revenue_qf", "total_revenue_af", "assets"]
    )
    assert sorted(library.requests[0]["facts"]) == ["us-gaap:Assets", "us-gaap:Revenues"]

    ltm = series_at(result, "total_revenue")
    assert ltm.loc[:"2020-02-13"].isna().all()
    for day, value in EXPECTED_LTM.items():
        assert ltm.loc[day] == pytest.approx(value), day
    assert ltm.loc["2020-05-07"] == pytest.approx(470.0)
    assert ltm.loc["2021-06-30"] == pytest.approx(620.0)

    quarters = series_at(result, "total_revenue_qf")
    for day, value in EXPECTED_QUARTERS.items():
        assert quarters.loc[day] == pytest.approx(value), day

    annual = series_at(result, "total_revenue_af")
    assert annual.loc[:"2020-02-13"].isna().all()
    assert annual.loc["2020-02-14":"2021-02-11"].eq(470.0).all()
    assert annual.loc["2021-02-12":].eq(620.0).all()

    assets = series_at(result, "assets")
    assert assets.loc["2020-02-14"] == 1000.0
    assert assets.loc["2020-05-08"] == 1050.0
    assert assets.loc["2021-03-01"] == 1200.0


@pytest.mark.usefixtures("library")
def test_unknown_company_yields_no_indicators() -> None:
    other: list[AssetInfo] = [
        {"id": "NYS:ZZZ", "symbol": "ZZZ", "exchange": "NYS", "cik": "0000999"}
    ]
    assert qnind.load_indicators(other, calendar(), standard_indicators=["assets"]) is None


def test_year_end_quarter_is_recovered_from_year_to_date_facts() -> None:
    facts = flow_facts("us-gaap:Revenues")
    quarterly = qnind.PeriodIndicatorBuilder("qf", ["us-gaap:Revenues"], True, "qf")
    assert quarterly.build_series_dict(list(facts))["2020-02-14"] == pytest.approx(140.0)

    quarters_only = [
        f
        for f in facts
        if f["period_length"] in (89, 90, 91, 364) and str(f["report_date"]) <= "2020-02-14"
    ]
    # Without the nine-month fact the recovered fourth quarter is 470 - 170/3 instead of 140:
    # previous_3_quarters returns an average of the remainder rather than the sum of 3 quarters.
    recovered = quarterly.build_series_dict(quarters_only)["2020-02-14"]
    assert recovered == pytest.approx(470.0 - (470.0 - 300.0) / 3.0)

    start = pd.Timestamp("2019-01-01").to_pydatetime()
    nine_months: list[tuple[str | list[str], float]] = [(["2019-01-01", "2019-09-30"], 330.0)]
    assert qnind.previous_3_quarters(nine_months, start, 470.0) == 330.0
    singles: list[tuple[str | list[str], float]] = [
        (["2019-01-01", "2019-03-31"], 100.0),
        (["2019-04-01", "2019-06-30"], 110.0),
        (["2019-07-01", "2019-09-30"], 120.0),
    ]
    assert qnind.previous_3_quarters(singles, start, 470.0) == pytest.approx(170.0 / 3.0)
    assert qnind.previous_3_quarters([], start, 400.0) == 100.0


def test_segment_facts_are_summed_when_no_total_is_reported() -> None:
    def fact(segment: str | None, value: float) -> Fact:
        return {
            "cik": CIK,
            "report_id": 1,
            "report_type": "10-K",
            "report_date": "2020-02-14",
            "fact_name": "us-gaap:Revenues",
            "period": ["2019-01-01", "2019-12-31"],
            "period_length": 364,
            "value": value,
            "segment": segment,
        }

    segments = [fact("north", 10.0), fact("south", 15.0)]
    assert [f["value"] for f in qnind.IndicatorUtils.get_restored_by_segment(segments)] == [25.0]
    with_total = [*segments, fact(None, 40.0)]
    assert [f["value"] for f in qnind.IndicatorUtils.get_restored_by_segment(with_total)] == [40.0]


def stocks_market() -> xr.DataArray:
    data = make_data(n_assets=1, n_days=150, name="stocks", start="2020-10-01")
    return renamed_assets(data, ["NYS:AAA"])


@pytest.mark.usefixtures("library")
def test_fundamental_indicators_are_derived_from_facts() -> None:
    market = stocks_market()
    names = ["total_revenue", "net_income", "assets", "equity", "shares", "roe"]
    result = qnfund.load_indicators_for(market, names)
    assert result.name == "secgov_indicators"
    assert result.dims == ("time", "field", "asset")
    assert list(result.coords["field"].values) == names
    times = result.coords["time"].values

    def at(field: str, day: str) -> float:
        index = int(np.searchsorted(times, np.datetime64(day)))
        return float(result.sel(field=field, asset="NYS:AAA").values[index])

    # reports older than the first market day all land on that day and the oldest one wins (540 expected)
    assert at("total_revenue", "2020-10-01") == pytest.approx(470.0)
    assert at("total_revenue", "2021-02-12") == pytest.approx(620.0)
    assert at("net_income", "2021-02-12") == pytest.approx(62.0)
    assert at("assets", "2020-10-01") == 1050.0
    assert at("assets", "2021-02-12") == 1200.0
    assert at("equity", "2021-02-12") == 600.0
    assert at("shares", "2021-02-12") == 12.0
    assert at("roe", "2021-02-12") == pytest.approx(62.0 / 600.0)
    assert math.isfinite(at("roe", "2021-04-01"))

    partial = qnfund.load_indicators_for(market, ["assets", "market_capitalization"])
    assert list(partial.coords["field"].values) == ["assets"]


def test_indicator_registry_is_inconsistent_with_complex_names() -> None:
    everything = qnfund.get_all_indicator_names()
    complex_names = qnfund.get_complex_indicator_names()
    standard = qnfund.get_standard_indicator_names()
    unregistered = {
        "market_capitalization",
        "ev",
        "ev_divide_by_ebitda",
        "p_divide_by_e",
        "p_divide_by_bv",
        "p_divide_by_s",
        "ev_divide_by_s",
    }
    assert set(complex_names) - set(everything) == unregistered
    assert standard == [name for name in everything if name not in complex_names]
    annual = qnfund.get_annual_indicator_names()
    assert {"assets", "equity", "liabilities", "shares", "debt"} <= set(annual)
    assert "total_revenue" not in annual


@pytest.mark.usefixtures("library")
def test_computed_facts_validate_period_and_map_series_to_dates() -> None:
    with pytest.raises(ValueError, match="time_period must be one of"):
        qnfund.get_computed_facts(stocks_market(), ["assets"], time_period="daily")

    computed = qnfund.get_computed_facts(
        stocks_market(), ["assets", "total_revenue"], time_period="qf"
    )
    assert sorted(computed.coords["field"].values) == ["us-gaap:Assets", "us-gaap:Revenues"]
    quarters = computed.sel(field="us-gaap:Revenues", asset="NYS:AAA").to_pandas()
    assert quarters.dropna().tolist() == [100.0, 160.0, 180.0]


def facts_cube(overrides: dict[str, list[float]]) -> xr.DataArray:
    names = sorted(
        {fact for spec in qnfund.GLOBAL_INDICATORS.values() for fact in spec["facts"]} | {"close"}
    )
    values = np.full((3, len(names), 1), np.nan)
    for name, series in overrides.items():
        values[:, names.index(name), 0] = series
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={"time": business_days(3), "field": names, "asset": ["a"]},
    )


def row(arr: xr.DataArray) -> list[float]:
    return [float(v) for v in arr.values[:, 0]]


def test_composite_builders_prefer_primary_facts() -> None:
    nan = np.nan
    facts = facts_cube(
        {
            "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
                nan,
                500.0,
                500.0,
            ],
            "us-gaap:StockholdersEquity": [400.0, 400.0, 400.0],
            "us-gaap:EarningsPerShareDiluted": [nan, nan, 2.0],
            "us-gaap:EarningsPerShare": [1.0, 1.0, 1.0],
            "us-gaap:Liabilities": [nan, nan, 300.0],
            "us-gaap:LiabilitiesAndStockholdersEquity": [1000.0, 1000.0, 1000.0],
            "us-gaap:CashAndCashEquivalentsAtCarryingValue": [10.0, 10.0, 10.0],
            "us-gaap:ShortTermInvestments": [5.0, nan, 5.0],
            "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent": [20.0, nan, 20.0],
            "us-gaap:LongTermDebtAndCapitalLeaseObligations": [30.0, nan, 30.0],
            "us-gaap:ShortTermBorrowings": [4.0, 4.0, 4.0],
            "us-gaap:OperatingLeaseLiabilityCurrent": [7.0, 7.0, 7.0],
            "us-gaap:OperatingIncomeLoss": [100.0, 100.0, 100.0],
            "us-gaap:Depreciation": [8.0, nan, 8.0],
            "us-gaap:AmortizationOfIntangibleAssets": [2.0, 2.0, 2.0],
            "close": [10.0, 10.0, 10.0],
            "dei:EntityCommonStockSharesOutstanding": [3.0, 3.0, 3.0],
        }
    )
    assert row(qnfund.build_equity(facts)) == [400.0, 500.0, 500.0]
    assert row(qnfund.build_eps(facts)) == [1.0, 1.0, 2.0]
    assert row(qnfund.build_liabilities(facts)) == [600.0, 500.0, 300.0]
    assert row(qnfund.build_cash_and_cash_equivalents_full(facts)) == [15.0, 10.0, 15.0]
    assert row(qnfund.build_debt(facts)) == [54.0, 11.0, 54.0]
    assert row(qnfund.build_net_debt(facts)) == [39.0, 1.0, 39.0]
    assert row(qnfund.build_market_capitalization(facts)) == [30.0, 30.0, 30.0]
    assert row(qnfund.build_ev(facts)) == [69.0, 31.0, 69.0]
    assert row(qnfund.build_depreciation_and_amortization(facts)) == [8.0, 10.0, 8.0]
    assert row(qnfund.build_ebitda_simple(facts)) == [108.0, 110.0, 108.0]
    assert row(qnfund.build_p_divide_by_bv(facts)) == pytest.approx([30 / 400, 30 / 500, 30 / 500])
    assert row(qnfund.build_ev_divide_by_ebitda(facts)) == pytest.approx(
        [69 / 108, 31 / 110, 69 / 108]
    )


def test_every_registered_indicator_builder_returns_an_aligned_series() -> None:
    facts = facts_cube(
        {
            "close": [10.0, 11.0, 12.0],
            "us-gaap:Revenues": [100.0, 110.0, 120.0],
            "us-gaap:NetIncomeLoss": [10.0, 11.0, 12.0],
            "us-gaap:StockholdersEquity": [50.0, 55.0, 60.0],
            "us-gaap:Liabilities": [20.0, 21.0, 22.0],
            "us-gaap:OperatingIncomeLoss": [30.0, 33.0, 36.0],
            "us-gaap:Depreciation": [3.0, 3.0, 3.0],
            "us-gaap:InterestExpense": [1.0, 1.0, 1.0],
            "us-gaap:CashAndCashEquivalentsAtCarryingValue": [5.0, 5.0, 5.0],
            "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent": [4.0, 4.0, 4.0],
            "us-gaap:LongTermDebtAndCapitalLeaseObligations": [6.0, 6.0, 6.0],
            "dei:EntityCommonStockSharesOutstanding": [2.0, 2.0, 2.0],
        }
    )
    for name, spec in qnfund.GLOBAL_INDICATORS.items():
        built = spec["build"](facts)
        assert built.dims == ("time", "asset"), name
        assert built.shape == (3, 1), name
    assert row(qnfund.build_net_income(facts)) == [10.0, 11.0, 12.0]
    assert row(qnfund.build_revenues(facts)) == [100.0, 110.0, 120.0]
    assert row(qnfund.build_roe(facts)) == pytest.approx([10 / 50, 11 / 55, 12 / 60])
    assert row(qnfund.build_ebitda_simple(facts)) == [33.0, 36.0, 39.0]
    assert row(qnfund.build_p_divide_by_e(facts)) == pytest.approx([2.0, 2.0, 2.0])
    assert row(qnfund.build_p_divide_by_s(facts)) == pytest.approx([0.2, 0.2, 0.2])
    assert row(qnfund.build_ev_divide_by_s(facts)) == pytest.approx([25 / 100, 27 / 110, 29 / 120])
