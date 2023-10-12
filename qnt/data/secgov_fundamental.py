from typing import List, Union, Callable

from qnt.data.common import *
from qnt.data.secgov import load_facts
from qnt.data.secgov_indicators import InstantIndicatorBuilder, PeriodIndicatorBuilder, IndicatorUtils
from qnt.data.stocks import load_list, load_ndx_list


def load_indicators_for(stocks_market_data: xr.DataArray,
                        indicator_names: List[str] = None,
                        time_period: str = 'ltm',
                        indicators_builders: dict = None
                        ) -> xr.DataArray:
    def fill_missing_time(data: xr.DataArray, fact_names: List[str]):
        for fact_name in fact_names:
            if fact_name not in GLOBAL_CUSTOM_LTM_US_GAAPS:
                ds_sel = data.sel(field=fact_name)
                data.loc[{"field": fact_name}] = ds_sel.ffill(dim="time")

    def build_indicators(data: xr.DataArray, names: List[str]) -> xr.DataArray:
        indicators_ = initialize_fundamental_data(data, names)
        for name in names:
            indicators_.loc[{'field': name}] = indicators_builders[name]['build'](data)
        return indicators_

    indicator_names = indicator_names or get_all_indicator_names()
    indicators_builders = indicators_builders or GLOBAL_INDICATORS

    fundamental_data = get_computed_facts(stocks_market_data, indicator_names, time_period, indicators_builders)
    fill_missing_time(fundamental_data, fundamental_data.field.values)
    merged_data = xr.concat([fundamental_data, stocks_market_data.sel(field='close')], dim="field")
    valid_names = [indicator for indicator in indicator_names if indicator in indicators_builders]

    indicators = build_indicators(merged_data, valid_names).ffill('time')
    indicators.name = "secgov_indicators"

    return indicators.transpose('time', 'field', 'asset')


def get_computed_facts(stocks_market_data: xr.DataArray,
                       indicator_names: Union[List[str], None] = None,
                       time_period: str = 'ltm',
                       indicators_builders: dict = None,
                       build_period_strategy: Callable = IndicatorUtils.build_ltm_with_remove_gaps,
                       build_instant_strategy: Callable = IndicatorUtils.build_series_fill_gaps) -> xr.DataArray:
    def load_all_facts(ciks, us_gaap_facts, min_date, max_date):
        columns = ['cik', 'report_id', 'report_type', 'report_date', 'fact_name', 'period', 'period_length', 'segment']
        for cik_reports in load_facts(ciks, us_gaap_facts, min_date=min_date, max_date=max_date, skip_segment=False,
                                      columns=columns, group_by_cik=True):
            yield cik_reports

    def compute(fundamental_data, all_facts, facts_names, build_period_strategy, build_instant_strategy, all_cik_asset):
        def get_computed(fact_name, reports):
            def _compute_annual(fact_name, reports):
                indicator = InstantIndicatorBuilder(fact_name, fact_name, True, build_instant_strategy)
                facts_no_segment = [f for f in reports if 'segment' not in f or f['segment'] is None]
                indicator_series = indicator.build_series_dict(facts_no_segment)
                return indicator_series

            def _compute_ltm(fact_name, reports):
                indicator = PeriodIndicatorBuilder(fact_name, [fact_name], True, time_period, build_period_strategy)
                series_data = indicator.build_series_dict(reports)
                return series_data

            facts = IndicatorUtils.get_filtered(reports, [fact_name])
            if fact_name in GLOBAL_ANNUAL_US_GAAPS:
                return _compute_annual(fact_name, facts)
            if fact_name in GLOBAL_CUSTOM_LTM_US_GAAPS:
                return GLOBAL_CUSTOM_LTM_US_GAAPS[fact_name](fact_name, reports)
            return _compute_ltm(fact_name, facts)

        fundamental_data_ = fundamental_data.copy()
        for cik_reports in all_facts:
            asset_name = all_cik_asset.get(cik_reports[0], None)
            if asset_name is None:
                continue

            for fact_name in facts_names:
                computed_data = get_computed(fact_name, cik_reports[1])
                for data, value in computed_data.items():

                    nearest_date = fundamental_data_.time.loc[data:]
                    is_wrong_date = len(nearest_date) < 1
                    if is_wrong_date:
                        continue

                    nearest_date = nearest_date[0].values
                    fundamental_data_.loc[{'asset': asset_name, 'time': nearest_date, 'field': fact_name}] = value

        return fundamental_data_

    if time_period not in ['ltm', 'qf', 'af']:
        raise ValueError("time_period must be one of 'ltm', 'qf', 'af'")

    indicator_names = indicator_names or get_all_indicator_names()
    indicators_builders = indicators_builders or GLOBAL_INDICATORS

    # Retrieve time and asset data
    start_date_offset = datetime.timedelta(days=730)
    min_date = (np.datetime64(stocks_market_data.time.min().values) - np.timedelta64(start_date_offset.days,
                                                                                     'D')).astype(datetime.date)
    max_date = stocks_market_data.time.max().values.astype(datetime.date)
    asset_names = stocks_market_data.asset.values

    assets_with_ciks = load_ndx_list(min_date, max_date) if 'NAS:' in asset_names[0] else load_list(min_date, max_date)
    ciks = [asset['cik'] for asset in assets_with_ciks if asset['id'] in asset_names and 'cik' in asset]
    facts_names = list(
        {fact for name in indicator_names if name in indicators_builders for fact in
         indicators_builders[name]['facts']})

    all_facts = load_all_facts(ciks, facts_names, min_date, max_date)
    fundamental_data = initialize_fundamental_data(stocks_market_data, facts_names)
    cik_asset_all = {asset['cik']: asset['id'] for asset in assets_with_ciks}

    return compute(fundamental_data, all_facts, facts_names, build_period_strategy, build_instant_strategy,
                   cik_asset_all)


def initialize_fundamental_data(data: xr.DataArray, facts_names: List[str]) -> xr.DataArray:
    r = xr.concat(
        [data.sel(field='close')] * len(facts_names),
        pd.Index(facts_names, name='field')
    )
    r[:] = np.nan
    return r


def build_equity(fundamental_facts: xr.DataArray) -> xr.DataArray:
    equity_full = fundamental_facts.sel(
        field='us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')
    equity_simple = fundamental_facts.sel(field='us-gaap:StockholdersEquity')
    equity = equity_full.where(~np.isnan(equity_full), equity_simple)
    return equity


def build_debt(fundamental_facts: xr.DataArray) -> xr.DataArray:
    primary_metrics = [
        'us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent',
        'us-gaap:LongTermDebtAndCapitalLeaseObligations'
    ]

    secondary_metrics = [
        'us-gaap:ShortTermBorrowings',
        'us-gaap:FinanceLeaseLiabilityCurrent',
        'us-gaap:FinanceLeaseLiabilityNoncurrent',
        'us-gaap:LongTermDebtCurrent',
        'us-gaap:LongTermDebtNoncurrent',
        'us-gaap:CapitalLeaseObligationsCurrent',
        'us-gaap:CapitalLeaseObligationsNoncurrent',
        'us-gaap:OperatingLeaseLiabilityCurrent',
        'us-gaap:OperatingLeaseLiabilityNoncurrent',
        'us-gaap:CommercialPaper'
    ]

    metrics = primary_metrics + secondary_metrics

    debt_data = {metric: fundamental_facts.sel(field=metric).fillna(0) for metric in metrics}

    mix_current = debt_data['us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent']
    mix_non_current = debt_data['us-gaap:LongTermDebtAndCapitalLeaseObligations']

    other_debts = sum(debt_data[metric] for metric in secondary_metrics)

    is_valid_debt = (~np.isnan(mix_current) & ~np.isnan(mix_non_current)) & (mix_current + mix_non_current != 0)
    combined_primary = mix_current + mix_non_current + debt_data['us-gaap:ShortTermBorrowings'] + debt_data[
        'us-gaap:CommercialPaper']

    debt = xr.where(is_valid_debt, combined_primary, other_debts)

    return debt


def build_net_debt(fundamental_facts: xr.DataArray) -> xr.DataArray:
    debt = build_debt(fundamental_facts)
    cash = build_cash_and_cash_equivalents_full(fundamental_facts)

    net_debt = debt.fillna(0) - cash.fillna(0)

    return net_debt


def build_losses_on_extinguishment_of_debt(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:GainsLossesOnExtinguishmentOfDebt')


def build_shares(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='dei:EntityCommonStockSharesOutstanding')


def build_assets(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:Assets')


def build_revenues(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:Revenues')


def build_income_before_income_taxes(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(
        field='us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest')


def build_operating_income(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:OperatingIncomeLoss')


def build_income_interest(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:InvestmentIncomeInterest')


def build_interest_expense(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:InterestExpense')


def build_interest_expense_debt(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:InterestExpenseDebt')


def build_interest_expense_capital_lease(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:InterestExpenseLesseeAssetsUnderCapitalLease')


def build_interest_income_expense_net(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:InterestIncomeExpenseNet')


def build_other_nonoperating_income_expense(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:OtherNonoperatingIncomeExpense')


def build_nonoperating_income_expense(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:NonoperatingIncomeExpense')


def build_cash_and_cash_equivalents(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:CashAndCashEquivalentsAtCarryingValue')


def build_short_term_investments(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:ShortTermInvestments')


def build_cash_and_cash_equivalents_full(fundamental_facts: xr.DataArray) -> xr.DataArray:
    cash = build_cash_and_cash_equivalents(fundamental_facts)
    short_term_investments = build_short_term_investments(fundamental_facts)
    available_for_sale_securities_current = fundamental_facts.sel(field='us-gaap:AvailableForSaleSecuritiesCurrent')
    marketable_securities_current = fundamental_facts.sel(field='us-gaap:MarketableSecuritiesCurrent')

    cash_and_cash_equivalents_full = cash.fillna(0) + short_term_investments.fillna(0) + \
                                     available_for_sale_securities_current.fillna(0) + \
                                     marketable_securities_current.fillna(0)

    return cash_and_cash_equivalents_full


def build_net_income(fundamental_facts: xr.DataArray) -> xr.DataArray:
    return fundamental_facts.sel(field='us-gaap:NetIncomeLoss')


def build_eps(fundamental_facts: xr.DataArray) -> xr.DataArray:
    eps_diluted = fundamental_facts.sel(field='us-gaap:EarningsPerShareDiluted')
    eps_simple = fundamental_facts.sel(field='us-gaap:EarningsPerShare')

    eps = eps_diluted.where(~np.isnan(eps_diluted), eps_simple)

    return eps


def build_liabilities(fundamental_facts: xr.DataArray) -> xr.DataArray:
    liabilities = fundamental_facts.sel(field='us-gaap:Liabilities')
    total = fundamental_facts.sel(field='us-gaap:LiabilitiesAndStockholdersEquity')
    equity = build_equity(fundamental_facts)

    return liabilities.where(~np.isnan(liabilities), total - equity)


def build_income_before_taxes(fundamental_facts: xr.DataArray) -> xr.DataArray:
    income_tax = fundamental_facts.sel(field='us-gaap:IncomeTaxExpenseBenefit')
    income_before_taxes = fundamental_facts.sel(
        field='us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments')
    net_income = build_net_income(fundamental_facts)

    return income_before_taxes.where(~np.isnan(income_before_taxes), net_income + income_tax)


def build_interest_net(fundamental_facts: xr.DataArray) -> xr.DataArray:
    operating_income = build_operating_income(fundamental_facts)
    income_before_taxes = build_income_before_taxes(fundamental_facts)

    interest_net = operating_income - income_before_taxes

    return interest_net


def build_depreciation_and_amortization(fundamental_facts: xr.DataArray) -> xr.DataArray:
    DepreciationAmortizationAndAccretionNet = fundamental_facts.sel(
        field='us-gaap:DepreciationAmortizationAndAccretionNet')
    DepreciationAndAmortization = fundamental_facts.sel(field='us-gaap:DepreciationAndAmortization')
    DepreciationDepletionAndAmortization = fundamental_facts.sel(field='us-gaap:DepreciationDepletionAndAmortization')
    Depreciation = fundamental_facts.sel(field='us-gaap:Depreciation')
    depreciation_for_restore_global = fundamental_facts.sel(field='us-gaap:Depreciation').ffill('time').fillna(0)
    AmortizationOfIntangibleAssets = fundamental_facts.sel(field='us-gaap:AmortizationOfIntangibleAssets')

    restored_value = xr.where(
        depreciation_for_restore_global > 0,
        depreciation_for_restore_global + AmortizationOfIntangibleAssets,
        np.nan
    )

    merged = DepreciationAmortizationAndAccretionNet.where(
        ~np.isnan(DepreciationAmortizationAndAccretionNet),
        DepreciationAndAmortization.where(
            ~np.isnan(DepreciationAndAmortization),
            DepreciationDepletionAndAmortization.where(
                ~np.isnan(DepreciationDepletionAndAmortization),
                Depreciation.where(
                    ~np.isnan(Depreciation),
                    restored_value
                )
            )
        )
    )

    return merged


def build_ebitda_use_income_before_taxes(fundamental_facts: xr.DataArray) -> xr.DataArray:
    income_before_taxes = build_income_before_taxes(fundamental_facts)
    depreciation_and_amortization = build_depreciation_and_amortization(fundamental_facts).fillna(0)
    interest_income_expense_net = build_interest_income_expense_net(fundamental_facts).fillna(0)
    income_interest = build_income_interest(fundamental_facts).fillna(0)
    interest_expense = build_interest_expense(fundamental_facts).fillna(0)

    interest = xr.where(interest_income_expense_net != 0,
                        interest_income_expense_net,
                        np.where(~np.isnan(interest_expense), interest_expense, income_interest))

    ebitda_use_income_before_taxes = income_before_taxes + depreciation_and_amortization - interest

    return ebitda_use_income_before_taxes


def build_ebitda_use_operating_income(fundamental_facts: xr.DataArray) -> xr.DataArray:
    interest_expense = build_interest_expense(fundamental_facts)
    merged_interest_expense = xr.where(~np.isnan(xr.where(interest_expense != 0, interest_expense, np.NaN)),
                                       interest_expense,
                                       build_income_interest(fundamental_facts)).fillna(0)

    ebitda_use_operating_income = (
            build_operating_income(fundamental_facts).fillna(0) +
            build_depreciation_and_amortization(fundamental_facts).fillna(0) +
            build_nonoperating_income_expense(fundamental_facts).fillna(0) +
            build_losses_on_extinguishment_of_debt(fundamental_facts).fillna(0) +
            merged_interest_expense
    )

    return ebitda_use_operating_income


def build_ebitda_simple(fundamental_facts: xr.DataArray) -> xr.DataArray:
    operating_income = build_operating_income(fundamental_facts)
    depreciation_and_amortization = build_depreciation_and_amortization(fundamental_facts)

    ebitda_simple = operating_income + depreciation_and_amortization

    return ebitda_simple


def build_liabilities_divide_by_ebitda(fundamental_facts: xr.DataArray) -> xr.DataArray:
    ebitda_simple = build_ebitda_simple(fundamental_facts)
    liabilities = build_liabilities(fundamental_facts)

    liabilities_divide_by_ebitda = liabilities / ebitda_simple

    liabilities_divide_by_ebitda = liabilities_divide_by_ebitda.where(np.isfinite(liabilities_divide_by_ebitda), np.nan)

    return liabilities_divide_by_ebitda


def build_net_debt_divide_by_ebitda(fundamental_facts: xr.DataArray) -> xr.DataArray:
    ebitda_simple = build_ebitda_simple(fundamental_facts)
    net_debt = build_net_debt(fundamental_facts)

    net_debt_divide_by_ebitda = net_debt / ebitda_simple

    net_debt_divide_by_ebitda = net_debt_divide_by_ebitda.where(np.isfinite(net_debt_divide_by_ebitda), np.nan)

    return net_debt_divide_by_ebitda


def build_roe(fundamental_facts: xr.DataArray) -> xr.DataArray:
    net_income = build_net_income(fundamental_facts)
    equity = build_equity(fundamental_facts)

    roe = net_income / equity

    roe = roe.where(np.isfinite(roe), np.nan)

    return roe


def build_ev(fundamental_facts: xr.DataArray) -> xr.DataArray:
    net_debt = build_net_debt(fundamental_facts)
    market_capitalization = build_market_capitalization(fundamental_facts)

    ev = market_capitalization + net_debt

    return ev


def build_market_capitalization(fundamental_facts: xr.DataArray) -> xr.DataArray:
    shares = build_shares(fundamental_facts)
    close_price = fundamental_facts.sel(field='close')

    market_capitalization = shares * close_price

    return market_capitalization


def build_ev_divide_by_ebitda(fundamental_facts: xr.DataArray) -> xr.DataArray:
    ebitda_simple = build_ebitda_simple(fundamental_facts)
    ev = build_ev(fundamental_facts)

    ev_divide_by_ebitda = ev / ebitda_simple

    ev_divide_by_ebitda = ev_divide_by_ebitda.where(np.isfinite(ev_divide_by_ebitda), np.nan)

    return ev_divide_by_ebitda


def build_p_divide_by_e(fundamental_facts: xr.DataArray) -> xr.DataArray:
    net_income = build_net_income(fundamental_facts)
    market_cap = build_market_capitalization(fundamental_facts)

    p_divide_by_e = market_cap / net_income

    p_divide_by_e = p_divide_by_e.where(np.isfinite(p_divide_by_e), np.nan)

    return p_divide_by_e


def build_p_divide_by_bv(fundamental_facts: xr.DataArray) -> xr.DataArray:
    equity = build_equity(fundamental_facts)
    market_cap = build_market_capitalization(fundamental_facts)

    p_divide_by_bv = market_cap / equity

    p_divide_by_bv = p_divide_by_bv.where(np.isfinite(p_divide_by_bv), np.nan)

    return p_divide_by_bv


def build_p_divide_by_s(fundamental_facts: xr.DataArray) -> xr.DataArray:
    revenues = build_revenues(fundamental_facts)
    market_cap = build_market_capitalization(fundamental_facts)

    p_divide_by_s = market_cap / revenues

    p_divide_by_s = p_divide_by_s.where(np.isfinite(p_divide_by_s), np.nan)

    return p_divide_by_s


def build_ev_divide_by_s(fundamental_facts: xr.DataArray) -> xr.DataArray:
    ev = build_ev(fundamental_facts)
    revenues = build_revenues(fundamental_facts)

    ev_divide_by_s = ev / revenues

    ev_divide_by_s = ev_divide_by_s.where(np.isfinite(ev_divide_by_s), np.nan)

    return ev_divide_by_s


def get_all_indicator_names():
    return list(GLOBAL_INDICATORS.keys())


def get_complex_indicator_names():
    return ['market_capitalization',
            'ebitda_use_income_before_taxes',
            'ebitda_use_operating_income',
            'ebitda_simple',
            'ev',
            'ev_divide_by_ebitda',
            'liabilities_divide_by_ebitda',
            'net_debt_divide_by_ebitda',
            'p_divide_by_e',
            'p_divide_by_bv',
            'p_divide_by_s',
            'ev_divide_by_s',
            'roe']


def get_standard_indicator_names():
    all_names = get_all_indicator_names()
    complex_names = get_complex_indicator_names()
    return [name for name in all_names if name not in complex_names]


def get_annual_indicator_names():
    annual_indicator_names = []

    for indicator_name, indicator_data in GLOBAL_INDICATORS.items():
        facts = indicator_data.get('facts', [])
        if all(fact in GLOBAL_ANNUAL_US_GAAPS for fact in facts):
            annual_indicator_names.append(indicator_name)

    return annual_indicator_names


GLOBAL_ANNUAL_US_GAAPS = [
    'us-gaap:GainsLossesOnExtinguishmentOfDebt',
    'us-gaap:Assets',
    'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    'us-gaap:StockholdersEquity',
    'us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent',
    'us-gaap:LongTermDebtAndCapitalLeaseObligations',
    'us-gaap:ShortTermBorrowings',
    'us-gaap:FinanceLeaseLiabilityCurrent',
    'us-gaap:FinanceLeaseLiabilityNoncurrent',
    'us-gaap:LongTermDebtCurrent',
    'us-gaap:LongTermDebtNoncurrent',
    'us-gaap:CapitalLeaseObligationsCurrent',
    'us-gaap:CapitalLeaseObligationsNoncurrent',
    'us-gaap:OperatingLeaseLiabilityCurrent',
    'us-gaap:OperatingLeaseLiabilityNoncurrent',
    'us-gaap:CommercialPaper',
    'us-gaap:CashAndCashEquivalentsAtCarryingValue',
    'us-gaap:ShortTermInvestments',
    'us-gaap:AvailableForSaleSecuritiesCurrent',
    'us-gaap:MarketableSecuritiesCurrent',
    'us-gaap:Liabilities',
    'us-gaap:LiabilitiesAndStockholdersEquity',
    'dei:EntityCommonStockSharesOutstanding'
]

GLOBAL_CUSTOM_LTM_US_GAAPS = {
    'us-gaap:DepreciationAmortizationAndAccretionNet': IndicatorUtils.get_ltm_amortization,
    'us-gaap:DepreciationAndAmortization': IndicatorUtils.get_ltm_amortization,
    'us-gaap:DepreciationDepletionAndAmortization': IndicatorUtils.get_ltm_amortization,
    'us-gaap:Depreciation': IndicatorUtils.get_ltm_amortization,
    'us-gaap:AmortizationOfIntangibleAssets': IndicatorUtils.get_ltm_amortization,
}

FACT_GROUPS = {
    'debt': [
        'us-gaap:LongTermDebtAndCapitalLeaseObligations',
        'us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent',
        'us-gaap:LongTermDebtCurrent',
        'us-gaap:LongTermDebtNoncurrent',
        'us-gaap:CapitalLeaseObligationsCurrent',
        'us-gaap:CapitalLeaseObligationsNoncurrent',
        'us-gaap:FinanceLeaseLiabilityCurrent',
        'us-gaap:FinanceLeaseLiabilityNoncurrent',
        'us-gaap:OperatingLeaseLiabilityCurrent',
        'us-gaap:OperatingLeaseLiabilityNoncurrent',
        'us-gaap:ShortTermBorrowings',
        'us-gaap:CommercialPaper',
    ],
    'cash_equivalents': [
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
    'ebitda': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',
    ],
    'shares': ['dei:EntityCommonStockSharesOutstanding'],
    'equity': [
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:StockholdersEquity',
    ],
    'income': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'us-gaap:NetIncomeLoss',
        'us-gaap:IncomeTaxExpenseBenefit'
    ],
    'interest': [
        'us-gaap:InvestmentIncomeInterest',
        'us-gaap:InterestExpense',
        'us-gaap:InterestIncomeExpenseNet'
    ],
    'depreciation_and_amortization': [
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets'
    ],
}

GLOBAL_INDICATORS = {
    'total_revenue': {'facts': ['us-gaap:Revenues'],
                      'build': build_revenues},

    'liabilities': {'facts': FACT_GROUPS['equity'] + ['us-gaap:Liabilities',
                                                      'us-gaap:LiabilitiesAndStockholdersEquity'],
                    'build': build_liabilities},

    'assets': {'facts': ['us-gaap:Assets'],
               'build': build_assets},

    'equity': {'facts': FACT_GROUPS['equity'],
               'build': build_equity},

    'net_income': {'facts': ['us-gaap:NetIncomeLoss'],
                   'build': build_net_income},

    'short_term_investments': {'facts': [
        'us-gaap:ShortTermInvestments',
    ],
        'build': build_short_term_investments},

    'cash_and_cash_equivalents': {'facts': [
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
    ],
        'build': build_cash_and_cash_equivalents},

    'cash_and_cash_equivalents_full': {'facts': FACT_GROUPS['cash_equivalents'],
                                       'build': build_cash_and_cash_equivalents_full},

    'operating_income': {'facts': [
        'us-gaap:OperatingIncomeLoss'],
        'build': build_operating_income},

    'income_before_taxes': {'facts': FACT_GROUPS['income'],
                            'build': build_income_before_taxes},

    'income_before_income_taxes': {'facts': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
        'build': build_income_before_income_taxes},

    'depreciation_and_amortization': {'facts': FACT_GROUPS['depreciation_and_amortization'],
                                      'build': build_depreciation_and_amortization},

    'interest_net': {'facts': FACT_GROUPS['income'] + [
        'us-gaap:OperatingIncomeLoss'
    ],
                     'build': build_interest_net},

    'income_interest': {'facts': [
        'us-gaap:InvestmentIncomeInterest',
    ],
        'build': build_income_interest},

    'interest_expense': {'facts': [
        'us-gaap:InterestExpense',
    ],
        'build': build_interest_expense},

    'interest_expense_debt': {'facts': [
        'us-gaap:InterestExpenseDebt',
    ],
        'build': build_interest_expense_debt},
    'interest_expense_capital_lease': {'facts': [
        'us-gaap:InterestExpenseLesseeAssetsUnderCapitalLease',
    ],
        'build': build_interest_expense_capital_lease},

    'interest_income_expense_net': {'facts': [
        'us-gaap:InterestIncomeExpenseNet',
    ],
        'build': build_interest_income_expense_net},

    'losses_on_extinguishment_of_debt': {'facts': [
        'us-gaap:GainsLossesOnExtinguishmentOfDebt',
    ],
        'build': build_losses_on_extinguishment_of_debt},

    'nonoperating_income_expense': {'facts': [
        'us-gaap:NonoperatingIncomeExpense',
    ],
        'build': build_nonoperating_income_expense},

    'other_nonoperating_income_expense': {'facts': [
        'us-gaap:OtherNonoperatingIncomeExpense',
    ],
        'build': build_other_nonoperating_income_expense},

    'debt': {
        'facts': FACT_GROUPS['debt'],
        'build': build_debt,
    },
    'net_debt': {
        'facts': FACT_GROUPS['debt'] + FACT_GROUPS['cash_equivalents'],
        'build': build_net_debt,
    },
    'eps': {
        'facts': [
            'us-gaap:EarningsPerShareDiluted',
            'us-gaap:EarningsPerShare'
        ],
        'build': build_eps,
    },
    'shares': {
        'facts': FACT_GROUPS['shares'],
        'build': build_shares,
    },
    'ebitda_use_income_before_taxes': {
        'facts': FACT_GROUPS['income'] + FACT_GROUPS['interest'] + FACT_GROUPS['ebitda'],
        'build': build_ebitda_use_income_before_taxes,
    },
    'ebitda_use_operating_income': {
        'facts': FACT_GROUPS['ebitda'] + [
            'us-gaap:NonoperatingIncomeExpense',
            'us-gaap:GainsLossesOnExtinguishmentOfDebt',
            'us-gaap:InvestmentIncomeInterest',
        ] + FACT_GROUPS['interest'],
        'build': build_ebitda_use_operating_income,
    },

    'ebitda_simple': {'facts': FACT_GROUPS['depreciation_and_amortization'] + [
        'us-gaap:OperatingIncomeLoss',
    ],
                      'build': build_ebitda_simple},

    'liabilities_divide_by_ebitda': {
        'facts': FACT_GROUPS['ebitda'] + [
            'us-gaap:Liabilities',
            'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
            'us-gaap:StockholdersEquity',
            'us-gaap:LiabilitiesAndStockholdersEquity'
        ] + FACT_GROUPS['cash_equivalents'],
        'build': build_liabilities_divide_by_ebitda
    },

    'net_debt_divide_by_ebitda': {
        'facts': FACT_GROUPS['ebitda'] + FACT_GROUPS['debt'] + FACT_GROUPS['cash_equivalents'],
        'build': build_net_debt_divide_by_ebitda
    },

    'roe': {'facts': [
        'us-gaap:NetIncomeLoss',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:StockholdersEquity'
    ],
        'build': build_roe},

    # 'market_capitalization': {
    #     'facts': FACT_GROUPS['shares'],
    #     'build': build_market_capitalization,
    # },
    # 'ev': {
    #     'facts': FACT_GROUPS['shares'] + FACT_GROUPS['debt'] + FACT_GROUPS['cash_equivalents'],
    #     'build': build_ev
    # },
    #
    # 'ev_divide_by_ebitda': {
    #     'facts': FACT_GROUPS['ebitda'] + FACT_GROUPS['shares'] + FACT_GROUPS['debt'] + FACT_GROUPS['cash_equivalents'],
    #     'build': build_ev_divide_by_ebitda
    # },
    #
    # 'p_divide_by_e': {'facts': [
    #     'us-gaap:NetIncomeLoss',
    #     'dei:EntityCommonStockSharesOutstanding',
    # ],
    #     'build': build_p_divide_by_e},
    #
    # 'p_divide_by_bv': {'facts': [
    #     'dei:EntityCommonStockSharesOutstanding',
    #     'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    #     'us-gaap:StockholdersEquity'
    # ],
    #     'build': build_p_divide_by_bv},
    #
    # 'p_divide_by_s': {'facts': [
    #     'dei:EntityCommonStockSharesOutstanding',
    #     'us-gaap:Revenues'
    # ],
    #     'build': build_p_divide_by_s},
    #
    # 'ev_divide_by_s': {'facts': FACT_GROUPS['debt'] + FACT_GROUPS['cash_equivalents'] + [
    #     'dei:EntityCommonStockSharesOutstanding',
    #     'us-gaap:Revenues',
    # ],
    #                    'build': build_ev_divide_by_s},

}
