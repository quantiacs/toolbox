import datetime as dt
import itertools

import pandas as pd
import numpy as np

from qnt.data.common import *
from qnt.data.secgov import load_facts
from qnt.data.secgov_indicators import InstantIndicatorBuilder, PeriodIndicatorBuilder
from qnt.data.stocks import load_list


class CacheHelper:
    cache = dict()

    def is_in_cache(self, key):
        if key not in self.cache:
            return False
        return True

    def get(self, key):
        # print("get " + key)
        return self.cache[key].copy(deep=True)

    def add(self, key, value):
        # print("add " + key)
        self.cache[key] = value

    def empty(self):
        self.cache = dict()

    @staticmethod
    def get_key_for(all_facts, market_data, fact_name, new_name, use_report_date):
        close_price = market_data.sel(field='close')
        close_price_df = close_price.to_pandas()
        name_ticker = close_price_df.columns[0]
        return name_ticker + "_" + fact_name + "_" + new_name + "_" + str(use_report_date)


global_cache = CacheHelper()


def get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy):
    key = global_cache.get_key_for(all_facts, market_data, fact_name, new_name, use_report_date)
    if global_cache.is_in_cache(key):
        return global_cache.get(key)
    facts = get_filtered(all_facts, [fact_name])
    indicator = PeriodIndicatorBuilder(new_name, [fact_name], use_report_date, 'ltm', build_ltm_strategy)
    r = indicator.build_series_dict(
        facts)

    result = get_df(r, market_data, new_name)
    global_cache.add(key, result)
    return result.copy()


def get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy):
    key = global_cache.get_key_for(all_facts, market_data, fact_name, new_name, use_report_date)
    if global_cache.is_in_cache(key):
        return global_cache.get(key)

    facts = get_filtered(all_facts, [fact_name])
    facts_no_segment = [f for f in facts if 'segment' not in f or f['segment'] is None]
    indicator_df = get_simple_indicator(facts_no_segment, market_data,
                                        fact_name, use_report_date, build_instant_strategy)
    indicator_df[new_name] = indicator_df[fact_name]
    result = indicator_df.drop(columns=[fact_name])
    global_cache.add(key, result)
    return result.copy()


def get_filtered(all_facts, facts_names, count_days_for_remove_old_period=366):
    r = []
    for fact in all_facts:
        if fact['fact_name'] in facts_names:
            r.append(fact)
    rr = get_filtered_by_period(r, count_days_for_remove_old_period)
    return rr


def get_filtered_by_period(all_facts, count_days_for_remove_old_period):
    r = []
    for fact in all_facts:
        report_date = fact['report_date']
        if type(fact['period']) not in [str, list]:
            continue
        if type(fact['period']) is str:
            period_end = fact['period']
        if type(fact['period']) is list:
            period_end = fact['period'][1]

        report_date_time = dt.datetime.strptime(report_date, '%Y-%m-%d')
        period_end_time = dt.datetime.strptime(period_end, '%Y-%m-%d')

        dist = (report_date_time - period_end_time).days

        if dist < count_days_for_remove_old_period:
            r.append(fact)

    return r


def get_simple_indicator(all_facts, market_data, fact_name, use_report_date, build_instant_strategy):
    indicator = InstantIndicatorBuilder(fact_name,
                                        fact_name,
                                        use_report_date, build_instant_strategy)
    f = get_filtered(all_facts, fact_name)
    indicator_series = indicator.build_series_dict(f)
    df = get_df(indicator_series, market_data, fact_name)
    return df


def get_df(serias_facts, market_data, name):
    time_coord = market_data.time
    min_date = pd.Timestamp(time_coord.min().values).to_pydatetime().date()
    series = [
        pd.Series(serias_facts if len(serias_facts) > 0 else {min_date.isoformat(): np.NaN}, dtype=np.float64,
                  name=name)]
    df = pd.concat(series, axis=1)
    df.index = df.index.astype(dtype=time_coord.dtype, copy=False)
    df.name = name

    time_coord_df = time_coord.to_pandas()
    time_coord_df.name = 'time'
    merge = df.join(time_coord_df, how='outer')
    merge[name] = merge[name].fillna(method="ffill")
    r = merge.drop(columns=['time'])
    return r


def build_losses_on_extinguishment_of_debt(all_facts, market_data, use_report_date, build_ltm_strategy,
                                           build_instant_strategy):
    fact_name = 'us-gaap:GainsLossesOnExtinguishmentOfDebt'
    new_name = 'losses_on_extinguishment_of_debt'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)


def build_shares(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    # https://www.sec.gov/structureddata/announcement/osd-announcement-110520-scaling-errors
    # https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0001628280-16-020309&xbrl_type=v#
    fact_name = 'dei:EntityCommonStockSharesOutstanding'
    new_name = 'shares'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)


def build_cash_and_cash_equivalent(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:CashAndCashEquivalentsAtCarryingValue'
    new_name = 'cash_and_cash_equivalent'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)


def build_assets(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:Assets'
    new_name = 'assets'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)


def build_equity(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    def get_merged(row):
        e = row['equity_full']
        if not math.isnan(e):
            return e

        return row['equity_simple']

    def build_equity_full(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
        fact_name = 'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
        new_name = 'equity_full'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_equity_simple(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
        fact_name = 'us-gaap:StockholdersEquity'
        new_name = 'equity_simple'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    equity = build_equity_full(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    equity['equity_simple'] = build_equity_simple(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                  build_instant_strategy)

    equity['equity'] = equity.apply(get_merged, axis=1)

    result = equity.drop(columns=['equity_simple', 'equity_full'])
    return result


def build_debt(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    def get_merged(row):
        mix_current = row['long_term_debt_and_capital_lease_obligations_current']
        mix_non_current = row['long_term_debt_and_capital_lease_obligations_non_current']

        long_term_debt_current = row['long_term_debt_current']
        long_term_debt_non_current = row['long_term_debt_non_current']

        capital_lease_obligations_current = row['capital_lease_obligations_current']
        capital_lease_obligations_non_current = row['capital_lease_obligations_non_current']

        short_term_borrowings = row['short_term_borrowings']
        commercial_paper = row['commercial_paper']

        if not math.isnan(mix_current) and not math.isnan(mix_non_current):
            if mix_current + mix_non_current != 0:
                return mix_current + mix_non_current + short_term_borrowings + commercial_paper

        debt = long_term_debt_current + long_term_debt_non_current + capital_lease_obligations_current + capital_lease_obligations_non_current

        finance_lease_liability_current = row['finance_lease_liability_current']
        finance_lease_liability_non_current = row['finance_lease_liability_non_current']

        operating_lease_liability_current = row['operating_lease_liability_current']
        operating_lease_liability_non_current = row['operating_lease_liability_non_current']

        return debt + finance_lease_liability_current + finance_lease_liability_non_current + \
               operating_lease_liability_current + operating_lease_liability_non_current + \
               short_term_borrowings + commercial_paper

    def build_long_term_debt_and_capital_lease_obligations_current(all_facts, market_data, use_report_date,
                                                                   build_ltm_strategy, build_instant_strategy):
        fact_name = 'us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent'
        new_name = 'long_term_debt_and_capital_lease_obligations_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_long_term_debt_and_capital_lease_obligations_non_current(all_facts, market_data, use_report_date,
                                                                       build_ltm_strategy, build_instant_strategy):
        fact_name = 'us-gaap:LongTermDebtAndCapitalLeaseObligations'
        new_name = 'long_term_debt_and_capital_lease_obligations_non_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_short_term_borrowings(all_facts, market_data, use_report_date, build_ltm_strategy,
                                    build_instant_strategy):
        fact_name = 'us-gaap:ShortTermBorrowings'
        new_name = 'short_term_borrowings'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_finance_lease_liability_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                              build_instant_strategy):
        fact_name = 'us-gaap:FinanceLeaseLiabilityCurrent'
        new_name = 'finance_lease_liability_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_finance_lease_liability_non_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                  build_instant_strategy):
        fact_name = 'us-gaap:FinanceLeaseLiabilityNoncurrent'
        new_name = 'finance_lease_liability_non_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_long_term_debt_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                     build_instant_strategy):
        fact_name = 'us-gaap:LongTermDebtCurrent'
        new_name = 'long_term_debt_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_long_term_debt_non_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                         build_instant_strategy):
        fact_name = 'us-gaap:LongTermDebtNoncurrent'
        new_name = 'long_term_debt_non_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_capital_lease_obligations_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                build_instant_strategy):
        fact_name = 'us-gaap:CapitalLeaseObligationsCurrent'
        new_name = 'capital_lease_obligations_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_capital_lease_obligations_non_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                    build_instant_strategy):
        fact_name = 'us-gaap:CapitalLeaseObligationsNoncurrent'
        new_name = 'capital_lease_obligations_non_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_operating_lease_liability_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                build_instant_strategy):
        fact_name = 'us-gaap:OperatingLeaseLiabilityCurrent'
        new_name = 'operating_lease_liability_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_operating_lease_liability_non_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                    build_instant_strategy):
        fact_name = 'us-gaap:OperatingLeaseLiabilityNoncurrent'
        new_name = 'operating_lease_liability_non_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_commercial_paper(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
        fact_name = 'us-gaap:CommercialPaper'
        new_name = 'commercial_paper'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    debt = build_long_term_debt_and_capital_lease_obligations_current(all_facts, market_data, use_report_date,
                                                                      build_ltm_strategy, build_instant_strategy)
    debt[
        'long_term_debt_and_capital_lease_obligations_non_current'] = build_long_term_debt_and_capital_lease_obligations_non_current(
        all_facts, market_data, use_report_date,
        build_ltm_strategy, build_instant_strategy)

    debt['long_term_debt_current'] = build_long_term_debt_current(all_facts, market_data, use_report_date,
                                                                  build_ltm_strategy, build_instant_strategy).fillna(0)
    debt['long_term_debt_non_current'] = build_long_term_debt_non_current(all_facts, market_data, use_report_date,
                                                                          build_ltm_strategy,
                                                                          build_instant_strategy).fillna(0)

    debt['capital_lease_obligations_current'] = build_capital_lease_obligations_current(all_facts,
                                                                                        market_data,
                                                                                        use_report_date,
                                                                                        build_ltm_strategy,
                                                                                        build_instant_strategy).fillna(
        0)

    debt['capital_lease_obligations_non_current'] = build_capital_lease_obligations_non_current(all_facts,
                                                                                                market_data,
                                                                                                use_report_date,
                                                                                                build_ltm_strategy,
                                                                                                build_instant_strategy).fillna(
        0)

    debt['operating_lease_liability_current'] = build_operating_lease_liability_current(all_facts, market_data,
                                                                                        use_report_date,
                                                                                        build_ltm_strategy,
                                                                                        build_instant_strategy).fillna(
        0)
    debt['operating_lease_liability_non_current'] = build_operating_lease_liability_non_current(all_facts, market_data,
                                                                                                use_report_date,
                                                                                                build_ltm_strategy,
                                                                                                build_instant_strategy).fillna(
        0)
    debt['finance_lease_liability_current'] = build_finance_lease_liability_current(all_facts, market_data,
                                                                                    use_report_date,
                                                                                    build_ltm_strategy,
                                                                                    build_instant_strategy).fillna(0)
    debt['finance_lease_liability_non_current'] = build_finance_lease_liability_non_current(all_facts, market_data,
                                                                                            use_report_date,
                                                                                            build_ltm_strategy,
                                                                                            build_instant_strategy).fillna(
        0)
    debt['short_term_borrowings'] = build_short_term_borrowings(all_facts, market_data, use_report_date,
                                                                build_ltm_strategy, build_instant_strategy).fillna(0)

    debt['commercial_paper'] = build_commercial_paper(all_facts, market_data, use_report_date,
                                                      build_ltm_strategy, build_instant_strategy).fillna(0)

    debt['debt'] = debt.apply(get_merged, axis=1)

    result = debt.drop(columns=['long_term_debt_and_capital_lease_obligations_current',
                                'long_term_debt_and_capital_lease_obligations_non_current',
                                'long_term_debt_current',
                                'long_term_debt_non_current',
                                'capital_lease_obligations_current',
                                'capital_lease_obligations_non_current',
                                'operating_lease_liability_current',
                                'operating_lease_liability_non_current',
                                'finance_lease_liability_current',
                                'finance_lease_liability_non_current',
                                'short_term_borrowings',
                                'commercial_paper'])
    return result


def build_net_debt(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    debt = build_debt(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    debt['cash'] = build_cash_and_cash_equivalents_full(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                        build_instant_strategy)
    debt['net_debt'] = debt['debt'].fillna(0) - debt['cash'].fillna(0)

    result = debt.drop(columns=['debt', 'cash'])
    return result


def build_revenues(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:Revenues'
    new_name = 'total_revenue'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_income_before_income_taxes(all_facts, market_data, use_report_date, build_ltm_strategy,
                                     build_instant_strategy):
    fact_name = 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
    new_name = 'income_before_income_taxes'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_operating_income(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:OperatingIncomeLoss'
    new_name = 'operating_income'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_income_interest(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:InvestmentIncomeInterest'
    new_name = 'income_interest'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_interest_expense(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:InterestExpense'
    new_name = 'interest_expense'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_interest_expense_debt(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:InterestExpenseDebt'
    new_name = 'interest_expense_debt'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_interest_expense_capital_lease(all_facts, market_data, use_report_date, build_ltm_strategy,
                                         build_instant_strategy):
    fact_name = 'us-gaap:InterestExpenseLesseeAssetsUnderCapitalLease'
    new_name = 'interest_expense_capital_lease'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_interest_income_expense_net(all_facts, market_data, use_report_date, build_ltm_strategy,
                                      build_instant_strategy):
    fact_name = 'us-gaap:InterestIncomeExpenseNet'
    new_name = 'interest_income_expense_net'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_other_nonoperating_income_expense(all_facts, market_data, use_report_date, build_ltm_strategy,
                                            build_instant_strategy):
    fact_name = 'us-gaap:OtherNonoperatingIncomeExpense'
    new_name = 'other_nonoperating_income_expense'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_nonoperating_income_expense(all_facts, market_data, use_report_date, build_ltm_strategy,
                                      build_instant_strategy):
    fact_name = 'us-gaap:NonoperatingIncomeExpense'
    new_name = 'nonoperating_income_expense'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_cash_and_cash_equivalents(all_facts, market_data, use_report_date, build_ltm_strategy,
                                    build_instant_strategy):
    fact_name = 'us-gaap:CashAndCashEquivalentsAtCarryingValue'
    new_name = 'cash_and_cash_equivalents'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)


def build_short_term_investments(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:ShortTermInvestments'
    new_name = 'short_term_investments'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)


def build_cash_and_cash_equivalents_full(all_facts, market_data, use_report_date, build_ltm_strategy,
                                         build_instant_strategy):
    def build_available_for_sale_securities_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                    build_instant_strategy):
        fact_name = 'us-gaap:AvailableForSaleSecuritiesCurrent'
        new_name = 'available_for_sale_securities_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def build_marketable_securities_current(all_facts, market_data, use_report_date, build_ltm_strategy,
                                            build_instant_strategy):
        fact_name = 'us-gaap:MarketableSecuritiesCurrent'
        new_name = 'marketable_securities_current'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    cash = build_cash_and_cash_equivalents(all_facts, market_data, use_report_date, build_ltm_strategy,
                                           build_instant_strategy)
    cash['short_term_investments'] = build_short_term_investments(all_facts, market_data, use_report_date,
                                                                  build_ltm_strategy, build_instant_strategy)
    cash['available_for_sale_securities_current'] = build_available_for_sale_securities_current(all_facts, market_data,
                                                                                                use_report_date,
                                                                                                build_ltm_strategy,
                                                                                                build_instant_strategy)
    cash['marketable_securities_current'] = build_marketable_securities_current(all_facts, market_data, use_report_date,
                                                                                build_ltm_strategy,
                                                                                build_instant_strategy)
    cash['cash_and_cash_equivalents_full'] = cash['cash_and_cash_equivalents'].fillna(0) \
                                             + cash['short_term_investments'].fillna(0) \
                                             + cash['available_for_sale_securities_current'].fillna(0) \
                                             + cash['marketable_securities_current'].fillna(0)
    result = cash.drop(columns=['cash_and_cash_equivalents',
                                'short_term_investments',
                                'available_for_sale_securities_current',
                                'marketable_securities_current'])
    return result


def build_net_income(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    fact_name = 'us-gaap:NetIncomeLoss'
    new_name = 'net_income'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)


def build_eps(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    def get_merged(row):
        diluted = row['eps_diluted']
        if not math.isnan(diluted):
            return diluted

        return row['eps_simple']

    fact_name = 'us-gaap:EarningsPerShareDiluted'
    new_name = 'eps_diluted'
    eps = get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)

    fact_name = 'us-gaap:EarningsPerShare'
    new_name = 'eps_simple'
    eps['eps_simple'] = get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)

    eps['eps'] = eps.apply(get_merged, axis=1)

    result = eps.drop(columns=['eps_simple', 'eps_diluted'])
    return result


def build_ev(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    net_debt = build_net_debt(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    net_debt['market_capitalization'] = build_market_capitalization(all_facts, market_data, use_report_date,
                                                                    build_ltm_strategy, build_instant_strategy)
    net_debt['ev'] = net_debt['market_capitalization'].fillna(0) + net_debt['net_debt'].fillna(0)

    r = net_debt.drop(columns=['market_capitalization', 'net_debt'])

    return r


def build_market_capitalization(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    shares_df = build_shares(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    close_price = market_data.sel(field='close')
    close_price_df = close_price.to_pandas()
    shares_df['price'] = close_price_df[close_price_df.columns[0]]

    shares_df['market_capitalization'] = shares_df['shares'] * shares_df['price']
    r = shares_df.drop(columns=['shares', 'price'])
    return r


def build_liabilities(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    def get_liabilities():
        fact_name = 'us-gaap:Liabilities'
        new_name = 'liabilities'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def get_total():
        fact_name = 'us-gaap:LiabilitiesAndStockholdersEquity'
        new_name = 'total'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date, build_instant_strategy)

    def get_merged(row):
        liabilities = row['liabilities']
        if not math.isnan(liabilities):
            return liabilities
        equity = row['equity']
        total = row['total']
        return total - equity

    liabilities = get_liabilities()
    liabilities['equity'] = \
        build_equity(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)[
            'equity'].fillna(
            0)
    liabilities['total'] = get_total()['total'].fillna(0)

    liabilities['liabilities'] = liabilities.apply(get_merged, axis=1)
    result = liabilities.drop(columns=['equity', 'total'])
    return result

    # alternative way

    # 'liabilities': {'facts': ['us-gaap:Liabilities', 'us-gaap:LiabilitiesCurrent', 'us-gaap:LongTermDebtNoncurrent',
    #                           'us-gaap:CapitalLeaseObligationsNoncurrent',
    #                           'us-gaap:DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent'],
    #                 'build': build_liabilities},

    # Liabilities_df = get_simple_indicator(all_facts, market_data, 'us-gaap:Liabilities', use_report_date)
    # LiabilitiesCurrent_df = get_simple_indicator(all_facts, market_data, 'us-gaap:LiabilitiesCurrent', use_report_date)
    # LongTermDebtNoncurrent_df = get_simple_indicator(all_facts, market_data, 'us-gaap:LongTermDebtNoncurrent',
    #                                                  use_report_date)
    # CapitalLeaseObligationsNoncurrent_df = get_simple_indicator(all_facts, market_data,
    #                                                             'us-gaap:CapitalLeaseObligationsNoncurrent',
    #                                                             use_report_date)
    # DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent_df = get_simple_indicator(all_facts, market_data,
    #                                                                            'us-gaap:DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent',
    #                                                                            use_report_date)
    # l_len = len(Liabilities_df['us-gaap:Liabilities'])
    # total_sum = Liabilities_df['us-gaap:Liabilities'].isnull().sum()
    # result = Liabilities_df.copy()
    # if total_sum == l_len:
    #     Current = LiabilitiesCurrent_df['us-gaap:LiabilitiesCurrent']
    #     LongTermDebt = LongTermDebtNoncurrent_df['us-gaap:LongTermDebtNoncurrent']
    #     CapitalLeaseObligations = CapitalLeaseObligationsNoncurrent_df['us-gaap:CapitalLeaseObligationsNoncurrent']
    #     DeferredIncomeTaxesAndOtherLiabilities = DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent_df[
    #         'us-gaap:DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent']
    #     result[
    #         'liabilities'] = Current + LongTermDebt + CapitalLeaseObligations + DeferredIncomeTaxesAndOtherLiabilities
    #     result = result.drop(columns=['us-gaap:Liabilities'])

    # return result


def build_income_before_taxes(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    def get_income_tax():
        fact_name = 'us-gaap:IncomeTaxExpenseBenefit'
        new_name = 'income_tax'
        return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)

    def get_income_before_taxes_():
        fact_name = 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
        new_name = 'income_before_taxes'
        return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date, build_ltm_strategy)

    def get_merged(row):
        income_before_taxes = row['income_before_taxes']
        if not math.isnan(income_before_taxes):
            return income_before_taxes
        net_income = row['net_income']
        income_tax = row['income_tax']
        return net_income + income_tax

    income_df = get_income_before_taxes_()
    income_df['net_income'] = build_net_income(all_facts, market_data, use_report_date, build_ltm_strategy,
                                               build_instant_strategy)
    income_df['income_tax'] = get_income_tax()
    income_df['merged'] = income_df.apply(get_merged, axis=1)
    result = income_df.drop(columns=['income_tax', 'net_income', 'income_before_taxes'])
    result = result.rename(columns={"merged": "income_before_taxes"})
    return result


def build_interest_net(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    operating_income_df = build_operating_income(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                 build_instant_strategy)
    income_before_taxes_df = build_income_before_taxes(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                       build_instant_strategy)
    r = operating_income_df.copy()
    r['interest_net'] = operating_income_df['operating_income'] - income_before_taxes_df['income_before_taxes']
    r = r.drop(columns=['operating_income'])
    return r


def build_depreciation_and_amortization(all_facts, market_data, use_report_date, build_ltm_strategy,
                                        build_instant_strategy):
    def get_ltm(fact_name, new_name):
        def get_accumulated_by_segment(facts_in_report):
            def get_key(fact):
                period = fact['period']
                period_str = period
                if type(period) == list:
                    r = ''
                    for p in period:
                        r += " " + p
                    period_str = r
                key = period_str + " " + fact['report_type'] + " " + str(fact['period_length'])
                return key

            r = {}
            for fact in facts_in_report:
                if fact['segment'] is not None:
                    key = get_key(fact)
                    if key in r and fact['value'] is not None:
                        if r[key]['value'] is not None:
                            r[key]['value'] += fact['value']
                        else:
                            r[key]['value'] = fact['value']
                    else:
                        r[key] = fact
            accumulated = []
            for key in r:
                accumulated.append(r[key])
            return accumulated

        def get_no_segment_fact(facts_in_report):
            r = []
            for fact in facts_in_report:
                if 'segment' not in fact or fact['segment'] is None:
                    r.append(fact)
            return r

        def get_filtered_facts(facts_in_report):
            no_segment_fact = get_no_segment_fact(facts_in_report)
            if len(no_segment_fact) != 0:
                return no_segment_fact
            return get_accumulated_by_segment(facts_in_report)

        def get_ltm_amortization(fact_name):
            def is_correct_year(fact):
                is_correct = fact['value'] is not None and fact['period_length'] is not None
                if is_correct \
                        and fact['report_type'] in ['10-K', '10-K/A', '8-K'] \
                        and (340 < fact['period_length'] < 380):
                    return True
                return False

            def is_correct_first_quarter(fact):
                is_correct = fact['value'] is not None and fact['period_length'] is not None
                if is_correct \
                        and fact['report_type'] in ['10-Q', '10-Q/A'] \
                        and (75 < fact['period_length'] < 120):
                    return True
                return False

            def is_correct_half_year(fact):
                is_correct = fact['value'] is not None and fact['period_length'] is not None
                if is_correct \
                        and fact['report_type'] in ['10-Q', '10-Q/A'] \
                        and (150 < fact['period_length'] < 210):
                    return True
                return False

            def is_correct_three_quarter(fact):
                is_correct = fact['value'] is not None and fact['period_length'] is not None
                if is_correct \
                        and fact['report_type'] in ['10-Q', '10-Q/A'] \
                        and (240 < fact['period_length'] < 310):
                    return True
                return False

            def get_correct_only_quarter_facts(facts):
                quarter_facts = []
                for fact in facts:
                    is_correct = fact['value'] is not None and fact['period_length'] is not None
                    if is_correct:
                        if (340 < fact['period_length'] < 380):
                            continue
                        quarter_facts.append(fact)
                return quarter_facts

            facts = get_filtered(all_facts, [fact_name], count_days_for_remove_old_period=370 * 2)

            group_key = (lambda f: f['report_date']) if use_report_date else (lambda f: f['period'][1])

            groups = itertools.groupby(facts, group_key)

            result = []
            last_year_value = None

            for g in groups:

                facts_in_report = list(g[1])
                filtered_facts_in_report = get_filtered_facts(facts_in_report)

                report_date = dt.datetime.strptime(g[0], '%Y-%m-%d').date().isoformat()
                sort_key = lambda f: (f['period'])
                facts_in_report_sorted = sorted(filtered_facts_in_report, key=sort_key)

                current = facts_in_report_sorted[-1]

                if current['report_type'] not in ['10-K', '10-K/A', '8-K', '10-Q', '10-Q/A']:
                    continue

                if is_correct_year(current):
                    last_year_value = current['value']
                    result.append([last_year_value, report_date])
                    continue

                if last_year_value is None:
                    continue

                quarter_facts = get_correct_only_quarter_facts(facts_in_report_sorted)

                if len(quarter_facts) == 0:
                    continue

                is_one_q_report = len(quarter_facts) == 1
                if is_one_q_report:
                    last_year_value = 0
                    result.append([current['value'], report_date])
                    continue

                previous = quarter_facts[-2]

                if is_correct_first_quarter(current) and is_correct_first_quarter(previous):
                    dif = last_year_value + current['value'] - previous['value']
                    result.append([dif, report_date])
                    continue

                if is_correct_half_year(current) and is_correct_half_year(previous):
                    dif = last_year_value + current['value'] - previous['value']
                    result.append([dif, report_date])
                    continue

                if is_correct_three_quarter(current) and is_correct_three_quarter(previous):
                    dif = last_year_value + current['value'] - previous['value']
                    result.append([dif, report_date])
                    continue

                result.append([np.nan, report_date])

            r = dict((item[1], item[0]) for item in reversed(result))
            return r

        def get_df(serias_facts, market_data, name):
            time_coord = market_data.time
            min_date = pd.Timestamp(time_coord.min().values).to_pydatetime().date()
            series = [
                pd.Series(serias_facts if len(serias_facts) > 0 else {min_date.isoformat(): np.NaN}, dtype=np.float64,
                          name=name)]
            df = pd.concat(series, axis=1)
            df.index = df.index.astype(dtype=time_coord.dtype, copy=False)
            df.name = name

            time_coord_df = time_coord.to_pandas()
            time_coord_df.name = 'time'
            merge = df.join(time_coord_df, how='outer')
            r = merge.drop(columns=['time'])
            return r

        ltm = get_ltm_amortization(fact_name)
        indicator_df = get_df(ltm, market_data, fact_name)
        indicator_df[new_name] = indicator_df[fact_name]
        r = indicator_df.drop(columns=[fact_name])
        return r

    def get_depreciation_and_amortization():
        fact_name = 'us-gaap:DepreciationAndAmortization'
        new_name = 'DepreciationAndAmortization'
        return get_ltm(fact_name, new_name)

    def get_depreciation_amortization_accretion_net():
        fact_name = 'us-gaap:DepreciationAmortizationAndAccretionNet'
        new_name = 'DepreciationAmortizationAndAccretionNet'
        return get_ltm(fact_name, new_name)

    def get_depreciation_depletion_amortization():
        fact_name = 'us-gaap:DepreciationDepletionAndAmortization'
        new_name = 'DepreciationDepletionAndAmortization'
        return get_ltm(fact_name, new_name)

    def get_depreciation():
        fact_name = 'us-gaap:Depreciation'
        new_name = 'Depreciation'
        return get_ltm(fact_name, new_name)

    def get_amortization():
        fact_name = 'us-gaap:AmortizationOfIntangibleAssets'
        new_name = 'AmortizationOfIntangibleAssets'
        return get_ltm(fact_name, new_name)

    depreciation_for_restore_global = 0

    def get_merged(row):
        DepreciationAmortizationAndAccretionNet = row['DepreciationAmortizationAndAccretionNet']
        DepreciationDepletionAndAmortization = row['DepreciationDepletionAndAmortization']
        DepreciationAndAmortization = row['DepreciationAndAmortization']

        if not math.isnan(DepreciationAmortizationAndAccretionNet):
            return DepreciationAmortizationAndAccretionNet

        if not math.isnan(DepreciationAndAmortization):
            return DepreciationAndAmortization

        if not math.isnan(DepreciationDepletionAndAmortization):
            return DepreciationDepletionAndAmortization

        # attempt to restore the value

        Depreciation = row['Depreciation']
        AmortizationOfIntangibleAssets = row['AmortizationOfIntangibleAssets']

        nonlocal depreciation_for_restore_global

        if not math.isnan(Depreciation) and not math.isnan(
                AmortizationOfIntangibleAssets):
            depreciation_for_restore_global = Depreciation
            return Depreciation + AmortizationOfIntangibleAssets

        if not math.isnan(Depreciation):
            depreciation_for_restore_global = Depreciation
            return Depreciation

        if not math.isnan(AmortizationOfIntangibleAssets) and depreciation_for_restore_global > 0:
            return depreciation_for_restore_global + AmortizationOfIntangibleAssets

        return DepreciationAmortizationAndAccretionNet

    amortization_df = get_depreciation_and_amortization()
    amortization_df['DepreciationAmortizationAndAccretionNet'] = get_depreciation_amortization_accretion_net()
    amortization_df['DepreciationDepletionAndAmortization'] = get_depreciation_depletion_amortization()
    amortization_df['Depreciation'] = get_depreciation()
    amortization_df['AmortizationOfIntangibleAssets'] = get_amortization()
    amortization_df['merged'] = amortization_df.apply(get_merged, axis=1)
    amortization_df['depreciation_and_amortization'] = amortization_df['merged'].fillna(method="ffill")
    result1 = amortization_df.drop(columns=['DepreciationAndAmortization',
                                            'DepreciationAmortizationAndAccretionNet',
                                            'DepreciationDepletionAndAmortization',
                                            'merged',
                                            'Depreciation',
                                            'AmortizationOfIntangibleAssets',
                                            ])

    return result1


def build_ebitda_use_income_before_taxes(all_facts, market_data, use_report_date, build_ltm_strategy,
                                         build_instant_strategy):
    def get_merged_interest(row):
        interest_expense = row['interest_expense']
        income_interest = row['income_interest']
        interest_income_expense_net = row['interest_income_expense_net']

        if interest_income_expense_net == 0:

            if not math.isnan(interest_expense):
                return interest_expense

            if not math.isnan(income_interest):
                return income_interest
        else:
            return interest_income_expense_net

        return interest_expense

    income_before_taxes_df = build_income_before_taxes(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                       build_instant_strategy)
    r = income_before_taxes_df.copy()
    r['depreciation_and_amortization'] = \
        build_depreciation_and_amortization(all_facts, market_data, use_report_date, build_ltm_strategy,
                                            build_instant_strategy)[
            'depreciation_and_amortization'].fillna(0)
    r['interest_income_expense_net'] = \
        build_interest_income_expense_net(all_facts, market_data, use_report_date, build_ltm_strategy,
                                          build_instant_strategy)[
            'interest_income_expense_net'].fillna(0)
    r['losses_on_extinguishment_of_debt'] = \
        build_losses_on_extinguishment_of_debt(all_facts, market_data, use_report_date, build_ltm_strategy,
                                               build_instant_strategy)[
            'losses_on_extinguishment_of_debt'].fillna(0)
    r['debt'] = \
        build_interest_expense_debt(all_facts, market_data, use_report_date, build_ltm_strategy,
                                    build_instant_strategy)[
            'interest_expense_debt'].fillna(0)

    r['interest_expense_capital_lease'] = \
        build_interest_expense_capital_lease(all_facts, market_data, use_report_date, build_ltm_strategy,
                                             build_instant_strategy)[
            'interest_expense_capital_lease'].fillna(0)
    r['income_interest'] = \
        build_income_interest(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)[
            'income_interest'].fillna(0)
    r['interest_expense'] = \
        build_interest_expense(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)[
            'interest_expense'].fillna(0)
    r['interest'] = r.apply(get_merged_interest, axis=1)

    r['ebitda_use_income_before_taxes'] = r['income_before_taxes'] + \
                                          r['depreciation_and_amortization'] - \
                                          r['interest']

    r = r.drop(columns=['interest',
                        'interest_expense',
                        'income_interest',
                        'income_before_taxes',
                        'depreciation_and_amortization',
                        'interest_income_expense_net',
                        'losses_on_extinguishment_of_debt',
                        'interest_expense_capital_lease',
                        'debt'])

    return r


def build_ebitda_use_operating_income(all_facts, market_data, use_report_date, build_ltm_strategy,
                                      build_instant_strategy):
    def get_merged_interest(row):
        interest_expense = row['interest_expense']
        income_interest = row['income_interest']

        if not math.isnan(interest_expense):
            return interest_expense

        if not math.isnan(income_interest):
            return income_interest

        return interest_expense

    operating_income_df = build_operating_income(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                 build_instant_strategy)
    depreciation_and_amortization_df = build_depreciation_and_amortization(all_facts, market_data, use_report_date,
                                                                           build_ltm_strategy, build_instant_strategy)
    income_interest_df = build_income_interest(all_facts, market_data, use_report_date, build_ltm_strategy,
                                               build_instant_strategy)
    onoperating_income_expense_df = build_nonoperating_income_expense(all_facts,
                                                                      market_data,
                                                                      use_report_date,
                                                                      build_ltm_strategy, build_instant_strategy)
    losses_on_extinguishment_of_debt_df = build_losses_on_extinguishment_of_debt(all_facts,
                                                                                 market_data,
                                                                                 use_report_date,
                                                                                 build_ltm_strategy,
                                                                                 build_instant_strategy)

    other_nonoperating_income_expense_df = build_other_nonoperating_income_expense(all_facts, market_data,
                                                                                   use_report_date, build_ltm_strategy,
                                                                                   build_instant_strategy)
    interest_income_expense_net_df = build_interest_income_expense_net(all_facts, market_data,
                                                                       use_report_date, build_ltm_strategy,
                                                                       build_instant_strategy)

    interest_expense_df = build_interest_expense(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                 build_instant_strategy)
    r = operating_income_df.copy()
    interest_expense_df['income_interest'] = income_interest_df['income_interest']
    interest_expense_df['other_nonoperating_income_expense_df'] = other_nonoperating_income_expense_df[
        'other_nonoperating_income_expense']
    interest_expense_df['interest_income_expense_net'] = interest_income_expense_net_df['interest_income_expense_net']
    interest_expense_df['interest'] = interest_expense_df.apply(get_merged_interest, axis=1)

    operating_income = operating_income_df['operating_income'].fillna(0)
    depreciation_and_amortization = depreciation_and_amortization_df['depreciation_and_amortization'].fillna(0)
    onoperating_income_expense = onoperating_income_expense_df['nonoperating_income_expense'].fillna(0)
    losses_on_extinguishment_of_debt = losses_on_extinguishment_of_debt_df['losses_on_extinguishment_of_debt'].fillna(0)
    merged_interest_expense = interest_expense_df['interest'].fillna(0)

    r['ebitda_use_operating_income'] = operating_income + \
                                       depreciation_and_amortization + \
                                       losses_on_extinguishment_of_debt + \
                                       onoperating_income_expense + \
                                       merged_interest_expense

    # def for_test_ebitda(depreciation_and_amortization_df, income_interest_df, losses_on_extinguishment_of_debt_df,
    #                     onoperating_income_expense_df, operating_income_df, r, interest_expense_df,
    #                     merged_interest_expense_df):
    #     date = '2017-03-01'
    #     date = '2021-01-28'
    #     operating_income = operating_income_df.loc[date].max() / 1000000
    #     depreciation_and_amortization = depreciation_and_amortization_df.loc[date].max() / 1000000
    #     income_interest = income_interest_df.loc[date].max() / 1000000
    #     onoperating_income_expense = onoperating_income_expense_df.loc[date].max() / 1000000
    #     losses_on_extinguishment_of_debt = losses_on_extinguishment_of_debt_df.loc[date].max() / 1000000
    #     interest_expense = interest_expense_df.loc[date].max() / 1000000
    #     merged_interest_expense = merged_interest_expense_df.loc[date].max() / 1000000
    #     operating_income_plys_amortization = operating_income + depreciation_and_amortization
    #     income_interest_onoperating_income_expense = income_interest + onoperating_income_expense
    #     # income_before_taxes = income_before_taxes_df.loc[date].max() / 1000000
    #     sum_interest = income_interest - interest_expense
    #     sum = r['ebitda_use_operating_income'].loc[date].max() / 1000000
    #     return
    #
    # for_test_ebitda(depreciation_and_amortization_df, income_interest_df, losses_on_extinguishment_of_debt_df,
    #                 onoperating_income_expense_df, operating_income_df, r, interest_expense_df, merged_interest_expense,
    #                 )

    r = r.drop(columns=['operating_income'])
    return r


def build_ebitda_simple(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    operating_income_df = build_operating_income(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                 build_instant_strategy)
    depreciation_and_amortization_df = build_depreciation_and_amortization(all_facts, market_data, use_report_date,
                                                                           build_ltm_strategy, build_instant_strategy)

    operating_income = operating_income_df['operating_income'].fillna(0)
    depreciation_and_amortization = depreciation_and_amortization_df['depreciation_and_amortization'].fillna(0)

    operating_income_df['ebitda_simple'] = operating_income + depreciation_and_amortization

    r = operating_income_df.drop(columns=['operating_income'])
    return r


def build_ev_divide_by_ebitda(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    ebitda_simple = build_ebitda_simple(all_facts, market_data, use_report_date, build_ltm_strategy,
                                        build_instant_strategy)
    ev = build_ev(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    ebitda_simple['ev_divide_by_ebitda'] = ev['ev'] / ebitda_simple['ebitda_simple']
    r = ebitda_simple.drop(columns=['ebitda_simple'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_liabilities_divide_by_ebitda(all_facts, market_data, use_report_date, build_ltm_strategy,
                                       build_instant_strategy):
    ebitda_simple = build_ebitda_simple(all_facts, market_data, use_report_date, build_ltm_strategy,
                                        build_instant_strategy)
    ebitda_simple['liabilities'] = build_liabilities(all_facts, market_data, use_report_date, build_ltm_strategy,
                                                     build_instant_strategy)
    ebitda_simple['liabilities_divide_by_ebitda'] = ebitda_simple['liabilities'] / ebitda_simple['ebitda_simple']
    r = ebitda_simple.drop(columns=['ebitda_simple', 'liabilities'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_net_debt_divide_by_ebitda(all_facts, market_data, use_report_date, build_ltm_strategy,
                                    build_instant_strategy):
    ebitda_simple = build_ebitda_simple(all_facts, market_data, use_report_date, build_ltm_strategy,
                                        build_instant_strategy)
    ebitda_simple['net_debt'] = build_net_debt(all_facts, market_data, use_report_date, build_ltm_strategy,
                                               build_instant_strategy)
    ebitda_simple['net_debt_divide_by_ebitda'] = ebitda_simple['net_debt'] / ebitda_simple['ebitda_simple']
    r = ebitda_simple.drop(columns=['ebitda_simple', 'net_debt'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_p_divide_by_e(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    net_income = build_net_income(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    market_cap = build_market_capitalization(all_facts, market_data, use_report_date, build_ltm_strategy,
                                             build_instant_strategy)
    net_income['p_divide_by_e'] = market_cap['market_capitalization'] / net_income['net_income']
    r = net_income.drop(columns=['net_income'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_p_divide_by_bv(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    equity = build_equity(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    market_cap = build_market_capitalization(all_facts, market_data, use_report_date, build_ltm_strategy,
                                             build_instant_strategy)
    equity['p_divide_by_bv'] = market_cap['market_capitalization'] / equity['equity']
    r = equity.drop(columns=['equity'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_p_divide_by_s(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    revenues = build_revenues(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    market_cap = build_market_capitalization(all_facts, market_data, use_report_date, build_ltm_strategy,
                                             build_instant_strategy)
    revenues['p_divide_by_s'] = market_cap['market_capitalization'] / revenues['total_revenue']
    r = revenues.drop(columns=['total_revenue'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_ev_divide_by_s(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    ev = build_ev(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    revenues = build_revenues(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    ev['ev_divide_by_s'] = ev['ev'] / revenues['total_revenue']
    r = ev.drop(columns=['ev'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


def build_roe(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy):
    net_income = build_net_income(all_facts, market_data, use_report_date, build_ltm_strategy, build_instant_strategy)
    net_income['equity'] = build_equity(all_facts, market_data, use_report_date, build_ltm_strategy,
                                        build_instant_strategy)

    net_income['roe'] = net_income['net_income'] / net_income['equity']
    r = net_income.drop(columns=['net_income', 'equity'])
    r.replace([np.inf, -np.inf], np.nan, inplace=True)
    return r


global_indicators = {
    'total_revenue': {'facts': ['us-gaap:Revenues'],
                      'build': build_revenues},

    'liabilities': {'facts': ['us-gaap:Liabilities',
                              'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                              'us-gaap:StockholdersEquity',
                              'us-gaap:LiabilitiesAndStockholdersEquity'],
                    'build': build_liabilities},

    'assets': {'facts': ['us-gaap:Assets'],
               'build': build_assets},

    'equity': {'facts': ['us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                         'us-gaap:StockholdersEquity'],
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

    'cash_and_cash_equivalents_full': {'facts': [
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_cash_and_cash_equivalents_full},

    'operating_income': {'facts': [
        'us-gaap:OperatingIncomeLoss'],
        'build': build_operating_income},

    'income_before_taxes': {'facts': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'us-gaap:NetIncomeLoss',
        'us-gaap:IncomeTaxExpenseBenefit',
    ],
        'build': build_income_before_taxes},

    'income_before_income_taxes': {'facts': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
        'build': build_income_before_income_taxes},

    'depreciation_and_amortization': {'facts': [
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets'

    ],
        'build': build_depreciation_and_amortization},

    'interest_net': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
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

    'debt': {'facts': [
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
        'us-gaap:LongTermDebtCurrent',
    ],
        'build': build_debt},

    'net_debt': {'facts': [
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
        'us-gaap:LongTermDebtCurrent',

        'us-gaap:DebtCurrent',

        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_net_debt},

    'eps': {'facts': [
        'us-gaap:EarningsPerShareDiluted',
        'us-gaap:EarningsPerShare'
    ],
        'build': build_eps},

    'shares': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
    ],
        'build': build_shares},

    'market_capitalization': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
    ],
        'build': build_market_capitalization},

    'ebitda_use_income_before_taxes': {'facts': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'us-gaap:NetIncomeLoss',
        'us-gaap:IncomeTaxExpenseBenefit',
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',
        'us-gaap:InterestIncomeExpenseNet',

        'us-gaap:InterestExpenseLesseeAssetsUnderCapitalLease',
        'us-gaap:InterestExpenseDebt',
    ],
        'build': build_ebitda_use_income_before_taxes},

    'ebitda_use_operating_income': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',
        'us-gaap:NonoperatingIncomeExpense',
        'us-gaap:GainsLossesOnExtinguishmentOfDebt',
        'us-gaap:InvestmentIncomeInterest',
        'us-gaap:InterestExpense',
        'us-gaap:OtherNonoperatingIncomeExpense',
        'us-gaap:InterestIncomeExpenseNet',
    ],
        'build': build_ebitda_use_operating_income},

    'ebitda_simple': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',
    ],
        'build': build_ebitda_simple},

    'ev': {'facts': [

        'dei:EntityCommonStockSharesOutstanding',

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
        'us-gaap:LongTermDebtCurrent',

        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_ev},

    'ev_divide_by_ebitda': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',

        'dei:EntityCommonStockSharesOutstanding',

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
        'us-gaap:LongTermDebtCurrent',

        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_ev_divide_by_ebitda},

    'liabilities_divide_by_ebitda': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',
        'us-gaap:Liabilities',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:StockholdersEquity',
        'us-gaap:LiabilitiesAndStockholdersEquity',

        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_liabilities_divide_by_ebitda},

    'net_debt_divide_by_ebitda': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',

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
        'us-gaap:LongTermDebtCurrent',

        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_net_debt_divide_by_ebitda},

    'p_divide_by_e': {'facts': [
        'us-gaap:NetIncomeLoss',
        'dei:EntityCommonStockSharesOutstanding',
    ],
        'build': build_p_divide_by_e},

    'p_divide_by_bv': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:StockholdersEquity'
    ],
        'build': build_p_divide_by_bv},

    'p_divide_by_s': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:Revenues'
    ],
        'build': build_p_divide_by_s},

    'ev_divide_by_s': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:Revenues',

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
        'us-gaap:LongTermDebtCurrent',

        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:ShortTermInvestments',
        'us-gaap:AvailableForSaleSecuritiesCurrent',
        'us-gaap:MarketableSecuritiesCurrent',
    ],
        'build': build_ev_divide_by_s},

    'roe': {'facts': [
        'us-gaap:NetIncomeLoss',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:StockholdersEquity'
    ],
        'build': build_roe},
}


def get_all_indicator_names():
    return list(global_indicators.keys())


def get_complex_indicator_names():
    return ['ebitda_simple',
            'ev',
            'ev_divide_by_ebitda',
            'liabilities_divide_by_ebitda',
            'net_debt_divide_by_ebitda',
            'p_divide_by_e',
            'p_divide_by_bv',
            'p_divide_by_s',
            'ev_divide_by_s',
            'roe']


def load_indicators_for(
        stocks_market_data,
        indicator_names=None,
        build_ltm_strategy=None,
        build_instant_strategy=None,
):
    if indicator_names is None:
        indicator_names = get_all_indicator_names()

    if build_ltm_strategy is None:
        build_ltm_strategy = PeriodIndicatorBuilder.build_ltm_with_remove_gaps

    if build_instant_strategy is None:
        build_instant_strategy = InstantIndicatorBuilder.build_series_fill_gaps

    global_cache.empty()
    fill_strategy = lambda xarr: xarr.ffill('time')
    start_date_offset = datetime.timedelta(days=365 * 2)

    def get_assets(time_coord):
        # assets = load_list(min_date=time_coord.min().values, max_date=time_coord.max().values)
        assets = load_list(min_date="2000-12-01", max_date=time_coord.max().values)
        return assets

    def get_ciks(market_data):
        asset_names = market_data.asset.to_pandas().to_list()

        assets = get_assets(market_data.time)

        assets_for_load = set()
        for asset in assets:
            if asset['id'] in asset_names and asset.get('cik') is not None:
                assets_for_load.add(asset['cik'])

        return list(assets_for_load)

    def get_us_gaap_facts_for_load(indicators):
        facts = []
        for name in indicators:
            if name in global_indicators:
                facts = facts + global_indicators[name]['facts']
        facts_unique = set(facts)
        facts_unique = list(facts_unique)
        return facts_unique

    def load_all_facts(ciks, us_gaap_facts, min_date, max_date):
        for cik_reports in load_facts(ciks, us_gaap_facts, min_date=min_date, max_date=max_date, skip_segment=False,
                                      columns=['cik', 'report_id', 'report_type', 'report_date', 'fact_name', 'period',
                                               'period_length', 'segment'],
                                      group_by_cik=True):
            yield cik_reports

    def build_indicators(all_facts, market_data, indicator_names, all_names):
        indicators_xr = []
        all_indicators_for_asset_df = None
        use_report_date = True
        for cik_reports in all_facts:
            asset_name = all_names[cik_reports[0]]
            for indicator in indicator_names:
                if indicator in global_indicators:
                    if all_indicators_for_asset_df is None:
                        all_indicators_for_asset_df = global_indicators[indicator]['build'](cik_reports[1],
                                                                                            market_data.sel(
                                                                                                asset=[asset_name]),
                                                                                            use_report_date,
                                                                                            build_ltm_strategy,
                                                                                            build_instant_strategy)
                    else:
                        all_indicators_for_asset_df[indicator] = global_indicators[indicator]['build'](cik_reports[1],
                                                                                                       market_data.sel(
                                                                                                           asset=[
                                                                                                               asset_name]),
                                                                                                       use_report_date,
                                                                                                       build_ltm_strategy,
                                                                                                       build_instant_strategy)
            if all_indicators_for_asset_df is None:
                continue

            df = all_indicators_for_asset_df.unstack().to_xarray().rename({'level_0': 'field', 'level_1': 'time'})
            df.name = asset_name
            indicators_xr.append(df)
            all_indicators_for_asset_df = None
            global_cache.empty()

        return indicators_xr

    def get_names(market_data):
        asset_names = market_data.asset.to_pandas().to_list()

        assets = get_assets(market_data.time)

        assets_for_load = {}
        for asset in assets:
            if asset['id'] in asset_names and asset.get('cik') is not None:
                assets_for_load[asset['cik']] = asset['id']

        return assets_for_load

    time_coord = stocks_market_data.time
    min_date = pd.Timestamp(time_coord.min().values).to_pydatetime().date() - parse_tail(start_date_offset)
    max_date = pd.Timestamp(time_coord.max().values).to_pydatetime().date()
    ciks = get_ciks(stocks_market_data)
    facts_names = get_us_gaap_facts_for_load(indicator_names)
    all_facts = load_all_facts(ciks, facts_names, min_date, max_date)

    all_names = get_names(stocks_market_data)
    builded_indicators = build_indicators(all_facts, stocks_market_data, indicator_names, all_names)

    if len(builded_indicators) is 0:
        return None  # TODO

    idc_arr = xr.concat(builded_indicators, pd.Index([d.name for d in builded_indicators], name='asset'))

    idc_arr = xr.align(idc_arr, time_coord, join='outer')[0]
    idc_arr = idc_arr.sel(time=np.sort(idc_arr.time.values))
    idc_arr = fill_strategy(idc_arr)
    idc_arr = idc_arr.sel(time=time_coord)

    idc_arr.name = "secgov_indicators"
    idc_arr = idc_arr.transpose('time', 'field', 'asset')
    return idc_arr
