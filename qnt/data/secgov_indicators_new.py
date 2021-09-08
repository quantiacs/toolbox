import datetime as dt
import itertools

import pandas as pd

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


def get_ltm(all_facts, market_data, fact_name, new_name, use_report_date=True):
    key = global_cache.get_key_for(all_facts, market_data, fact_name, new_name, use_report_date)
    if global_cache.is_in_cache(key):
        return global_cache.get(key)
    facts = get_filtered(all_facts, [fact_name])
    indicator = PeriodIndicatorBuilder(new_name, [fact_name], use_report_date, 'ltm')
    r = indicator.build_series_dict(
        facts)

    result = get_df(r, market_data, new_name)
    global_cache.add(key, result)
    return result.copy()


def get_annual(all_facts, market_data, fact_name, new_name, use_report_date=True):
    key = global_cache.get_key_for(all_facts, market_data, fact_name, new_name, use_report_date)
    if global_cache.is_in_cache(key):
        return global_cache.get(key)

    facts = get_filtered(all_facts, [fact_name])
    indicator_df = get_simple_indicator(facts, market_data,
                                        fact_name, use_report_date)
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


def get_simple_indicator(all_facts, market_data, fact_name, use_report_date):
    indicator = InstantIndicatorBuilder(fact_name,
                                        fact_name,
                                        use_report_date)
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


def build_losses_on_extinguishment_of_debt(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:GainsLossesOnExtinguishmentOfDebt'
    new_name = 'losses_on_extinguishment_of_debt'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)


def build_shares(all_facts, market_data, use_report_date=True):
    # https://www.sec.gov/structureddata/announcement/osd-announcement-110520-scaling-errors
    fact_name = 'dei:EntityCommonStockSharesOutstanding'
    new_name = 'shares'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)


def build_cash_and_cash_equivalent(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:CashAndCashEquivalentsAtCarryingValue'
    new_name = 'cash_and_cash_equivalent'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)


def build_assets(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:Assets'
    new_name = 'assets'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)


def build_equity(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
    new_name = 'equity'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)


def build_revenues(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:Revenues'
    new_name = 'total_revenue'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_income_before_income_taxes(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
    new_name = 'income_before_income_taxes'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_operating_income(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:OperatingIncomeLoss'
    new_name = 'operating_income'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_income_interest(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:InvestmentIncomeInterest'
    new_name = 'income_interest'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_interest_expense(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:InterestExpense'
    new_name = 'interest_expense'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_interest_expense_debt(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:InterestExpenseDebt'
    new_name = 'interest_expense_debt'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_interest_expense_capital_lease(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:InterestExpenseLesseeAssetsUnderCapitalLease'
    new_name = 'interest_expense_capital_lease'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_interest_income_expense_net(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:InterestIncomeExpenseNet'
    new_name = 'interest_income_expense_net'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_other_nonoperating_income_expense(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:OtherNonoperatingIncomeExpense'
    new_name = 'other_nonoperating_income_expense'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_nonoperating_income_expense(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:NonoperatingIncomeExpense'
    new_name = 'nonoperating_income_expense'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_cash_and_cash_equivalents(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:CashAndCashEquivalentsAtCarryingValue'
    new_name = 'cash_and_cash_equivalents'
    return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)


def build_net_income(all_facts, market_data, use_report_date=True):
    fact_name = 'us-gaap:NetIncomeLoss'
    new_name = 'net_income'
    return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)


def build_eps(all_facts, market_data, use_report_date=True):
    def get_merged(row):
        diluted = row['eps_diluted']
        if not math.isnan(diluted):
            return diluted

        return row['eps_simple']

    fact_name = 'us-gaap:EarningsPerShareDiluted'
    new_name = 'eps_diluted'
    eps = get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)

    fact_name = 'us-gaap:EarningsPerShare'
    new_name = 'eps_simple'
    eps['eps_simple'] = get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)

    eps['eps'] = eps.apply(get_merged, axis=1)

    result = eps.drop(columns=['eps_simple', 'eps_diluted'])
    return result


def build_ev(all_facts, market_data, use_report_date=True):
    cash = build_cash_and_cash_equivalents(all_facts, market_data, use_report_date)
    cash['liabilities'] = build_liabilities(all_facts, market_data, use_report_date)
    cash['market_capitalization'] = build_market_capitalization(all_facts, market_data, use_report_date)
    cash['ev'] = cash['market_capitalization'] + cash['liabilities'] - cash['cash_and_cash_equivalents']
    r = cash.drop(columns=['market_capitalization', 'liabilities', 'cash_and_cash_equivalents'])
    return r


def build_market_capitalization(all_facts, market_data, use_report_date=True):
    shares_df = build_shares(all_facts, market_data, use_report_date)
    close_price = market_data.sel(field='close')
    close_price_df = close_price.to_pandas()
    shares_df['price'] = close_price_df[close_price_df.columns[0]]

    shares_df['market_capitalization'] = shares_df['shares'] * shares_df['price']
    r = shares_df.drop(columns=['shares', 'price'])
    return r


def build_liabilities(all_facts, market_data, use_report_date=True):
    def get_liabilities():
        fact_name = 'us-gaap:Liabilities'
        new_name = 'liabilities'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)

    def get_stockholders_equity():
        fact_name = 'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
        new_name = 'tockholders_equity'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)

    def get_total():
        fact_name = 'us-gaap:LiabilitiesAndStockholdersEquity'
        new_name = 'total'
        return get_annual(all_facts, market_data, fact_name, new_name, use_report_date)

    liabilities = get_liabilities()
    count_v = len(liabilities['liabilities'])
    count_null = liabilities['liabilities'].isnull().sum()
    is_all_null = count_v == count_null
    if is_all_null:
        stockholders_equity = get_stockholders_equity()['tockholders_equity']
        total = get_total()['total']
        liabilities['liabilities'] = total - stockholders_equity

    return liabilities

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


def build_income_before_taxes(all_facts, market_data, use_report_date=True):
    def get_income_tax():
        fact_name = 'us-gaap:IncomeTaxExpenseBenefit'
        new_name = 'income_tax'
        return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)

    def get_income_before_taxes_():
        fact_name = 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
        new_name = 'income_before_taxes'
        return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)

    def get_merged(row):
        income_before_taxes = row['income_before_taxes']
        if not math.isnan(income_before_taxes):
            return income_before_taxes
        net_income = row['net_income']
        income_tax = row['income_tax']
        return net_income + income_tax

    income_df = get_income_before_taxes_()
    income_df['net_income'] = build_net_income(all_facts, market_data, use_report_date)
    income_df['income_tax'] = get_income_tax()
    income_df['merged'] = income_df.apply(get_merged, axis=1)
    result = income_df.drop(columns=['income_tax', 'net_income', 'income_before_taxes'])
    result = result.rename(columns={"merged": "income_before_taxes"})
    return result


def build_interest_net(all_facts, market_data, use_report_date=True):
    operating_income_df = build_operating_income(all_facts, market_data, use_report_date)
    income_before_taxes_df = build_income_before_taxes(all_facts, market_data, use_report_date)
    r = operating_income_df.copy()
    r['interest_net'] = operating_income_df['operating_income'] - income_before_taxes_df['income_before_taxes']
    r = r.drop(columns=['operating_income'])
    return r


def build_depreciation_and_amortization(all_facts, market_data, use_report_date=True):
    def get_ltm(fact_name, new_name):
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
                report_date = dt.datetime.strptime(g[0], '%Y-%m-%d').date().isoformat()
                sort_key = lambda f: (f['period'])
                facts_in_report_sorted = sorted(facts_in_report, key=sort_key)

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


def build_ebitda_use_income_before_taxes(all_facts, market_data, use_report_date=True):
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

    income_before_taxes_df = build_income_before_taxes(all_facts, market_data, use_report_date)
    r = income_before_taxes_df.copy()
    r['depreciation_and_amortization'] = build_depreciation_and_amortization(all_facts, market_data, use_report_date)[
        'depreciation_and_amortization'].fillna(0)
    r['interest_income_expense_net'] = build_interest_income_expense_net(all_facts, market_data, use_report_date)[
        'interest_income_expense_net'].fillna(0)
    r['losses_on_extinguishment_of_debt'] = \
        build_losses_on_extinguishment_of_debt(all_facts, market_data, use_report_date)[
            'losses_on_extinguishment_of_debt'].fillna(0)
    r['debt'] = \
        build_interest_expense_debt(all_facts, market_data, use_report_date)['interest_expense_debt'].fillna(0)

    r['interest_expense_capital_lease'] = \
        build_interest_expense_capital_lease(all_facts, market_data, use_report_date)[
            'interest_expense_capital_lease'].fillna(0)
    r['income_interest'] = \
        build_income_interest(all_facts, market_data, use_report_date)[
            'income_interest'].fillna(0)
    r['interest_expense'] = \
        build_interest_expense(all_facts, market_data, use_report_date)[
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


def build_ebitda_use_operating_income(all_facts, market_data, use_report_date=True):
    def get_merged_interest(row):
        interest_expense = row['interest_expense']
        income_interest = row['income_interest']

        if not math.isnan(interest_expense):
            return interest_expense

        if not math.isnan(income_interest):
            return income_interest

        return interest_expense

    operating_income_df = build_operating_income(all_facts, market_data, use_report_date)
    depreciation_and_amortization_df = build_depreciation_and_amortization(all_facts, market_data, use_report_date)
    income_interest_df = build_income_interest(all_facts, market_data, use_report_date)
    onoperating_income_expense_df = build_nonoperating_income_expense(all_facts, market_data, use_report_date)
    losses_on_extinguishment_of_debt_df = build_losses_on_extinguishment_of_debt(all_facts, market_data,
                                                                                 use_report_date)

    other_nonoperating_income_expense_df = build_other_nonoperating_income_expense(all_facts, market_data,
                                                                                   use_report_date)
    interest_income_expense_net_df = build_interest_income_expense_net(all_facts, market_data,
                                                                       use_report_date)

    interest_expense_df = build_interest_expense(all_facts, market_data, use_report_date)
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


    def for_test_ebitda(depreciation_and_amortization_df, income_interest_df, losses_on_extinguishment_of_debt_df,
                        onoperating_income_expense_df, operating_income_df, r, interest_expense_df,
                        merged_interest_expense_df):
        date = '2017-03-01'
        date = '2021-01-28'
        operating_income = operating_income_df.loc[date].max() / 1000000
        depreciation_and_amortization = depreciation_and_amortization_df.loc[date].max() / 1000000
        income_interest = income_interest_df.loc[date].max() / 1000000
        onoperating_income_expense = onoperating_income_expense_df.loc[date].max() / 1000000
        losses_on_extinguishment_of_debt = losses_on_extinguishment_of_debt_df.loc[date].max() / 1000000
        interest_expense = interest_expense_df.loc[date].max() / 1000000
        merged_interest_expense = merged_interest_expense_df.loc[date].max() / 1000000
        operating_income_plys_amortization = operating_income + depreciation_and_amortization
        income_interest_onoperating_income_expense = income_interest + onoperating_income_expense
        # income_before_taxes = income_before_taxes_df.loc[date].max() / 1000000
        sum_interest = income_interest - interest_expense
        sum = r['ebitda_use_operating_income'].loc[date].max() / 1000000
        return

    for_test_ebitda(depreciation_and_amortization_df, income_interest_df, losses_on_extinguishment_of_debt_df,
                    onoperating_income_expense_df, operating_income_df, r, interest_expense_df, merged_interest_expense,
                    )

    r = r.drop(columns=['operating_income'])
    return r


def build_ebitda_simple(all_facts, market_data, use_report_date=True):
    operating_income_df = build_operating_income(all_facts, market_data, use_report_date)
    depreciation_and_amortization_df = build_depreciation_and_amortization(all_facts, market_data, use_report_date)

    operating_income = operating_income_df['operating_income'].fillna(0)
    depreciation_and_amortization = depreciation_and_amortization_df['depreciation_and_amortization'].fillna(0)

    operating_income_df['ebitda_simple'] = operating_income + depreciation_and_amortization

    r = operating_income_df.drop(columns=['operating_income'])
    return r


def build_ev_divide_by_ebitda(all_facts, market_data, use_report_date=True):
    ebitda_simple = build_ebitda_simple(all_facts, market_data, use_report_date)
    ev = build_ev(all_facts, market_data, use_report_date)
    ebitda_simple['ev_divide_by_ebitda'] = ev['ev'] / ebitda_simple['ebitda_simple']
    r = ebitda_simple.drop(columns=['ebitda_simple'])
    return r


def build_liabilities_divide_by_ebitda(all_facts, market_data, use_report_date=True):
    ebitda_simple = build_ebitda_simple(all_facts, market_data, use_report_date)
    liabilities = build_liabilities(all_facts, market_data, use_report_date)
    ebitda_simple['liabilities_divide_by_ebitda'] = liabilities['liabilities'] / ebitda_simple['ebitda_simple']
    r = ebitda_simple.drop(columns=['ebitda_simple'])
    return r


def build_p_divide_by_e(all_facts, market_data, use_report_date=True):
    net_income = build_net_income(all_facts, market_data, use_report_date)
    market_cap = build_market_capitalization(all_facts, market_data, use_report_date)
    net_income['p_divide_by_e'] = market_cap['market_capitalization'] / net_income['net_income']
    r = net_income.drop(columns=['net_income'])
    return r


def build_p_divide_by_bv(all_facts, market_data, use_report_date=True):
    equity = build_equity(all_facts, market_data, use_report_date)
    market_cap = build_market_capitalization(all_facts, market_data, use_report_date)
    equity['p_divide_by_bv'] = market_cap['market_capitalization'] / equity['equity']
    r = equity.drop(columns=['equity'])
    return r


def build_p_divide_by_s(all_facts, market_data, use_report_date=True):
    revenues = build_revenues(all_facts, market_data, use_report_date)
    market_cap = build_market_capitalization(all_facts, market_data, use_report_date)
    revenues['p_divide_by_s'] = market_cap['market_capitalization'] / revenues['total_revenue']
    r = revenues.drop(columns=['total_revenue'])
    return r


def build_ev_divide_by_s(all_facts, market_data, use_report_date=True):
    ev = build_ev(all_facts, market_data, use_report_date)
    revenues = build_revenues(all_facts, market_data, use_report_date)
    ev['ev_divide_by_s'] = ev['ev'] / revenues['total_revenue']
    r = ev.drop(columns=['ev'])
    return r


def build_roe(all_facts, market_data, use_report_date=True):
    net_income = build_net_income(all_facts, market_data, use_report_date)
    net_income['equity'] = build_equity(all_facts, market_data, use_report_date)

    net_income['roe'] = net_income['net_income'] / net_income['equity']
    r = net_income.drop(columns=['net_income', 'equity'])
    return r


global_indicators = {
    'total_revenue': {'facts': ['us-gaap:Revenues'],
                      'build': build_revenues},

    'liabilities': {'facts': ['us-gaap:Liabilities',
                              'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                              'us-gaap:LiabilitiesAndStockholdersEquity'],
                    'build': build_liabilities},

    'assets': {'facts': ['us-gaap:Assets'],
               'build': build_assets},

    'equity': {'facts': ['us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
               'build': build_equity},

    'net_income': {'facts': ['us-gaap:NetIncomeLoss'],
                   'build': build_net_income},

    'cash_and_cash_equivalents': {'facts': [
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
    ],
        'build': build_cash_and_cash_equivalents},

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
        'us-gaap:Liabilities',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:LiabilitiesAndStockholdersEquity',
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
    ],
        'build': build_ev},

    'ev_divide_by_ebitda': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets',
        'us-gaap:Liabilities',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:LiabilitiesAndStockholdersEquity',
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
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
        'us-gaap:LiabilitiesAndStockholdersEquity',
    ],
        'build': build_liabilities_divide_by_ebitda},

    'p_divide_by_e': {'facts': [
        'us-gaap:NetIncomeLoss',
        'dei:EntityCommonStockSharesOutstanding',
    ],
        'build': build_p_divide_by_e},

    'p_divide_by_bv': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
    ],
        'build': build_p_divide_by_bv},

    'p_divide_by_s': {'facts': [
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:Revenues'
    ],
        'build': build_p_divide_by_s},

    'ev_divide_by_s': {'facts': [
        'us-gaap:Liabilities',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'us-gaap:LiabilitiesAndStockholdersEquity',
        'dei:EntityCommonStockSharesOutstanding',
        'us-gaap:CashAndCashEquivalentsAtCarryingValue',
        'us-gaap:Revenues'
    ],
        'build': build_ev_divide_by_s},

    'roe': {'facts': [
        'us-gaap:NetIncomeLoss',
        'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
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
            'p_divide_by_e',
            'p_divide_by_bv',
            'p_divide_by_s',
            'ev_divide_by_s',
            'roe']


def load_fundamental_indicators_for(
        stocks_market_data,
        indicator_names=None,
):
    if indicator_names is None:
        indicator_names = get_all_indicator_names()

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

        assets_for_load = []
        for asset in assets:
            if asset['id'] in asset_names and asset.get('cik') is not None:
                assets_for_load.append(asset['cik'])

        return assets_for_load

    def get_us_gaap_facts_for_load(indicators):
        facts = []
        for name in indicators:
            if name in global_indicators:
                facts = facts + global_indicators[name]['facts']
        facts_unique = set(facts)
        facts_unique = list(facts_unique)
        return facts_unique

    def load_all_facts(ciks, us_gaap_facts, min_date, max_date):
        for cik_reports in load_facts(ciks, us_gaap_facts, min_date=min_date, max_date=max_date, skip_segment=True,
                                      columns=['cik', 'report_id', 'report_type', 'report_date', 'fact_name', 'period',
                                               'period_length'],
                                      group_by_cik=True):
            yield cik_reports

    def build_indicators(all_facts, market_data, indicator_names, all_names):
        indicators_xr = []
        all_indicators_for_asset_df = None
        for cik_reports in all_facts:
            asset_name = all_names[cik_reports[0]]
            for indicator in indicator_names:
                if indicator in global_indicators:
                    if all_indicators_for_asset_df is None:
                        all_indicators_for_asset_df = global_indicators[indicator]['build'](cik_reports[1],
                                                                                            market_data.sel(
                                                                                                asset=[asset_name]))
                    else:
                        all_indicators_for_asset_df[indicator] = global_indicators[indicator]['build'](cik_reports[1],
                                                                                                       market_data.sel(
                                                                                                           asset=[
                                                                                                               asset_name]))
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
