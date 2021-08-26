from qnt.data.common import *
from qnt.data.stocks import load_list
from qnt.data.secgov import load_facts
import itertools
import pandas as pd
import datetime as dt
from qnt.log import log_info, log_err


class IndicatorBuilder:
    facts = None
    alias = None
    use_report_date = None
    sort_key = None
    group_key = None

    def __init__(self, alias, facts, use_report_date):
        self.facts = facts
        self.alias = alias
        self.use_report_date = use_report_date
        if (self.use_report_date):
            self.sort_key = lambda f: (f['report_date'], f['period'], f['report_id'], -self.facts.index(f['fact_name']))
        else:
            self.sort_key = lambda f: (f['period'], f['report_date'], f['report_id'], -self.facts.index(f['fact_name']))

    def build_series_dict(self, fact_data):
        pass


class InstantIndicatorBuilder(IndicatorBuilder):

    def __init__(self, alias, facts, use_report_date):
        super().__init__(alias, facts, use_report_date)
        self.group_key = (lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'])

    def build_series_dict(self, fact_data):
        fact_data = sorted(fact_data, key=self.sort_key, reverse=True)
        groups = itertools.groupby(fact_data, self.group_key)
        return dict((g[0], next(g[1])['value']) for g in groups)


class SimplePeriodIndicatorBuilder(IndicatorBuilder):
    periods = None
    """
    qf, representing quarterly values
    af, representing annual values
    saf, representing semi-annual values
    """

    def __init__(self, alias, facts, use_report_date, periods):
        super().__init__(alias, facts, use_report_date)
        self.periods = periods
        self.group_key = (lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'][1])

    def build_series_dict(self, fact_data):
        fact_data = sorted(fact_data, key=self.sort_key, reverse=True)

        # TODO restore missed semi-annual facts
        # TODO restore missed quarter facts
        # TODO ltm

        if self.periods == 'qf':
            fact_data = [f for f in fact_data if 80 < f['period_length'] < 100]
        elif self.periods == 'saf':
            fact_data = [f for f in fact_data if 170 < f['period_length'] < 190]
        elif self.periods == 'af':
            fact_data = [f for f in fact_data if 355 < f['period_length'] < 375]

        groups = itertools.groupby(fact_data, self.group_key)
        return dict((g[0], next(g[1])['value']) for g in groups)


class PeriodIndicatorBuilder(IndicatorBuilder):
    periods = None
    """    
    qf, representing quarterly values
    af, representing annual values
    ltm, representing LTM (last twelve months) values
    """

    def __init__(self, alias, facts, use_report_date, periods):
        super().__init__(alias, facts, use_report_date)
        self.periods = periods

        if self.use_report_date:
            self.sort_key = lambda f: (f['report_date'], f['period'], f['report_id'], -self.facts.index(f['fact_name']))
        else:
            self.sort_key = lambda f: (f['period'], f['report_date'], f['report_id'], -self.facts.index(f['fact_name']))

        self.group_key = (lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'][1])

    def build_series_dict(self, fact_data):
        fact_data = sorted(fact_data, key=self.sort_key, reverse=True)

        if self.periods == 'ltm':
            result = self.build_ltm(fact_data)
            return dict((item[1].date().isoformat(), item[0]) for item in reversed(result))
        elif self.periods == 'qf':
            q_value_date_all = self.build_series_qf(fact_data)
            return dict((q_value_date[1], q_value_date[0]) for q_value_date in reversed(q_value_date_all))
        elif self.periods == 'af':
            fact_data = [f for f in fact_data if 340 < f['period_length'] < 380]
            groups = itertools.groupby(fact_data, self.group_key)
            return dict((g[0], next(g[1])['value']) for g in groups)

    def build_series_qf(self, fact_data):
        reports_all = sorted(fact_data, key=self.sort_key)
        q_value_date_all = []
        all_facts_for_recovered_q_values = []

        groups = itertools.groupby(reports_all, self.group_key)

        for g in groups:

            q_indexis = []
            k_indexis = []

            facts_in_report = list(g[1])
            report_date = g[0]

            for i, f in enumerate(facts_in_report):
                if f['value'] is not None:
                    all_facts_for_recovered_q_values.append([f['period'], f['value']])

                if f['value'] is not None \
                        and f['period_length'] is not None \
                        and f['report_type'] in ['10-Q', '10-Q/A',
                                                 '10-K', '10-K/A']:
                    if (75 < f['period_length'] < 120): q_indexis.append(i)
                    if (340 < f['period_length'] < 380): k_indexis.append(i)

            is_Q_report_exist = (len(q_indexis) > 0)
            is_K_report_exist = (len(k_indexis) > 0)
            if is_Q_report_exist and is_K_report_exist == False:
                q_value_date_all.append([facts_in_report[q_indexis[-1]]['value'], report_date])
                continue

            if is_K_report_exist and is_Q_report_exist == False:
                first_k_date = dt.datetime.strptime(facts_in_report[k_indexis[-1]]['period'][0], '%Y-%m-%d')
                k_value = facts_in_report[k_indexis[-1]]['value']

                if k_value is None:
                    q_value_date_all.append([np.nan, report_date])
                    continue

                if len(q_value_date_all) == 0:
                    q_value_date_all.append([k_value / 4, report_date])
                    continue

                previous_report_date = dt.datetime.strptime(q_value_date_all[-1][1], '%Y-%m-%d')
                dist = (dt.datetime.strptime(report_date, '%Y-%m-%d') - previous_report_date).days

                is_one_year_gap_in_reports = dist > 360
                if is_one_year_gap_in_reports:
                    recovered_q_value = k_value / 4
                else:
                    previous_3q = previous_3_quarters(all_facts_for_recovered_q_values, first_k_date,
                                                      facts_in_report[k_indexis[-1]]['value'])
                    recovered_q_value = k_value - previous_3q

                q_value_date_all.append([recovered_q_value, report_date])
                continue

            if is_K_report_exist and is_Q_report_exist:
                q_fact = facts_in_report[q_indexis[-1]]
                last_q_date = dt.datetime.strptime(q_fact['period'][1], '%Y-%m-%d')

                k_fact = facts_in_report[k_indexis[-1]]
                last_k_date = dt.datetime.strptime(k_fact['period'][1], '%Y-%m-%d')
                first_k_date = dt.datetime.strptime(k_fact['period'][0], '%Y-%m-%d')

                if (last_k_date - dt.timedelta(days=5)) < last_q_date < (last_k_date + dt.timedelta(days=5)):
                    q_value_date_all.append([q_fact['value'], report_date])
                else:
                    k_value = k_fact['value']
                    if k_value is None:
                        recovered_q_value = np.nan
                    else:
                        previous_3q = previous_3_quarters(all_facts_for_recovered_q_values, first_k_date,
                                                          facts_in_report[k_indexis[-1]]['value'])
                        recovered_q_value = k_value - previous_3q
                    q_value_date_all.append([recovered_q_value, report_date])
                continue

            q_value_date_all.append([np.nan, report_date])

        return q_value_date_all

    def build_ltm(self, fact_data):
        def get_annual(fact_data):
            annual = [f for f in fact_data if 340 < f['period_length'] < 380]
            groups = itertools.groupby(annual, self.group_key)
            return dict((g[0], next(g[1])['value']) for g in groups)

        def get_merge(annuals, quarters):
            copy = quarters + []
            all_times = []
            for q in quarters:
                all_times.append(q[1])
            for time_annual in annuals:
                time = dt.datetime.strptime(time_annual, '%Y-%m-%d')
                if time not in all_times:
                    copy.append([annuals[time_annual], time])
            return copy

        def remove_gaps(series_dict):
            if len(series_dict) == 0:
                return series_dict

            r = []
            sort_f = lambda f: (f[1])
            sort_series = sorted(series_dict, key=sort_f)
            previous_fact = None
            for fact in sort_series:
                if previous_fact is None:
                    previous_fact = fact
                    r.append(fact)
                    continue

                previous_date = previous_fact[1]
                current_date = fact[1]

                dist = (current_date - previous_date).days

                if dist > 380:
                    stop = 18

                r.append(fact)
                previous_fact = fact

            today = dt.datetime.today()
            previous_date = previous_fact[1]
            dist_today = (today - previous_date).days

            if dist_today > 380:
                restore_null_value = previous_date + dt.timedelta(days=75)
                r.append([0, restore_null_value])

            return r

        values_for_annual = []
        date_for_annual = []
        value = 0
        period = 1
        result_new = []
        sort_type = lambda f: (f[period])
        quarter_all = self.build_series_qf(fact_data)
        quarter_all = sorted(quarter_all, key=sort_type)
        for quarter in quarter_all:

            if len(date_for_annual) != 0:
                previous_report_date = dt.datetime.strptime(date_for_annual[-1], '%Y-%m-%d')
                report_date = dt.datetime.strptime(quarter[period], '%Y-%m-%d')
                dist = (report_date - previous_report_date).days

                is_gap_in_reports = dist > 120  # 120 - randomly selected
                if is_gap_in_reports:
                    values_for_annual = []
                    date_for_annual = []

            values_for_annual.append(quarter[value])
            date_for_annual.append(quarter[period])

            if len(date_for_annual) == 4:
                total_for_quarter = np.nansum(values_for_annual)
                date_report_appearance = dt.datetime.strptime(date_for_annual[-1], '%Y-%m-%d')
                result_new.append([total_for_quarter, date_report_appearance])
                values_for_annual.pop(0)
                date_for_annual.pop(0)

        annual = get_annual(fact_data)

        merged = get_merge(annual, result_new)

        rr = remove_gaps(merged)

        sort_f = lambda f: (f[1])
        sort_merged = sorted(rr, key=sort_f)

        return sort_merged


def previous_3_quarters(full_list, start_time, val):
    ind1 = 0
    ind2 = 0
    ind3 = 0
    ind12 = 0
    ind23 = 0

    local_index = []
    # Searching for available timeframes
    for i, info in enumerate(full_list):
        left_bound = dt.datetime.strptime(info[0][0], '%Y-%m-%d')
        right_bound = dt.datetime.strptime(info[0][1], '%Y-%m-%d')

        left_index1 = (left_bound - dt.timedelta(days=10)) < start_time < (left_bound + dt.timedelta(days=10))
        left_index2 = (left_bound - dt.timedelta(days=110)) < start_time < (left_bound - dt.timedelta(days=70))
        left_index3 = (left_bound - dt.timedelta(days=210)) < start_time < (left_bound - dt.timedelta(days=150))

        if left_index1:
            dist = (right_bound - left_bound).days

            if 80 < dist < 120:
                local_index.extend([info[1], '1'])  # first quarter
            elif 150 < dist < 200:
                local_index.extend([info[1], '12'])  # first and second quarters
            elif 250 < dist < 290:
                local_index.extend([info[1], '123'])  # first, second and third quarters -> exit
                return info[1]

        if left_index2:

            dist = (right_bound - left_bound).days
            if 80 < dist < 100:
                local_index.extend([info[1], '2'])  # second quarter
            elif 150 < dist < 200:
                local_index.extend([info[1], '23'])  # second and third quarters

        if left_index3:
            dist = (right_bound - left_bound).days
            if 80 < dist < 120: local_index.extend([info[1], '3'])  # third quarter

    # Now let's collect information about all 3 quarters from the available data
    if len(local_index) > 0:
        if '1' in local_index: ind1 = local_index.index('1') - 1
        if '2' in local_index: ind2 = local_index.index('2') - 1
        if '3' in local_index: ind3 = local_index.index('3') - 1
        if '12' in local_index: ind12 = local_index.index('12') - 1
        if '23' in local_index: ind23 = local_index.index('23') - 1

        if '1' in local_index:
            if '2' in local_index:
                if '3' in local_index:
                    return (val - local_index[ind1] - local_index[ind2] - local_index[ind3])

            if '23' in local_index:
                return (val - local_index[ind1] - local_index[ind23])

            return (val - local_index[ind1]) / 3

        elif '12' in local_index:
            if '3' in local_index:
                return (val - local_index[ind12] - local_index[ind3])
            else:
                return (val - local_index[ind12]) / 2

        elif '2' in local_index:
            if '3' in local_index:
                return (val - local_index[ind2] - local_index[ind3]) / 2
            else:
                return (val - local_index[ind2]) / 3

        elif '23' in local_index:
            return (val - local_index[ind23]) / 2

        elif '3' in local_index:
            return (val - local_index[ind3]) / 3
    else:
        return val / 4


def get_ltm(all_facts, market_data, fact_name, new_name, use_report_date=True):
    facts = get_filtered(all_facts, [fact_name])
    indicator = PeriodIndicatorBuilder(new_name, [fact_name], use_report_date, 'ltm')
    r = indicator.build_series_dict(
        facts)

    result = get_df(r, market_data, new_name)
    return result


def get_annual(all_facts, market_data, fact_name, new_name, use_report_date=True):
    facts = get_filtered(all_facts, [fact_name])
    indicator_df = get_simple_indicator(facts, market_data,
                                        fact_name, use_report_date)
    indicator_df[new_name] = indicator_df[fact_name]
    result = indicator_df.drop(columns=[fact_name])
    return result


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
    def get_net_income():
        fact_name = 'us-gaap:NetIncomeLoss'
        new_name = 'net_income'
        return get_ltm(all_facts, market_data, fact_name, new_name, use_report_date)

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
    income_df['net_income'] = get_net_income()
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

                if is_correct_year(current):
                    last_year_value = current['value']
                    result.append([last_year_value, report_date])
                    continue

                if last_year_value is None:
                    continue

                quarter_facts = get_correct_only_quarter_facts(facts_in_report_sorted)
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
    # r['debt'] + \
    # r['interest_expense_capital_lease']
    date = '2017-02-03'
    # date = '2015-04-01'
    operating_income = r.loc[date] / 1000000
    r = r.drop(columns=['income_before_taxes', 'depreciation_and_amortization', 'interest_income_expense_net',
                        'losses_on_extinguishment_of_debt', 'interest_expense_capital_lease', 'debt'])

    return r


def build_ebitda_experiment(all_facts, market_data, use_report_date=True):
    income_before_taxes_df = build_income_before_taxes(all_facts, market_data, use_report_date)
    depreciation_and_amortization_df = build_depreciation_and_amortization(all_facts, market_data, use_report_date)
    interest_net_df = build_interest_net(all_facts, market_data, use_report_date)
    r = income_before_taxes_df.copy()
    r['ebitda_use_income_before_taxes'] = income_before_taxes_df['income_before_taxes'] + \
                                          depreciation_and_amortization_df[
                                              'depreciation_and_amortization'] - interest_net_df['interest_net']
    r = r.drop(columns=['income_before_taxes'])
    return r


def build_ebitda_use_operating_income(all_facts, market_data, use_report_date=True):
    def get_merged_interest(row):
        interest_expense = row['interest_expense']
        income_interest = row['income_interest']
        # interest_income_expense_net = row['interest_income_expense_net']
        #
        # if not math.isnan(interest_income_expense_net):
        #     return interest_income_expense_net

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

    income_before_taxes_df = build_income_before_taxes(all_facts, market_data, use_report_date)

    operating_income = operating_income_df['operating_income'].fillna(0)
    depreciation_and_amortization = depreciation_and_amortization_df['depreciation_and_amortization'].fillna(0)
    onoperating_income_expense = onoperating_income_expense_df['nonoperating_income_expense'].fillna(0)
    losses_on_extinguishment_of_debt = losses_on_extinguishment_of_debt_df['losses_on_extinguishment_of_debt'].fillna(0)
    interest_income_expense_net = interest_income_expense_net_df['interest_income_expense_net'].fillna(0)
    other_nonoperating_income_expense = other_nonoperating_income_expense_df[
        'other_nonoperating_income_expense'].fillna(0)
    merged_interest_expense = interest_expense_df['interest'].fillna(0)

    r['ebitda_use_operating_income'] = operating_income + \
                                       depreciation_and_amortization + \
                                       losses_on_extinguishment_of_debt + \
                                       onoperating_income_expense + \
                                       merged_interest_expense

    # r['ebitda_use_operating_income'] = operating_income + \
    #                                    depreciation_and_amortization + \
    #                                    losses_on_extinguishment_of_debt + \
    #                                    onoperating_income_expense + \
    #                                    merged_interest_expense + \
    #                                    onoperating_income_expense - interest_income_expense_net

    for_test_ebitda(depreciation_and_amortization_df, income_interest_df, losses_on_extinguishment_of_debt_df,
                    onoperating_income_expense_df, operating_income_df, r, interest_expense_df, merged_interest_expense,
                    income_before_taxes_df)

    r = r.drop(columns=['operating_income'])
    return r


def for_test_ebitda(depreciation_and_amortization_df, income_interest_df, losses_on_extinguishment_of_debt_df,
                    onoperating_income_expense_df, operating_income_df, r, interest_expense_df,
                    merged_interest_expense_df, income_before_taxes_df):
    date = '2017-03-01'
    date = '2016-03-30'
    operating_income = operating_income_df.loc[date].max() / 1000000
    depreciation_and_amortization = depreciation_and_amortization_df.loc[date].max() / 1000000
    income_interest = income_interest_df.loc[date].max() / 1000000
    onoperating_income_expense = onoperating_income_expense_df.loc[date].max() / 1000000
    losses_on_extinguishment_of_debt = losses_on_extinguishment_of_debt_df.loc[date].max() / 1000000
    interest_expense = interest_expense_df.loc[date].max() / 1000000
    merged_interest_expense = merged_interest_expense_df.loc[date].max() / 1000000
    operating_income_plys_amortization = operating_income + depreciation_and_amortization
    income_interest_onoperating_income_expense = income_interest + onoperating_income_expense
    income_before_taxes = income_before_taxes_df.loc[date].max() / 1000000
    sum_interest = income_interest - interest_expense
    sum = r['ebitda_use_operating_income'].loc[date].max() / 1000000
    return


global_indicators = {
    'total_revenue': {'facts': ['us-gaap:Revenues'],
                      'build': build_revenues},

    'liabilities': {'facts': ['us-gaap:Liabilities',
                              'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                              'us-gaap:LiabilitiesAndStockholdersEquity'],
                    'build': build_liabilities},

    'cash_and_cash_equivalent': {'facts': ['us-gaap:CashAndCashEquivalentsAtCarryingValue'],
                                 'build': build_cash_and_cash_equivalent},

    'assets': {'facts': ['us-gaap:Assets'],
               'build': build_assets},

    'equity': {'facts': ['us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
               'build': build_equity},

    'income_before_taxes': {'facts': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
        'us-gaap:NetIncomeLoss',
        'us-gaap:IncomeTaxExpenseBenefit',
    ],
        'build': build_income_before_taxes},

    'income_before_income_taxes': {'facts': [
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'],
        'build': build_income_before_income_taxes},

    'operating_income': {'facts': [
        'us-gaap:OperatingIncomeLoss'],
        'build': build_operating_income},

    'interest_net': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
    ],
        'build': build_interest_net},

    'depreciation_and_amortization': {'facts': [
        # 'us-gaap:DepreciationAndAmortization',
        # 'us-gaap:DepreciationAmortizationAndAccretionNet',
        # 'msft:DepreciationAmortizationAndOther',
        # 'us-gaap:Depreciation',
        # 'us-gaap:AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment',
        # 'us-gaap:DepreciationDepletionAndAmortization',
        # 'us-gaap:AmortizationOfIntangibleAssets',
        # 'us-gaap:AmortizationOfIntangibleAssets',
        # 'us-gaap:ScheduleofFiniteLivedIntangibleAssetsFutureAmortizationExpenseTableTextBlock',
        # 'us-gaap:DeferredTaxAssetsDepreciationAndAmortization',
        # 'us-gaap:FinanceLeaseRightOfUseAssetAmortization',
        # 'us-gaap:FinanceLeaseRightOfUseAssetAmortization',
        'us-gaap:DepreciationAndAmortization',
        'us-gaap:DepreciationAmortizationAndAccretionNet',
        'us-gaap:DepreciationDepletionAndAmortization',
        'us-gaap:Depreciation',
        'us-gaap:AmortizationOfIntangibleAssets'

    ],
        'build': build_depreciation_and_amortization},

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

    'nonoperating_income_expense': {'facts': [
        'us-gaap:NonoperatingIncomeExpense',
    ],
        'build': build_nonoperating_income_expense},

    'other_nonoperating_income_expense': {'facts': [
        'us-gaap:OtherNonoperatingIncomeExpense',
    ],
        'build': build_other_nonoperating_income_expense},

    'ebitda_use_income_before_taxes': {'facts': [
        'us-gaap:OperatingIncomeLoss',
        'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
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
        # 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
    ],
        'build': build_ebitda_use_operating_income},

    'losses_on_extinguishment_of_debt': {'facts': [
        'us-gaap:GainsLossesOnExtinguishmentOfDebt',
    ],
        'build': build_losses_on_extinguishment_of_debt},
}


def get_all_indicators():
    return list(global_indicators.keys())


def load_indicators_for(
        market_data,
        indicators=None,
):
    fill_strategy = lambda xarr: xarr.ffill('time')
    start_date_offset = datetime.timedelta(days=365 * 2)

    def get_ciks(market_data):
        asset_names = market_data.asset.to_pandas().to_list()
        time_coord = market_data.time

        # assets = load_list(min_date=time_coord.min().values, max_date=time_coord.max().values)
        assets = load_list(min_date="2000-12-01", max_date=time_coord.max().values)
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

    def build_indicators(all_facts, market_data, indicators, all_names):
        indicators_xr = []
        for cik_reports in all_facts:
            for indicator in indicators:
                if indicator in global_indicators:
                    res_df = global_indicators[indicator]['build'](cik_reports[1], market_data)
                    df = res_df.unstack().to_xarray().rename({'level_0': 'field', 'level_1': 'time'})
                    df.name = all_names[cik_reports[0]]
                    indicators_xr.append(df)

        return indicators_xr

    def get_names(market_data):
        asset_names = market_data.asset.to_pandas().to_list()
        time_coord = market_data.time

        # assets = load_list(min_date=time_coord.min().values, max_date=time_coord.max().values)
        assets = load_list(min_date="2000-12-01", max_date=time_coord.max().values)
        assets_for_load = {}
        for asset in assets:
            if asset['id'] in asset_names and asset.get('cik') is not None:
                assets_for_load[asset['cik']] = asset['id']

        return assets_for_load

    time_coord = market_data.time
    min_date = pd.Timestamp(time_coord.min().values).to_pydatetime().date() - parse_tail(start_date_offset)
    max_date = pd.Timestamp(time_coord.max().values).to_pydatetime().date()
    ciks = get_ciks(market_data)
    facts_names = get_us_gaap_facts_for_load(indicators)
    all_facts = load_all_facts(ciks, facts_names, min_date, max_date)

    all_names = get_names(market_data)
    builded_indicators = build_indicators(all_facts, market_data, indicators, all_names)

    if len(builded_indicators) is 0:
        return None  # TODO

    idc_arr = xr.concat(builded_indicators, pd.Index([d.name for d in builded_indicators], name='asset'))

    idc_arr = xr.align(idc_arr, time_coord, join='outer')[0]
    idc_arr = idc_arr.sel(time=np.sort(idc_arr.time.values))
    idc_arr = fill_strategy(idc_arr)
    idc_arr = idc_arr.sel(time=time_coord)

    idc_arr.name = "secgov_indicators"
    return idc_arr
    #
    # for (cik, inds) in builded_indicators:
    #     items = inds.items()
    #     series = [pd.Series(v if len(v) > 0 else {min_date.isoformat(): np.nan}, dtype=np.float64, name=k)
    #               for (k, v) in items]
    #     df = pd.concat(series, axis=1)
    #     df.index = df.index.astype(dtype=time_coord.dtype, copy=False)
    #     df = df.unstack().to_xarray().rename({'level_0': 'field', 'level_1': 'time'})
    #     df.name = all_names[cik]
    #     dfs.append(df)
    #
    # if len(dfs) is 0:
    #     return None  # TODO
    #
    # idc_arr = xr.concat(dfs, pd.Index([d.name for d in dfs], name='asset'))
    #
    # idc_arr = xr.align(idc_arr, time_coord, join='outer')[0]
    # idc_arr = idc_arr.sel(time=np.sort(idc_arr.time.values))
    # idc_arr = fill_strategy(idc_arr)
    # idc_arr = idc_arr.sel(time=time_coord)
    #
    # idc_arr.name = "secgov_indicators"
    # return idc_arr

    # Liabilities = InstantIndicatorBuilder('Liabilities', ['us-gaap:Liabilities'], use_report_date).build_series_dict(
    #     facts)
    # LiabilitiesCurrent = InstantIndicatorBuilder('LiabilitiesCurrent', ['us-gaap:LiabilitiesCurrent'],
    #                                              use_report_date).build_series_dict(facts)
    # DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent = InstantIndicatorBuilder(
    #     'DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent',
    #     ['us-gaap:DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent'], use_report_date).build_series_dict(facts)

    # Liabilities_df = get_df(Liabilities, market_data, 'us-gaap:Liabilities')
    # LiabilitiesCurrent_df = get_df(LiabilitiesCurrent, market_data, 'us-gaap:LiabilitiesCurrent')
    # LongTermDebtNoncurrent_df = get_df(LongTermDebtNoncurrent, market_data, 'us-gaap:LongTermDebtNoncurrent')
    # # DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent_df = get_df(DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent,
    # #                                                              market_data,
    # #                                                              'us-gaap:DeferredIncomeTaxesAndOtherLiabilitiesNoncurrent')
    # result = Liabilities_df.copy()
    # result['liabilities'] = Liabilities_df['us-gaap:Liabilities'] + LiabilitiesCurrent_df['us-gaap:LiabilitiesCurrent']
    # r = result.drop(columns=['us-gaap:Liabilities'])
