from qnt.data.common import *
from qnt.data.secgov import load_facts
import itertools
import pandas as pd
import datetime as dt
from qnt.log import log_info, log_err


def load_indicators(
        assets,
        time_coord,
        standard_indicators=None,
        builders=None,
        start_date_offset=datetime.timedelta(days=365 * 2),
        fill_strategy=lambda xarr: xarr.ffill('time')
):
    cik2id = dict((a['cik'], a['id']) for a in assets if a.get('cik') is not None)
    min_date = pd.Timestamp(time_coord.min().values).to_pydatetime().date() - parse_tail(start_date_offset)
    max_date = pd.Timestamp(time_coord.max().values).to_pydatetime().date()
    indicator_dicts = load_indicator_dicts(list(cik2id.keys()), standard_indicators, builders, min_date, max_date)

    dfs = []

    for (cik, inds) in indicator_dicts:
        series = [pd.Series(v if len(v) > 0 else {min_date.isoformat(): np.nan}, dtype=np.float64, name=k)
                  for (k, v) in inds.items()]
        df = pd.concat(series, axis=1)
        df.index = df.index.astype(dtype=time_coord.dtype, copy=False)
        df = df.unstack().to_xarray().rename({'level_0': 'field', 'level_1': 'time'})
        df.name = cik2id[cik]
        dfs.append(df)

    if len(dfs) is 0:
        return None  # TODO

    idc_arr = xr.concat(dfs, pd.Index([d.name for d in dfs], name='asset'))

    idc_arr = xr.align(idc_arr, time_coord, join='outer')[0]
    idc_arr = idc_arr.sel(time=np.sort(idc_arr.time.values))
    idc_arr = fill_strategy(idc_arr)
    idc_arr = idc_arr.sel(time=time_coord)

    idc_arr.name = "secgov_indicators"
    idc_arr = idc_arr.transpose('time', 'field', 'asset')

    return idc_arr


secgov_load_indicators = deprecated_wrap(load_indicators)


def load_indicator_dicts(ciks, standard_indicators=None, builders=None, min_date=None, max_date=None,
                         tail=DEFAULT_TAIL):
    if builders is None:
        builders = []
    else:
        builders = list(builders)

    if standard_indicators is None:
        builders = builders + standard_indicator_builders
    else:
        for a in standard_indicators:
            for si in standard_indicator_builders:
                if si.alias == a:
                    builders.append(si)

    fact_names = [f for b in builders for f in b.facts]
    fact_names = set(fact_names)
    fact_names = list(fact_names)

    for g in load_facts(ciks, fact_names, min_date=min_date, max_date=max_date, skip_segment=True, tail=tail,
                        columns=['cik', 'report_id', 'report_type', 'report_date', 'fact_name', 'period',
                                 'period_length'],
                        group_by_cik=True):
        indicators = dict()
        for b in builders:
            data = [d for d in g[1] if d['fact_name'] in b.facts]
            indicators[b.alias] = b.build_series_dict(data)
        yield (g[0], indicators)


secgov_load_indicator_dicts = deprecated_wrap(load_indicator_dicts)


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

    @staticmethod
    def get_restored_by_segment(facts_in_report):
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

        no_segment_fact = get_no_segment_fact(facts_in_report)
        if len(no_segment_fact) != 0:
            return no_segment_fact
        return get_accumulated_by_segment(facts_in_report)


class InstantIndicatorBuilder(IndicatorBuilder):

    def __init__(self, alias, facts, use_report_date, build_strategy=None):
        super().__init__(alias, facts, use_report_date)
        self.group_key = (lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'])

        if build_strategy is None:
            self.build = InstantIndicatorBuilder.build_series_fill_gaps
        else:
            self.build = build_strategy

    def build_series_dict(self, fact_data):
        if len(fact_data) == 0:
            return []

        fact_data = sorted(fact_data, key=self.sort_key, reverse=True)
        groups = itertools.groupby(fact_data, self.group_key)

        series_ = []

        for g in groups:
            facts_in_report = list(g[1])
            filtered_facts_in_report = IndicatorBuilder.get_restored_by_segment(facts_in_report)
            report_date = dt.datetime.strptime(g[0], '%Y-%m-%d')
            value = filtered_facts_in_report[0]['value']
            series_.append((value, report_date))

        restore = self.build(series_)
        return dict((g[1].strftime('%Y-%m-%d'), g[0]) for g in restore)

    @staticmethod
    def build_series_fill_gaps(fact_series_dict):

        border_max_gap_days = 380

        def fill_gaps(series_dict):
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

                if dist >= border_max_gap_days:
                    restore_null_value = previous_date + dt.timedelta(days=border_max_gap_days)
                    r.append([0, restore_null_value])

                r.append(fact)
                previous_fact = fact

            today = dt.datetime.today()
            last_fact_date = previous_fact[1]
            dist_today = (today - last_fact_date).days

            if dist_today >= border_max_gap_days:
                restore_null_value = last_fact_date + dt.timedelta(days=border_max_gap_days)
                r.append([0, restore_null_value])

            return r

        r = fill_gaps(fact_series_dict)

        sort_merged = sorted(r, key=lambda f: (f[1]))

        return sort_merged


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

    def __init__(self, alias, facts, use_report_date, periods, build_ltm_strategy=None):
        super().__init__(alias, facts, use_report_date)
        self.periods = periods

        if self.use_report_date:
            self.sort_key = lambda f: (f['report_date'], f['period'], f['report_id'], -self.facts.index(f['fact_name']))
        else:
            self.sort_key = lambda f: (f['period'], f['report_date'], f['report_id'], -self.facts.index(f['fact_name']))

        self.group_key = (lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'][1])

        if build_ltm_strategy is None:
            self.build_ltm = PeriodIndicatorBuilder.build_ltm_with_remove_gaps
        else:
            self.build_ltm = build_ltm_strategy

    def build_series_dict(self, fact_data):
        def get_annual(fact_data):
            annual = [f for f in fact_data if 340 < f['period_length'] < 380]
            groups = itertools.groupby(annual, self.group_key)
            return dict((g[0], next(g[1])['value']) for g in groups)

        def get_quarter_all(fact_data):
            q_value_date_all = self.build_series_qf(fact_data)
            return dict((q_value_date[1], q_value_date[0]) for q_value_date in reversed(q_value_date_all))

        fact_data = sorted(fact_data, key=self.sort_key, reverse=True)

        if self.periods == 'ltm':
            quarter_all = self.build_series_qf(fact_data)
            annual_all = get_annual(fact_data)
            result = self.build_ltm(quarter_all, annual_all)
            return dict((item[1].date().isoformat(), item[0]) for item in reversed(result))
        elif self.periods == 'qf':
            return get_quarter_all(fact_data)
        elif self.periods == 'af':
            return get_annual(fact_data)

    def build_series_qf(self, fact_data):
        reports_all = sorted(fact_data, key=self.sort_key)
        q_value_date_all = []
        all_facts_for_recovered_q_values = []

        groups = itertools.groupby(reports_all, self.group_key)

        for g in groups:

            q_indexis = []
            k_indexis = []

            facts_in_report = list(g[1])
            filtered_facts_in_report = IndicatorBuilder.get_restored_by_segment(facts_in_report)

            report_date = g[0]

            for i, f in enumerate(filtered_facts_in_report):
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
                q_value_date_all.append([filtered_facts_in_report[q_indexis[-1]]['value'], report_date])
                continue

            if is_K_report_exist and is_Q_report_exist == False:
                first_k_date = dt.datetime.strptime(filtered_facts_in_report[k_indexis[-1]]['period'][0], '%Y-%m-%d')
                k_value = filtered_facts_in_report[k_indexis[-1]]['value']

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
                                                      filtered_facts_in_report[k_indexis[-1]]['value'])
                    recovered_q_value = k_value - previous_3q

                q_value_date_all.append([recovered_q_value, report_date])
                continue

            if is_K_report_exist and is_Q_report_exist:
                q_fact = filtered_facts_in_report[q_indexis[-1]]
                last_q_date = dt.datetime.strptime(q_fact['period'][1], '%Y-%m-%d')

                k_fact = filtered_facts_in_report[k_indexis[-1]]
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
                                                          filtered_facts_in_report[k_indexis[-1]]['value'])
                        recovered_q_value = k_value - previous_3q
                    q_value_date_all.append([recovered_q_value, report_date])
                continue

            q_value_date_all.append([np.nan, report_date])

        return q_value_date_all

    @staticmethod
    def build_ltm_with_remove_gaps(quarter_all, annual_all):

        border_quarter_days = 120
        border_max_gap_days = 380

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

        def fill_gaps(series_dict):
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

                if dist >= border_max_gap_days:
                    restore_null_value = previous_date + dt.timedelta(days=border_max_gap_days)
                    r.append([0, restore_null_value])

                r.append(fact)
                previous_fact = fact

            today = dt.datetime.today()
            last_fact_date = previous_fact[1]
            dist_today = (today - last_fact_date).days

            if dist_today >= border_max_gap_days:
                restore_null_value = last_fact_date + dt.timedelta(days=border_max_gap_days)
                r.append([0, restore_null_value])

            return r

        values_for_annual = []
        date_for_annual = []
        value = 0
        period = 1
        result_new = []
        sort_type = lambda f: (f[period])
        quarter_all = sorted(quarter_all, key=sort_type)
        for quarter in quarter_all:

            if len(date_for_annual) != 0:
                previous_report_date = dt.datetime.strptime(date_for_annual[-1], '%Y-%m-%d')
                report_date = dt.datetime.strptime(quarter[period], '%Y-%m-%d')
                dist = (report_date - previous_report_date).days

                is_gap_in_reports = dist > border_quarter_days
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

        merged = get_merge(annual_all, result_new)

        rr = fill_gaps(merged)

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


standard_indicator_builders = [
    # instant
    InstantIndicatorBuilder('assets', ['us-gaap:Assets'], True),
    InstantIndicatorBuilder('assets_curr', ['us-gaap:AssetsCurrent'], True),
    InstantIndicatorBuilder('equity', ['us-gaap:StockholdersEquity'], True),
    InstantIndicatorBuilder('liabilities', ['us-gaap:Liabilities'], True),
    InstantIndicatorBuilder('liabilities_curr', ['us-gaap:LiabilitiesCurrent'], True),
    InstantIndicatorBuilder('market_cap', ['us-gaap:CapitalizationLongtermDebtAndEquity'], True),
    InstantIndicatorBuilder('debt_lt', ['us-gaap:LongTermDebt'], True),
    InstantIndicatorBuilder('debt_st', ['us-gaap:ShortTermBorrowings'], True),
    InstantIndicatorBuilder('goodwill', ['us-gaap:Goodwill'], True),
    InstantIndicatorBuilder('inventory', ['us-gaap:InventoryNet'], True),
    InstantIndicatorBuilder('ivestment_short_term',
                            ['us-gaap:AvailableForSaleSecuritiesCurrent', 'us-gaap:MarketableSecuritiesCurrent'], True),
    InstantIndicatorBuilder('invested_capital', ['us-gaap:MarketableSecurities'], True),
    InstantIndicatorBuilder('shares', ['us-gaap:CommonStockSharesOutstanding', 'us-gaap:CommonStockSharesIssued'],
                            True),
    InstantIndicatorBuilder('ppent', ['us-gaap:PropertyPlantAndEquipmentNet'], True),
    InstantIndicatorBuilder('cash_equivalent', ['us-gaap:CashAndCashEquivalentsAtCarryingValue'], True),
    # period
    PeriodIndicatorBuilder('sales_revenue', [
        'us-gaap:SalesRevenueGoodsNet',
        'us-gaap:SalesRevenueNet',
        'us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax'
    ], True, 'ltm'),
    PeriodIndicatorBuilder('total_revenue', ['us-gaap:Revenues'], True, 'ltm'),
    PeriodIndicatorBuilder('capex', ['us-gaap:CapitalExpenditureDiscontinuedOperations'], True, 'ltm'),
    PeriodIndicatorBuilder('cashflow_op', [
        'us-gaap:OtherOperatingActivitiesCashFlowStatement',
        'us-gaap:NetCashProvidedByUsedInOperatingActivities',
        'us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'
    ], True, 'ltm'),
    PeriodIndicatorBuilder('cogs', [
        'us-gaap:CostOfGoodsAndServicesSold',
        'us-gaap:CostOfGoodsSold',
        'us-gaap:CostOfRevenue'
    ], True, 'ltm'),
    PeriodIndicatorBuilder('divs', ['us-gaap:Dividends'], True, 'ltm'),
    PeriodIndicatorBuilder('eps', [
        'us-gaap:EarningsPerShareDiluted',
        'us-gaap:EarningsPerShare'
    ], True, 'ltm'),
    PeriodIndicatorBuilder('income', ['us-gaap:NetIncomeLoss'], True, 'ltm'),
    PeriodIndicatorBuilder('interest_expense', ['us-gaap:InterestExpense'], True, 'ltm'),
    PeriodIndicatorBuilder('operating_expense', ['us-gaap:OperatingExpenses'], True, 'ltm'),
    PeriodIndicatorBuilder('operating_income', ['us-gaap:OperatingIncomeLoss'], True, 'ltm'),
    PeriodIndicatorBuilder('rd_expense', ['us-gaap:ResearchAndDevelopmentExpense'], True, 'ltm'),
    PeriodIndicatorBuilder('retained_earnings', ['us-gaap:PostconfirmationRetainedEarningsDeficit'], True, 'ltm'),
    PeriodIndicatorBuilder('sales_ps', ['us-gaap:EarningsPerShareBasic'], True, 'ltm'),
    PeriodIndicatorBuilder('sga_expense', ['us-gaap:SellingGeneralAndAdministrativeExpense'], True, 'ltm'),
]
for s in standard_indicator_builders + []:
    if type(s) == PeriodIndicatorBuilder:
        standard_indicator_builders.append(PeriodIndicatorBuilder(s.alias + '_ltm', s.facts, s.use_report_date, 'ltm'))
        standard_indicator_builders.append(PeriodIndicatorBuilder(s.alias + '_af', s.facts, s.use_report_date, 'af'))
        standard_indicator_builders.append(PeriodIndicatorBuilder(s.alias + '_qf', s.facts, s.use_report_date, 'qf'))

if __name__ == '__main__':
    import qnt.data.stocks as qds
    import qnt.data.common as qdc

    qdc.BASE_URL = 'http://127.0.0.1:8001/'

    assets = qds.load_assets()
    data = qds.load_data(assets)

    indicators = secgov_load_indicators(assets, data.time)
    log_info(indicators)
