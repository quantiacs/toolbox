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
    builders = list(builders or [])

    if standard_indicators is None:
        builders.extend(standard_indicator_builders)
    else:
        builders.extend(filter(lambda si: si.alias in standard_indicators, standard_indicator_builders))

    fact_names = list(set(fact for builder in builders for fact in builder.facts))

    for group in load_facts(ciks, fact_names, min_date=min_date, max_date=max_date, skip_segment=True, tail=tail,
                            columns=['cik', 'report_id', 'report_type', 'report_date', 'fact_name', 'period',
                                     'period_length'],
                            group_by_cik=True):
        cik, facts_for_cik = group
        indicators = {builder.alias: builder.build_series_dict(
            [fact for fact in facts_for_cik if fact['fact_name'] in builder.facts])
            for builder in builders}

        yield (cik, indicators)


secgov_load_indicator_dicts = deprecated_wrap(load_indicator_dicts)


class IndicatorUtils:
    BORDER_MAX_GAP_DAYS = 380
    BORDER_QUARTER_DAYS = 120

    @staticmethod
    def _key_for_fact(fact):
        period = fact.get('period', '')
        period_str = " ".join(period) if isinstance(period, list) else period
        return f"{period_str} {fact['report_type']} {fact['period_length']}"

    @staticmethod
    def _accumulate_by_key(facts_in_report):
        accumulated_dict = {}
        for fact in facts_in_report:
            key = IndicatorUtils._key_for_fact(fact)
            if key in accumulated_dict:
                existing_value = accumulated_dict[key].get('value', 0)
                accumulated_dict[key]['value'] = existing_value + fact.get('value', 0)
            else:
                accumulated_dict[key] = fact
        return list(accumulated_dict.values())

    @staticmethod
    def get_restored_by_segment(facts_in_report):
        no_segment_fact = [fact for fact in facts_in_report if fact.get('segment') is None]
        return no_segment_fact if no_segment_fact else IndicatorUtils._accumulate_by_key(facts_in_report)

    @staticmethod
    def _fill_series_gaps(series, max_gap_days):
        if not series:
            return []

        series.sort(key=lambda x: x[1])
        filled_series, previous_fact = [series[0]], series[0]

        for fact in series[1:]:
            gap_days = (fact[1] - previous_fact[1]).days
            if gap_days >= max_gap_days:
                restore_null_date = previous_fact[1] + dt.timedelta(days=max_gap_days)
                filled_series.append([0, restore_null_date])
            filled_series.append(fact)
            previous_fact = fact

        if (dt.datetime.today() - filled_series[-1][1]).days >= max_gap_days:
            restore_null_date = filled_series[-1][1] + dt.timedelta(days=max_gap_days)
            filled_series.append([0, restore_null_date])

        return filled_series

    @staticmethod
    def build_series_fill_gaps(fact_series_dict):
        return sorted(IndicatorUtils._fill_series_gaps(fact_series_dict, IndicatorUtils.BORDER_MAX_GAP_DAYS),
                      key=lambda x: x[1])

    @staticmethod
    def _merge_series(annuals, quarters):
        merged = quarters.copy()
        quarter_dates = {q[1] for q in quarters}
        for date_str, val in annuals.items():
            date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj not in quarter_dates:
                merged.append([val, date_obj])
        return merged

    @staticmethod
    def build_ltm_with_remove_gaps(quarter_all, annual_all):
        def _annualize_quarters(quarter_all):
            quarter_all = sorted(quarter_all, key=lambda x: x[1])
            annualized_data, values_for_annual, dates_for_annual = [], [], []

            for quarter_value, date_str in quarter_all:
                report_date = dt.datetime.strptime(date_str, '%Y-%m-%d')
                if dates_for_annual and (
                        report_date - dt.datetime.strptime(dates_for_annual[-1],
                                                           '%Y-%m-%d')).days > IndicatorUtils.BORDER_QUARTER_DAYS:
                    values_for_annual, dates_for_annual = [], []

                values_for_annual.append(quarter_value)
                dates_for_annual.append(date_str)

                if len(dates_for_annual) == 4:
                    total_for_quarter = np.nansum(values_for_annual)
                    annualized_data.append([total_for_quarter, report_date])
                    values_for_annual.pop(0)
                    dates_for_annual.pop(0)

            return annualized_data

        annualized_data = _annualize_quarters(quarter_all)
        merged_data = IndicatorUtils._merge_series(annual_all, annualized_data)
        return sorted(IndicatorUtils._fill_series_gaps(merged_data, IndicatorUtils.BORDER_MAX_GAP_DAYS),
                      key=lambda x: x[1])


class IndicatorBuilder:
    def __init__(self, alias, facts, use_report_date):
        self.facts = facts
        self.alias = alias
        self.use_report_date = use_report_date

        if self.use_report_date:
            self.sort_key = lambda f: (
                f['report_date'], f['period'], f['report_id'], -self.facts.index(f['fact_name']))
            self.group_key = lambda f: f['report_date']
        else:
            self.sort_key = lambda f: (
                f['period'], f['report_date'], f['report_id'], -self.facts.index(f['fact_name']))
            self.group_key = lambda f: f['period'][1]

    def build_series_dict(self, fact_data):
        raise NotImplementedError("Method must be implemented in child class.")


class InstantIndicatorBuilder(IndicatorBuilder):
    def __init__(self, alias, facts, use_report_date, build_strategy=None):
        super().__init__(alias, facts, use_report_date)
        self.build = build_strategy or IndicatorUtils.build_series_fill_gaps

    def build_series_dict(self, fact_data):
        if not fact_data:
            return {}

        fact_data.sort(key=self.sort_key, reverse=True)
        groups = itertools.groupby(fact_data, self.group_key)
        series_ = [
            (f[0]['value'], dt.datetime.strptime(g[0], '%Y-%m-%d'))
            for g in groups
            for f in [IndicatorUtils.get_restored_by_segment(list(g[1]))]
        ]

        restore = self.build(series_)
        return {g[1].strftime('%Y-%m-%d'): g[0] for g in restore}


class PeriodIndicatorBuilder(IndicatorBuilder):
    PERIOD_LENGTHS = {
        'qf': (80, 100),
        'saf': (170, 190),
        'af': (355, 375)
    }

    def __init__(self, alias, facts, use_report_date, periods, build_ltm_strategy=None):
        super().__init__(alias, facts, use_report_date)
        self.periods = periods
        self.build_ltm = build_ltm_strategy or IndicatorUtils.build_ltm_with_remove_gaps

    def build_series_dict(self, fact_data):
        fact_data.sort(key=self.sort_key, reverse=True)

        if self.periods == 'ltm':
            return self._process_ltm_data(fact_data)
        elif self.periods == 'qf':
            return self._get_quarters_dict(fact_data)
        elif self.periods == 'af':
            return self._get_annual_dict(fact_data)

    def _process_ltm_data(self, fact_data):
        quarter_data = self._build_series_qf(fact_data)
        annual_data = self._get_annual_dict(fact_data)
        result = self.build_ltm(quarter_data, annual_data)
        return {item[1].date().isoformat(): item[0] for item in reversed(result)}

    def _get_annual_dict(self, fact_data):
        annual = [f for f in fact_data if 340 < f['period_length'] < 380]
        groups = itertools.groupby(annual, self.group_key)
        return {g[0]: next(g[1])['value'] for g in groups}

    def _get_quarters_dict(self, fact_data):
        q_values = self._build_series_qf(fact_data)
        return {q[1]: q[0] for q in reversed(q_values)}

    def _build_series_qf(self, fact_data):
        def is_valid_fact(f):
            return f['value'] is not None and f['period_length'] is not None and f['report_type'] in ['10-Q', '10-Q/A',
                                                                                                      '10-K', '10-K/A']

        def get_recovered_q_value(all_facts, first_k_date, k_fact_value):
            previous_3q = previous_3_quarters(all_facts, first_k_date, k_fact_value)
            return k_fact_value - previous_3q

        reports_all = sorted(fact_data, key=self.sort_key)
        q_value_date_all = []
        all_facts_for_recovered_q_values = []
        groups = itertools.groupby(reports_all, self.group_key)

        for report_date, group in groups:
            q_indices = []
            k_indices = []

            facts_in_report = list(group)
            filtered_facts_in_report = IndicatorUtils.get_restored_by_segment(facts_in_report)

            for i, f in enumerate(filtered_facts_in_report):
                if f['value'] is not None:
                    all_facts_for_recovered_q_values.append([f['period'], f['value']])
                if is_valid_fact(f):
                    if 75 < f['period_length'] < 120:
                        q_indices.append(i)
                    elif 340 < f['period_length'] < 380:
                        k_indices.append(i)

            is_Q_report_exist = bool(q_indices)
            is_K_report_exist = bool(k_indices)

            if is_Q_report_exist and not is_K_report_exist:
                q_value_date_all.append([filtered_facts_in_report[q_indices[-1]]['value'], report_date])
                continue

            if is_K_report_exist:
                first_k_date = dt.datetime.strptime(filtered_facts_in_report[k_indices[-1]]['period'][0], '%Y-%m-%d')
                k_value = filtered_facts_in_report[k_indices[-1]]['value']

                if k_value is None:
                    q_value_date_all.append([np.nan, report_date])
                    continue

                if not q_value_date_all:
                    q_value_date_all.append([k_value / 4, report_date])
                    continue

                if not is_Q_report_exist or (is_Q_report_exist and is_K_report_exist):
                    previous_report_date = dt.datetime.strptime(q_value_date_all[-1][1], '%Y-%m-%d')
                    days_diff = (dt.datetime.strptime(report_date, '%Y-%m-%d') - previous_report_date).days
                    is_one_year_gap_in_reports = days_diff > 360

                    if is_one_year_gap_in_reports or not is_Q_report_exist:
                        recovered_q_value = get_recovered_q_value(all_facts_for_recovered_q_values, first_k_date,
                                                                  k_value)
                    else:
                        q_fact = filtered_facts_in_report[q_indices[-1]]
                        last_q_date = dt.datetime.strptime(q_fact['period'][1], '%Y-%m-%d')
                        last_k_date = dt.datetime.strptime(filtered_facts_in_report[k_indices[-1]]['period'][1],
                                                           '%Y-%m-%d')

                        if (last_k_date - dt.timedelta(days=5)) <= last_q_date <= (last_k_date + dt.timedelta(days=5)):
                            recovered_q_value = q_fact['value']
                        else:
                            recovered_q_value = get_recovered_q_value(all_facts_for_recovered_q_values, first_k_date,
                                                                      k_value)

                    q_value_date_all.append([recovered_q_value, report_date])
                    continue

            q_value_date_all.append([np.nan, report_date])

        return q_value_date_all


def previous_3_quarters(full_list, start_time, val):
    def calculate_average(val, *args):
        """
        Calculate average of val minus each value in args.
        """
        total = val - sum(args)
        return total / len(args)

    def find_quarter_index(full_list, start_time, offset, dist_bounds):
        """
        Find the index of the quarter with the specified offset and distance bounds.
        """
        for info in full_list:
            left_bound = dt.datetime.strptime(info[0][0], '%Y-%m-%d')
            right_bound = dt.datetime.strptime(info[0][1], '%Y-%m-%d')
            dist = (right_bound - left_bound).days

            if (left_bound - dt.timedelta(days=offset[0])) < start_time < (left_bound + dt.timedelta(days=offset[1])) \
                    and dist_bounds[0] < dist < dist_bounds[1]:
                return info[1]

        return 0

    q1 = find_quarter_index(full_list, start_time, (10, 10), (80, 120))
    q12 = find_quarter_index(full_list, start_time, (10, 10), (150, 200))
    q123 = find_quarter_index(full_list, start_time, (10, 10), (250, 290))

    q2 = find_quarter_index(full_list, start_time, (110, 70), (80, 120))
    q23 = find_quarter_index(full_list, start_time, (110, 70), (150, 200))

    q3 = find_quarter_index(full_list, start_time, (210, 150), (80, 120))

    if q123: return q123
    if q1: return calculate_average(val, q1, q2, q3)
    if q12: return calculate_average(val, q12, q3)
    if q2: return calculate_average(val, q2, q3)
    if q23: return calculate_average(val, q23)
    if q3: return calculate_average(val, q3)

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


def append_period_variants(builder_list, s):
    return builder_list + [
        PeriodIndicatorBuilder(s.alias + '_ltm', s.facts, s.use_report_date, 'ltm'),
        PeriodIndicatorBuilder(s.alias + '_af', s.facts, s.use_report_date, 'af'),
        PeriodIndicatorBuilder(s.alias + '_qf', s.facts, s.use_report_date, 'qf')
    ]


additional_builders = []
for s in standard_indicator_builders:
    if isinstance(s, PeriodIndicatorBuilder):
        additional_builders.extend(append_period_variants([], s))

standard_indicator_builders.extend(additional_builders)

if __name__ == '__main__':
    import qnt.data.stocks as qds
    import qnt.data.common as qdc

    qdc.BASE_URL = 'http://127.0.0.1:8001/'

    assets = qds.load_assets()
    data = qds.load_data(assets)

    indicators = secgov_load_indicators(assets, data.time)
    log_info(indicators)
