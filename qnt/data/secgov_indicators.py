from qnt.data.common import *
from qnt.data.secgov import load_facts
import itertools
import pandas as pd
import datetime as dt


def load_indicators(
        assets,
        time_coord,
        standard_indicators=None,
        builders = None,
        start_date_offset = datetime.timedelta(days=365*2),
        fill_strategy=lambda xarr: xarr.ffill('time')
):

    cik2id = dict((a['cik'], a['id']) for a in assets if a.get('cik') is not None)
    min_date = pd.Timestamp(time_coord.min().values).to_pydatetime().date() - parse_tail(start_date_offset)
    max_date = pd.Timestamp(time_coord.max().values).to_pydatetime().date()
    indicator_dicts = load_indicator_dicts(list(cik2id.keys()),  standard_indicators, builders, min_date, max_date)

    dfs = []

    for (cik, inds) in indicator_dicts:
        series = [pd.Series(v if len(v) > 0 else {min_date.isoformat(): np.nan}, dtype=np.float64, name=k)
                  for (k,v) in inds.items()]
        df = pd.concat(series, axis=1)
        df.index = df.index.astype(dtype=time_coord.dtype,copy=False)
        df = df.unstack().to_xarray().rename({'level_0':'field', 'level_1': 'time'})
        df.name = cik2id[cik]
        dfs.append(df)

    if len(dfs) is 0:
        return None # TODO

    idc_arr = xr.concat(dfs, pd.Index([d.name for d in dfs], name='asset'))

    idc_arr = xr.align(idc_arr, time_coord, join='outer')[0]
    idc_arr = idc_arr.sel(time = np.sort(idc_arr.time.values))
    idc_arr = fill_strategy(idc_arr)
    idc_arr = idc_arr.sel(time=time_coord)

    idc_arr.name = "secgov_indicators"
    return idc_arr


secgov_load_indicators = deprecated_wrap(load_indicators)


def load_indicator_dicts(ciks, standard_indicators=None, builders=None, min_date=None, max_date=None, tail=DEFAULT_TAIL):
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
                    columns=['cik', 'report_id', 'report_type', 'report_date', 'fact_name', 'period', 'period_length'],
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
        if(self.use_report_date):
            self.sort_key = lambda f: (f['report_date'], f['period'], f['report_id'], -self.facts.index(f['fact_name']))
        else:
            self.sort_key = lambda f: (f['period'], f['report_date'], f['report_id'], -self.facts.index(f['fact_name']))

    def build_series_dict(self, fact_data):
        pass


class InstantIndicatorBuilder(IndicatorBuilder):

    def __init__(self, alias, facts, use_report_date):
        super().__init__(alias, facts, use_report_date)
        self.group_key=(lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'])

    def build_series_dict(self, fact_data):
        fact_data = sorted(fact_data, key=self.sort_key, reverse=True)
        groups = itertools.groupby(fact_data,self.group_key)
        return dict((g[0],  next(g[1])['value']) for g in groups)


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
        self.group_key=(lambda f: f['report_date']) if self.use_report_date else (lambda f: f['period'][1])

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

        groups = itertools.groupby(fact_data,self.group_key)
        return dict((g[0] ,  next(g[1])['value']) for g in groups)


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
            result = self.build_series_qf(fact_data)
            return dict((item[1] ,  item[0]) for item in reversed(result))
        elif self.periods == 'af':
            fact_data = [f for f in fact_data if 340 < f['period_length'] < 380]
            groups = itertools.groupby(fact_data,self.group_key)
            return dict((g[0] ,  next(g[1])['value']) for g in groups)

    def build_series_qf(self, fact_data):
        # from the earliest reports to new ones
        fact_data = sorted(fact_data, key=self.sort_key)
        result = []
        all_info = []

        # For each report...
        groups = itertools.groupby(fact_data,self.group_key)
        for g in groups:

            local_facts = list(g[1])

            #identify the report type
            Q_report = False
            K_report = False

            q_indexis = []
            k_indexis = []

            # Form all info list and find indexis for quarter and annual facts
            for i, f in enumerate(local_facts):
                if f['value'] is not None:
                    all_info.append([f['period'],f['value']])

                if f['period_length'] is not None:
                    if (75 < f['period_length'] < 120): q_indexis.append(i)
                    if (340 < f['period_length'] < 380): k_indexis.append(i)

                if f['report_type'] in ['10-Q','10-Q/A']: Q_report = True
                if f['report_type'] in ['10-K','10-K/A']: K_report = True

            # Quarter info only
            if Q_report and (len(q_indexis) > 0) and not K_report:
                result.append([local_facts[q_indexis[-1]]['value'],g[0]])

            # Annual report but all periods are quarters
            elif K_report and (len(k_indexis)) == 0 and (len(q_indexis) > 0) and not Q_report:
                result.append([local_facts[q_indexis[-1]]['value'],g[0]])

            # Both reports at the same report date - take the most actual info
            elif Q_report and K_report and (len(k_indexis)) > 0 and (len(q_indexis) > 0):
                last_k_date = dt.datetime.strptime(local_facts[k_indexis[-1]]['period'][1], '%Y-%m-%d')
                last_q_date = dt.datetime.strptime(local_facts[q_indexis[-1]]['period'][1], '%Y-%m-%d')
                if last_q_date > last_k_date:
                    result.append([local_facts[q_indexis[-1]]['value'],g[0]])
                else:
                    result.append([local_facts[k_indexis[-1]]['value'],g[0]])

            # Mixed info
            elif K_report and (len(k_indexis)) > 0 and (len(q_indexis) > 0) and not Q_report:
                last_q_date = dt.datetime.strptime(local_facts[q_indexis[-1]]['period'][1], '%Y-%m-%d')

                last_k_date = dt.datetime.strptime(local_facts[k_indexis[-1]]['period'][1], '%Y-%m-%d')
                first_k_date = dt.datetime.strptime(local_facts[k_indexis[-1]]['period'][0], '%Y-%m-%d')

                # I may contains 4th quarter info separately
                if (last_k_date - dt.timedelta(days = 5)) < last_q_date < (last_k_date + dt.timedelta(days = 5)):
                    result.append([local_facts[q_indexis[-1]]['value'],g[0]])
                # If not, one can exctract it from other periods
                else:
                    local_value = local_facts[k_indexis[-1]]['value']
                    if local_value is None:
                        temp = np.nan
                    else:
                        temp = previous_3_quarters(all_info, first_k_date, local_facts[k_indexis[-1]]['value'])
                    result.append([temp,g[0]])

            # Annual info only
            elif K_report and (len(k_indexis)) > 0 and (len(q_indexis) == 0) and not Q_report:
                first_k_date = dt.datetime.strptime(local_facts[k_indexis[-1]]['period'][0], '%Y-%m-%d')
                local_value = local_facts[k_indexis[-1]]['value']

                if local_value is None:
                    temp = np.nan
                else:
                    temp = previous_3_quarters(all_info, first_k_date, local_facts[k_indexis[-1]]['value'])
                result.append([temp,g[0]])

            #All other cases
            elif (K_report or Q_report) and len(local_facts) > 0:
                if local_facts[-1]['value'] is not None \
                        and local_facts[-1]['period_length'] is not None \
                        and local_facts[-1]['period_length'] > 0:
                    temp = local_facts[-1]['value']/local_facts[-1]['period_length']*90
                    result.append([temp,g[0]])

            #We have tried
            else:
                result.append([np.nan,g[0]])

        return result

    def build_ltm(self, fact_data):
        # averaging period
        avg_time_frame = 360

        sort_type = lambda f: (f[1])
        data_list = self.build_series_qf(fact_data)

        #check data
        if len(data_list) == 0:
            return []

        annual_value_list = []
        annual_date_list = []
        result = []

        #sort data
        data_list = sorted(data_list,key=sort_type)

        # the day we stop ltm
        end_date = dt.datetime.strptime(data_list[-1][1], '%Y-%m-%d')

        # add new events to a data: end of info shelf life
        add_list = []
        for item in data_list:
            loop_date = dt.datetime.strptime(item[1], '%Y-%m-%d') + dt.timedelta(days = 365)
            add_list.append([0,loop_date.strftime('%Y-%m-%d')])

        data_list = data_list + add_list
        data_list = sorted(data_list,key=sort_type)

        # for event in data list..
        for item in data_list:
            loop_date = dt.datetime.strptime(item[1], '%Y-%m-%d')

            if (len(annual_value_list)==0) or (len(annual_date_list)==0):
                start_date = loop_date
                annual_value_list = []
                annual_date_list = []

            dist = (loop_date - start_date).days

            # In case of weak data, we will not create synthetic one
            if (end_date - loop_date).days < 0: break

            # If less than year -> take into account new data
            if dist < avg_time_frame:
                if (item[0] is not None):
                    if (item[0] != 0) and (~np.isnan(item[0])):
                        annual_value_list.append(item[0])
                        annual_date_list.append(loop_date)

            # otherwise -> save result and drop it
            else:
                # Company might have a lot of reports. We might have some overlaps.
                # But there is only 4 quarters per year anyway
                local_value = np.nansum(annual_value_list)/len(annual_value_list)*4
                result.append([local_value,loop_date])

                if (item[0] is not None):
                    if (item[0] != 0) and (~np.isnan(item[0])):
                        annual_value_list.append(item[0])
                        annual_date_list.append(loop_date)

                annual_value_list.pop(0)
                annual_date_list.pop(0)
                if len(annual_date_list) > 0:
                    start_date = annual_date_list[0]
                else:
                    start_date = 0
        return result


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

        left_index1 = (left_bound - dt.timedelta(days = 10)) < start_time < (left_bound + dt.timedelta(days = 10))
        left_index2 = (left_bound - dt.timedelta(days = 110)) < start_time < (left_bound - dt.timedelta(days = 70))
        left_index3 = (left_bound - dt.timedelta(days = 210)) < start_time < (left_bound - dt.timedelta(days = 150))

        if left_index1:
            dist = (right_bound - left_bound).days

            if 80< dist< 120:
                local_index.extend([info[1], '1']) # first quarter
            elif 150 < dist< 200 :
                local_index.extend([info[1], '12']) # first and second quarters
            elif 250 < dist< 290 :
                local_index.extend([info[1], '123']) # first, second and third quarters -> exit
                return info[1]

        if left_index2:

            dist = (right_bound - left_bound).days
            if 80< dist< 100:
                local_index.extend([info[1], '2']) #second quarter
            elif 150 < dist< 200 :
                local_index.extend([info[1], '23']) #second and third quarters

        if left_index3:
            dist = (right_bound - left_bound).days
            if 80< dist< 120: local_index.extend([info[1], '3']) # third quarter

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
                    return (val-local_index[ind1]-local_index[ind2]-local_index[ind3])

            if '23' in local_index:
                return (val-local_index[ind1]-local_index[ind23])

            return (val-local_index[ind1])/3

        elif '12' in local_index:
            if '3' in local_index:
                return (val-local_index[ind12]-local_index[ind3])
            else:
                return (val-local_index[ind12])/2

        elif '2' in local_index:
            if '3' in local_index:
                return (val-local_index[ind2]-local_index[ind3])/2
            else:
                return (val-local_index[ind2])/3

        elif '23' in local_index:
            return (val-local_index[ind23])/2

        elif '3' in local_index:
            return (val-local_index[ind3])/3
    else:
        return val/4


standard_indicator_builders = [
    #instant
    InstantIndicatorBuilder('assets' , ['us-gaap:Assets'], True),
    InstantIndicatorBuilder('assets_curr', ['us-gaap:AssetsCurrent'], True),
    InstantIndicatorBuilder('equity' , ['us-gaap:StockholdersEquity'], True),
    InstantIndicatorBuilder('liabilities', ['us-gaap:Liabilities'], True),
    InstantIndicatorBuilder('liabilities_curr', ['us-gaap:LiabilitiesCurrent'], True),
    InstantIndicatorBuilder('market_cap', ['us-gaap:CapitalizationLongtermDebtAndEquity'], True),
    InstantIndicatorBuilder('debt_lt', ['us-gaap:LongTermDebt'], True),
    InstantIndicatorBuilder('debt_st', ['us-gaap:ShortTermBorrowings'], True),
    InstantIndicatorBuilder('goodwill', ['us-gaap:Goodwill'], True),
    InstantIndicatorBuilder('inventory', ['us-gaap:InventoryNet'], True),
    InstantIndicatorBuilder('ivestment_short_term', ['us-gaap:AvailableForSaleSecuritiesCurrent', 'us-gaap:MarketableSecuritiesCurrent'], True),
    InstantIndicatorBuilder('invested_capital', ['us-gaap:MarketableSecurities'], True),
    InstantIndicatorBuilder('shares', ['us-gaap:CommonStockSharesOutstanding', 'us-gaap:CommonStockSharesIssued'], True),
    InstantIndicatorBuilder('ppent', ['us-gaap:PropertyPlantAndEquipmentNet'], True),
    InstantIndicatorBuilder('cash_equivalent', ['us-gaap:CashAndCashEquivalentsAtCarryingValue'], True),
    #period
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
    print(indicators)
