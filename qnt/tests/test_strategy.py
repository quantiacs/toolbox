import sys
import unittest
import pandas as pd
import xarray as xr
import numpy as np

import json
import os

os.environ['API_KEY'] = "default"

import xarray as xr
import qnt.output as qnout
import qnt.ta as qnta
import qnt.stats as qnstats


def calculate_weights(data):
    def multi_trix_v3(data, params):
        s_ = qnta.trix(data.sel(field='high'), params[0])
        w_1 = s_.shift(time=params[1]) > s_.shift(time=params[2])
        w_2 = s_.shift(time=params[3]) > s_.shift(time=params[4])
        weights = (w_1 * w_2) * data.sel(field="is_liquid")
        return weights.fillna(0)

    def multi_ema_v3(data, params):
        s_ = qnta.ema(data.sel(field='high'), params[0])
        w_1 = s_.shift(time=params[1]) > s_.shift(time=params[2])
        w_2 = s_.shift(time=params[3]) > s_.shift(time=params[4])
        weights = (w_1 * w_2) * data.sel(field="is_liquid")
        return weights.fillna(0)

    def multi_ema_v4(data, params):
        s_ = qnta.trix(data.sel(field='high'), 30)
        w_1 = s_.shift(time=params[0]) > s_.shift(time=params[1])
        s_ = qnta.ema(data.sel(field='high'), params[2])
        w_2 = s_.shift(time=params[3]) > s_.shift(time=params[4])
        weights = (w_1 * w_2) * data.sel(field="is_liquid")
        return weights.fillna(0)

    def get_enough_bid_for(weights_):
        time_traded = weights_.time[abs(weights_).fillna(0).sum('asset') > 0]
        is_strategy_traded = len(time_traded)
        if is_strategy_traded:
            return xr.where(weights_.time < time_traded.min(), data.sel(field="is_liquid"), weights_)
        return weights_

    weights_1 = multi_trix_v3(data, [87, 135, 108, 13, 114])
    weights_2 = multi_trix_v3(data, [89, 8, 101, 148, 36])
    weights_3 = multi_trix_v3(data, [196, 125, 76, 12, 192])
    weights_4 = multi_ema_v3(data, [69, 47, 57, 7, 41])

    weights_f = (weights_1 + weights_2) * weights_3 * weights_4

    weights_5 = multi_trix_v3(data, [89, 139, 22, 8, 112])
    weights_6 = multi_trix_v3(data, [92, 139, 20, 10, 110])
    weights_7 = multi_ema_v4(data, [13, 134, 42, 66, 133])

    weights_t = (weights_5 + weights_6) * weights_7 + weights_3

    weights_all = 4 * weights_f + weights_t

    weights_new = get_enough_bid_for(weights_all)
    weights_new = weights_new.sel(time=slice("2006-01-01", None))

    return qnout.clean(output=weights_new, data=data, kind="stocks_nasdaq100")


def calculate_weights_ta(data):
    fields = data.sel(field='high')
    k, d = qnta.stochastic(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 30)
    macd_line, signal_line, hist = qnta.macd(fields, 30)
    indicators = [
        qnta.sma(fields, 30),
        qnta.ema(fields, 30),
        qnta.dema(fields, 30),
        qnta.rsi(fields, 30),
        qnta.roc(fields, 30),
        qnta.trix(fields, 30),
        qnta.atr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 30),
        qnta.wma(fields, 30),
        k, d, macd_line, signal_line, hist
    ]

    total_sum = sum(indicators)

    return total_sum.fillna(0)


def load_data_and_create_data_array(filename, dims, transpose_order):
    ds = xr.open_dataset(filename).load()
    dataset_name = list(ds.data_vars)[0]
    values = ds[dataset_name].transpose(*transpose_order).values
    coords = {dim: ds[dim].values for dim in dims}
    return xr.DataArray(values, dims=dims, coords=coords)


schema_global = {'fields': [{'name': 'time', 'type': 'datetime'},
                            {'name': 'equity', 'type': 'number'},
                            {'name': 'relative_return', 'type': 'number'},
                            {'name': 'volatility', 'type': 'number'},
                            {'name': 'underwater', 'type': 'number'},
                            {'name': 'max_drawdown', 'type': 'number'},
                            {'name': 'sharpe_ratio', 'type': 'number'},
                            {'name': 'mean_return', 'type': 'number'},
                            {'name': 'bias', 'type': 'number'},
                            {'name': 'instruments', 'type': 'number'},
                            {'name': 'avg_turnover', 'type': 'number'},
                            {'name': 'avg_holding_time', 'type': 'number'}],
                 'pandas_version': '1.4.0',
                 'primaryKey': ['time']}


class TestBaseStatistic(unittest.TestCase):

    def test_statistic(self):
        dir = os.path.abspath(os.curdir)

        dims = ['field', 'time', 'asset']
        data = load_data_and_create_data_array(f"{dir}/data/data_2005-01-01.nc", dims, dims)

        weights = calculate_weights(data)

        stat = qnstats.calc_stat(data.sel(time=slice("2006-01-01", None)), weights)
        import qnt.graph as qngraph
        qngraph.make_major_plots(stat)  # works in jupyter
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0959920818,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 89.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2006-01-20T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0908065015,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 89.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2006-01-23T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0856038356,
                                    'bias': 1.0,
                                    'equity': 1.003131434,
                                    'instruments': 89.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0539298509,
                                    'relative_return': 0.003131434,
                                    'sharpe_ratio': 4.3492355839,
                                    'time': '2006-01-24T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0123998459},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0813008815,
                                    'bias': 1.0,
                                    'equity': 0.9973223154,
                                    'instruments': 89.0,
                                    'max_drawdown': -0.0057909845,
                                    'mean_return': -0.0413508275,
                                    'relative_return': -0.0057909845,
                                    'sharpe_ratio': -1.5908117877,
                                    'time': '2006-01-25T00:00:00.000',
                                    'underwater': -0.0057909845,
                                    'volatility': 0.0259935386},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0776666778,
                                    'bias': 1.0,
                                    'equity': 1.0122943845,
                                    'instruments': 89.0,
                                    'max_drawdown': -0.0057909845,
                                    'mean_return': 0.1985769359,
                                    'relative_return': 0.0150122672,
                                    'sharpe_ratio': 3.2002864614,
                                    'time': '2006-01-26T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0620497378},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0744561837,
                                    'bias': 1.0,
                                    'equity': 1.0236275706,
                                    'instruments': 89.0,
                                    'max_drawdown': -0.0057909845,
                                    'mean_return': 0.3867163833,
                                    'relative_return': 0.0111955438,
                                    'sharpe_ratio': 5.4228787842,
                                    'time': '2006-01-27T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0713120095},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0714235235,
                                    'bias': 1.0,
                                    'equity': 1.0229479593,
                                    'instruments': 89.0,
                                    'max_drawdown': -0.0057909845,
                                    'mean_return': 0.351104874,
                                    'relative_return': -0.0006639244,
                                    'sharpe_ratio': 5.032955633,
                                    'time': '2006-01-30T00:00:00.000',
                                    'underwater': -0.0006639244,
                                    'volatility': 0.0697611701},
                                   {'avg_holding_time': 1.0,
                                    'avg_turnover': 0.0685437159,
                                    'bias': 1.0,
                                    'equity': 1.0230233343,
                                    'instruments': 91.0,
                                    'max_drawdown': -0.0057909845,
                                    'mean_return': 0.3321643283,
                                    'relative_return': 7.36841e-05,
                                    'sharpe_ratio': 4.8770788374,
                                    'time': '2006-01-31T00:00:00.000',
                                    'underwater': -0.0005902892,
                                    'volatility': 0.0681072296}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'equity', 'type': 'number'},
                                                {'name': 'relative_return', 'type': 'number'},
                                                {'name': 'volatility', 'type': 'number'},
                                                {'name': 'underwater', 'type': 'number'},
                                                {'name': 'max_drawdown', 'type': 'number'},
                                                {'name': 'sharpe_ratio', 'type': 'number'},
                                                {'name': 'mean_return', 'type': 'number'},
                                                {'name': 'bias', 'type': 'number'},
                                                {'name': 'instruments', 'type': 'number'},
                                                {'name': 'avg_turnover', 'type': 'number'},
                                                {'name': 'avg_holding_time', 'type': 'number'}],
                                     'pandas_version': '1.4.0',
                                     'primaryKey': ['time']}}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': 11.4334672231,
                                    'avg_turnover': 0.1709156431,
                                    'bias': 1.0,
                                    'equity': 60.1058753093,
                                    'instruments': 214.0,
                                    'max_drawdown': -0.3918111995,
                                    'mean_return': 0.2589817235,
                                    'relative_return': -0.0132098125,
                                    'sharpe_ratio': 1.0512328839,
                                    'time': '2023-10-20T00:00:00.000',
                                    'underwater': -0.1747530725,
                                    'volatility': 0.2463599907},
                                   {'avg_holding_time': 11.4270481715,
                                    'avg_turnover': 0.1711531269,
                                    'bias': 1.0,
                                    'equity': 59.8281280711,
                                    'instruments': 214.0,
                                    'max_drawdown': -0.3918111995,
                                    'mean_return': 0.2585893223,
                                    'relative_return': -0.0046209665,
                                    'sharpe_ratio': 1.0497416282,
                                    'time': '2023-10-23T00:00:00.000',
                                    'underwater': -0.1785665109,
                                    'volatility': 0.2463361606},
                                   {'avg_holding_time': 11.4260866278,
                                    'avg_turnover': 0.1711702456,
                                    'bias': 1.0,
                                    'equity': 59.6803868318,
                                    'instruments': 214.0,
                                    'max_drawdown': -0.3918111995,
                                    'mean_return': 0.2583499052,
                                    'relative_return': -0.0024694277,
                                    'sharpe_ratio': 1.0488807224,
                                    'time': '2023-10-24T00:00:00.000',
                                    'underwater': -0.1805949816,
                                    'volatility': 0.2463100901},
                                   {'avg_holding_time': 11.4260866278,
                                    'avg_turnover': 0.1711329854,
                                    'bias': 1.0,
                                    'equity': 58.8273082956,
                                    'instruments': 214.0,
                                    'max_drawdown': -0.3918111995,
                                    'mean_return': 0.257267962,
                                    'relative_return': -0.0142941187,
                                    'sharpe_ratio': 1.0444909681,
                                    'time': '2023-10-25T00:00:00.000',
                                    'underwater': -0.1923076542,
                                    'volatility': 0.2463094175},
                                   {'avg_holding_time': 11.4052568064,
                                    'avg_turnover': 0.1711003015,
                                    'bias': 1.0,
                                    'equity': 59.7372670849,
                                    'instruments': 214.0,
                                    'max_drawdown': -0.3918111995,
                                    'mean_return': 0.2582883243,
                                    'relative_return': 0.0154683057,
                                    'sharpe_ratio': 1.0486492544,
                                    'time': '2023-10-26T00:00:00.000',
                                    'underwater': -0.1798140221,
                                    'volatility': 0.246305734}],
                          'schema': schema_global}, json.loads(stat_tail))

    def test_ta(self):
        dir = os.path.abspath(os.curdir)

        dims = ['field', 'time', 'asset']
        data = load_data_and_create_data_array(f"{dir}/data/data_2005-01-01.nc", dims, dims)

        weights = calculate_weights_ta(data)

        stat = qnstats.calc_stat(data.sel(time=slice("2006-01-01", None)), weights)
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': 5.403206699,
                                    'avg_turnover': 0.1172396991,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 177.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2006-01-20T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 5.8232899207,
                                    'avg_turnover': 0.1127922284,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 177.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2006-01-23T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 6.2022043097,
                                    'avg_turnover': 0.1079194804,
                                    'bias': 1.0,
                                    'equity': 1.0057016235,
                                    'instruments': 177.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.100225586,
                                    'relative_return': 0.0057016235,
                                    'sharpe_ratio': 4.4392237851,
                                    'time': '2006-01-24T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0225772772},
                                   {'avg_holding_time': 6.6564639415,
                                    'avg_turnover': 0.1042517229,
                                    'bias': 1.0,
                                    'equity': 1.0016806707,
                                    'instruments': 177.0,
                                    'max_drawdown': -0.0039981569,
                                    'mean_return': 0.0268012049,
                                    'relative_return': -0.0039981569,
                                    'sharpe_ratio': 0.971593733,
                                    'time': '2006-01-25T00:00:00.000',
                                    'underwater': -0.0039981569,
                                    'volatility': 0.0275847857},
                                   {'avg_holding_time': 7.0491094129,
                                    'avg_turnover': 0.101180465,
                                    'bias': 1.0,
                                    'equity': 1.0156166964,
                                    'instruments': 177.0,
                                    'max_drawdown': -0.0039981569,
                                    'mean_return': 0.2582294939,
                                    'relative_return': 0.0139126432,
                                    'sharpe_ratio': 4.4446592837,
                                    'time': '2006-01-26T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0580988277},
                                   {'avg_holding_time': 7.5741173727,
                                    'avg_turnover': 0.0991532498,
                                    'bias': 1.0,
                                    'equity': 1.0256836602,
                                    'instruments': 177.0,
                                    'max_drawdown': -0.0039981569,
                                    'mean_return': 0.4262252403,
                                    'relative_return': 0.0099121685,
                                    'sharpe_ratio': 6.5322979965,
                                    'time': '2006-01-27T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0652488972},
                                   {'avg_holding_time': 7.9011714004,
                                    'avg_turnover': 0.096228215,
                                    'bias': 1.0,
                                    'equity': 1.0272889472,
                                    'instruments': 177.0,
                                    'max_drawdown': -0.0039981569,
                                    'mean_return': 0.4291605029,
                                    'relative_return': 0.0015650897,
                                    'sharpe_ratio': 6.757289561,
                                    'time': '2006-01-30T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0635107463},
                                   {'avg_holding_time': 8.2094209738,
                                    'avg_turnover': 0.0933774664,
                                    'bias': 1.0,
                                    'equity': 1.0293025113,
                                    'instruments': 177.0,
                                    'max_drawdown': -0.0039981569,
                                    'mean_return': 0.4389384182,
                                    'relative_return': 0.0019600757,
                                    'sharpe_ratio': 7.0876319579,
                                    'time': '2006-01-31T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0619301934}],
                          'schema': schema_global}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': 55.3650244403,
                                    'avg_turnover': 0.0344372677,
                                    'bias': 1.0,
                                    'equity': 8.5569523417,
                                    'instruments': 244.0,
                                    'max_drawdown': -0.5457454949,
                                    'mean_return': 0.1282869071,
                                    'relative_return': -0.0125095042,
                                    'sharpe_ratio': 0.5881131398,
                                    'time': '2023-10-20T00:00:00.000',
                                    'underwater': -0.1187989924,
                                    'volatility': 0.2181330401},
                                   {'avg_holding_time': 55.3681497711,
                                    'avg_turnover': 0.0344355257,
                                    'bias': 1.0,
                                    'equity': 8.5603986428,
                                    'instruments': 244.0,
                                    'max_drawdown': -0.5457454949,
                                    'mean_return': 0.1282820678,
                                    'relative_return': 0.0004027487,
                                    'sharpe_ratio': 0.5881565464,
                                    'time': '2023-10-23T00:00:00.000',
                                    'underwater': -0.1184440899,
                                    'volatility': 0.2181087136},
                                   {'avg_holding_time': 55.3700804061,
                                    'avg_turnover': 0.0344321047,
                                    'bias': 1.0,
                                    'equity': 8.6462073614,
                                    'instruments': 244.0,
                                    'max_drawdown': -0.5457454949,
                                    'mean_return': 0.1288843029,
                                    'relative_return': 0.0100239162,
                                    'sharpe_ratio': 0.5909524455,
                                    'time': '2023-10-24T00:00:00.000',
                                    'underwater': -0.1096074474,
                                    'volatility': 0.2180958956},
                                   {'avg_holding_time': 55.380983874,
                                    'avg_turnover': 0.0344312849,
                                    'bias': 1.0,
                                    'equity': 8.4650565279,
                                    'instruments': 244.0,
                                    'max_drawdown': -0.5457454949,
                                    'mean_return': 0.1275115749,
                                    'relative_return': -0.0209514792,
                                    'sharpe_ratio': 0.5845635076,
                                    'time': '2023-10-25T00:00:00.000',
                                    'underwater': -0.1282624884,
                                    'volatility': 0.2181312608},
                                   {'avg_holding_time': 56.5752709478,
                                    'avg_turnover': 0.0344288354,
                                    'bias': 1.0,
                                    'equity': 8.382361628,
                                    'instruments': 244.0,
                                    'max_drawdown': -0.5457454949,
                                    'mean_return': 0.126859813,
                                    'relative_return': -0.009768972,
                                    'sharpe_ratio': 0.5816036997,
                                    'time': '2023-10-26T00:00:00.000',
                                    'underwater': -0.1367784677,
                                    'volatility': 0.2181207119}],
                          'schema': schema_global}, json.loads(stat_tail))

    def test_backtester(self):
        import qnt.data as qndata
        data = qndata.cryptodaily.load_data(min_date="2024-01-01")  # load data

        import qnt.backtester as qnbt
        # import memory_profiler
        # # Measure memory usage before backtesting
        # mem_usage_before = memory_profiler.memory_usage()[0]
        weights = qnbt.backtest(
            competition_type="crypto_daily",
            lookback_period=150,
            start_date="2024-01-01",
            strategy=calculate_weights_ta,
            analyze=False,
            build_plots=False
        )

        # # Measure memory usage after backtesting
        # mem_usage_after = memory_profiler.memory_usage()[0]
        # mem_usage_diff = mem_usage_after - mem_usage_before
        #
        # print(f"Memory usage before backtest: {mem_usage_before:.2f} MiB")
        # print(f"Memory usage after backtest: {mem_usage_after:.2f} MiB")
        # print(f"Memory usage increased by: {mem_usage_diff:.2f} MiB")

        weights_slice = weights.sel(time=slice("2024-01-01", "2024-09-15"))

        stat = qnstats.calc_stat(data.sel(time=slice("2024-01-01", "2024-09-15")), weights_slice)
        stat_df = stat.to_pandas()
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")

        self.assertEqual({'data': [{'avg_holding_time': 6.0689614135,
                                    'avg_turnover': 0.0996294211,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2024-01-13T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 6.2211125895,
                                    'avg_turnover': 0.0927320566,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2024-01-14T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 6.3292917837,
                                    'avg_turnover': 0.0894374411,
                                    'bias': 1.0,
                                    'equity': 1.0166046177,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.4929109373,
                                    'relative_return': 0.0166046177,
                                    'sharpe_ratio': 6.2290294189,
                                    'time': '2024-01-15T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0791312585},
                                   {'avg_holding_time': 6.6182285833,
                                    'avg_turnover': 0.0840899208,
                                    'bias': 1.0,
                                    'equity': 1.0324454462,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 1.0717794302,
                                    'relative_return': 0.0155820937,
                                    'sharpe_ratio': 10.5342355361,
                                    'time': '2024-01-16T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.1017424973},
                                   {'avg_holding_time': 6.9022154585,
                                    'avg_turnover': 0.0806504898,
                                    'bias': 1.0,
                                    'equity': 1.0224868321,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0096456564,
                                    'mean_return': 0.6119702035,
                                    'relative_return': -0.0096456564,
                                    'sharpe_ratio': 5.476103669,
                                    'time': '2024-01-17T00:00:00.000',
                                    'underwater': -0.0096456564,
                                    'volatility': 0.1117528521},
                                   {'avg_holding_time': 7.1041000269,
                                    'avg_turnover': 0.0775257808,
                                    'bias': 1.0,
                                    'equity': 0.9884640627,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0425992323,
                                    'mean_return': -0.209652772,
                                    'relative_return': -0.0332745306,
                                    'sharpe_ratio': -1.1251086417,
                                    'time': '2024-01-18T00:00:00.000',
                                    'underwater': -0.0425992323,
                                    'volatility': 0.1863400246},
                                   {'avg_holding_time': 7.4371379545,
                                    'avg_turnover': 0.0756378181,
                                    'bias': 1.0,
                                    'equity': 0.9964004094,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0425992323,
                                    'mean_return': -0.0669297492,
                                    'relative_return': 0.0080289683,
                                    'sharpe_ratio': -0.3616553663,
                                    'time': '2024-01-19T00:00:00.000',
                                    'underwater': -0.0349122919,
                                    'volatility': 0.1850649967},
                                   {'avg_holding_time': 7.8301638709,
                                    'avg_turnover': 0.0722277612,
                                    'bias': 1.0,
                                    'equity': 0.9970880711,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0425992323,
                                    'mean_return': -0.0518288228,
                                    'relative_return': 0.000690146,
                                    'sharpe_ratio': -0.2872797948,
                                    'time': '2024-01-20T00:00:00.000',
                                    'underwater': -0.0342462405,
                                    'volatility': 0.1804123496}],
                          'schema': schema_global}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': 74.0936092456,
                                    'avg_turnover': 0.0355326819,
                                    'bias': 1.0,
                                    'equity': 1.3443737489,
                                    'instruments': 12.0,
                                    'max_drawdown': -0.2742275581,
                                    'mean_return': 0.5274254894,
                                    'relative_return': -0.0056752981,
                                    'sharpe_ratio': 0.9913687936,
                                    'time': '2024-09-11T00:00:00.000',
                                    'underwater': -0.2288017127,
                                    'volatility': 0.5320174417},
                                   {'avg_holding_time': 74.2601516608,
                                    'avg_turnover': 0.0354690772,
                                    'bias': 1.0,
                                    'equity': 1.3629813322,
                                    'instruments': 12.0,
                                    'max_drawdown': -0.2742275581,
                                    'mean_return': 0.555081621,
                                    'relative_return': 0.013841079,
                                    'sharpe_ratio': 1.0449982469,
                                    'time': '2024-09-12T00:00:00.000',
                                    'underwater': -0.2181274963,
                                    'volatility': 0.5311794758},
                                   {'avg_holding_time': 74.7851021567,
                                    'avg_turnover': 0.0354322956,
                                    'bias': 1.0,
                                    'equity': 1.4187781489,
                                    'instruments': 12.0,
                                    'max_drawdown': -0.2742275581,
                                    'mean_return': 0.6434407215,
                                    'relative_return': 0.0409373301,
                                    'sharpe_ratio': 1.2090059815,
                                    'time': '2024-09-13T00:00:00.000',
                                    'underwater': -0.1861197236,
                                    'volatility': 0.5322064004},
                                   {'avg_holding_time': 75.4096052503,
                                    'avg_turnover': 0.0354924776,
                                    'bias': 1.0,
                                    'equity': 1.4060390349,
                                    'instruments': 12.0,
                                    'max_drawdown': -0.2742275581,
                                    'mean_return': 0.6194820374,
                                    'relative_return': -0.008978933,
                                    'sharpe_ratio': 1.1659157925,
                                    'time': '2024-09-14T00:00:00.000',
                                    'underwater': -0.1934275001,
                                    'volatility': 0.5313265687},
                                   {'avg_holding_time': 226.4227215731,
                                    'avg_turnover': 0.0353834459,
                                    'bias': 1.0,
                                    'equity': 1.3839198074,
                                    'instruments': 12.0,
                                    'max_drawdown': -0.2742275581,
                                    'mean_return': 0.5807488949,
                                    'relative_return': -0.0157315885,
                                    'sharpe_ratio': 1.0943029486,
                                    'time': '2024-09-15T00:00:00.000',
                                    'underwater': -0.2061161667,
                                    'volatility': 0.5307021202}],
                          'schema': schema_global}, json.loads(stat_tail))

    def test_backtester_ml(self):
        import logging
        def create_model():
            from sklearn.linear_model import Ridge
            model = Ridge(random_state=18)
            return model

        def get_features(data):
            close_price = data.sel(field="close").ffill('time').bfill('time').fillna(1)
            close_price_lwma = qnta.lwma(close_price, 2)
            features = xr.concat([close_price_lwma], "feature")
            return features

        def get_target_classes(data):
            price_current = data.sel(field='close')
            price_future = qnta.shift(price_current, -1)

            class_positive = 1  # prices goes up
            class_negative = 0  # price goes down

            target_is_price_up = xr.where(price_future > price_current, class_positive, class_negative)

            return target_is_price_up

        def create_and_train_models(data):
            """Create and train the models working on an asset-by-asset basis."""

            asset_name_all = data.coords['asset'].values

            data = data.sel(time=slice('2013-05-01', None))  # Cut the noisy data head before 2013-05-01

            features_all = get_features(data)
            target_all = get_target_classes(data)

            models = dict()

            for asset_name in asset_name_all:

                # Drop missing values:
                target_cur = target_all.sel(asset=asset_name).dropna('time', how='any')
                features_cur = features_all.sel(asset=asset_name).dropna('time', how='any')

                # Align features and targets:
                target_for_learn_df, feature_for_learn_df = xr.align(target_cur, features_cur, join='inner')

                if len(features_cur.time) < 30:
                    # Not enough points for training
                    continue

                # Transpose features so that 'time' is the first dimension
                feature_for_learn_df = feature_for_learn_df.transpose('time', 'feature')

                # Convert to NumPy arrays
                X = feature_for_learn_df.values
                y = target_for_learn_df.values

                model = create_model()
                model.fit(X, y)
                models[asset_name] = model

            return models

        def predict(models, data):
            """Performs prediction and generates output weights.
               Generation is performed for several days in order to speed
               up the evaluation.
            """

            asset_name_all = data.coords['asset'].values
            weights = xr.zeros_like(data.sel(field='close'))

            # Compute features once outside the loop
            features_all = get_features(data)

            for asset_name in asset_name_all:
                if asset_name in models:
                    model = models[asset_name]

                    # Select and process features for the current asset
                    features_cur = features_all.sel(asset=asset_name).dropna('time', how='any')

                    if len(features_cur.time) < 1:
                        continue

                    try:
                        # Transpose features so that 'time' is the first dimension
                        features_cur = features_cur.transpose('time', 'feature')

                        # Convert to NumPy array with correct shape
                        X = features_cur.values  # Shape: (n_samples, n_features)

                        # Make predictions
                        predictions = model.predict(X)

                        # Assign predictions to the weights DataArray
                        weights.loc[dict(asset=asset_name, time=features_cur.time.values)] = predictions

                    except KeyboardInterrupt as e:
                        raise e
                    except:
                        logging.exception('model prediction failed')

            return weights

        import qnt.data as qndata
        data = qndata.cryptodaily.load_data(min_date="2024-01-01")  # load data

        import qnt.backtester as qnbt
        # import memory_profiler
        # # Measure memory usage before backtesting
        # mem_usage_before = memory_profiler.memory_usage()[0]
        weights = qnbt.backtest_ml(
            train=create_and_train_models,
            predict=predict,
            train_period=30,
            retrain_interval=35,
            retrain_interval_after_submit=1,
            predict_each_day=False,
            competition_type='crypto_daily',
            lookback_period=365,
            start_date='2024-01-01',
            analyze=False,
            build_plots=False
        )

        # # Measure memory usage after backtesting
        # mem_usage_after = memory_profiler.memory_usage()[0]
        # mem_usage_diff = mem_usage_after - mem_usage_before
        #
        # print(f"Memory usage before backtest: {mem_usage_before:.2f} MiB")
        # print(f"Memory usage after backtest: {mem_usage_after:.2f} MiB")
        # print(f"Memory usage increased by: {mem_usage_diff:.2f} MiB")

        weights_slice = weights.sel(time=slice("2024-01-01", "2024-09-15"))

        stat = qnstats.calc_stat(data.sel(time=slice("2024-01-01", "2024-09-15")), weights_slice)
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")

        self.assertEqual({'data': [{'avg_holding_time': 4.1413701278,
                                    'avg_turnover': 0.2156677129,
                                    'bias': 0.7233558842,
                                    'equity': 1.0,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2024-01-13T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 4.4614343823,
                                    'avg_turnover': 0.2087766172,
                                    'bias': 0.8100853746,
                                    'equity': 1.0,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2024-01-14T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': 4.4374581195,
                                    'avg_turnover': 0.2014618396,
                                    'bias': 0.8212211564,
                                    'equity': 1.0054312854,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.151675937,
                                    'relative_return': 0.0054312854,
                                    'sharpe_ratio': 5.675759534,
                                    'time': '2024-01-15T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0267234607},
                                   {'avg_holding_time': 4.4766162194,
                                    'avg_turnover': 0.1907313672,
                                    'bias': 0.7185972788,
                                    'equity': 1.008085582,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.216475822,
                                    'relative_return': 0.0026399582,
                                    'sharpe_ratio': 7.7425744072,
                                    'time': '2024-01-16T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0279591529},
                                   {'avg_holding_time': 4.5559756067,
                                    'avg_turnover': 0.1853304406,
                                    'bias': 0.7389233577,
                                    'equity': 1.0096329371,
                                    'instruments': 10.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.2444575096,
                                    'relative_return': 0.0015349442,
                                    'sharpe_ratio': 8.901969462,
                                    'time': '2024-01-17T00:00:00.000',
                                    'underwater': 0.0,
                                    'volatility': 0.0274610591},
                                   {'avg_holding_time': 4.6058130726,
                                    'avg_turnover': 0.1769334858,
                                    'bias': 0.8431159583,
                                    'equity': 0.9791264553,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0302154186,
                                    'mean_return': -0.3642251766,
                                    'relative_return': -0.0302154186,
                                    'sharpe_ratio': -2.5819770718,
                                    'time': '2024-01-18T00:00:00.000',
                                    'underwater': -0.0302154186,
                                    'volatility': 0.1410644504},
                                   {'avg_holding_time': 4.7880736333,
                                    'avg_turnover': 0.1752919632,
                                    'bias': 0.8555829526,
                                    'equity': 0.9757200667,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0335893067,
                                    'mean_return': -0.3925094534,
                                    'relative_return': -0.0034790078,
                                    'sharpe_ratio': -2.8556855471,
                                    'time': '2024-01-19T00:00:00.000',
                                    'underwater': -0.0335893067,
                                    'volatility': 0.1374484154},
                                   {'avg_holding_time': 4.936379071,
                                    'avg_turnover': 0.1684786726,
                                    'bias': 0.8625655489,
                                    'equity': 0.9935672296,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0335893067,
                                    'mean_return': -0.1165990365,
                                    'relative_return': 0.0182912738,
                                    'sharpe_ratio': -0.7387633663,
                                    'time': '2024-01-20T00:00:00.000',
                                    'underwater': -0.0159124241,
                                    'volatility': 0.1578300195}],
                          'schema': schema_global}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual(
            {'data': [{'avg_holding_time': 22.3440486567,
                       'avg_turnover': 0.0941371228,
                       'bias': 1.0,
                       'equity': 0.7069747542,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.3924357289,
                       'relative_return': -0.0074254203,
                       'sharpe_ratio': -0.7138668736,
                       'time': '2024-09-11T00:00:00.000',
                       'underwater': -0.4358865738,
                       'volatility': 0.5497323708},
                      {'avg_holding_time': 22.3771036119,
                       'avg_turnover': 0.0939443166,
                       'bias': 1.0,
                       'equity': 0.7210114743,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.373873146,
                       'relative_return': 0.0198546272,
                       'sharpe_ratio': -0.6807404773,
                       'time': '2024-09-12T00:00:00.000',
                       'underwater': -0.424686312,
                       'volatility': 0.5492153889},
                      {'avg_holding_time': 22.3888904398,
                       'avg_turnover': 0.0937012321,
                       'bias': 1.0,
                       'equity': 0.7400214102,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.3490150429,
                       'relative_return': 0.0263656496,
                       'sharpe_ratio': -0.6356111551,
                       'time': '2024-09-13T00:00:00.000',
                       'underwater': -0.4095177929,
                       'volatility': 0.5491015067},
                      {'avg_holding_time': 22.4236500237,
                       'avg_turnover': 0.0936374249,
                       'bias': 1.0,
                       'equity': 0.7366375311,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.3521574567,
                       'relative_return': -0.0045726773,
                       'sharpe_ratio': -0.6425633679,
                       'time': '2024-09-14T00:00:00.000',
                       'underwater': -0.4122178775,
                       'volatility': 0.5480509383},
                      {'avg_holding_time': 24.2074270916,
                       'avg_turnover': 0.0933959104,
                       'bias': 1.0,
                       'equity': 0.715314806,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.3774803911,
                       'relative_return': -0.0289460205,
                       'sharpe_ratio': -0.6888212963,
                       'time': '2024-09-15T00:00:00.000',
                       'underwater': -0.4292318309,
                       'volatility': 0.5480091762}],
             'schema': schema_global}, json.loads(stat_tail))

        weights = qnbt.backtest_ml(
            train=create_and_train_models,
            predict=predict,
            train_period=30,
            retrain_interval=35,
            retrain_interval_after_submit=1,
            predict_each_day=True,
            competition_type='crypto_daily',
            lookback_period=365,
            start_date='2024-01-01',
            analyze=False,
            build_plots=False
        )

        weights_slice = weights.sel(time=slice("2024-01-01", "2024-09-15"))

        stat = qnstats.calc_stat(data.sel(time=slice("2024-01-01", "2024-09-15")), weights_slice)
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(2).to_json(orient="table")

        self.assertEqual({'data': [{'avg_holding_time': 4.7880736333,
                                    'avg_turnover': 0.1752919632,
                                    'bias': 0.8555829526,
                                    'equity': 0.9757200667,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0335893067,
                                    'mean_return': -0.3925094534,
                                    'relative_return': -0.0034790078,
                                    'sharpe_ratio': -2.8556855471,
                                    'time': '2024-01-19T00:00:00.000',
                                    'underwater': -0.0335893067,
                                    'volatility': 0.1374484154},
                                   {'avg_holding_time': 4.936379071,
                                    'avg_turnover': 0.1684786726,
                                    'bias': 0.8625655489,
                                    'equity': 0.9935672296,
                                    'instruments': 10.0,
                                    'max_drawdown': -0.0335893067,
                                    'mean_return': -0.1165990365,
                                    'relative_return': 0.0182912738,
                                    'sharpe_ratio': -0.7387633663,
                                    'time': '2024-01-20T00:00:00.000',
                                    'underwater': -0.0159124241,
                                    'volatility': 0.1578300195}],
                          'schema': schema_global}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail(2).to_json(orient="table")
        self.assertEqual(
            {'data': [{'avg_holding_time': 22.4236500237,
                       'avg_turnover': 0.0936374249,
                       'bias': 1.0,
                       'equity': 0.7366375311,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.3521574567,
                       'relative_return': -0.0045726773,
                       'sharpe_ratio': -0.6425633679,
                       'time': '2024-09-14T00:00:00.000',
                       'underwater': -0.4122178775,
                       'volatility': 0.5480509383},
                      {'avg_holding_time': 24.2074270916,
                       'avg_turnover': 0.0933959104,
                       'bias': 1.0,
                       'equity': 0.715314806,
                       'instruments': 12.0,
                       'max_drawdown': -0.4850712428,
                       'mean_return': -0.3774803911,
                       'relative_return': -0.0289460205,
                       'sharpe_ratio': -0.6888212963,
                       'time': '2024-09-15T00:00:00.000',
                       'underwater': -0.4292318309,
                       'volatility': 0.5480091762}],
             'schema': schema_global}, json.loads(stat_tail))


if __name__ == '__main__':
    unittest.main()
