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
                                     'primaryKey': ['time']}}, json.loads(stat_tail))

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
                                     'primaryKey': ['time']}}, json.loads(stat_tail))


if __name__ == '__main__':
    unittest.main()
