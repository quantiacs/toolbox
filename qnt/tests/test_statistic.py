import sys
import unittest
import pandas as pd
import xarray as xr
import numpy as np

import json
import os

os.environ['API_KEY'] = "default"

import qnt.data as qndata
import qnt.stats as qnstats

dir2_path = os.path.dirname(os.path.abspath(__file__))

dir1_path = os.path.join(dir2_path, 'data')

sys.path.append(dir1_path)

import stats_data

EXPECTED_JSON_SCHEMA = {
    "base": {"fields": [{"name": "time", "type": "datetime"}, {"name": "BTC", "type": "number"}],
             "primaryKey": ["time"], "pandas_version": "0.20.0"},
    "base_values": {"fields": [{"name": "time", "type": "datetime"}, {"name": "values", "type": "number"}],
                    "primaryKey": ["time"], "pandas_version": "0.20.0"},
    "statistic": {'fields': [{'name': 'time', 'type': 'datetime'},
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
                  'pandas_version': '0.20.0',
                  'primaryKey': ['time']},
}

POINTS_PER_YEAR = 365


def calculate_relative_return(data, portfolio_history, per_asset=False):
    return qnstats.calc_relative_return(
        data, portfolio_history, slippage_factor=0.0,
        roll_slippage_factor=0.0, per_asset=per_asset,
        points_per_year=POINTS_PER_YEAR
    )


def generate_stat_head(dataframe, head=20, tail=8):
    stat_head = dataframe.head(head).tail(tail).to_json(orient="table")
    return json.loads(stat_head)


def get_rr_dataframe_and_json(data, portfolio_history, per_asset=False, head=20, tail=8):
    rr = calculate_relative_return(data, portfolio_history, per_asset=per_asset)
    rr_df = rr.to_pandas()
    stat_head = generate_stat_head(rr_df, head=head, tail=tail)
    return rr_df, stat_head


def load_data_and_create_data_array(filename, dims, transpose_order):
    ds = xr.open_dataset(filename).load()
    dataset_name = list(ds.data_vars)[0]
    values = ds[dataset_name].transpose(*transpose_order).values
    coords = {dim: ds[dim].values for dim in dims}
    return xr.DataArray(values, dims=dims, coords=coords)


class TestBaseStatistic(unittest.TestCase):

    def test_relative_return_border_case_per_asset(self):
        dir = os.path.abspath(os.curdir)

        dims = ['time', 'asset']
        weights = load_data_and_create_data_array(f"{dir}/data/weights_2022_02_09_no_short.nc", dims, dims)

        dims = ['field', 'time', 'asset']
        data = load_data_and_create_data_array(f"{dir}/data/data_2022_01_09.nc", dims, dims)

        assets = data.asset.values
        for asset in assets:
            if asset in ['NAS:WBD', 'NAS:XLNX']:
                continue

            asset_data = data.sel(asset=[asset])
            asset_weights = weights.sel(asset=[asset])

            asset_data_pd = asset_data.sel(field='close').to_pandas()
            asset_weights_pd = asset_weights.to_pandas()

            is_liquid = asset_data.sel(field="is_liquid") * asset_weights

            portfolio_history = qnstats.output_normalize(is_liquid, per_asset=False)
            rr_df, stat_head = get_rr_dataframe_and_json(asset_data, portfolio_history)

            is_liquid_slice = is_liquid.sel(time=slice("2022-02-10", None))

            portfolio_history_slice = qnstats.output_normalize(is_liquid_slice, per_asset=False)
            rr_df_slice, stat_head_slice = get_rr_dataframe_and_json(asset_data, portfolio_history_slice,
                                                                     per_asset=False, head=19, tail=8)
            print(asset)
            self.assertEqual(stat_head, stat_head_slice)

    def test_relative_return_border_case(self):
        dir = os.path.abspath(os.curdir)

        dims = ['time', 'asset']
        weights = load_data_and_create_data_array(f"{dir}/data/weights_2022_02_09_no_short.nc", dims, dims)

        dims = ['field', 'time', 'asset']
        data = load_data_and_create_data_array(f"{dir}/data/data_2022_01_09.nc", dims, dims)

        assets = data.asset.values
        # filtered_assets = [asset for asset in assets if asset not in ['NAS:WBD', 'NAS:XLNX']]
        filtered_assets = assets

        data = data.sel(asset=filtered_assets)
        weights = weights.sel(asset=filtered_assets)

        is_liquid = data.sel(field="is_liquid") * weights

        portfolio_history = qnstats.output_normalize(is_liquid, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        is_liquid_slice = is_liquid.sel(time=slice("2022-02-10", None))

        portfolio_history_slice = qnstats.output_normalize(is_liquid_slice, per_asset=False)
        rr_df_slice, stat_head_slice = get_rr_dataframe_and_json(data, portfolio_history_slice,
                                                                 per_asset=False, head=19, tail=8)

        self.assertEqual(stat_head, stat_head_slice)

    def test_relative_return_no_liquid(self):
        data = stats_data.get_base_df()
        is_liquid = data.sel(field="is_liquid")

        portfolio_history = qnstats.output_normalize(is_liquid, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)
        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0625},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0588235294},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        is_liquid.loc[dict(time='2021-02-15T00:00:00.000Z', asset='BTC')] = 0

        portfolio_history = qnstats.output_normalize(is_liquid, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0625},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0588235294},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

    def test_stats_border_case(self):
        data = stats_data.get_xr_with_NaN()
        one = data.sel(field="close") - data.sel(field="close") + 1

        strategy_full = one
        strategy_eth = one.sel(asset=['ETH'])

        statsETH_full = qnstats.calc_stat(data.sel(asset=['ETH']), strategy_full)
        statsETH_simple = qnstats.calc_stat(data.sel(asset=['ETH']), strategy_eth)

        stats_ETH_full_ = statsETH_full.to_pandas().tail(1).to_json(orient="table")
        stats_ETH_simple_ = statsETH_simple.to_pandas().tail(1).to_json(orient="table")
        # This happens because after normalization, the weights for ETH decrease by half.
        self.assertEqual(
            {'data': [{'avg_holding_time': 8.0,
                       'avg_turnover': 0.1054668795,
                       'bias': 0.0,
                       'equity': 1.0232916667,
                       'instruments': 1.0,
                       'max_drawdown': -0.0323101777,
                       'mean_return': 0.5222631434,
                       'relative_return': 0.0,
                       'sharpe_ratio': 2.3784078181,
                       'time': '2021-02-18T00:00:00.000Z',
                       'underwater': -0.0081179321,
                       'volatility': 0.2195851945}],
             'schema': EXPECTED_JSON_SCHEMA['statistic']}, json.loads(stats_ETH_full_))

        self.assertEqual(
            {'data': [{'avg_holding_time': 8.0,
                       'avg_turnover': 0.2093873416,
                       'bias': 0.0,
                       'equity': 1.0465,
                       'instruments': 1.0,
                       'max_drawdown': -0.0626959248,
                       'mean_return': 1.2921392413,
                       'relative_return': 0.0,
                       'sharpe_ratio': 2.9783175293,
                       'time': '2021-02-18T00:00:00.000Z',
                       'underwater': -0.015830721,
                       'volatility': 0.4338487178}],
             'schema': EXPECTED_JSON_SCHEMA['statistic']}, json.loads(stats_ETH_simple_))

    def test_relative_return_NaN(self):
        data = stats_data.get_xr_with_NaN()
        one = data.sel(field="close") - data.sel(field="close") + 1

        one = one.fillna(1)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': -0.0625},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.0}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        portfolio_history = qnstats.output_normalize(one, per_asset=True)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0666666667,
                                    'ETH': 0.0666666667,
                                    'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.1875, 'ETH': 0.1875, 'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0526315789,
                                    'ETH': 0.0526315789,
                                    'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'},
                                                {'name': 'ETH', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0333333333,
                                    'ETH': 0.0333333333,
                                    'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.0977822581,
                                    'ETH': 0.0977822581,
                                    'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0263157895,
                                    'ETH': 0.0263157895,
                                    'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'},
                                                {'name': 'ETH', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

    def test_relative_return_NaN_one(self):
        data_two = stats_data.get_xr_with_NaN()
        data = data_two.sel(asset=['BTC'])
        one = data.sel(field="close") - data.sel(field="close") + 1

        one = one.fillna(1)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': -0.0625},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.0}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0666666667, 'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.1875, 'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0526315789, 'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

    def test_relative_return_complex(self):
        data = stats_data.get_base_df()
        one = data.sel(field="close") - data.sel(field="close") + 1

        data.loc[dict(time='2021-02-15T00:00:00.000Z', asset='BTC', field='open')] = 4

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)
        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0625},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': -0.5709342561},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

    def test_relative_return_one(self):
        data_two = stats_data.get_base_df()
        data = data_two.sel(asset=['BTC'])
        one = data.sel(field="close") - data.sel(field="close") + 1

        one = one.fillna(1)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0625},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0588235294},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        portfolio_history = qnstats.output_normalize(one, per_asset=True)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0666666667, 'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': 0.0625, 'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': 0.0588235294, 'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.0555555556, 'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0526315789, 'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

        data_slice = data.sel(time=slice("2021-01-31", None))
        one_slice = one.sel(time=slice('2021-01-31', None))

        portfolio_history = qnstats.output_normalize(one_slice, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data_slice, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0625},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0588235294},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05},
                                   {'time': '2021-02-19T00:00:00.000Z', 'values': 0.0476190476}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

    def test_relative_return_one_border_case(self):
        data_two = stats_data.get_xr_correct()
        data = data_two.sel(asset=['BTC'])

        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='open')] = 15
        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='close')] = 16
        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='low')] = 15
        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='high')] = 16

        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='open')] = 15
        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='close')] = 16
        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='low')] = 15
        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='high')] = 16
        #
        # {"time": "2021-02-13T00:00:00.000Z", "open": 15, "close": 16, "low": 15, "high": 16, "vol": 1000, "divs": 0,
        #  "split": 0, "split_cumprod": 0, "is_liquid": 1},
        # {"time": "2021-02-14T00:00:00.000Z", "open": 15, "close": 16, "low": 15, "high": 16, "vol": 1000, "divs": 0,
        #  "split": 0, "split_cumprod": 0, "is_liquid": 1},
        # {"time": "2021-02-15T00:00:00.000Z", "open": 15, "close": 16, "low": 15, "high": 16, "vol": 1000, "divs": 0,
        #  "split": 0, "split_cumprod": 0, "is_liquid": 1},
        # {"time": "2021-02-16T00:00:00.000Z", "open": 18, "close": 19, "low": 18, "high": 19, "vol": 1000, "divs": 0,
        #  "split": 0, "split_cumprod": 0, "is_liquid": 1},

        one = data.sel(field="close") - data.sel(field="close") + 1
        one = one.fillna(1)

        portfolio_history = qnstats.output_normalize(one, per_asset=True)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0666666667, 'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': -0.00390625, 'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.1797385621, 'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0526315789, 'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': -0.00390625},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.1797385621},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='open')] = 18
        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='close')] = 19
        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='low')] = 18
        data.loc[dict(time='2021-02-14T00:00:00.000Z', field='high')] = 19

        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='open')] = 18
        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='close')] = 19
        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='low')] = 18
        data.loc[dict(time='2021-02-15T00:00:00.000Z', field='high')] = 19

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.1875},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': -0.0027700831},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        portfolio_history = qnstats.output_normalize(one, per_asset=True)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0666666667, 'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': 0.1875, 'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': -0.0027700831, 'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.0, 'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0526315789, 'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

    def test_relative_return_two(self):
        data = stats_data.get_xr_correct()
        one = data.sel(field="close") - data.sel(field="close") + 1
        one = one.fillna(1)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history)

        self.assertEqual({'data': [{'time': '2021-02-11T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-12T00:00:00.000Z', 'values': 0.0},
                                   {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0666666667},
                                   {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0546875},
                                   {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0588235294},
                                   {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0555555556},
                                   {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0526315789},
                                   {'time': '2021-02-18T00:00:00.000Z', 'values': 0.05}],
                          'schema': EXPECTED_JSON_SCHEMA['base_values']}, stat_head)

        portfolio_history = qnstats.output_normalize(one, per_asset=True)
        rr_df, stat_head = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)

        self.assertEqual({'data': [{'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                                   {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                                   {'BTC': 0.0666666667,
                                    'ETH': 0.0666666667,
                                    'time': '2021-02-13T00:00:00.000Z'},
                                   {'BTC': 0.0546875,
                                    'ETH': 0.0546875,
                                    'time': '2021-02-14T00:00:00.000Z'},
                                   {'BTC': 0.0588235294,
                                    'ETH': 0.0588235294,
                                    'time': '2021-02-15T00:00:00.000Z'},
                                   {'BTC': 0.0555555556,
                                    'ETH': 0.0555555556,
                                    'time': '2021-02-16T00:00:00.000Z'},
                                   {'BTC': 0.0526315789,
                                    'ETH': 0.0526315789,
                                    'time': '2021-02-17T00:00:00.000Z'},
                                   {'BTC': 0.05, 'ETH': 0.05, 'time': '2021-02-18T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'},
                                                {'name': 'ETH', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, stat_head)

    def test_slippage_factor(self):
        def get_slippage_json(data, head):
            slippage = qnstats.calc_slippage(data,
                                             period_days=1,
                                             fract=0.05,
                                             points_per_year=365)
            r = slippage.to_pandas().head(head).to_json(orient="table")
            return json.loads(r)

        data = stats_data.get_base_df()

        r = get_slippage_json(data, 5)

        self.assertEqual({'data': [{'BTC': None, 'time': '2021-01-30T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-01-31T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-01T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-02T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-03T00:00:00.000Z'}],
                          'schema': EXPECTED_JSON_SCHEMA['base']}, r)

        data.loc[dict(time='2021-01-31T00:00:00.000Z', asset='BTC', field='close')] = 4
        data.loc[dict(time='2021-01-31T00:00:00.000Z', asset='BTC', field='high')] = 4

        r = get_slippage_json(data, 3)

        self.assertEqual({'data': [{'BTC': None, 'time': '2021-01-30T00:00:00.000Z'},
                                   {'BTC': 0.1, 'time': '2021-01-31T00:00:00.000Z'},
                                   {'BTC': 0.05, 'time': '2021-02-01T00:00:00.000Z'}],
                          'schema': EXPECTED_JSON_SCHEMA['base']}, r)

    def test_statistic(self):
        data = stats_data.get_base_df()
        buy_and_hold = data.sel(field="close") - data.sel(field="close") + 1

        stat = qnstats.calc_stat(data, buy_and_hold)
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': None,
                                    'avg_turnover': 0.2387085137,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2021-02-11T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.2271524111,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2021-02-12T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.2125164554,
                                    'bias': 1.0,
                                    'equity': 1.0633333333,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 3.4561119146,
                                    'relative_return': 0.0633333333,
                                    'sharpe_ratio': 11.4508113541,
                                    'time': '2021-02-13T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.301822448},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1994798535,
                                    'bias': 1.0,
                                    'equity': 1.12978125,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 15.1783379265,
                                    'relative_return': 0.0624902038,
                                    'sharpe_ratio': 38.1834983226,
                                    'time': '2021-02-14T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.3975104062},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1879499944,
                                    'bias': 1.0,
                                    'equity': 1.19623894,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 45.8594135909,
                                    'relative_return': 0.0588235023,
                                    'sharpe_ratio': 102.2394822568,
                                    'time': '2021-02-15T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.4485489615},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1776803267,
                                    'bias': 1.0,
                                    'equity': 1.2626966588,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 112.2711331786,
                                    'relative_return': 0.0555555555,
                                    'sharpe_ratio': 234.9847755271,
                                    'time': '2021-02-16T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.4777804559},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1684749293,
                                    'bias': 1.0,
                                    'equity': 1.3291543776,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 235.5641731673,
                                    'relative_return': 0.0526315789,
                                    'sharpe_ratio': 476.5440779962,
                                    'time': '2021-02-17T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.4943177012},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1601764962,
                                    'bias': 1.0,
                                    'equity': 1.3956120965,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 437.4865053257,
                                    'relative_return': 0.05,
                                    'sharpe_ratio': 869.9875335492,
                                    'time': '2021-02-18T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5028652578}],
                          'schema': EXPECTED_JSON_SCHEMA['statistic']}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': None,
                                    'avg_turnover': 0.0337705883,
                                    'bias': 1.0,
                                    'equity': 6.4463987315,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 1193.2391460706,
                                    'relative_return': 0.0104166667,
                                    'sharpe_ratio': 4237.1617981522,
                                    'time': '2021-05-05T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.2816128349},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0334235337,
                                    'bias': 1.0,
                                    'equity': 6.5128564504,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 1152.7982974101,
                                    'relative_return': 0.0103092784,
                                    'sharpe_ratio': 4106.228531354,
                                    'time': '2021-05-06T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.2807438233},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0330835398,
                                    'bias': 1.0,
                                    'equity': 6.5793141693,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 1114.0783627083,
                                    'relative_return': 0.0102040816,
                                    'sharpe_ratio': 3980.4004956444,
                                    'time': '2021-05-07T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.279891022},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0327503934,
                                    'bias': 1.0,
                                    'equity': 6.6457718882,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 1076.9956058566,
                                    'relative_return': 0.0101010101,
                                    'sharpe_ratio': 3859.4543277577,
                                    'time': '2021-05-08T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.2790538544},
                                   {'avg_holding_time': 98.0,
                                    'avg_turnover': 0.0324238895,
                                    'bias': 1.0,
                                    'equity': 6.712229607,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 1041.4706461667,
                                    'relative_return': 0.01,
                                    'sharpe_ratio': 3743.1765832705,
                                    'time': '2021-05-09T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.27823177}],
                          'schema': EXPECTED_JSON_SCHEMA['statistic']}, json.loads(stat_tail))

    def test_cryptofutures(self):
        d = stats_data.get_cripto_futures()
        buy_and_hold = d.sel(field="close") - d.sel(field="close") + 1
        stat = qnstats.calc_stat(d, buy_and_hold)
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': None,
                                    'avg_turnover': 0.1073344404,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2015-01-13T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1087649158,
                                    'bias': 1.0,
                                    'equity': 1.0,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 0.0,
                                    'relative_return': 0.0,
                                    'sharpe_ratio': None,
                                    'time': '2015-01-14T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.0},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1216795153,
                                    'bias': 1.0,
                                    'equity': 1.1155921267,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 13.32074858,
                                    'relative_return': 0.1155921267,
                                    'sharpe_ratio': 24.1813858204,
                                    'time': '2015-01-15T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5508678733},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1214536932,
                                    'bias': 1.0,
                                    'equity': 1.1203868373,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 12.3723413064,
                                    'relative_return': 0.0042979065,
                                    'sharpe_ratio': 23.1861834887,
                                    'time': '2015-01-16T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5336083583},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1175824065,
                                    'bias': 1.0,
                                    'equity': 1.0694302345,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.045481258,
                                    'mean_return': 3.2259117591,
                                    'relative_return': -0.045481258,
                                    'sharpe_ratio': 5.6612638682,
                                    'time': '2015-01-17T00:00:00.000Z',
                                    'underwater': -0.045481258,
                                    'volatility': 0.5698218338},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1165122641,
                                    'bias': 1.0,
                                    'equity': 1.1283481789,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.045481258,
                                    'mean_return': 10.572579761,
                                    'relative_return': 0.0550928359,
                                    'sharpe_ratio': 17.7217488389,
                                    'time': '2015-01-18T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5965878344},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1112712787,
                                    'bias': 1.0,
                                    'equity': 1.1665716793,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.045481258,
                                    'mean_return': 18.2931777607,
                                    'relative_return': 0.0338756255,
                                    'sharpe_ratio': 30.9149139399,
                                    'time': '2015-01-19T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5917266274},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1088763711,
                                    'bias': 1.0,
                                    'equity': 1.1400976294,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.045481258,
                                    'mean_return': 9.94440808,
                                    'relative_return': -0.0226938905,
                                    'sharpe_ratio': 16.8184706304,
                                    'time': '2015-01-20T00:00:00.000Z',
                                    'underwater': -0.0226938905,
                                    'volatility': 0.5912789753}],
                          'schema': EXPECTED_JSON_SCHEMA['statistic']}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': None,
                                    'avg_turnover': 0.0677416729,
                                    'bias': 1.0,
                                    'equity': 1.2816699212,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083435733,
                                    'mean_return': 4.0404559571,
                                    'relative_return': -0.0053911755,
                                    'sharpe_ratio': 5.5658601428,
                                    'time': '2015-02-25T00:00:00.000Z',
                                    'underwater': -0.1345838404,
                                    'volatility': 0.7259355883},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0665608175,
                                    'bias': 1.0,
                                    'equity': 1.2750458866,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083435733,
                                    'mean_return': 3.7395322077,
                                    'relative_return': -0.0051682844,
                                    'sharpe_ratio': 5.1937539624,
                                    'time': '2015-02-26T00:00:00.000Z',
                                    'underwater': -0.1390565573,
                                    'volatility': 0.7200056519},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0668562147,
                                    'bias': 1.0,
                                    'equity': 1.384022378,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083435733,
                                    'mean_return': 6.7308670766,
                                    'relative_return': 0.085468682,
                                    'sharpe_ratio': 9.0797073142,
                                    'time': '2015-02-27T00:00:00.000Z',
                                    'underwater': -0.0654728559,
                                    'volatility': 0.7413088158},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0672549949,
                                    'bias': 1.0,
                                    'equity': 1.3780978689,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083435733,
                                    'mean_return': 6.2718989565,
                                    'relative_return': -0.0042806455,
                                    'sharpe_ratio': 8.5277607068,
                                    'time': '2015-02-28T00:00:00.000Z',
                                    'underwater': -0.0694732353,
                                    'volatility': 0.7354684509},
                                   {'avg_holding_time': 58.0,
                                    'avg_turnover': 0.0665924113,
                                    'bias': 1.0,
                                    'equity': 1.4109467797,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083435733,
                                    'mean_return': 7.1193827619,
                                    'relative_return': 0.0238364136,
                                    'sharpe_ratio': 9.7446973709,
                                    'time': '2015-03-01T00:00:00.000Z',
                                    'underwater': -0.0472928145,
                                    'volatility': 0.7305904423}],
                          'schema': EXPECTED_JSON_SCHEMA['statistic']}, json.loads(stat_tail))

    def test_relative_return(self):
        # this code demonstrates the problem of calculating if nan is found in the sample
        #  {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0},
        #           {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0},
        #           {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0},
        #           {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0},
        #           {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0},
        #           {'time': '2021-02-18T00:00:00.000Z', 'values': 0.0},
        #           {'time': '2021-02-19T00:00:00.000Z', 'values': 0.0},

        data = stats_data.get_base_df()
        weights_two_day = data.head(100)
        one = weights_two_day.sel(field="close") - weights_two_day.sel(field="close") + 1

        points_per_year = qnstats.calc_avg_points_per_year(weights_two_day)
        self.assertEqual(365, points_per_year)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)

        rr_df, _ = get_rr_dataframe_and_json(data, portfolio_history)
        stat_head = json.loads(rr_df.to_json(orient="table"))
        self.assertEqual(
            {"schema": EXPECTED_JSON_SCHEMA["base_values"],
             "data": [{"time": "2021-01-30T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-01-31T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-01T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-02T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-03T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-04T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-05T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-06T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-07T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-08T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-09T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-10T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-11T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-12T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-13T00:00:00.000Z", "values": 0.0666666667},
                      {"time": "2021-02-14T00:00:00.000Z", "values": 0.0625},
                      {"time": "2021-02-15T00:00:00.000Z", "values": 0.0588235294},
                      {"time": "2021-02-16T00:00:00.000Z", "values": 0.0555555556},
                      {"time": "2021-02-17T00:00:00.000Z", "values": 0.0526315789},
                      {"time": "2021-02-18T00:00:00.000Z", "values": 0.05},
                      {"time": "2021-02-19T00:00:00.000Z", "values": 0.0476190476},
                      {"time": "2021-02-20T00:00:00.000Z", "values": 0.0454545455},
                      {"time": "2021-02-21T00:00:00.000Z", "values": 0.0434782609},
                      {"time": "2021-02-22T00:00:00.000Z", "values": 0.0416666667},
                      {"time": "2021-02-23T00:00:00.000Z", "values": 0.04},
                      {"time": "2021-02-24T00:00:00.000Z", "values": 0.0384615385},
                      {"time": "2021-02-25T00:00:00.000Z", "values": 0.037037037},
                      {"time": "2021-02-26T00:00:00.000Z", "values": 0.0357142857},
                      {"time": "2021-02-27T00:00:00.000Z", "values": 0.0344827586},
                      {"time": "2021-02-28T00:00:00.000Z", "values": 0.0333333333},
                      {"time": "2021-03-01T00:00:00.000Z", "values": 0.0322580645},
                      {"time": "2021-03-02T00:00:00.000Z", "values": 0.03125},
                      {"time": "2021-03-03T00:00:00.000Z", "values": 0.0303030303},
                      {"time": "2021-03-04T00:00:00.000Z", "values": 0.0294117647},
                      {"time": "2021-03-05T00:00:00.000Z", "values": 0.0285714286},
                      {"time": "2021-03-06T00:00:00.000Z", "values": 0.0277777778},
                      {"time": "2021-03-07T00:00:00.000Z", "values": 0.027027027},
                      {"time": "2021-03-08T00:00:00.000Z", "values": 0.0263157895},
                      {"time": "2021-03-09T00:00:00.000Z", "values": 0.0256410256},
                      {"time": "2021-03-10T00:00:00.000Z", "values": 0.025},
                      {"time": "2021-03-11T00:00:00.000Z", "values": 0.0243902439},
                      {"time": "2021-03-12T00:00:00.000Z", "values": 0.0238095238},
                      {"time": "2021-03-13T00:00:00.000Z", "values": 0.023255814},
                      {"time": "2021-03-14T00:00:00.000Z", "values": 0.0227272727},
                      {"time": "2021-03-15T00:00:00.000Z", "values": 0.0222222222},
                      {"time": "2021-03-16T00:00:00.000Z", "values": 0.0217391304},
                      {"time": "2021-03-17T00:00:00.000Z", "values": 0.0212765957},
                      {"time": "2021-03-18T00:00:00.000Z", "values": 0.0208333333},
                      {"time": "2021-03-19T00:00:00.000Z", "values": 0.0204081633},
                      {"time": "2021-03-20T00:00:00.000Z", "values": 0.02},
                      {"time": "2021-03-21T00:00:00.000Z", "values": 0.0196078431},
                      {"time": "2021-03-22T00:00:00.000Z", "values": 0.0192307692},
                      {"time": "2021-03-23T00:00:00.000Z", "values": 0.0188679245},
                      {"time": "2021-03-24T00:00:00.000Z", "values": 0.0185185185},
                      {"time": "2021-03-25T00:00:00.000Z", "values": 0.0181818182},
                      {"time": "2021-03-26T00:00:00.000Z", "values": 0.0178571429},
                      {"time": "2021-03-27T00:00:00.000Z", "values": 0.0175438596},
                      {"time": "2021-03-28T00:00:00.000Z", "values": 0.0172413793},
                      {"time": "2021-03-29T00:00:00.000Z", "values": 0.0169491525},
                      {"time": "2021-03-30T00:00:00.000Z", "values": 0.0166666667},
                      {"time": "2021-03-31T00:00:00.000Z", "values": 0.0163934426},
                      {"time": "2021-04-01T00:00:00.000Z", "values": 0.0161290323},
                      {"time": "2021-04-02T00:00:00.000Z", "values": 0.0158730159},
                      {"time": "2021-04-03T00:00:00.000Z", "values": 0.015625},
                      {"time": "2021-04-04T00:00:00.000Z", "values": 0.0153846154},
                      {"time": "2021-04-05T00:00:00.000Z", "values": 0.0151515152},
                      {"time": "2021-04-06T00:00:00.000Z", "values": 0.0149253731},
                      {"time": "2021-04-07T00:00:00.000Z", "values": 0.0147058824},
                      {"time": "2021-04-08T00:00:00.000Z", "values": 0.0144927536},
                      {"time": "2021-04-09T00:00:00.000Z", "values": 0.0142857143},
                      {"time": "2021-04-10T00:00:00.000Z", "values": 0.014084507},
                      {"time": "2021-04-11T00:00:00.000Z", "values": 0.0138888889},
                      {"time": "2021-04-12T00:00:00.000Z", "values": 0.0136986301},
                      {"time": "2021-04-13T00:00:00.000Z", "values": 0.0135135135},
                      {"time": "2021-04-14T00:00:00.000Z", "values": 0.0133333333},
                      {"time": "2021-04-15T00:00:00.000Z", "values": 0.0131578947},
                      {"time": "2021-04-16T00:00:00.000Z", "values": 0.012987013},
                      {"time": "2021-04-17T00:00:00.000Z", "values": 0.0128205128},
                      {"time": "2021-04-18T00:00:00.000Z", "values": 0.0126582278},
                      {"time": "2021-04-19T00:00:00.000Z", "values": 0.0125},
                      {"time": "2021-04-20T00:00:00.000Z", "values": 0.012345679},
                      {"time": "2021-04-21T00:00:00.000Z", "values": 0.012195122},
                      {"time": "2021-04-22T00:00:00.000Z", "values": 0.0120481928},
                      {"time": "2021-04-23T00:00:00.000Z", "values": 0.0119047619},
                      {"time": "2021-04-24T00:00:00.000Z", "values": 0.0117647059},
                      {"time": "2021-04-25T00:00:00.000Z", "values": 0.011627907},
                      {"time": "2021-04-26T00:00:00.000Z", "values": 0.0114942529},
                      {"time": "2021-04-27T00:00:00.000Z", "values": 0.0113636364},
                      {"time": "2021-04-28T00:00:00.000Z", "values": 0.0112359551},
                      {"time": "2021-04-29T00:00:00.000Z", "values": 0.0111111111},
                      {"time": "2021-04-30T00:00:00.000Z", "values": 0.010989011},
                      {"time": "2021-05-01T00:00:00.000Z", "values": 0.0108695652},
                      {"time": "2021-05-02T00:00:00.000Z", "values": 0.0107526882},
                      {"time": "2021-05-03T00:00:00.000Z", "values": 0.0106382979},
                      {"time": "2021-05-04T00:00:00.000Z", "values": 0.0105263158},
                      {"time": "2021-05-05T00:00:00.000Z", "values": 0.0104166667},
                      {"time": "2021-05-06T00:00:00.000Z", "values": 0.0103092784},
                      {"time": "2021-05-07T00:00:00.000Z", "values": 0.0102040816},
                      {"time": "2021-05-08T00:00:00.000Z", "values": 0.0101010101},
                      {"time": "2021-05-09T00:00:00.000Z", "values": 0.01}]}, stat_head)

        data_two = stats_data.get_base_df_two_with_NaN()

        data_ = data_two.sel(asset=['BTC'])

        one_ = data_.sel(field="close") - data_.sel(field="close") + 1
        one_ = one_.fillna(1)

        portfolio_history_ = qnstats.output_normalize(one, per_asset=False)
        rr_df, _ = get_rr_dataframe_and_json(data_, portfolio_history_)
        stat_head = json.loads(rr_df.head(30).to_json(orient="table"))
        self.assertEqual(
            {"schema": EXPECTED_JSON_SCHEMA["base_values"],
             "data": [{"time": "2021-01-30T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-01-31T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-01T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-02T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-03T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-04T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-05T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-06T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-07T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-08T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-09T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-10T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-11T00:00:00.000Z", "values": 0.0},
                      {"time": "2021-02-12T00:00:00.000Z", "values": 0.0},
                      {'time': '2021-02-13T00:00:00.000Z', 'values': 0.0},
                      {'time': '2021-02-14T00:00:00.000Z', 'values': 0.0},
                      {'time': '2021-02-15T00:00:00.000Z', 'values': 0.0},
                      {'time': '2021-02-16T00:00:00.000Z', 'values': 0.0},
                      {'time': '2021-02-17T00:00:00.000Z', 'values': 0.0},
                      {'time': '2021-02-18T00:00:00.000Z', 'values': 0.0},
                      {'time': '2021-02-19T00:00:00.000Z', 'values': 0.0},
                      {"time": "2021-02-20T00:00:00.000Z", "values": 0.0454545455},
                      {"time": "2021-02-21T00:00:00.000Z", "values": 0.0434782609},
                      {"time": "2021-02-22T00:00:00.000Z", "values": 0.0416666667},
                      {"time": "2021-02-23T00:00:00.000Z", "values": 0.04},
                      {"time": "2021-02-24T00:00:00.000Z", "values": 0.0384615385},
                      {"time": "2021-02-25T00:00:00.000Z", "values": 0.037037037},
                      {"time": "2021-02-26T00:00:00.000Z", "values": 0.0357142857},
                      {"time": "2021-02-27T00:00:00.000Z", "values": 0.0344827586},
                      ]}, stat_head)

    def test_relative_return_per_asset(self):
        # this code demonstrates the problem of calculating if nan is found in the sample
        #  {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-13T00:00:00.000Z'},
        #           {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-14T00:00:00.000Z'},
        #           {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-15T00:00:00.000Z'},
        #           {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-16T00:00:00.000Z'},
        #           {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-17T00:00:00.000Z'},
        #           {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-18T00:00:00.000Z'},
        #           {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-19T00:00:00.000Z'},

        data = stats_data.get_base_df_two()
        one = data.sel(field="close") - data.sel(field="close") + 1
        one = one.fillna(1)

        portfolio_history = qnstats.output_normalize(one, per_asset=False)

        rr_df, _ = get_rr_dataframe_and_json(data, portfolio_history, per_asset=True)
        stat_head = json.loads(rr_df.to_json(orient="table"))
        self.assertEqual(
            {'data': [{'BTC': 0.0, 'ETH': 0.0, 'time': '2021-01-30T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-01-31T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-01T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-02T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-03T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-04T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-05T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-06T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-07T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-08T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-09T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-10T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                      {'BTC': 0.0333333333,
                       'ETH': 0.0333333333,
                       'time': '2021-02-13T00:00:00.000Z'},
                      {'BTC': 0.03125, 'ETH': 0.03125, 'time': '2021-02-14T00:00:00.000Z'},
                      {'BTC': 0.0294117647,
                       'ETH': 0.0294117647,
                       'time': '2021-02-15T00:00:00.000Z'},
                      {'BTC': 0.0277777778,
                       'ETH': 0.0277777778,
                       'time': '2021-02-16T00:00:00.000Z'},
                      {'BTC': 0.0263157895,
                       'ETH': 0.0263157895,
                       'time': '2021-02-17T00:00:00.000Z'},
                      {'BTC': 0.025, 'ETH': 0.025, 'time': '2021-02-18T00:00:00.000Z'},
                      {'BTC': 0.0238095238,
                       'ETH': 0.0238095238,
                       'time': '2021-02-19T00:00:00.000Z'},
                      {'BTC': 0.0227272727,
                       'ETH': 0.0227272727,
                       'time': '2021-02-20T00:00:00.000Z'},
                      {'BTC': 0.0217391304,
                       'ETH': 0.0217391304,
                       'time': '2021-02-21T00:00:00.000Z'},
                      {'BTC': 0.0208333333,
                       'ETH': 0.0208333333,
                       'time': '2021-02-22T00:00:00.000Z'},
                      {'BTC': 0.02, 'ETH': 0.02, 'time': '2021-02-23T00:00:00.000Z'},
                      {'BTC': 0.0192307692,
                       'ETH': 0.0192307692,
                       'time': '2021-02-24T00:00:00.000Z'},
                      {'BTC': 0.0185185185,
                       'ETH': 0.0185185185,
                       'time': '2021-02-25T00:00:00.000Z'},
                      {'BTC': 0.0178571429,
                       'ETH': 0.0178571429,
                       'time': '2021-02-26T00:00:00.000Z'},
                      {'BTC': 0.0172413793,
                       'ETH': 0.0172413793,
                       'time': '2021-02-27T00:00:00.000Z'}],
             'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                   {'name': 'BTC', 'type': 'number'},
                                   {'name': 'ETH', 'type': 'number'}],
                        'pandas_version': '0.20.0',
                        'primaryKey': ['time']}}, stat_head)

        data_two = stats_data.get_base_df_two_with_NaN()

        one_ = data_two.sel(field="close") - data_two.sel(field="close") + 1
        one_ = one_.fillna(1)

        portfolio_history_ = qnstats.output_normalize(one_, per_asset=False)

        rr_df, _ = get_rr_dataframe_and_json(data_two, portfolio_history_, per_asset=True)
        stat_head = json.loads(rr_df.head(30).to_json(orient="table"))
        self.assertEqual(
            {'data': [{'BTC': 0.0, 'ETH': 0.0, 'time': '2021-01-30T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-01-31T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-01T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-02T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-03T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-04T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-05T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-06T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-07T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-08T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-09T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-10T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-11T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-12T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-13T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-14T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-15T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-16T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-17T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-18T00:00:00.000Z'},
                      {'BTC': 0.0, 'ETH': 0.0, 'time': '2021-02-19T00:00:00.000Z'},
                      {'BTC': 0.0227272727,
                       'ETH': 0.0227272727,
                       'time': '2021-02-20T00:00:00.000Z'},
                      {'BTC': 0.0217391304,
                       'ETH': 0.0217391304,
                       'time': '2021-02-21T00:00:00.000Z'},
                      {'BTC': 0.0208333333,
                       'ETH': 0.0208333333,
                       'time': '2021-02-22T00:00:00.000Z'},
                      {'BTC': 0.02, 'ETH': 0.02, 'time': '2021-02-23T00:00:00.000Z'},
                      {'BTC': 0.0192307692,
                       'ETH': 0.0192307692,
                       'time': '2021-02-24T00:00:00.000Z'},
                      {'BTC': 0.0185185185,
                       'ETH': 0.0185185185,
                       'time': '2021-02-25T00:00:00.000Z'},
                      {'BTC': 0.0178571429,
                       'ETH': 0.0178571429,
                       'time': '2021-02-26T00:00:00.000Z'},
                      {'BTC': 0.0172413793,
                       'ETH': 0.0172413793,
                       'time': '2021-02-27T00:00:00.000Z'}],
             'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                   {'name': 'BTC', 'type': 'number'},
                                   {'name': 'ETH', 'type': 'number'}],
                        'pandas_version': '0.20.0',
                        'primaryKey': ['time']}}, stat_head)


if __name__ == '__main__':
    unittest.main()
