import unittest

import json
import os

os.environ['API_KEY'] = "default"

import qnt.data as qndata
import qnt.stats as qnstats


class Fields:
    OPEN = "open"
    LOW = "low"
    HIGH = "high"
    CLOSE = "close"
    VOL = "vol"
    DIVS = "divs"
    SPLIT = "split"
    SPLIT_CUMPROD = "split_cumprod"
    IS_LIQUID = 'is_liquid'


f = Fields


class Dimensions:
    TIME = 'time'
    FIELD = 'field'
    ASSET = 'asset'


ds = Dimensions

dims = (ds.FIELD, ds.TIME, ds.ASSET)

import pandas as pd
import xarray as xr
import numpy as np


def load_data_and_create_data_array(filename, dims, transpose_order):
    ds = xr.open_dataset(filename).load()
    dataset_name = list(ds.data_vars)[0]
    values = ds[dataset_name].transpose(*transpose_order).values
    coords = {dim: ds[dim].values for dim in dims}
    return xr.DataArray(values, dims=dims, coords=coords)


class TestBaseStatistic(unittest.TestCase):

    def test_commodity_imf(self):
        self.maxDiff = None
        dir = os.path.abspath(os.curdir)

        dims = ['time', 'asset']
        commodity = load_data_and_create_data_array(f"{dir}/data/commodity_imf.nc", dims, dims)

        commodity_server = qndata.imf_load_commodity_data(min_date="1990-01-01", max_date="2021-05-25")
        commodity_df = commodity.to_pandas().to_json(orient="table")
        commodity_server_df = commodity_server.to_pandas().to_json(orient="table")

        self.assertEqual(json.loads(commodity_df), json.loads(commodity_server_df))

    def test_futures(self):
        dir = os.path.abspath(os.curdir)

        dims = ['field', 'time', 'asset']
        futures = load_data_and_create_data_array(f"{dir}/data/futures.nc", dims, dims)

        futures_server = qndata.futures_load_data(min_date="1990-01-01", max_date="2021-05-25")

        futures_df = futures.sel(field="close").to_pandas().to_json(orient="table")
        futures_server_df = futures_server.sel(field="close").to_pandas().to_json(orient="table")
        self.assertEqual(futures_df, futures_server_df)

        futures_df = futures.sel(field="open").to_pandas().to_json(orient="table")
        futures_server_df = futures_server.sel(field="open").to_pandas().to_json(orient="table")
        self.assertEqual(futures_df, futures_server_df)

        futures_df = futures.sel(field="vol").to_pandas().to_json(orient="table")
        futures_server_df = futures_server.sel(field="vol").to_pandas().to_json(orient="table")
        self.assertEqual(futures_df, futures_server_df)

    def test_crypto_futures(self):
        dir = os.path.abspath(os.curdir)

        dims = ['field', 'time', 'asset']
        crypto_futures = load_data_and_create_data_array(f"{dir}/data/crypto_futures.nc", dims, dims)

        crypto_futures_server = qndata.cryptofutures_load_data(min_date="1990-01-01", max_date="2021-05-25")

        f_df = crypto_futures.sel(field="close").to_pandas().tail().to_json(orient="table")
        f_server_df = crypto_futures_server.sel(field="close").to_pandas().tail().to_json(orient="table")
        self.assertEqual(f_df, f_server_df)

        f_df = crypto_futures.sel(field="open").to_pandas().to_json(orient="table")
        f_server_df = crypto_futures_server.sel(field="open").to_pandas().to_json(orient="table")
        self.assertEqual(f_df, f_server_df)

        f_df = crypto_futures.sel(field="vol").to_pandas().to_json(orient="table")
        f_server_df = crypto_futures_server.sel(field="vol").to_pandas().to_json(orient="table")
        self.assertEqual(f_df, f_server_df)

    def test_global_cryptofutures_load_data(self):
        d = qndata.cryptofutures_load_data(min_date="2015-01-01",
                                           max_date="2015-03-01",
                                           dims=('time', 'field', 'asset'))
        buy_and_hold = d.sel(field="close") - d.sel(field="close") + 1
        stat = qnstats.calc_stat(d,
                                 buy_and_hold)
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
                                    'avg_turnover': 0.1217410247,
                                    'bias': 1.0,
                                    'equity': 1.1170697321,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 13.7895032698,
                                    'relative_return': 0.1170697321,
                                    'sharpe_ratio': 24.7163767489,
                                    'time': '2015-01-15T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5579095759},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.121510183,
                                    'bias': 1.0,
                                    'equity': 1.1218896609,
                                    'instruments': 1.0,
                                    'max_drawdown': 0.0,
                                    'mean_return': 12.7875689899,
                                    'relative_return': 0.0043147967,
                                    'sharpe_ratio': 23.6616374485,
                                    'time': '2015-01-16T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.54043466},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1176355782,
                                    'bias': 1.0,
                                    'equity': 1.070864625,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.0454813318,
                                    'mean_return': 3.3492939227,
                                    'relative_return': -0.0454813318,
                                    'sharpe_ratio': 5.8145562506,
                                    'time': '2015-01-17T00:00:00.000Z',
                                    'underwater': -0.0454813318,
                                    'volatility': 0.5760188359},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1165624919,
                                    'bias': 1.0,
                                    'equity': 1.1298618212,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.0454813318,
                                    'mean_return': 10.8914806998,
                                    'relative_return': 0.0550930481,
                                    'sharpe_ratio': 18.0909344181,
                                    'time': '2015-01-18T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.6020408039},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1113186252,
                                    'bias': 1.0,
                                    'equity': 1.1681417869,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.0454813318,
                                    'mean_return': 18.7981781722,
                                    'relative_return': 0.0338802187,
                                    'sharpe_ratio': 31.4944780481,
                                    'time': '2015-01-19T00:00:00.000Z',
                                    'underwater': 0.0,
                                    'volatility': 0.5968721928},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.1089211142,
                                    'bias': 1.0,
                                    'equity': 1.1416371714,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.0454813318,
                                    'mean_return': 10.2172872267,
                                    'relative_return': -0.0226895534,
                                    'sharpe_ratio': 17.1362010282,
                                    'time': '2015-01-20T00:00:00.000Z',
                                    'underwater': -0.0226895534,
                                    'volatility': 0.5962399256}],
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
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': None,
                                    'avg_turnover': 0.0677576685,
                                    'bias': 1.0,
                                    'equity': 1.283406468,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083424427,
                                    'mean_return': 4.0851354947,
                                    'relative_return': -0.0053911747,
                                    'sharpe_ratio': 5.6160104646,
                                    'time': '2015-02-25T00:00:00.000Z',
                                    'underwater': -0.1345804446,
                                    'volatility': 0.727408811},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0665765325,
                                    'bias': 1.0,
                                    'equity': 1.2767734584,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083424427,
                                    'mean_return': 3.7808040474,
                                    'relative_return': -0.0051682844,
                                    'sharpe_ratio': 5.2404365811,
                                    'time': '2015-02-26T00:00:00.000Z',
                                    'underwater': -0.139053179,
                                    'volatility': 0.7214673794},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0668716606,
                                    'bias': 1.0,
                                    'equity': 1.3858977648,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083424427,
                                    'mean_return': 6.7970275324,
                                    'relative_return': 0.0854688087,
                                    'sharpe_ratio': 9.1519468642,
                                    'time': '2015-02-27T00:00:00.000Z',
                                    'underwater': -0.0654690799,
                                    'volatility': 0.7426865161},
                                   {'avg_holding_time': None,
                                    'avg_turnover': 0.0672701772,
                                    'bias': 1.0,
                                    'equity': 1.3799653632,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083424427,
                                    'mean_return': 6.3330768196,
                                    'relative_return': -0.0042805478,
                                    'sharpe_ratio': 8.5949630968,
                                    'time': '2015-02-28T00:00:00.000Z',
                                    'underwater': -0.0694693841,
                                    'volatility': 0.7368358361},
                                   {'avg_holding_time': 58.0,
                                    'avg_turnover': 0.0666073409,
                                    'bias': 1.0,
                                    'equity': 1.4128588134,
                                    'instruments': 1.0,
                                    'max_drawdown': -0.2083424427,
                                    'mean_return': 7.1865481596,
                                    'relative_return': 0.0238364317,
                                    'sharpe_ratio': 9.8184890165,
                                    'time': '2015-03-01T00:00:00.000Z',
                                    'underwater': -0.0472888546,
                                    'volatility': 0.731940337}],
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
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_tail))

    def test_global_futures_load_data(self):
        d = qndata.futures_load_data(min_date="2015-01-01",
                                     max_date="2015-03-01",
                                     dims=('time', 'field', 'asset'))
        buy_and_hold = d.sel(field="close") - d.sel(field="close") + 1
        stat = qnstats.calc_stat(d,
                                 buy_and_hold)
        stat_head_df = stat.to_pandas().head(20)
        stat_head = stat_head_df.tail(8).to_json(orient="table")
        self.assertEqual(
            {'data': [{'avg_holding_time': 1.4975963217,
                       'avg_turnover': 0.2240413778,
                       'bias': 1.0,
                       'equity': 1.0,
                       'instruments': 71.0,
                       'max_drawdown': 0.0,
                       'mean_return': 0.0,
                       'relative_return': 0.0,
                       'sharpe_ratio': None,
                       'time': '2015-01-19T00:00:00.000Z',
                       'underwater': 0.0,
                       'volatility': 0.0},
                      {'avg_holding_time': 3.5063911766,
                       'avg_turnover': 0.2283826683,
                       'bias': 1.0,
                       'equity': 1.0,
                       'instruments': 71.0,
                       'max_drawdown': 0.0,
                       'mean_return': 0.0,
                       'relative_return': 0.0,
                       'sharpe_ratio': None,
                       'time': '2015-01-20T00:00:00.000Z',
                       'underwater': 0.0,
                       'volatility': 0.0},
                      {'avg_holding_time': 4.435443262,
                       'avg_turnover': 0.2527382758,
                       'bias': 1.0,
                       'equity': 0.9985330555,
                       'instruments': 71.0,
                       'max_drawdown': -0.0014669445,
                       'mean_return': -0.0242656321,
                       'relative_return': -0.0014669445,
                       'sharpe_ratio': -4.1857019583,
                       'time': '2015-01-21T00:00:00.000Z',
                       'underwater': -0.0014669445,
                       'volatility': 0.0057972671},
                      {'avg_holding_time': 4.435443262,
                       'avg_turnover': 0.2374045948,
                       'bias': 1.0,
                       'equity': 1.0006414755,
                       'instruments': 71.0,
                       'max_drawdown': -0.0014669445,
                       'mean_return': 0.0101106923,
                       'relative_return': 0.0021115175,
                       'sharpe_ratio': 0.9948183022,
                       'time': '2015-01-22T00:00:00.000Z',
                       'underwater': 0.0,
                       'volatility': 0.0101633558},
                      {'avg_holding_time': 4.4139810561,
                       'avg_turnover': 0.2243219851,
                       'bias': 1.0,
                       'equity': 1.0025655131,
                       'instruments': 71.0,
                       'max_drawdown': -0.0014669445,
                       'mean_return': 0.0385552246,
                       'relative_return': 0.0019228042,
                       'sharpe_ratio': 3.1857939553,
                       'time': '2015-01-23T00:00:00.000Z',
                       'underwater': 0.0,
                       'volatility': 0.0121022342},
                      {'avg_holding_time': 4.4346309434,
                       'avg_turnover': 0.2130929435,
                       'bias': 1.0,
                       'equity': 1.00062137,
                       'instruments': 71.0,
                       'max_drawdown': -0.0019391682,
                       'mean_return': 0.0086995913,
                       'relative_return': -0.0019391682,
                       'sharpe_ratio': 0.6216143491,
                       'time': '2015-01-26T00:00:00.000Z',
                       'underwater': -0.0019391682,
                       'volatility': 0.0139951585},
                      {'avg_holding_time': 4.4815124875,
                       'avg_turnover': 0.2035406573,
                       'bias': 1.0,
                       'equity': 1.0008131205,
                       'instruments': 71.0,
                       'max_drawdown': -0.0019391682,
                       'mean_return': 0.0107952374,
                       'relative_return': 0.0001916315,
                       'sharpe_ratio': 0.7918363801,
                       'time': '2015-01-27T00:00:00.000Z',
                       'underwater': -0.0017479084,
                       'volatility': 0.0136331668},
                      {'avg_holding_time': 4.4815124875,
                       'avg_turnover': 0.1937784263,
                       'bias': 1.0,
                       'equity': 0.9969018847,
                       'instruments': 71.0,
                       'max_drawdown': -0.0056491355,
                       'mean_return': -0.0381932198,
                       'relative_return': -0.003908058,
                       'sharpe_ratio': -2.0054402205,
                       'time': '2015-01-28T00:00:00.000Z',
                       'underwater': -0.0056491355,
                       'volatility': 0.0190448059}],
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
                        'pandas_version': '0.20.0',
                        'primaryKey': ['time']}}
            , json.loads(stat_head))

        stat_tail = stat.to_pandas().tail().to_json(orient="table")
        self.assertEqual({'data': [{'avg_holding_time': 8.4702783084,
                                    'avg_turnover': 0.1326775112,
                                    'bias': 1.0,
                                    'equity': 1.0138500757,
                                    'instruments': 71.0,
                                    'max_drawdown': -0.0079828466,
                                    'mean_return': 0.0951109235,
                                    'relative_return': -0.0031417308,
                                    'sharpe_ratio': 2.0337197313,
                                    'time': '2015-02-23T00:00:00.000Z',
                                    'underwater': -0.0067521642,
                                    'volatility': 0.0467669768},
                                   {'avg_holding_time': 8.5726998302,
                                    'avg_turnover': 0.130256892,
                                    'bias': 1.0,
                                    'equity': 1.015843861,
                                    'instruments': 71.0,
                                    'max_drawdown': -0.0079828466,
                                    'mean_return': 0.1064648372,
                                    'relative_return': 0.0019665485,
                                    'sharpe_ratio': 2.2976158438,
                                    'time': '2015-02-24T00:00:00.000Z',
                                    'underwater': -0.0047988942,
                                    'volatility': 0.0463370922},
                                   {'avg_holding_time': 8.5276427194,
                                    'avg_turnover': 0.1273602796,
                                    'bias': 1.0,
                                    'equity': 1.0181617235,
                                    'instruments': 71.0,
                                    'max_drawdown': -0.0079828466,
                                    'mean_return': 0.1195673077,
                                    'relative_return': 0.0022817114,
                                    'sharpe_ratio': 2.5999400363,
                                    'time': '2015-02-25T00:00:00.000Z',
                                    'underwater': -0.0025281326,
                                    'volatility': 0.0459884867},
                                   {'avg_holding_time': 8.5867761138,
                                    'avg_turnover': 0.1250404875,
                                    'bias': 1.0,
                                    'equity': 1.0178108272,
                                    'instruments': 71.0,
                                    'max_drawdown': -0.0079828466,
                                    'mean_return': 0.1141339463,
                                    'relative_return': -0.0003446371,
                                    'sharpe_ratio': 2.5103070071,
                                    'time': '2015-02-26T00:00:00.000Z',
                                    'underwater': -0.0028718984,
                                    'volatility': 0.0454661306},
                                   {'avg_holding_time': 11.7121038954,
                                    'avg_turnover': 0.1227141634,
                                    'bias': 1.0,
                                    'equity': 1.0196771927,
                                    'instruments': 71.0,
                                    'max_drawdown': -0.0079828466,
                                    'mean_return': 0.1235043021,
                                    'relative_return': 0.0018337057,
                                    'sharpe_ratio': 2.7415850925,
                                    'time': '2015-02-27T00:00:00.000Z',
                                    'underwater': -0.0010434589,
                                    'volatility': 0.0450485022}],
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
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_tail))

    def test_global_cryptofutures_load_data_all(self):
        d = qndata.cryptofutures_load_data(min_date="2015-01-01",
                                           max_date="2015-03-01",
                                           dims=('time', 'field', 'asset'))

        stat_head = d.sel(field="close").to_pandas().to_json(orient="table")
        self.assertEqual({'data': [{'BTC': 315.53, 'time': '2015-01-01T00:00:00.000Z'},
                                   {'BTC': 315.3, 'time': '2015-01-02T00:00:00.000Z'},
                                   {'BTC': 291.6, 'time': '2015-01-03T00:00:00.000Z'},
                                   {'BTC': 264.0, 'time': '2015-01-04T00:00:00.000Z'},
                                   {'BTC': 271.01, 'time': '2015-01-05T00:00:00.000Z'},
                                   {'BTC': 288.5, 'time': '2015-01-06T00:00:00.000Z'},
                                   {'BTC': 296.46, 'time': '2015-01-07T00:00:00.000Z'},
                                   {'BTC': 292.54, 'time': '2015-01-08T00:00:00.000Z'},
                                   {'BTC': 288.39, 'time': '2015-01-09T00:00:00.000Z'},
                                   {'BTC': 279.27, 'time': '2015-01-10T00:00:00.000Z'},
                                   {'BTC': 268.79, 'time': '2015-01-11T00:00:00.000Z'},
                                   {'BTC': 271.52, 'time': '2015-01-12T00:00:00.000Z'},
                                   {'BTC': 237.0, 'time': '2015-01-13T00:00:00.000Z'},
                                   {'BTC': 184.42, 'time': '2015-01-14T00:00:00.000Z'},
                                   {'BTC': 207.1, 'time': '2015-01-15T00:00:00.000Z'},
                                   {'BTC': 208.0, 'time': '2015-01-16T00:00:00.000Z'},
                                   {'BTC': 198.54, 'time': '2015-01-17T00:00:00.000Z'},
                                   {'BTC': 209.48, 'time': '2015-01-18T00:00:00.000Z'},
                                   {'BTC': 216.61, 'time': '2015-01-19T00:00:00.000Z'},
                                   {'BTC': 211.7, 'time': '2015-01-20T00:00:00.000Z'},
                                   {'BTC': 226.8, 'time': '2015-01-21T00:00:00.000Z'},
                                   {'BTC': 232.0, 'time': '2015-01-22T00:00:00.000Z'},
                                   {'BTC': 234.81, 'time': '2015-01-23T00:00:00.000Z'},
                                   {'BTC': 245.31, 'time': '2015-01-24T00:00:00.000Z'},
                                   {'BTC': 251.49, 'time': '2015-01-25T00:00:00.000Z'},
                                   {'BTC': 275.0, 'time': '2015-01-26T00:00:00.000Z'},
                                   {'BTC': 262.66, 'time': '2015-01-27T00:00:00.000Z'},
                                   {'BTC': 234.43, 'time': '2015-01-28T00:00:00.000Z'},
                                   {'BTC': 234.2, 'time': '2015-01-29T00:00:00.000Z'},
                                   {'BTC': 226.89, 'time': '2015-01-30T00:00:00.000Z'},
                                   {'BTC': 222.51, 'time': '2015-01-31T00:00:00.000Z'},
                                   {'BTC': 233.32, 'time': '2015-02-01T00:00:00.000Z'},
                                   {'BTC': 227.36, 'time': '2015-02-02T00:00:00.000Z'},
                                   {'BTC': 226.89, 'time': '2015-02-03T00:00:00.000Z'},
                                   {'BTC': 225.33, 'time': '2015-02-04T00:00:00.000Z'},
                                   {'BTC': 217.7, 'time': '2015-02-05T00:00:00.000Z'},
                                   {'BTC': 223.68, 'time': '2015-02-06T00:00:00.000Z'},
                                   {'BTC': 228.0, 'time': '2015-02-07T00:00:00.000Z'},
                                   {'BTC': 222.84, 'time': '2015-02-08T00:00:00.000Z'},
                                   {'BTC': 221.52, 'time': '2015-02-09T00:00:00.000Z'},
                                   {'BTC': 219.91, 'time': '2015-02-10T00:00:00.000Z'},
                                   {'BTC': 219.42, 'time': '2015-02-11T00:00:00.000Z'},
                                   {'BTC': 222.19, 'time': '2015-02-12T00:00:00.000Z'},
                                   {'BTC': 235.39, 'time': '2015-02-13T00:00:00.000Z'},
                                   {'BTC': 255.3, 'time': '2015-02-14T00:00:00.000Z'},
                                   {'BTC': 232.32, 'time': '2015-02-15T00:00:00.000Z'},
                                   {'BTC': 235.08, 'time': '2015-02-16T00:00:00.000Z'},
                                   {'BTC': 241.37, 'time': '2015-02-17T00:00:00.000Z'},
                                   {'BTC': 232.6, 'time': '2015-02-18T00:00:00.000Z'},
                                   {'BTC': 241.26, 'time': '2015-02-19T00:00:00.000Z'},
                                   {'BTC': 244.9, 'time': '2015-02-20T00:00:00.000Z'},
                                   {'BTC': 244.84, 'time': '2015-02-21T00:00:00.000Z'},
                                   {'BTC': 236.41, 'time': '2015-02-22T00:00:00.000Z'},
                                   {'BTC': 240.16, 'time': '2015-02-23T00:00:00.000Z'},
                                   {'BTC': 239.28, 'time': '2015-02-24T00:00:00.000Z'},
                                   {'BTC': 237.99, 'time': '2015-02-25T00:00:00.000Z'},
                                   {'BTC': 236.76, 'time': '2015-02-26T00:00:00.000Z'},
                                   {'BTC': 257.0, 'time': '2015-02-27T00:00:00.000Z'},
                                   {'BTC': 255.9, 'time': '2015-02-28T00:00:00.000Z'},
                                   {'BTC': 262.0, 'time': '2015-03-01T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_head))

        stat_head = d.sel(field="open").to_pandas().to_json(orient="table")
        self.assertEqual({'data': [{'BTC': 317.99, 'time': '2015-01-01T00:00:00.000Z'},
                                   {'BTC': 315.23, 'time': '2015-01-02T00:00:00.000Z'},
                                   {'BTC': 315.3, 'time': '2015-01-03T00:00:00.000Z'},
                                   {'BTC': 291.58, 'time': '2015-01-04T00:00:00.000Z'},
                                   {'BTC': 264.0, 'time': '2015-01-05T00:00:00.000Z'},
                                   {'BTC': 271.42, 'time': '2015-01-06T00:00:00.000Z'},
                                   {'BTC': 288.5, 'time': '2015-01-07T00:00:00.000Z'},
                                   {'BTC': 296.64, 'time': '2015-01-08T00:00:00.000Z'},
                                   {'BTC': 292.07, 'time': '2015-01-09T00:00:00.000Z'},
                                   {'BTC': 287.96, 'time': '2015-01-10T00:00:00.000Z'},
                                   {'BTC': 279.27, 'time': '2015-01-11T00:00:00.000Z'},
                                   {'BTC': 269.0, 'time': '2015-01-12T00:00:00.000Z'},
                                   {'BTC': 271.59, 'time': '2015-01-13T00:00:00.000Z'},
                                   {'BTC': 237.0, 'time': '2015-01-14T00:00:00.000Z'},
                                   {'BTC': 184.42, 'time': '2015-01-15T00:00:00.000Z'},
                                   {'BTC': 207.08, 'time': '2015-01-16T00:00:00.000Z'},
                                   {'BTC': 208.66, 'time': '2015-01-17T00:00:00.000Z'},
                                   {'BTC': 198.51, 'time': '2015-01-18T00:00:00.000Z'},
                                   {'BTC': 208.71, 'time': '2015-01-19T00:00:00.000Z'},
                                   {'BTC': 216.9, 'time': '2015-01-20T00:00:00.000Z'},
                                   {'BTC': 211.7, 'time': '2015-01-21T00:00:00.000Z'},
                                   {'BTC': 227.3, 'time': '2015-01-22T00:00:00.000Z'},
                                   {'BTC': 231.94, 'time': '2015-01-23T00:00:00.000Z'},
                                   {'BTC': 234.84, 'time': '2015-01-24T00:00:00.000Z'},
                                   {'BTC': 245.5, 'time': '2015-01-25T00:00:00.000Z'},
                                   {'BTC': 251.49, 'time': '2015-01-26T00:00:00.000Z'},
                                   {'BTC': 275.0, 'time': '2015-01-27T00:00:00.000Z'},
                                   {'BTC': 262.59, 'time': '2015-01-28T00:00:00.000Z'},
                                   {'BTC': 234.58, 'time': '2015-01-29T00:00:00.000Z'},
                                   {'BTC': 234.21, 'time': '2015-01-30T00:00:00.000Z'},
                                   {'BTC': 227.16, 'time': '2015-01-31T00:00:00.000Z'},
                                   {'BTC': 222.69, 'time': '2015-02-01T00:00:00.000Z'},
                                   {'BTC': 233.32, 'time': '2015-02-02T00:00:00.000Z'},
                                   {'BTC': 227.27, 'time': '2015-02-03T00:00:00.000Z'},
                                   {'BTC': 226.89, 'time': '2015-02-04T00:00:00.000Z'},
                                   {'BTC': 225.4, 'time': '2015-02-05T00:00:00.000Z'},
                                   {'BTC': 217.7, 'time': '2015-02-06T00:00:00.000Z'},
                                   {'BTC': 223.8, 'time': '2015-02-07T00:00:00.000Z'},
                                   {'BTC': 227.98, 'time': '2015-02-08T00:00:00.000Z'},
                                   {'BTC': 222.85, 'time': '2015-02-09T00:00:00.000Z'},
                                   {'BTC': 221.72, 'time': '2015-02-10T00:00:00.000Z'},
                                   {'BTC': 219.91, 'time': '2015-02-11T00:00:00.000Z'},
                                   {'BTC': 219.42, 'time': '2015-02-12T00:00:00.000Z'},
                                   {'BTC': 222.19, 'time': '2015-02-13T00:00:00.000Z'},
                                   {'BTC': 235.39, 'time': '2015-02-14T00:00:00.000Z'},
                                   {'BTC': 255.71, 'time': '2015-02-15T00:00:00.000Z'},
                                   {'BTC': 232.08, 'time': '2015-02-16T00:00:00.000Z'},
                                   {'BTC': 236.58, 'time': '2015-02-17T00:00:00.000Z'},
                                   {'BTC': 241.46, 'time': '2015-02-18T00:00:00.000Z'},
                                   {'BTC': 232.53, 'time': '2015-02-19T00:00:00.000Z'},
                                   {'BTC': 241.29, 'time': '2015-02-20T00:00:00.000Z'},
                                   {'BTC': 244.89, 'time': '2015-02-21T00:00:00.000Z'},
                                   {'BTC': 244.72, 'time': '2015-02-22T00:00:00.000Z'},
                                   {'BTC': 236.35, 'time': '2015-02-23T00:00:00.000Z'},
                                   {'BTC': 240.16, 'time': '2015-02-24T00:00:00.000Z'},
                                   {'BTC': 239.33, 'time': '2015-02-25T00:00:00.000Z'},
                                   {'BTC': 237.99, 'time': '2015-02-26T00:00:00.000Z'},
                                   {'BTC': 236.71, 'time': '2015-02-27T00:00:00.000Z'},
                                   {'BTC': 257.0, 'time': '2015-02-28T00:00:00.000Z'},
                                   {'BTC': 255.89, 'time': '2015-03-01T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_head))

        stat_head = d.sel(field="high").to_pandas().to_json(orient="table")
        self.assertEqual({'data': [{'BTC': 322.9, 'time': '2015-01-01T00:00:00.000Z'},
                                   {'BTC': 316.74, 'time': '2015-01-02T00:00:00.000Z'},
                                   {'BTC': 315.99, 'time': '2015-01-03T00:00:00.000Z'},
                                   {'BTC': 292.76, 'time': '2015-01-04T00:00:00.000Z'},
                                   {'BTC': 279.5, 'time': '2015-01-05T00:00:00.000Z'},
                                   {'BTC': 290.45, 'time': '2015-01-06T00:00:00.000Z'},
                                   {'BTC': 303.33, 'time': '2015-01-07T00:00:00.000Z'},
                                   {'BTC': 300.0, 'time': '2015-01-08T00:00:00.000Z'},
                                   {'BTC': 297.32, 'time': '2015-01-09T00:00:00.000Z'},
                                   {'BTC': 292.1, 'time': '2015-01-10T00:00:00.000Z'},
                                   {'BTC': 282.86, 'time': '2015-01-11T00:00:00.000Z'},
                                   {'BTC': 274.44, 'time': '2015-01-12T00:00:00.000Z'},
                                   {'BTC': 272.1, 'time': '2015-01-13T00:00:00.000Z'},
                                   {'BTC': 238.5, 'time': '2015-01-14T00:00:00.000Z'},
                                   {'BTC': 230.74, 'time': '2015-01-15T00:00:00.000Z'},
                                   {'BTC': 223.1, 'time': '2015-01-16T00:00:00.000Z'},
                                   {'BTC': 214.0, 'time': '2015-01-17T00:00:00.000Z'},
                                   {'BTC': 222.0, 'time': '2015-01-18T00:00:00.000Z'},
                                   {'BTC': 219.95, 'time': '2015-01-19T00:00:00.000Z'},
                                   {'BTC': 219.25, 'time': '2015-01-20T00:00:00.000Z'},
                                   {'BTC': 230.0, 'time': '2015-01-21T00:00:00.000Z'},
                                   {'BTC': 242.0, 'time': '2015-01-22T00:00:00.000Z'},
                                   {'BTC': 236.98, 'time': '2015-01-23T00:00:00.000Z'},
                                   {'BTC': 248.3, 'time': '2015-01-24T00:00:00.000Z'},
                                   {'BTC': 259.39, 'time': '2015-01-25T00:00:00.000Z'},
                                   {'BTC': 315.0, 'time': '2015-01-26T00:00:00.000Z'},
                                   {'BTC': 280.37, 'time': '2015-01-27T00:00:00.000Z'},
                                   {'BTC': 269.97, 'time': '2015-01-28T00:00:00.000Z'},
                                   {'BTC': 239.33, 'time': '2015-01-29T00:00:00.000Z'},
                                   {'BTC': 243.89, 'time': '2015-01-30T00:00:00.000Z'},
                                   {'BTC': 234.03, 'time': '2015-01-31T00:00:00.000Z'},
                                   {'BTC': 233.32, 'time': '2015-02-01T00:00:00.000Z'},
                                   {'BTC': 233.32, 'time': '2015-02-02T00:00:00.000Z'},
                                   {'BTC': 248.42, 'time': '2015-02-03T00:00:00.000Z'},
                                   {'BTC': 233.0, 'time': '2015-02-04T00:00:00.000Z'},
                                   {'BTC': 228.8, 'time': '2015-02-05T00:00:00.000Z'},
                                   {'BTC': 225.88, 'time': '2015-02-06T00:00:00.000Z'},
                                   {'BTC': 239.78, 'time': '2015-02-07T00:00:00.000Z'},
                                   {'BTC': 232.9, 'time': '2015-02-08T00:00:00.000Z'},
                                   {'BTC': 225.98, 'time': '2015-02-09T00:00:00.000Z'},
                                   {'BTC': 223.4, 'time': '2015-02-10T00:00:00.000Z'},
                                   {'BTC': 224.4, 'time': '2015-02-11T00:00:00.000Z'},
                                   {'BTC': 223.17, 'time': '2015-02-12T00:00:00.000Z'},
                                   {'BTC': 242.5, 'time': '2015-02-13T00:00:00.000Z'},
                                   {'BTC': 259.0, 'time': '2015-02-14T00:00:00.000Z'},
                                   {'BTC': 268.54, 'time': '2015-02-15T00:00:00.000Z'},
                                   {'BTC': 243.65, 'time': '2015-02-16T00:00:00.000Z'},
                                   {'BTC': 246.28, 'time': '2015-02-17T00:00:00.000Z'},
                                   {'BTC': 244.99, 'time': '2015-02-18T00:00:00.000Z'},
                                   {'BTC': 243.42, 'time': '2015-02-19T00:00:00.000Z'},
                                   {'BTC': 248.98, 'time': '2015-02-20T00:00:00.000Z'},
                                   {'BTC': 247.73, 'time': '2015-02-21T00:00:00.000Z'},
                                   {'BTC': 247.78, 'time': '2015-02-22T00:00:00.000Z'},
                                   {'BTC': 241.0, 'time': '2015-02-23T00:00:00.000Z'},
                                   {'BTC': 240.99, 'time': '2015-02-24T00:00:00.000Z'},
                                   {'BTC': 240.83, 'time': '2015-02-25T00:00:00.000Z'},
                                   {'BTC': 238.33, 'time': '2015-02-26T00:00:00.000Z'},
                                   {'BTC': 262.8, 'time': '2015-02-27T00:00:00.000Z'},
                                   {'BTC': 258.38, 'time': '2015-02-28T00:00:00.000Z'},
                                   {'BTC': 266.91, 'time': '2015-03-01T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_head))

        stat_head = d.sel(field="low").to_pandas().to_json(orient="table")
        self.assertEqual({'data': [{'BTC': 313.81, 'time': '2015-01-01T00:00:00.000Z'},
                                   {'BTC': 313.28, 'time': '2015-01-02T00:00:00.000Z'},
                                   {'BTC': 290.11, 'time': '2015-01-03T00:00:00.000Z'},
                                   {'BTC': 255.03, 'time': '2015-01-04T00:00:00.000Z'},
                                   {'BTC': 258.06, 'time': '2015-01-05T00:00:00.000Z'},
                                   {'BTC': 270.31, 'time': '2015-01-06T00:00:00.000Z'},
                                   {'BTC': 284.0, 'time': '2015-01-07T00:00:00.000Z'},
                                   {'BTC': 282.01, 'time': '2015-01-08T00:00:00.000Z'},
                                   {'BTC': 280.4, 'time': '2015-01-09T00:00:00.000Z'},
                                   {'BTC': 275.0, 'time': '2015-01-10T00:00:00.000Z'},
                                   {'BTC': 265.83, 'time': '2015-01-11T00:00:00.000Z'},
                                   {'BTC': 266.24, 'time': '2015-01-12T00:00:00.000Z'},
                                   {'BTC': 226.1, 'time': '2015-01-13T00:00:00.000Z'},
                                   {'BTC': 166.45, 'time': '2015-01-14T00:00:00.000Z'},
                                   {'BTC': 172.51, 'time': '2015-01-15T00:00:00.000Z'},
                                   {'BTC': 197.72, 'time': '2015-01-16T00:00:00.000Z'},
                                   {'BTC': 192.2, 'time': '2015-01-17T00:00:00.000Z'},
                                   {'BTC': 194.0, 'time': '2015-01-18T00:00:00.000Z'},
                                   {'BTC': 207.0, 'time': '2015-01-19T00:00:00.000Z'},
                                   {'BTC': 203.7, 'time': '2015-01-20T00:00:00.000Z'},
                                   {'BTC': 210.0, 'time': '2015-01-21T00:00:00.000Z'},
                                   {'BTC': 223.3, 'time': '2015-01-22T00:00:00.000Z'},
                                   {'BTC': 225.13, 'time': '2015-01-23T00:00:00.000Z'},
                                   {'BTC': 229.03, 'time': '2015-01-24T00:00:00.000Z'},
                                   {'BTC': 245.0, 'time': '2015-01-25T00:00:00.000Z'},
                                   {'BTC': 251.48, 'time': '2015-01-26T00:00:00.000Z'},
                                   {'BTC': 250.0, 'time': '2015-01-27T00:00:00.000Z'},
                                   {'BTC': 231.01, 'time': '2015-01-28T00:00:00.000Z'},
                                   {'BTC': 220.0, 'time': '2015-01-29T00:00:00.000Z'},
                                   {'BTC': 224.22, 'time': '2015-01-30T00:00:00.000Z'},
                                   {'BTC': 221.21, 'time': '2015-01-31T00:00:00.000Z'},
                                   {'BTC': 211.02, 'time': '2015-02-01T00:00:00.000Z'},
                                   {'BTC': 222.5, 'time': '2015-02-02T00:00:00.000Z'},
                                   {'BTC': 222.66, 'time': '2015-02-03T00:00:00.000Z'},
                                   {'BTC': 220.23, 'time': '2015-02-04T00:00:00.000Z'},
                                   {'BTC': 210.12, 'time': '2015-02-05T00:00:00.000Z'},
                                   {'BTC': 215.0, 'time': '2015-02-06T00:00:00.000Z'},
                                   {'BTC': 222.66, 'time': '2015-02-07T00:00:00.000Z'},
                                   {'BTC': 221.1, 'time': '2015-02-08T00:00:00.000Z'},
                                   {'BTC': 215.33, 'time': '2015-02-09T00:00:00.000Z'},
                                   {'BTC': 214.0, 'time': '2015-02-10T00:00:00.000Z'},
                                   {'BTC': 218.1, 'time': '2015-02-11T00:00:00.000Z'},
                                   {'BTC': 217.8, 'time': '2015-02-12T00:00:00.000Z'},
                                   {'BTC': 221.46, 'time': '2015-02-13T00:00:00.000Z'},
                                   {'BTC': 234.87, 'time': '2015-02-14T00:00:00.000Z'},
                                   {'BTC': 228.2, 'time': '2015-02-15T00:00:00.000Z'},
                                   {'BTC': 228.62, 'time': '2015-02-16T00:00:00.000Z'},
                                   {'BTC': 231.5, 'time': '2015-02-17T00:00:00.000Z'},
                                   {'BTC': 231.01, 'time': '2015-02-18T00:00:00.000Z'},
                                   {'BTC': 231.93, 'time': '2015-02-19T00:00:00.000Z'},
                                   {'BTC': 238.95, 'time': '2015-02-20T00:00:00.000Z'},
                                   {'BTC': 243.28, 'time': '2015-02-21T00:00:00.000Z'},
                                   {'BTC': 232.1, 'time': '2015-02-22T00:00:00.000Z'},
                                   {'BTC': 232.61, 'time': '2015-02-23T00:00:00.000Z'},
                                   {'BTC': 236.7, 'time': '2015-02-24T00:00:00.000Z'},
                                   {'BTC': 235.53, 'time': '2015-02-25T00:00:00.000Z'},
                                   {'BTC': 233.62, 'time': '2015-02-26T00:00:00.000Z'},
                                   {'BTC': 236.64, 'time': '2015-02-27T00:00:00.000Z'},
                                   {'BTC': 251.25, 'time': '2015-02-28T00:00:00.000Z'},
                                   {'BTC': 245.65, 'time': '2015-03-01T00:00:00.000Z'}],
                          'schema': {'fields': [{'name': 'time', 'type': 'datetime'},
                                                {'name': 'BTC', 'type': 'number'}],
                                     'pandas_version': '0.20.0',
                                     'primaryKey': ['time']}}, json.loads(stat_head))


if __name__ == '__main__':
    unittest.main()

# import pickle
# import pickle5 as pickle
# futures = qndata.futures_load_data(min_date="1990-01-01", max_date="2021-05-25")
# commodity = qndata.imf_load_commodity_data(min_date="1990-01-01", max_date="2021-05-25")
# crypto_futures = qndata.cryptofutures_load_data(min_date="1990-01-01", max_date="2021-05-25")
#
# def save_object(obj, filename):
#     with open(filename, 'wb') as output:  # Overwrites any existing file.
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
#
# save_object(futures, 'futures.pkl')
# save_object(commodity, 'commodity_imf.pkl')
# save_object(crypto_futures, 'crypto_futures.pkl')

# ds_disk = xr.open_dataset(dir + "/data/commodity_imf.nc")
# ds_disk_ = ds_disk.transpose(  dims=('time', 'field', 'asset'), transpose_coords=True)
