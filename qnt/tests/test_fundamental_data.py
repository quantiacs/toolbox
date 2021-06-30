import unittest
import pandas as pd

import json
import os
import pickle

os.environ['API_KEY'] = "default"

pd.set_option('display.max_rows', None)


def get_data():
    import pandas as pd
    import qnt.data    as qndata
    import datetime    as dt

    import qnt.data.secgov_indicators

    pd.set_option('display.max_rows', None)

    def get_data_filter(data, assets):
        filler = data.sel(asset=assets)
        return filler

    assets = qndata.load_assets(min_date="2010-01-01", max_date="2021-06-25")

    WMT = {}
    for a in assets:
        if a['id'] == 'NYSE:WMT':
            WMT = a

    data = qndata.load_data(min_date="2010-01-01", max_date="2021-06-25",
                            dims=("time", "field", "asset"),
                            assets=['NYSE:WMT'],
                            forward_order=True)

    data_lbls = ['total_revenue']
    fun_indicators = qnt.data.secgov_load_indicators([WMT], time_coord=data.time, standard_indicators=data_lbls)

    # def save_object(obj, filename):
    #     with open(filename, 'wb') as output:  # Overwrites any existing file.
    #         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    #
    # save_object(fun_indicators, 'fundamental_NYSE_WMT_total_revenue.pkl')

    NYSE_WMT = get_data_filter(fun_indicators, ['NYSE:WMT'])

    q = NYSE_WMT.sel(field='total_revenue').to_pandas()
    return q


def print_adecvate_values(df_):
    df_copy = df_.copy(True)
    print(df_copy.columns)
    df_copy['NYSE:WMT'] = df_copy['NYSE:WMT'] / 1000000
    print(df_copy)


class TestBaseStatistic(unittest.TestCase):
    maxDiff = None

    def test_total_revenue(self):
        dir = os.path.abspath(os.curdir)
        with open(dir + '/data/fundamental_NYSE_WMT_total_revenue.pkl', 'rb') as input:
            total_revenue_default = pickle.load(input)
        NYSE_WMT = total_revenue_default.sel(asset=['NYSE:WMT'])
        NYSE_WMT_default = NYSE_WMT.sel(field='total_revenue').to_pandas()

        total_revenue = get_data()
        NYSE_WMT_df = total_revenue.T
        print_adecvate_values(NYSE_WMT_df)

        revenue = total_revenue.to_json(orient="table")
        self.assertEqual(NYSE_WMT_default.to_json(orient="table"), revenue)


if __name__ == '__main__':
    unittest.main()
