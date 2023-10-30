import unittest
import pandas as pd

import json
import os

import pandas as pd
import xarray as xr
import numpy as np

os.environ['API_KEY'] = "default"

pd.set_option('display.max_rows', None)


class TestBaseFundamentalData(unittest.TestCase):
    maxDiff = None

    def test_total_revenue_wmt_quarter(self):
        wmt_indicators = get_data_wmt()
        indicator = wmt_indicators.sel(field='total_revenue_qf').to_pandas()
        print_normed(indicator)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf page 61
        self.assertEqual(114826, indicator.loc['2015-06-05'].max() / 1000000)  # q1 2016
        self.assertEqual(120229, indicator.loc['2015-09-09'].max() / 1000000)  # q2
        self.assertEqual(117408, indicator.loc['2015-12-02'].max() / 1000000)  # q3
        self.assertEqual(129667, indicator.loc['2016-03-30'].max() / 1000000)  # q4
        self.assertEqual(115904, indicator.loc['2016-06-03'].max() / 1000000)  # q1 2017
        self.assertEqual(120854, indicator.loc['2016-08-31'].max() / 1000000)  # q2
        self.assertEqual(118179, indicator.loc['2016-12-01'].max() / 1000000)  # q3
        self.assertEqual(118179, indicator.loc['2017-03-30'].max() / 1000000)  # q3
        self.assertEqual(130936, indicator.loc['2017-03-31'].max() / 1000000)  # q4

    def test_total_revenue_wmt_annual(self):
        wmt_indicators = get_data_wmt()
        indicator = wmt_indicators.sel(field='total_revenue_af').to_pandas()
        print_normed(indicator)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 61
        self.assertEqual(482130, indicator.loc['2016-03-31'].max() / 1000000)
        self.assertEqual(482130, indicator.loc['2016-12-01'].max() / 1000000)
        self.assertEqual(482130, indicator.loc['2017-03-30'].max() / 1000000)

        self.assertEqual(485873, indicator.loc['2017-03-31'].max() / 1000000)
        self.assertEqual(485873, indicator.loc['2018-03-29'].max() / 1000000)
        self.assertEqual(523964, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(559151, indicator.loc['2021-03-19'].max() / 1000000)

    def test_total_revenue_wmt_ltm(self):
        wmt_indicators = get_data_wmt()
        indicator = wmt_indicators.sel(field='total_revenue').to_pandas()
        print_normed(indicator)

        revenue = indicator.head(10).to_json(orient="table")
        NYSE_WMT_default = get_default_WMT_total_revenue().head(10).to_json(orient="table")
        self.assertEqual(json.loads(NYSE_WMT_default), json.loads(revenue))
        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 61
        self.assertEqual(482130, indicator.loc['2016-03-31'].max() / 1000000)
        self.assertEqual(484604, indicator.loc['2016-12-01'].max() / 1000000)  # 485873 - 130936 + 129667
        self.assertEqual(484604, indicator.loc['2017-03-30'].max() / 1000000)  # 485873 - 130936 + 129667
        self.assertEqual(485873, indicator.loc['2017-03-31'].max() / 1000000)

    def test_liabilities_wmt(self):
        name_indicator = 'liabilities'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 61
        self.assertEqual(126382, indicator.loc['2017-03-30'].max() / 1000000)
        self.assertEqual(118290, indicator.loc['2017-03-31'].max() / 1000000)
        self.assertEqual(123604, indicator.loc['2017-06-02'].max() / 1000000)
        self.assertEqual(122520, indicator.loc['2017-08-31'].max() / 1000000)
        self.assertEqual(130508, indicator.loc['2017-12-01'].max() / 1000000)

        self.assertEqual(123700, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(139661, indicator.loc['2019-03-28'].max() / 1000000)

        # https://s2.q4cdn.com/056532643/files/doc_financials/2020/ar/Walmart_2020_Annual_Report.pdf page 54
        self.assertEqual(154943, indicator.loc['2020-03-31'].max() / 1000000)
        self.assertEqual(154943, indicator.loc['2020-03-31'].max() / 1000000)
        self.assertEqual(164965, indicator.loc['2021-03-31'].max() / 1000000)
        self.assertEqual(151989, indicator.loc['2021-06-04'].max() / 1000000)

    def test_cash_and_cash_equivalents_wmt(self):
        name_indicator = 'cash_and_cash_equivalents'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 40
        self.assertEqual(8705, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(6867, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(6756, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(7722, indicator.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(9465, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(17741, indicator.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(14930, indicator.loc['2020-06-03'].max() / 1000000)
        self.assertEqual(16906, indicator.loc['2020-09-02'].max() / 1000000)
        self.assertEqual(14325, indicator.loc['2020-12-02'].max() / 1000000)

    def test_assets_wmt(self):
        name_indicator = 'assets'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 40
        self.assertEqual(199581, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(198825, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(204522, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(219295, indicator.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(236495, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(252496, indicator.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(232892, indicator.loc['2020-06-03'].max() / 1000000)
        self.assertEqual(237382, indicator.loc['2020-09-02'].max() / 1000000)
        self.assertEqual(250863, indicator.loc['2020-12-02'].max() / 1000000)

    def test_assets_apple(self):
        name_indicator = 'assets'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(290479, indicator.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(321686, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(375319, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(365725, indicator.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(338516, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(323888, indicator.loc['2020-10-30'].max() / 1000000)
        self.assertEqual(337158, indicator.loc['2021-04-29'].max() / 1000000)

    def test_equity_wmt(self):
        name_indicator = 'equity'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 40
        self.assertEqual(83611, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(80535, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(80822, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(79634, indicator.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(81552, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(87531, indicator.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(74110, indicator.loc['2020-06-03'].max() / 1000000)
        self.assertEqual(81197, indicator.loc['2020-09-02'].max() / 1000000)
        self.assertEqual(87504, indicator.loc['2020-12-02'].max() / 1000000)

    def test_income_before_taxes_wmt(self):
        name_indicator = 'income_before_taxes'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(24799, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(21638, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(20497, indicator.loc['2017-03-31 '].max() / 1000000)

    def test_net_income_wmt(self):
        name_indicator = 'net_income'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(16363, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(14694, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(13643, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(13510, indicator.loc['2021-03-19'].max() / 1000000)
        self.assertEqual(12250, indicator.loc['2021-06-04'].max() / 1000000)

    def test_operating_income_wmt(self):
        name_indicator = 'operating_income'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(27147, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(24105, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(22764, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(20437, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(21957, indicator.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(20568, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(22548, indicator.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(5322 + 5224 + 6059 + 5778, indicator.loc['2020-12-02'].max() / 1000000)
        self.assertEqual(24233, indicator.loc['2021-06-04'].max() / 1000000)

    def test_interest_net_wmt(self):
        name_indicator = 'interest_net'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(2348, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(2467, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(2267, indicator.loc['2017-03-31 '].max() / 1000000)

    def test_depreciation_and_amortization_wmt(self):
        name_indicator = 'depreciation_and_amortization'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        # https://s2.q4cdn.com/056532643/files/doc_financials/2020/ar/Walmart_2020_Annual_Report.pdf 35
        self.assertEqual(9173, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(9454, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(10080, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(10678, indicator.loc['2019-03-28 '].max() / 1000000)
        self.assertEqual(10987, indicator.loc['2020-03-20 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2021/ar/WMT_2021_AnnualReport.pdf 81
        self.assertEqual(11152, indicator.loc['2021-03-19'].max() / 1000000)

        # https://s2.q4cdn.com/056532643/files/doc_financials/2022/q1/Earnings-Release-(FY22-Q1).pdf page 5
        self.assertEqual(11152 + 2661 - 2791, indicator.loc['2021-06-04 '].max() / 1000000)

    def test_income_interest_wmt(self):
        name_indicator = 'income_interest'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(113, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(81, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(100, indicator.loc['2017-03-31 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2020/ar/Walmart_2020_Annual_Report.pdf page 52
        self.assertEqual(152, indicator.loc['2018-04-02 '].max() / 1000000)
        self.assertEqual(217, indicator.loc['2019-03-28 '].max() / 1000000)
        self.assertEqual(189, indicator.loc['2020-03-20 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2021/ar/WMT_2021_AnnualReport.pdf page 56
        self.assertEqual(121, indicator.loc['2021-03-19 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2022/q1/Earnings-Release-(FY22-Q1).pdf page 5
        self.assertEqual(121 + 30 - 43, indicator.loc['2021-06-04 '].max() / 1000000)

    def test_nonoperating_income_expense_wmt(self):
        name_indicator = 'nonoperating_income_expense'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(-8368, indicator.loc['2019-03-28 '].max() / 1000000)
        self.assertEqual(1958, indicator.loc['2020-03-20 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2021/ar/WMT_2021_AnnualReport.pdf page 56
        self.assertEqual(210, indicator.loc['2021-03-19 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2022/q1/Earnings-Release-(FY22-Q1).pdf page 5
        self.assertEqual(210 + (-2529 - 721), indicator.loc['2021-06-04 '].max() / 1000000)

    def test_ebitda_use_operating_income_wmt(self):
        name_indicator = 'ebitda_use_operating_income'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(36433, indicator.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(33640, indicator.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(32944, indicator.loc['2017-03-31 '].max() / 1000000)

        self.assertEqual(27982, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(24484, indicator.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(33702, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(34031, indicator.loc['2021-03-19'].max() / 1000000)

        # self.assertEqual(5322 + 5224 + 6059 + 5778, total_liabilities.T.loc['2020-12-02'].max() / 1000000)
        self.assertEqual(32323, indicator.loc['2021-06-04'].max() / 1000000)

    def test_shares_wmt(self):
        name_indicator = 'shares'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(3033.009079, indicator.loc['2017-03-31 '].max() / 1000000)

        self.assertEqual(2950.696818, indicator.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(2869.684230, indicator.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(2832.277220, indicator.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(2817.071695, indicator.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(2802.145927, indicator.loc['2021-06-04'].max() / 1000000)

    def test_market_capitalization_wmt(self):
        name_indicator = 'market_capitalization'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(218619, round(indicator.loc['2017-03-31 '].max() / 1000000))

        self.assertEqual(252432, round(indicator.loc['2018-04-02'].max() / 1000000))
        self.assertEqual(278732, round(indicator.loc['2019-03-28'].max() / 1000000))
        self.assertEqual(322795, round(indicator.loc['2020-03-20'].max() / 1000000))
        self.assertEqual(371121, round(indicator.loc['2021-03-19'].max() / 1000000))

        self.assertEqual(397484, round(indicator.loc['2021-06-04'].max() / 1000000))

    def test_eps_wmt(self):
        name_indicator = 'eps'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(4.39, indicator.loc['2017-03-31 '].max())

        self.assertEqual(3.27, indicator.loc['2018-04-02'].max())
        self.assertEqual(2.2800000000000002, indicator.loc['2019-03-28'].max())
        self.assertEqual(5.1899999999999995, indicator.loc['2020-03-20'].max())
        self.assertEqual(4.739999999999999, indicator.loc['2021-03-19'].max())

        self.assertEqual(4.31, indicator.loc['2021-06-04'].max())

    def test_debt_wmt(self):
        name_indicator = 'debt'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(1099 + 2256 + 565 + 36015 + 6003, indicator.loc['2017-03-31 '].max() / 1000000)

        # https://www.sec.gov/cgi-bin/browse-edgar?filenum=001-06991&action=getcompany

        # <us-gaap:OperatingLeaseLiabilityCurrent contextRef="FI2020Q4" decimals="-6" id="d14043724e1814-wk-Fact-792744BB9A725D81B56097016860A62A" unitRef="usd">1793000000</us-gaap:OperatingLeaseLiabilityCurrent>
        self.assertEqual(72433, indicator.loc['2020-03-20 '].max() / 1000000)

    def test_ev_tesla(self):
        name_indicator = 'ev'
        name_asset = 'NASDAQ:TSLA'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertAlmostEqual(11404.127117677155, indicator.loc['2017-03-01 '].max() / 1000000)  # did not check

    def test_net_debt_divide_by_ebitda_wmt(self):
        name_indicator = 'net_debt_divide_by_ebitda'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(1.1895932285957862, indicator.loc['2017-03-31 '].max())
        self.assertEqual(1.995499920773253, indicator.loc['2020-03-20 '].max())

    def test_net_debt_wmt(self):
        name_indicator = 'net_debt'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(39071, indicator.loc['2017-03-31 '].max() / 1000000)

        # https://www.sec.gov/cgi-bin/browse-edgar?filenum=001-06991&action=getcompany

        # <us-gaap:OperatingLeaseLiabilityCurrent contextRef="FI2020Q4" decimals="-6" id="d14043724e1814-wk-Fact-792744BB9A725D81B56097016860A62A" unitRef="usd">1793000000</us-gaap:OperatingLeaseLiabilityCurrent>
        self.assertEqual(62968, indicator.loc['2020-03-20 '].max() / 1000000)

    def test_ev_wmt(self):
        name_indicator = 'ev'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator)

        self.assertAlmostEqual(257690.2944143203, indicator.loc['2017-03-31 '].max() / 1000000)
        self.assertAlmostEqual(437460.39974495, indicator.loc['2021-06-04'].max() / 1000000)

    def test_ev_divide_by_ebitda_wmt(self):
        name_indicator = 'ev_divide_by_ebitda'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertAlmostEqual(7.845886445448797, indicator.loc['2017-03-31 '].max())
        self.assertAlmostEqual(12.40846404041838, indicator.loc['2021-06-04'].max())

    def test_p_divide_by_bv_wmt(self):
        name_indicator = 'p_divide_by_bv'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertAlmostEqual(2.714587377094683, indicator.loc['2017-03-31 '].max())

    def test_p_divide_by_s_wmt(self):
        name_indicator = 'p_divide_by_s'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertAlmostEqual(0.4499515190478176, indicator.loc['2017-03-31 '].max())

    def test_ev_divide_by_s_wmt(self):
        name_indicator = 'ev_divide_by_s'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertAlmostEqual(0.5303655367026369, indicator.loc['2017-03-31 '].max())

    def test_roe_wmt(self):
        name_indicator = 'roe'
        name_asset = 'NYSE:WMT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertEqual(0.1694046066927423, indicator.loc['2017-03-31 '].max())

    def test_equity_dal(self):
        # https://www.sec.gov/cgi-bin/viewer?action=view&cik=27904&accession_number=0000027904-21-000003&xbrl_type=v#
        name_indicator = 'equity'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(12287000000, indicator.loc['2017-02-13 '].max())
        self.assertEqual(13910000000, indicator.loc['2018-02-23 '].max())
        self.assertEqual(13687000000, indicator.loc['2019-02-15'].max())
        self.assertEqual(15358000000, indicator.loc['2020-02-13'].max())
        self.assertEqual(1534000000, indicator.loc['2021-02-18'].max())

    def test_net_income_dal(self):
        # https://www.sec.gov/cgi-bin/viewer?action=view&cik=27904&accession_number=0000027904-21-000003&xbrl_type=v#
        name_indicator = 'net_income'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(4373000000, indicator.loc['2017-02-13 '].max())
        self.assertEqual(3577000000, indicator.loc['2018-02-23 '].max())
        self.assertEqual(3935000000, indicator.loc['2019-02-15'].max())
        self.assertEqual(4767000000, indicator.loc['2020-02-13'].max())
        self.assertEqual(-12385000000, indicator.loc['2021-02-18'].max())

    def test_ev_dal(self):
        name_indicator = 'ev'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(40665.37813828, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(43986.58811868, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(42684, round(indicator.loc['2019-02-15'].max() / 1000000))
        self.assertEqual(45924, round(indicator.loc['2020-02-13'].max() / 1000000))
        self.assertEqual(43382, round(indicator.loc['2021-02-18'].max() / 1000000))

    def test_ebitda_simple_dal(self):
        name_indicator = 'ebitda_simple'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(8854, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(8349, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(7593, indicator.loc['2019-02-15'].max() / 1000000)
        self.assertEqual(9199, indicator.loc['2020-02-13'].max() / 1000000)
        self.assertEqual(-10157, indicator.loc['2021-02-18'].max() / 1000000)

    def test_ev_divide_by_ebitda_dal(self):
        name_indicator = 'ev_divide_by_ebitda'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(5, round(indicator.loc['2017-02-13 '].max()))
        self.assertEqual(5, round(indicator.loc['2018-02-23 '].max()))
        self.assertEqual(5.62146332224944, indicator.loc['2019-02-15'].max())
        self.assertEqual(5, round(indicator.loc['2020-02-13'].max()))
        self.assertEqual(-4, round(indicator.loc['2021-02-18'].max()))

    def test_net_debt_divide_by_ebitda_dal(self):
        name_indicator = 'net_debt_divide_by_ebitda'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(0, round(indicator.loc['2017-02-13 '].max()))
        self.assertEqual(1, round(indicator.loc['2018-02-23 '].max()))
        self.assertEqual(1.053997102594495, indicator.loc['2019-02-15'].max())
        self.assertEqual(1, round(indicator.loc['2020-02-13'].max()))
        self.assertEqual(-1, round(indicator.loc['2021-02-18'].max()))

    def test_roe_dal(self):
        name_indicator = 'roe'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(4373000000 / 12287000000, indicator.loc['2017-02-13 '].max())

    def test_p_divide_by_e_dal(self):
        name_indicator = 'p_divide_by_e'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print(indicator)

        self.assertEqual(8.365510665053739, indicator.loc['2017-02-13 '].max())

    def test_debt_dal(self):
        name_indicator = 'debt'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(7332, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(8834, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(9771, indicator.loc['2019-02-15'].max() / 1000000)
        self.assertEqual(11160, indicator.loc['2020-02-13'].max() / 1000000)
        self.assertEqual(29157, indicator.loc['2021-02-18'].max() / 1000000)

    def test_cash_and_cash_equivalents_dal(self):
        name_indicator = 'cash_and_cash_equivalents'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(2762, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(1814, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(1565, indicator.loc['2019-02-15'].max() / 1000000)
        self.assertEqual(2882, indicator.loc['2020-02-13'].max() / 1000000)
        self.assertEqual(8307, indicator.loc['2021-02-18'].max() / 1000000)

    def test_cash_and_cash_equivalents_full_dal(self):
        # https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum=001-05424&type=10-K&dateb=&owner=include&count=40&search_text=
        name_indicator = 'cash_and_cash_equivalents_full'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(2762 + 487, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(2639, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(1768, indicator.loc['2019-02-15'].max() / 1000000)
        # self.assertEqual(2882, indicator.loc['2020-02-13'].max() / 1000000) see test_short_term_investments_special_case_dal
        self.assertEqual(14096, indicator.loc['2021-02-18'].max() / 1000000)

    def test_short_term_investments_dal(self):
        name_indicator = 'short_term_investments'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(487, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(825, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(203, indicator.loc['2019-02-15'].max() / 1000000)
        self.assertEqual(5789, indicator.loc['2021-02-18'].max() / 1000000)

    # def test_short_term_investments_special_case_dal(self):
    #     name_indicator = 'short_term_investments'
    #     name_asset = 'NYSE:DAL'
    #     wmt_indicators = get_data_new_for([name_asset], [name_indicator])
    #     indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
    #     print_normed(indicator, name_asset)
    #     self.assertEqual(0, indicator.loc['2020-02-13'].max() / 1000000)

    def test_net_debt_dal(self):
        name_indicator = 'net_debt'
        name_asset = 'NYSE:DAL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(7332 - 3249, indicator.loc['2017-02-13 '].max() / 1000000)
        self.assertEqual(6195, indicator.loc['2018-02-23 '].max() / 1000000)
        self.assertEqual(8003, indicator.loc['2019-02-15'].max() / 1000000)
        # self.assertEqual(8278, indicator.loc['2020-02-13'].max() / 1000000)
        self.assertEqual(27425 + 1732 - (8307 + 5789), indicator.loc['2021-02-18'].max() / 1000000)

    def test_debt_apple(self):
        name_indicator = 'debt'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(87027, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(115680, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(114483, indicator.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(108047, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(122278, indicator.loc['2020-10-30'].max() / 1000000)  # + OperatingLease + FinanceLease

    # def test_debt_apple(self):
    #     name_indicator = 'debt'
    #     name_asset = 'NASDAQ:HSIC'
    #     wmt_indicators = get_data_new_for([name_asset], [name_indicator])
    #     indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
    #     print_normed(indicator, name_asset)
    #
    #     self.assertEqual(87027, indicator.loc['2020-02-20'].max() / 1000000)

    # def test_ev_apple(self):
    #     name_indicator = 'ev'
    #     name_asset = 'NASDAQ:AAPL'
    #     wmt_indicators = get_data_new_for([name_asset], [name_indicator])
    #     indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
    #     print(indicator)
    #
    #     self.assertEqual(87027 - 67155, indicator.loc['2016-10-26'].max() / 1000000) #19877
    #     self.assertEqual(41499, indicator.loc['2017-11-03'].max() / 1000000)
    #     self.assertEqual(48182, indicator.loc['2018-11-05'].max() / 1000000)
    #     # self.assertEqual(7490, indicator.loc['2019-10-31'].max() / 1000000)
    #     self.assertEqual(21493, indicator.loc['2020-10-30'].max() / 1000000)

    def test_cash_and_cash_equivalents_full_apple(self):
        # https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-19-000119&xbrl_type=v#
        name_indicator = 'cash_and_cash_equivalents_full'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(67155, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(74181, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(66301, indicator.loc['2018-11-05'].max() / 1000000)
        # self.assertEqual(100557, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(90943, indicator.loc['2020-10-30'].max() / 1000000)  # + OperatingLease + FinanceLease

    def test_depreciation_and_amortization_ASRT(self):
        # https://www.sec.gov/cgi-bin/viewer?action=view&cik=1005201&accession_number=0001005201-19-000040&xbrl_type=v
        name_indicator = 'depreciation_and_amortization'
        name_asset = 'NASDAQ:ASRT'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(109.375, indicator.loc['2017-02-24'].max() / 1000000)
        self.assertEqual(105.502, indicator.loc['2018-03-01'].max() / 1000000)
        self.assertEqual(106.426, indicator.loc['2019-03-11'].max() / 1000000)
        self.assertEqual(102.946, indicator.loc['2020-03-10'].max() / 1000000)

    # def test_perform_wmt(self):
    #     import qnt.data    as qndata
    #     import qnt.data.secgov_fundamental
    #     name_indicator = 'total_revenue'
    #     name_asset = 'NYSE:WMT'
    #     data = qndata.stocks_load_data(tail=5 * 365, dims=("time", "field", "asset"))
    #
    #     name_assets = data.asset.to_pandas().to_list()
    #
    #     import qnt.data    as qndata
    #     import qnt.data.secgov_fundamental
    #
    #     market_data = qndata.load_data(min_date="2010-01-01", max_date="2021-07-28",
    #                                    dims=("time", "field", "asset"),
    #                                    assets=name_assets,
    #                                    forward_order=True)
    #
    #     wmt_indicators = qnt.data.load_fundamental_indicators_for(market_data)
    #
    #     # wmt_indicators = get_data_new_for(name_assets, qnt.data.get_all_indicators())
    #     # wmt_indicators_df = wmt_indicators.asset
    #     print(wmt_indicators.asset)
    #     indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
    #     print(indicator.T)
    #
    #     self.assertEqual(0.1694046066927423, indicator.loc['2017-03-31 '].max())

    # def test_depreciation_and_amortization_microsoft(self):
    #     wmt_indicators = get_data_new_for(['NASDAQ:MSFT'])
    #     ebitda = wmt_indicators.sel(field='depreciation_and_amortization').to_pandas()
    #     print_normed(ebitda.T, 'NASDAQ:MSFT')
    #
    #     # https://www.sec.gov/Archives/edgar/data/789019/000156459020034944/msft-10k_20200630_htm.xml
    #     # https://www.sec.gov/Archives/edgar/data/789019/000156459020034944
    #     # https://www.sec.gov/edgar/browse/?CIK=789019&owner=exclude
    #     # https://www.sec.gov/edgar/search-and-access
    #     self.assertEqual(12796, ebitda.T.loc['2015-04-01'].max() / 1000000)

    def test_operating_income_apple(self):
        name_indicator = 'operating_income'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(71230, indicator.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(60024, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(61344, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(70898, indicator.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(63930, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(66288, indicator.loc['2020-10-30'].max() / 1000000)
        self.assertEqual(88903, indicator.loc['2021-04-29'].max() / 1000000)

    def test_depreciation_and_amortization_apple(self):
        name_indicator = 'depreciation_and_amortization'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(11257, indicator.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(10505, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(10157, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(10903, indicator.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(12547, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(11056, indicator.loc['2020-10-30'].max() / 1000000)

    def test_interest_expense_apple(self):
        name_indicator = 'interest_expense'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(733, indicator.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(1456, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(2323, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(3240, indicator.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(3576, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(2873, indicator.loc['2020-10-30'].max() / 1000000)

    def test_ebitda_use_operating_income_apple(self):
        name_indicator = 'ebitda_use_operating_income'
        name_asset = 'NASDAQ:AAPL'
        wmt_indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = wmt_indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(84505, indicator.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(73333, indicator.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(76569, indicator.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(87046, indicator.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(81860, indicator.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(81020, indicator.loc['2020-10-30'].max() / 1000000)
        # self.assertEqual(128026, indicator.loc['2021-04-29'].max() / 1000000)

    def test_operating_income_amazon(self):
        name_indicator = 'operating_income'
        name_asset = 'NASDAQ:AMZN'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(2233, indicator.loc['2016-01-29'].max() / 1000000)
        self.assertEqual(4186, indicator.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(4107, indicator.loc['2018-02-02'].max() / 1000000)
        self.assertEqual(12420, indicator.loc['2019-02-01'].max() / 1000000)
        self.assertEqual(14540, indicator.loc['2020-01-31'].max() / 1000000)
        self.assertEqual(22899, indicator.loc['2021-02-03'].max() / 1000000)
        self.assertEqual(27775, indicator.loc['2021-04-30'].max() / 1000000)

    def test_depreciation_and_amortization_amazon(self):
        name_indicator = 'depreciation_and_amortization'
        name_asset = 'NASDAQ:AMZN'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # https://www.stock-analysis-on.net/NASDAQ/Company/Amazoncom-Inc/Valuation/EV-to-EBITDA
        # self.assertEqual(2233, indicator.loc['2016-01-29'].max() / 1000000)
        self.assertEqual(8116, indicator.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(11478, indicator.loc['2018-02-02'].max() / 1000000)
        self.assertEqual(15341, indicator.loc['2019-02-01'].max() / 1000000)
        self.assertEqual(21789, indicator.loc['2020-01-31'].max() / 1000000)
        self.assertEqual(25251, indicator.loc['2021-02-03'].max() / 1000000)
        self.assertEqual(27397, indicator.loc['2021-04-30'].max() / 1000000)

    def test_ebitda_use_operating_income_amazon(self):
        name_indicator = 'ebitda_use_operating_income'
        name_asset = 'NASDAQ:AMZN'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        # https://www.stock-analysis-on.net/NASDAQ/Company/Amazoncom-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(8308, indicator.loc['2016-01-29'].max() / 1000000)
        self.assertEqual(12492, indicator.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(16132, indicator.loc['2018-02-02'].max() / 1000000)

        self.assertEqual(28020, indicator.loc['2019-02-01'].max() / 1000000)  # bt provide value = 26416
        self.assertEqual(37364, indicator.loc['2020-01-31'].max() / 1000000)
        self.assertEqual(51076, indicator.loc['2021-02-03'].max() / 1000000)

        self.assertEqual(60104, indicator.loc['2021-04-30'].max() / 1000000)

    def test_operating_income_fb(self):
        name_indicator = 'operating_income'
        name_asset = 'NASDAQ:FB'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(12428, indicator.loc['2017-02-03'].max() / 1000000)
        self.assertEqual(20202, indicator.loc['2018-02-01'].max() / 1000000)
        self.assertEqual(24913, indicator.loc['2019-01-31'].max() / 1000000)
        self.assertEqual(23986, indicator.loc['2020-01-30'].max() / 1000000)
        self.assertEqual(32671, indicator.loc['2021-01-28'].max() / 1000000)

        self.assertEqual(38156, indicator.loc['2021-04-29'].max() / 1000000)

    def test_depreciation_and_amortization_fb(self):
        name_indicator = 'depreciation_and_amortization'
        name_asset = 'NASDAQ:FB'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(2342, indicator.loc['2017-02-03'].max() / 1000000)
        self.assertEqual(3025, indicator.loc['2018-02-01'].max() / 1000000)
        self.assertEqual(4315, indicator.loc['2019-01-31'].max() / 1000000)
        self.assertEqual(5741, indicator.loc['2020-01-30'].max() / 1000000)
        self.assertEqual(6862, indicator.loc['2021-01-28'].max() / 1000000)

    def test_ebitda_use_operating_income_fb(self):
        name_indicator = 'ebitda_use_operating_income'
        name_asset = 'NASDAQ:FB'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(14869, indicator.loc['2017-02-03'].max() / 1000000)
        self.assertEqual(23625, indicator.loc['2018-02-01'].max() / 1000000)
        self.assertEqual(29685, indicator.loc['2019-01-31'].max() / 1000000)
        self.assertEqual(30573, indicator.loc['2020-01-30'].max() / 1000000)
        self.assertEqual(40062, indicator.loc['2021-01-28'].max() / 1000000)  # 39533 bt value

        # self.assertEqual(45393, indicator.loc['2021-04-29'].max() / 1000000)bt value

    # def test_ebitda_use_income_before_taxes_fb(self):
    #     indicators = get_data_new_for(['NASDAQ:FB'])
    #     indicator = indicators.sel(field='ebitda_use_income_before_taxes').to_pandas()
    #     print_normed(indicator, 'NASDAQ:FB')
    #
    #     # https://www.stock-analysis-on.net/NASDAQ/Company/Facebook-Inc/Valuation/EV-to-EBITDA
    #     self.assertEqual(14859, indicator.loc['2017-02-03'].max() / 1000000)
    #     self.assertEqual(23619, indicator.loc['2018-02-01'].max() / 1000000)
    #     self.assertEqual(29676, indicator.loc['2019-01-31'].max() / 1000000)
    #     self.assertEqual(30553, indicator.loc['2020-01-30'].max() / 1000000)
    #     self.assertEqual(40042, indicator.loc['2021-01-28'].max() / 1000000)

    # def test_ebitda_use_income_before_taxes_wmt(self):
    #     indicators = get_data_new_for(['NYSE:WMT'])
    #     indicator = indicators.sel(field='ebitda_use_income_before_taxes').to_pandas()
    #     print_normed(indicator, 'NYSE:WMT')
    #
    #     # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
    #     self.assertEqual(36433, indicator.loc['2015-04-01'].max() / 1000000)
    #     self.assertEqual(33640, indicator.loc['2016-03-30 '].max() / 1000000)
    #     self.assertEqual(32944, indicator.loc['2017-03-31 '].max() / 1000000)
    #
    #     self.assertEqual(27982, indicator.loc['2018-04-02'].max() / 1000000)
    #     self.assertEqual(24484, indicator.loc['2019-03-28'].max() / 1000000)
    #     self.assertEqual(33702, indicator.loc['2020-03-20'].max() / 1000000)
    #     self.assertEqual(34031, indicator.loc['2021-03-19'].max() / 1000000)
    #
    #     # self.assertEqual(5322 + 5224 + 6059 + 5778, total_liabilities.T.loc['2020-12-02'].max() / 1000000)
    #     self.assertEqual(32323, indicator.loc['2021-06-04'].max() / 1000000)

    def test_operating_income_visa(self):
        name_indicator = 'operating_income'
        name_asset = 'NYSE:V'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(7883, indicator.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(12144, indicator.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(12954, indicator.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(15001, indicator.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(14081, indicator.loc['2020-11-19'].max() / 1000000)

    def test_depreciation_and_amortization_visa(self):
        name_indicator = 'depreciation_and_amortization'
        name_asset = 'NYSE:V'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(502, indicator.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(556, indicator.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(613, indicator.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(656, indicator.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(767, indicator.loc['2020-11-19'].max() / 1000000)

    # def test_income_before_income_taxes_visa(self):
    #     indicators = get_data_new_for(['NYSE:V'])
    #     indicator = indicators.sel(field='income_before_income_taxes').to_pandas()
    #     print_normed(indicator, 'NYSE:V')
    #
    #     self.assertEqual(502, indicator.loc['2016-11-15'].max() / 1000000)
    #     self.assertEqual(556, indicator.loc['2017-11-17'].max() / 1000000)
    #     self.assertEqual(613, indicator.loc['2018-11-16'].max() / 1000000)
    #     self.assertEqual(656, indicator.loc['2019-11-14'].max() / 1000000)
    #     self.assertEqual(767, indicator.loc['2020-11-19'].max() / 1000000)

    # def test_ebitda_use_operating_income_visa(self):
    #     import qnt.data    as qndata
    #     futures_server = qndata.load_data(min_date="2020-01-01")
    #     indicators = get_data_new_for(['NYSE:V'])
    #     indicator = indicators.sel(field='ebitda_use_operating_income').to_pandas()
    #     print_normed(indicator, 'NYSE:V')
    #
    #     self.assertEqual(8941, indicator.loc['2016-11-15'].max() / 1000000)
    #     self.assertEqual(12813, indicator.loc['2017-11-17'].max() / 1000000)
    #     self.assertEqual(14031, indicator.loc['2018-11-16'].max() / 1000000)
    #     self.assertEqual(16073, indicator.loc['2019-11-14'].max() / 1000000)
    #     self.assertEqual(15073, indicator.loc['2020-11-19'].max() / 1000000)

    def test_ebitda_use_income_before_taxes_visa(self):
        name_indicator = 'ebitda_use_income_before_taxes'
        name_asset = 'NYSE:V'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(8941, indicator.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(12813, indicator.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(14031, indicator.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(16073, indicator.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(15073, indicator.loc['2020-11-19'].max() / 1000000)

    def test_income_before_taxes_visa(self):
        name_indicator = 'income_before_taxes'
        name_asset = 'NYSE:V'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(8012, indicator.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(11694, indicator.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(12806, indicator.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(14884, indicator.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(13790, indicator.loc['2020-11-19'].max() / 1000000)

    def test_operating_income_nvidia(self):
        name_indicator = 'operating_income'
        name_asset = 'NASDAQ:NVDA'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(1934, indicator.loc['2017-03-01'].max() / 1000000)
        self.assertEqual(3210, indicator.loc['2018-02-28'].max() / 1000000)
        self.assertEqual(3804, indicator.loc['2019-02-21'].max() / 1000000)
        self.assertEqual(2846, indicator.loc['2020-02-20'].max() / 1000000)
        self.assertEqual(4532, indicator.loc['2021-02-26'].max() / 1000000)

    # def test_ebitda_nvidia(self):
    #     indicators = get_data_new_for(['NASDAQ:NVDA'])
    #     indicator = indicators.sel(field='ebitda_use_operating_income').to_pandas()
    #     print_normed(indicator, 'NASDAQ:NVDA')
    #
    #     self.assertEqual(2150, indicator.loc['2017-03-01'].max() / 1000000)
    #     self.assertEqual(3210, indicator.loc['2018-02-28'].max() / 1000000)
    #     self.assertEqual(3804, indicator.loc['2019-02-21'].max() / 1000000)
    #     self.assertEqual(2846, indicator.loc['2020-02-20'].max() / 1000000)
    #     self.assertEqual(4532, indicator.loc['2021-02-26'].max() / 1000000)

    def test_interest_income_expense_net_visa(self):
        name_indicator = 'interest_income_expense_net'
        name_asset = 'NYSE:V'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(-427, indicator.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(-563, indicator.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(-612, indicator.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(-533, indicator.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(-516, indicator.loc['2020-11-19'].max() / 1000000)

    def test_other_nonoperating_income_expense_visa(self):
        name_indicator = 'other_nonoperating_income_expense'
        name_asset = 'NYSE:V'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(556, indicator.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(113, indicator.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(464, indicator.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(416, indicator.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(225, indicator.loc['2020-11-19'].max() / 1000000)

    def test_nonoperating_income_expense_amazon(self):
        name_indicator = 'nonoperating_income_expense'
        name_asset = 'NASDAQ:AMZN'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print_normed(indicator, name_asset)

        self.assertEqual(-294, indicator.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(-301, indicator.loc['2018-02-02'].max() / 1000000)
        self.assertEqual(-565, indicator.loc['2020-01-31'].max() / 1000000)

    def test_liabilities_divide_by_ebitda_wmt(self):
        name_indicator = 'liabilities_divide_by_ebitda'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertEqual(3.829061382778889, indicator.loc['2017-02-10'].max())
        self.assertEqual(3.990094166564755, indicator.loc['2018-02-02'].max())
        self.assertEqual(5.015774927801758, indicator.loc['2020-01-31'].max())

    def test_p_divide_by_e_wmt(self):
        name_indicator = 'p_divide_by_e'
        name_asset = 'NYSE:WMT'
        indicators = get_data_new_for([name_asset], [name_indicator])
        indicator = indicators.sel(field=name_indicator).to_pandas()
        print(indicator.T)

        self.assertEqual(14.456321204295989, indicator.loc['2017-02-10'].max())
        self.assertEqual(27.045579637679136, indicator.loc['2018-02-02'].max())
        self.assertEqual(22.51529482377764, indicator.loc['2020-01-31'].max())

    # def test_ev_tesla_last_day(self):
    #     import qnt.data    as qndata
    #     import qnt.data.secgov_fundamental
    #     import qnt.graph   as qngraph
    #
    #     market_data = qndata.stocks_load_data(tail=2 * 365, dims=("time", "field", "asset"), assets=['NYSE:GM'])
    #     indicators = qnt.data.load_fundamental_indicators_for(market_data, ['net_debt'])
    #
    #     TSLA = indicators.sel(asset='NYSE:GM')
    #     print(TSLA)
    #
    #     self.assertEqual(14.456321204295989, indicators.loc['2017-02-10'].max())
    #     self.assertEqual(27.045579637679136, indicators.loc['2018-02-02'].max())
    #     self.assertEqual(22.51529482377764, indicators.loc['2020-01-31'].max())

    def test_all_indicators_list(self):
        import qnt.data.secgov_fundamental as fundamental
        all_indicators = fundamental.get_all_indicator_names()

        self.assertEqual(
            ['total_revenue',
             'liabilities',
             'assets',
             'equity',
             'net_income',
             'short_term_investments',
             'cash_and_cash_equivalents',
             'cash_and_cash_equivalents_full',
             'operating_income',
             'income_before_taxes',
             'income_before_income_taxes',
             'depreciation_and_amortization',
             'interest_net',
             'income_interest',
             'interest_expense',
             'interest_expense_debt',
             'interest_expense_capital_lease',
             'interest_income_expense_net',
             'losses_on_extinguishment_of_debt',
             'nonoperating_income_expense',
             'other_nonoperating_income_expense',
             'debt',
             'net_debt',
             'eps',
             'shares',
             'market_capitalization',
             'ebitda_use_income_before_taxes',
             'ebitda_use_operating_income',
             'ebitda_simple',
             'ev',
             'ev_divide_by_ebitda',
             'liabilities_divide_by_ebitda',
             'net_debt_divide_by_ebitda',
             'p_divide_by_e',
             'p_divide_by_bv',
             'p_divide_by_s',
             'ev_divide_by_s',
             'roe']

            , all_indicators)

        self.assertEqual(
            ['market_capitalization',
             'ebitda_use_income_before_taxes',
             'ebitda_use_operating_income',
             'ebitda_simple',
             'ev',
             'ev_divide_by_ebitda',
             'liabilities_divide_by_ebitda',
             'net_debt_divide_by_ebitda',
             'p_divide_by_e',
             'p_divide_by_bv',
             'p_divide_by_s',
             'ev_divide_by_s',
             'roe']
            , fundamental.get_complex_indicator_names())

        self.assertEqual(
            ['total_revenue',
             'liabilities',
             'assets',
             'equity',
             'net_income',
             'short_term_investments',
             'cash_and_cash_equivalents',
             'cash_and_cash_equivalents_full',
             'operating_income',
             'income_before_taxes',
             'income_before_income_taxes',
             'depreciation_and_amortization',
             'interest_net',
             'income_interest',
             'interest_expense',
             'interest_expense_debt',
             'interest_expense_capital_lease',
             'interest_income_expense_net',
             'losses_on_extinguishment_of_debt',
             'nonoperating_income_expense',
             'other_nonoperating_income_expense',
             'debt',
             'net_debt',
             'eps',
             'shares', ]
            , fundamental.get_standard_indicator_names())

        self.assertEqual(
            ['liabilities',
             'assets',
             'equity',
             'short_term_investments',
             'cash_and_cash_equivalents',
             'cash_and_cash_equivalents_full',
             'losses_on_extinguishment_of_debt',
             'debt',
             'net_debt',
             'shares',
             'market_capitalization',
             'ev',
             'p_divide_by_bv']
            , fundamental.get_annual_indicator_names())

    # def test_ebitda_use_income_before_taxes(self):
    #     name_indicator = 'ebitda_use_income_before_taxes'
    #     name_asset = 'NYSE:WMT'
    #     indicators = get_data_new_for([name_asset], [name_indicator])
    #     indicator = indicators.sel(field=name_indicator).to_pandas()
    #     print_normed(indicator, name_asset)
    #
    #     # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
    #     self.assertEqual(9173, indicator.loc['2015-04-01'].max() / 1000000)
    #     self.assertEqual(9454, indicator.loc['2016-03-30 '].max() / 1000000)
    #     self.assertEqual(10080, indicator.loc['2017-03-31 '].max() / 1000000)


def load_data_and_create_data_array(filename, dims, transpose_order):
    ds = xr.open_dataset(filename).load()
    dataset_name = list(ds.data_vars)[0]
    values = ds[dataset_name].transpose(*transpose_order).values
    coords = {dim: ds[dim].values for dim in dims}
    return xr.DataArray(values, dims=dims, coords=coords)


def get_default_WMT_total_revenue():
    dir = os.path.abspath(os.curdir)

    dims = ['time', 'field', 'asset']
    total_revenue_default = load_data_and_create_data_array(f"{dir}/data/fundamental_NYSE_WMT_total_revenue.nc", dims,
                                                            dims)

    NYSE_WMT = total_revenue_default.sel(asset=['NYSE:WMT'])
    NYSE_WMT_default = NYSE_WMT.sel(field='total_revenue').to_pandas()
    return NYSE_WMT_default


def print_normed(df_, ticker='NYSE:WMT'):
    df_copy = df_.copy(True)
    print(df_copy.columns)
    df_copy[ticker] = df_copy[ticker] / 1000000
    print(df_copy)


def get_data_wmt():
    import pandas as pd
    import qnt.data as qndata
    import datetime as dt

    import qnt.data.secgov_indicators

    pd.set_option('display.max_rows', None)

    def get_data_filter(data, assets):
        filler = data.sel(asset=assets)
        return filler

    import qnt.data as qndata
    assets = qndata.load_assets(min_date="2000-01-01", max_date="2021-06-25")

    WMT = {}
    DOCU = {}
    for a in assets:
        if a['id'] == 'NYSE:WMT':
            WMT = a
        if a['id'] == 'NASDAQ:DOCU':
            DOCU = a

    data = qndata.load_data(min_date="2010-01-01", max_date="2021-06-25",
                            dims=("time", "field", "asset"),
                            assets=['NYSE:WMT', 'NASDAQ:DOCU'],
                            forward_order=True)

    data_lbls = ['total_revenue',
                 'total_revenue_qf', 'total_revenue_af', 'liabilities', 'liabilities_curr', 'debt_lt', 'debt_st',
                 'liabilities_calculated']
    fun_indicators = qnt.data.secgov_load_indicators([WMT, DOCU], time_coord=data.time, standard_indicators=data_lbls)

    # import pickle
    # def save_object(obj, filename):
    #     with open(filename, 'wb') as output:  # Overwrites any existing file.
    #         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    #
    # save_object(fun_indicators, 'fundamental_NYSE_WMT_total_revenue.pkl')

    NYSE_WMT = get_data_filter(fun_indicators, ['NYSE:WMT'])
    return NYSE_WMT


def test_all1():
    import pandas as pd
    import qnt.data as qndata
    import datetime as dt

    import qnt.data.secgov_indicators

    pd.set_option('display.max_rows', None)
    assets = qndata.load_assets(min_date="2010-01-01", max_date="2021-06-25")

    WMT = {}
    for a in assets:
        if a['id'] == 'NYSE:WMT':
            WMT = a

    data = qndata.stocks.load_ndx_data(min_date="2010-01-01", max_date="2021-07-28",
                                       dims=("time", "field", "asset"),
                                       assets=['NYSE:WMT'],
                                       forward_order=True)

    data_lbls = ['assets', 'liabilities', 'operating_expense', 'ivestment_short_term', 'shares', 'invested_capital',
                 'assets_curr', 'equity', 'liabilities_curr', 'debt_lt', 'debt_st', 'goodwill', 'inventory', 'ppent',
                 'cash_equivalent',
                 'sales_revenue', 'total_revenue', 'cashflow_op', 'cogs', 'divs', 'eps', 'income', 'interest_expense',
                 'operating_income', 'rd_expense', 'sales_ps', 'sga_expense', 'sales_revenue_qf', 'sales_revenue_af',
                 'total_revenue_qf', 'total_revenue_af', 'cashflow_op_qf', 'cashflow_op_af', 'cogs_qf', 'cogs_af',
                 'divs_qf', 'divs_af',
                 'eps_qf', 'eps_af', 'income_qf', 'income_af', 'interest_expense_qf', 'interest_expense_af',
                 'operating_expense_qf',
                 'operating_expense_af', 'operating_income_qf', 'operating_income_af', 'rd_expense_qf', 'rd_expense_af',
                 'sales_ps_qf',
                 'sga_expense_qf', 'sga_expense_af']

    fun_data1 = qnt.data.secgov_load_indicators(assets, time_coord=data.time, standard_indicators=data_lbls)

    return fun_data1


def test_all2():
    import qnt.data as qndata
    import qnt.data.secgov_fundamental as fundamental

    market_data = qndata.stocks.load_ndx_data(min_date="2010-01-01", max_date="2021-09-28",
                                              dims=("time", "field", "asset"),
                                              forward_order=True)

    indicators = fundamental.load_indicators_for(market_data)

    return indicators


def get_data_new_for(asset_names, indicators_names):
    import qnt.data as qndata
    import qnt.data.secgov_fundamental as fundamental

    # import pickle

    # with open('stocks_all.pkl', 'rb') as handle:
    #     market_data = pickle.load(handle)
    #
    # market_data = market_data.sel(asset=asset_names)

    # market_data = qndata.load_data(min_date="2010-01-01", max_date="2021-09-28",
    #                                dims=("time", "field", "asset"),
    #                                assets=asset_names,
    #                                forward_order=True)

    market_data = qndata.stocks.load_ndx_data(min_date="2010-01-01", max_date="2021-09-28",
                                              dims=("time", "field", "asset"),
                                              assets=asset_names,
                                              forward_order=True)

    indicators = fundamental.load_indicators_for(market_data, indicators_names)

    return indicators


if __name__ == '__main__':
    unittest.main()
