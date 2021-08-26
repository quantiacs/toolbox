import unittest
import pandas as pd

import json
import os
import pickle

os.environ['API_KEY'] = "default"

pd.set_option('display.max_rows', None)


class TestBaseFundamentalData(unittest.TestCase):
    maxDiff = None

    def test_total_revenue_quarter(self):
        wmt_indicators = get_data_wmt()
        indicator = wmt_indicators.sel(field='total_revenue_qf').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf page 61
        self.assertEqual(114826, indicator.T.loc['2015-06-05'].max() / 1000000)  # q1 2016
        self.assertEqual(120229, indicator.T.loc['2015-09-09'].max() / 1000000)  # q2
        self.assertEqual(117408, indicator.T.loc['2015-12-02'].max() / 1000000)  # q3
        self.assertEqual(129667, indicator.T.loc['2016-03-30'].max() / 1000000)  # q4
        self.assertEqual(115904, indicator.T.loc['2016-06-03'].max() / 1000000)  # q1 2017
        self.assertEqual(120854, indicator.T.loc['2016-08-31'].max() / 1000000)  # q2
        self.assertEqual(118179, indicator.T.loc['2016-12-01'].max() / 1000000)  # q3
        self.assertEqual(118179, indicator.T.loc['2017-03-30'].max() / 1000000)  # q3
        self.assertEqual(130936, indicator.T.loc['2017-03-31'].max() / 1000000)  # q4

    def test_total_revenue_annual(self):
        wmt_indicators = get_data_wmt()
        indicator = wmt_indicators.sel(field='total_revenue_af').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 61
        self.assertEqual(482130, indicator.T.loc['2016-03-31'].max() / 1000000)
        self.assertEqual(482130, indicator.T.loc['2016-12-01'].max() / 1000000)
        self.assertEqual(482130, indicator.T.loc['2017-03-30'].max() / 1000000)

        self.assertEqual(485873, indicator.T.loc['2017-03-31'].max() / 1000000)
        self.assertEqual(485873, indicator.T.loc['2018-03-29'].max() / 1000000)
        self.assertEqual(523964, indicator.T.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(559151, indicator.T.loc['2021-03-19'].max() / 1000000)

    def test_total_revenue_ltm(self):
        wmt_indicators = get_data_wmt()
        indicator = wmt_indicators.sel(field='total_revenue').to_pandas()
        print_normed(indicator.T)

        revenue = indicator.head(10).to_json(orient="table")
        NYSE_WMT_default = get_default_WMT_total_revenue().head(10).to_json(orient="table")
        self.assertEqual(json.loads(NYSE_WMT_default), json.loads(revenue))
        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 61
        self.assertEqual(482130, indicator.T.loc['2016-03-31'].max() / 1000000)
        self.assertEqual(484604, indicator.T.loc['2016-12-01'].max() / 1000000)  # 485873 - 130936 + 129667
        self.assertEqual(484604, indicator.T.loc['2017-03-30'].max() / 1000000)  # 485873 - 130936 + 129667
        self.assertEqual(485873, indicator.T.loc['2017-03-31'].max() / 1000000)

    def test_liabilities(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='liabilities').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 61
        self.assertEqual(126382, indicator.T.loc['2017-03-30'].max() / 1000000)
        self.assertEqual(118290, indicator.T.loc['2017-03-31'].max() / 1000000)
        self.assertEqual(123604, indicator.T.loc['2017-06-02'].max() / 1000000)
        self.assertEqual(122520, indicator.T.loc['2017-08-31'].max() / 1000000)
        self.assertEqual(130508, indicator.T.loc['2017-12-01'].max() / 1000000)

        self.assertEqual(123700, indicator.T.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(139661, indicator.T.loc['2019-03-28'].max() / 1000000)

        # https://s2.q4cdn.com/056532643/files/doc_financials/2020/ar/Walmart_2020_Annual_Report.pdf page 54
        self.assertEqual(154943, indicator.T.loc['2020-03-31'].max() / 1000000)
        self.assertEqual(154943, indicator.T.loc['2020-03-31'].max() / 1000000)
        self.assertEqual(164965, indicator.T.loc['2021-03-31'].max() / 1000000)
        self.assertEqual(151989, indicator.T.loc['2021-06-04'].max() / 1000000)

    def test_cash_and_cash_equivalents(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='cash_and_cash_equivalent').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 40
        self.assertEqual(8705, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(6867, indicator.T.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(6756, indicator.T.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(7722, indicator.T.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(9465, indicator.T.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(17741, indicator.T.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(14930, indicator.T.loc['2020-06-03'].max() / 1000000)
        self.assertEqual(16906, indicator.T.loc['2020-09-02'].max() / 1000000)
        self.assertEqual(14325, indicator.T.loc['2020-12-02'].max() / 1000000)

    def test_assets(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='assets').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 40
        self.assertEqual(199581, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(198825, indicator.T.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(204522, indicator.T.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(219295, indicator.T.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(236495, indicator.T.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(252496, indicator.T.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(232892, indicator.T.loc['2020-06-03'].max() / 1000000)
        self.assertEqual(237382, indicator.T.loc['2020-09-02'].max() / 1000000)
        self.assertEqual(250863, indicator.T.loc['2020-12-02'].max() / 1000000)

    def test_assets_apple(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AAPL'])
        indicator = wmt_indicators.sel(field='assets').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AAPL')

        self.assertEqual(290479, indicator.T.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(321686, indicator.T.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(375319, indicator.T.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(365725, indicator.T.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(338516, indicator.T.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(323888, indicator.T.loc['2020-10-30'].max() / 1000000)
        self.assertEqual(337158, indicator.T.loc['2021-04-29'].max() / 1000000)

    def test_equity(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='equity').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 40
        self.assertEqual(83611, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(80535, indicator.T.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(80822, indicator.T.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(79634, indicator.T.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(81552, indicator.T.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(87531, indicator.T.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(74110, indicator.T.loc['2020-06-03'].max() / 1000000)
        self.assertEqual(81197, indicator.T.loc['2020-09-02'].max() / 1000000)
        self.assertEqual(87504, indicator.T.loc['2020-12-02'].max() / 1000000)

    def test_income_before_taxes(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='income_before_taxes').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(24799, indicator.T.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(21638, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(20497, indicator.T.loc['2017-03-31 '].max() / 1000000)

    def test_operating_income(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='operating_income').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(27147, indicator.T.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(24105, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(22764, indicator.T.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(20437, indicator.T.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(21957, indicator.T.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(20568, indicator.T.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(22548, indicator.T.loc['2021-03-19'].max() / 1000000)

        self.assertEqual(5322 + 5224 + 6059 + 5778, indicator.T.loc['2020-12-02'].max() / 1000000)
        self.assertEqual(24233, indicator.T.loc['2021-06-04'].max() / 1000000)

    def test_interest_net(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='interest_net').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 38
        self.assertEqual(2348, indicator.T.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(2467, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(2267, indicator.T.loc['2017-03-31 '].max() / 1000000)

    def test_depreciation_and_amortization(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='depreciation_and_amortization').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        # https://s2.q4cdn.com/056532643/files/doc_financials/2020/ar/Walmart_2020_Annual_Report.pdf 35
        self.assertEqual(9173, indicator.T.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(9454, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(10080, indicator.T.loc['2017-03-31 '].max() / 1000000)
        self.assertEqual(10678, indicator.T.loc['2019-03-28 '].max() / 1000000)
        self.assertEqual(10987, indicator.T.loc['2020-03-20 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2021/ar/WMT_2021_AnnualReport.pdf 81
        self.assertEqual(11152, indicator.T.loc['2021-03-19'].max() / 1000000)

        # https://s2.q4cdn.com/056532643/files/doc_financials/2022/q1/Earnings-Release-(FY22-Q1).pdf page 5
        self.assertEqual(11152 + 2661 - 2791, indicator.T.loc['2021-06-04 '].max() / 1000000)

    def test_income_interest(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='income_interest').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(113, indicator.T.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(81, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(100, indicator.T.loc['2017-03-31 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2020/ar/Walmart_2020_Annual_Report.pdf page 52
        self.assertEqual(152, indicator.T.loc['2018-04-02 '].max() / 1000000)
        self.assertEqual(217, indicator.T.loc['2019-03-28 '].max() / 1000000)
        self.assertEqual(189, indicator.T.loc['2020-03-20 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2021/ar/WMT_2021_AnnualReport.pdf page 56
        self.assertEqual(121, indicator.T.loc['2021-03-19 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2022/q1/Earnings-Release-(FY22-Q1).pdf page 5
        self.assertEqual(121 + 30 - 43, indicator.T.loc['2021-06-04 '].max() / 1000000)

    def test_nonoperating_income_expense(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='nonoperating_income_expense').to_pandas()
        print_normed(indicator.T)

        self.assertEqual(-8368, indicator.T.loc['2019-03-28 '].max() / 1000000)
        self.assertEqual(1958, indicator.T.loc['2020-03-20 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2021/ar/WMT_2021_AnnualReport.pdf page 56
        self.assertEqual(210, indicator.T.loc['2021-03-19 '].max() / 1000000)
        # https://s2.q4cdn.com/056532643/files/doc_financials/2022/q1/Earnings-Release-(FY22-Q1).pdf page 5
        self.assertEqual(210 + (-2529 - 721), indicator.T.loc['2021-06-04 '].max() / 1000000)

    def test_ebitda_use_operating_income_wmt(self):
        wmt_indicators = get_data_wmt_new()
        indicator = wmt_indicators.sel(field='ebitda_use_operating_income').to_pandas()
        print_normed(indicator.T)

        # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
        self.assertEqual(36433, indicator.T.loc['2015-04-01'].max() / 1000000)
        self.assertEqual(33640, indicator.T.loc['2016-03-30 '].max() / 1000000)
        self.assertEqual(32944, indicator.T.loc['2017-03-31 '].max() / 1000000)

        self.assertEqual(27982, indicator.T.loc['2018-04-02'].max() / 1000000)
        self.assertEqual(24484, indicator.T.loc['2019-03-28'].max() / 1000000)
        self.assertEqual(33702, indicator.T.loc['2020-03-20'].max() / 1000000)
        self.assertEqual(34031, indicator.T.loc['2021-03-19'].max() / 1000000)

        # self.assertEqual(5322 + 5224 + 6059 + 5778, total_liabilities.T.loc['2020-12-02'].max() / 1000000)
        self.assertEqual(32323, indicator.T.loc['2021-06-04'].max() / 1000000)

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
        wmt_indicators = get_data_new_for(['NASDAQ:AAPL'])
        indicator = wmt_indicators.sel(field='operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AAPL')

        self.assertEqual(71230, indicator.T.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(60024, indicator.T.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(61344, indicator.T.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(70898, indicator.T.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(63930, indicator.T.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(66288, indicator.T.loc['2020-10-30'].max() / 1000000)
        self.assertEqual(88903, indicator.T.loc['2021-04-29'].max() / 1000000)

    def test_depreciation_and_amortization_apple(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AAPL'])
        indicator = wmt_indicators.sel(field='depreciation_and_amortization').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AAPL')

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(11257, indicator.T.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(10505, indicator.T.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(10157, indicator.T.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(10903, indicator.T.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(12547, indicator.T.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(11056, indicator.T.loc['2020-10-30'].max() / 1000000)

    def test_interest_expense_apple(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AAPL'])
        indicator = wmt_indicators.sel(field='interest_expense').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AAPL')

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(733, indicator.T.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(1456, indicator.T.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(2323, indicator.T.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(3240, indicator.T.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(3576, indicator.T.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(2873, indicator.T.loc['2020-10-30'].max() / 1000000)

    def test_ebitda_use_operating_income_apple(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AAPL'])
        indicator = wmt_indicators.sel(field='ebitda_use_operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AAPL')

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(84505, indicator.T.loc['2015-10-28'].max() / 1000000)
        self.assertEqual(73333, indicator.T.loc['2016-10-26'].max() / 1000000)
        self.assertEqual(76569, indicator.T.loc['2017-11-03'].max() / 1000000)
        self.assertEqual(87046, indicator.T.loc['2018-11-05'].max() / 1000000)
        self.assertEqual(81860, indicator.T.loc['2019-10-31'].max() / 1000000)
        self.assertEqual(81020, indicator.T.loc['2020-10-30'].max() / 1000000)
        # self.assertEqual(128026, indicator.T.loc['2021-04-29'].max() / 1000000)

    def test_operating_income_amazon(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AMZN'])
        indicator = wmt_indicators.sel(field='operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AMZN')

        # https://www.stock-analysis-on.net/NASDAQ/Company/Apple-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(2233, indicator.T.loc['2016-01-29'].max() / 1000000)
        self.assertEqual(4186, indicator.T.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(4107, indicator.T.loc['2018-02-02'].max() / 1000000)
        self.assertEqual(12420, indicator.T.loc['2019-02-01'].max() / 1000000)
        self.assertEqual(14540, indicator.T.loc['2020-01-31'].max() / 1000000)
        self.assertEqual(22899, indicator.T.loc['2021-02-03'].max() / 1000000)
        self.assertEqual(27775, indicator.T.loc['2021-04-30'].max() / 1000000)

    def test_depreciation_and_amortization_amazon(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AMZN'])
        indicator = wmt_indicators.sel(field='depreciation_and_amortization').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AMZN')

        # https://www.stock-analysis-on.net/NASDAQ/Company/Amazoncom-Inc/Valuation/EV-to-EBITDA
        # self.assertEqual(2233, indicator.T.loc['2016-01-29'].max() / 1000000)
        self.assertEqual(8116, indicator.T.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(11478, indicator.T.loc['2018-02-02'].max() / 1000000)
        self.assertEqual(15341, indicator.T.loc['2019-02-01'].max() / 1000000)
        self.assertEqual(21789, indicator.T.loc['2020-01-31'].max() / 1000000)
        self.assertEqual(25251, indicator.T.loc['2021-02-03'].max() / 1000000)
        self.assertEqual(27397, indicator.T.loc['2021-04-30'].max() / 1000000)

    def test_ebitda_use_operating_income_amazon(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AMZN'])
        indicator = wmt_indicators.sel(field='ebitda_use_operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AMZN')

        # https://www.stock-analysis-on.net/NASDAQ/Company/Amazoncom-Inc/Valuation/EV-to-EBITDA
        self.assertEqual(8308, indicator.T.loc['2016-01-29'].max() / 1000000)
        self.assertEqual(12492, indicator.T.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(16132, indicator.T.loc['2018-02-02'].max() / 1000000)

        self.assertEqual(28020, indicator.T.loc['2019-02-01'].max() / 1000000)  # bt provide value = 26416
        self.assertEqual(37364, indicator.T.loc['2020-01-31'].max() / 1000000)
        self.assertEqual(51076, indicator.T.loc['2021-02-03'].max() / 1000000)

        self.assertEqual(60104, indicator.T.loc['2021-04-30'].max() / 1000000)

    def test_operating_income_fb(self):
        indicators = get_data_new_for(['NASDAQ:FB'])
        indicator = indicators.sel(field='operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:FB')

        self.assertEqual(12428, indicator.T.loc['2017-02-03'].max() / 1000000)
        self.assertEqual(20202, indicator.T.loc['2018-02-01'].max() / 1000000)
        self.assertEqual(24913, indicator.T.loc['2019-01-31'].max() / 1000000)
        self.assertEqual(23986, indicator.T.loc['2020-01-30'].max() / 1000000)
        self.assertEqual(32671, indicator.T.loc['2021-01-28'].max() / 1000000)

        self.assertEqual(38156, indicator.T.loc['2021-04-29'].max() / 1000000)

    def test_depreciation_and_amortization_fb(self):
        indicators = get_data_new_for(['NASDAQ:FB'])
        indicator = indicators.sel(field='depreciation_and_amortization').to_pandas()
        print_normed(indicator.T, 'NASDAQ:FB')

        self.assertEqual(2342, indicator.T.loc['2017-02-03'].max() / 1000000)
        self.assertEqual(3025, indicator.T.loc['2018-02-01'].max() / 1000000)
        self.assertEqual(4315, indicator.T.loc['2019-01-31'].max() / 1000000)
        self.assertEqual(5741, indicator.T.loc['2020-01-30'].max() / 1000000)
        self.assertEqual(6862, indicator.T.loc['2021-01-28'].max() / 1000000)

    def test_ebitda_use_operating_income_fb(self):
        indicators = get_data_new_for(['NASDAQ:FB'])
        indicator = indicators.sel(field='ebitda_use_operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:FB')

        self.assertEqual(14869, indicator.T.loc['2017-02-03'].max() / 1000000)
        self.assertEqual(23625, indicator.T.loc['2018-02-01'].max() / 1000000)
        self.assertEqual(29685, indicator.T.loc['2019-01-31'].max() / 1000000)
        self.assertEqual(30573, indicator.T.loc['2020-01-30'].max() / 1000000)
        self.assertEqual(40042, indicator.T.loc['2021-01-28'].max() / 1000000)  # 39533 bt value

        # self.assertEqual(45393, indicator.T.loc['2021-04-29'].max() / 1000000)bt value

    # def test_ebitda_use_income_before_taxes_fb(self):
    #     indicators = get_data_new_for(['NASDAQ:FB'])
    #     indicator = indicators.sel(field='ebitda_use_income_before_taxes').to_pandas()
    #     print_normed(indicator.T, 'NASDAQ:FB')
    #
    #     # https://www.stock-analysis-on.net/NASDAQ/Company/Facebook-Inc/Valuation/EV-to-EBITDA
    #     self.assertEqual(14859, indicator.T.loc['2017-02-03'].max() / 1000000)
    #     self.assertEqual(23619, indicator.T.loc['2018-02-01'].max() / 1000000)
    #     self.assertEqual(29676, indicator.T.loc['2019-01-31'].max() / 1000000)
    #     self.assertEqual(30553, indicator.T.loc['2020-01-30'].max() / 1000000)
    #     self.assertEqual(40042, indicator.T.loc['2021-01-28'].max() / 1000000)

    # def test_ebitda_use_income_before_taxes_wmt(self):
    #     indicators = get_data_new_for(['NYSE:WMT'])
    #     indicator = indicators.sel(field='ebitda_use_income_before_taxes').to_pandas()
    #     print_normed(indicator.T, 'NYSE:WMT')
    #
    #     # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
    #     self.assertEqual(36433, indicator.T.loc['2015-04-01'].max() / 1000000)
    #     self.assertEqual(33640, indicator.T.loc['2016-03-30 '].max() / 1000000)
    #     self.assertEqual(32944, indicator.T.loc['2017-03-31 '].max() / 1000000)
    #
    #     self.assertEqual(27982, indicator.T.loc['2018-04-02'].max() / 1000000)
    #     self.assertEqual(24484, indicator.T.loc['2019-03-28'].max() / 1000000)
    #     self.assertEqual(33702, indicator.T.loc['2020-03-20'].max() / 1000000)
    #     self.assertEqual(34031, indicator.T.loc['2021-03-19'].max() / 1000000)
    #
    #     # self.assertEqual(5322 + 5224 + 6059 + 5778, total_liabilities.T.loc['2020-12-02'].max() / 1000000)
    #     self.assertEqual(32323, indicator.T.loc['2021-06-04'].max() / 1000000)

    def test_operating_income_visa(self):
        indicators = get_data_new_for(['NYSE:V'])
        indicator = indicators.sel(field='operating_income').to_pandas()
        print_normed(indicator.T, 'NYSE:V')

        self.assertEqual(7883, indicator.T.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(12144, indicator.T.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(12954, indicator.T.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(15001, indicator.T.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(14081, indicator.T.loc['2020-11-19'].max() / 1000000)

    def test_depreciation_and_amortization_visa(self):
        indicators = get_data_new_for(['NYSE:V'])
        indicator = indicators.sel(field='depreciation_and_amortization').to_pandas()
        print_normed(indicator.T, 'NYSE:V')

        self.assertEqual(502, indicator.T.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(556, indicator.T.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(613, indicator.T.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(656, indicator.T.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(767, indicator.T.loc['2020-11-19'].max() / 1000000)

    # def test_income_before_income_taxes_visa(self):
    #     indicators = get_data_new_for(['NYSE:V'])
    #     indicator = indicators.sel(field='income_before_income_taxes').to_pandas()
    #     print_normed(indicator.T, 'NYSE:V')
    #
    #     self.assertEqual(502, indicator.T.loc['2016-11-15'].max() / 1000000)
    #     self.assertEqual(556, indicator.T.loc['2017-11-17'].max() / 1000000)
    #     self.assertEqual(613, indicator.T.loc['2018-11-16'].max() / 1000000)
    #     self.assertEqual(656, indicator.T.loc['2019-11-14'].max() / 1000000)
    #     self.assertEqual(767, indicator.T.loc['2020-11-19'].max() / 1000000)

    # def test_ebitda_use_operating_income_visa(self):
    #     import qnt.data    as qndata
    #     futures_server = qndata.load_data(min_date="2020-01-01")
    #     indicators = get_data_new_for(['NYSE:V'])
    #     indicator = indicators.sel(field='ebitda_use_operating_income').to_pandas()
    #     print_normed(indicator.T, 'NYSE:V')
    #
    #     self.assertEqual(8941, indicator.T.loc['2016-11-15'].max() / 1000000)
    #     self.assertEqual(12813, indicator.T.loc['2017-11-17'].max() / 1000000)
    #     self.assertEqual(14031, indicator.T.loc['2018-11-16'].max() / 1000000)
    #     self.assertEqual(16073, indicator.T.loc['2019-11-14'].max() / 1000000)
    #     self.assertEqual(15073, indicator.T.loc['2020-11-19'].max() / 1000000)

    def test_ebitda_use_income_before_taxes_visa(self):
        import qnt.data    as qndata
        futures_server = qndata.load_data(min_date="2020-01-01")
        indicators = get_data_new_for(['NYSE:V'])
        indicator = indicators.sel(field='ebitda_use_income_before_taxes').to_pandas()
        print_normed(indicator.T, 'NYSE:V')

        self.assertEqual(8941, indicator.T.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(12813, indicator.T.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(14031, indicator.T.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(16073, indicator.T.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(15073, indicator.T.loc['2020-11-19'].max() / 1000000)

    def test_income_before_taxes_visa(self):
        indicators = get_data_new_for(['NYSE:V'])
        indicator = indicators.sel(field='income_before_taxes').to_pandas()
        print_normed(indicator.T, 'NYSE:V')

        self.assertEqual(8012, indicator.T.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(11694, indicator.T.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(12806, indicator.T.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(14884, indicator.T.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(13790, indicator.T.loc['2020-11-19'].max() / 1000000)

    def test_operating_income_nvidia(self):
        indicators = get_data_new_for(['NASDAQ:NVDA'])
        indicator = indicators.sel(field='operating_income').to_pandas()
        print_normed(indicator.T, 'NASDAQ:NVDA')

        self.assertEqual(1934, indicator.T.loc['2017-03-01'].max() / 1000000)
        self.assertEqual(3210, indicator.T.loc['2018-02-28'].max() / 1000000)
        self.assertEqual(3804, indicator.T.loc['2019-02-21'].max() / 1000000)
        self.assertEqual(2846, indicator.T.loc['2020-02-20'].max() / 1000000)
        self.assertEqual(4532, indicator.T.loc['2021-02-26'].max() / 1000000)

    # def test_ebitda_nvidia(self):
    #     indicators = get_data_new_for(['NASDAQ:NVDA'])
    #     indicator = indicators.sel(field='ebitda_use_operating_income').to_pandas()
    #     print_normed(indicator.T, 'NASDAQ:NVDA')
    #
    #     self.assertEqual(2150, indicator.T.loc['2017-03-01'].max() / 1000000)
    #     self.assertEqual(3210, indicator.T.loc['2018-02-28'].max() / 1000000)
    #     self.assertEqual(3804, indicator.T.loc['2019-02-21'].max() / 1000000)
    #     self.assertEqual(2846, indicator.T.loc['2020-02-20'].max() / 1000000)
    #     self.assertEqual(4532, indicator.T.loc['2021-02-26'].max() / 1000000)

    def test_interest_income_expense_net_visa(self):
        indicators = get_data_new_for(['NYSE:V'])
        indicator = indicators.sel(field='interest_income_expense_net').to_pandas()
        print_normed(indicator.T, 'NYSE:V')

        self.assertEqual(-427, indicator.T.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(-563, indicator.T.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(-612, indicator.T.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(-533, indicator.T.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(-516, indicator.T.loc['2020-11-19'].max() / 1000000)

    def test_other_nonoperating_income_expense_visa(self):
        indicators = get_data_new_for(['NYSE:V'])
        indicator = indicators.sel(field='other_nonoperating_income_expense').to_pandas()
        print_normed(indicator.T, 'NYSE:V')

        self.assertEqual(556, indicator.T.loc['2016-11-15'].max() / 1000000)
        self.assertEqual(113, indicator.T.loc['2017-11-17'].max() / 1000000)
        self.assertEqual(464, indicator.T.loc['2018-11-16'].max() / 1000000)
        self.assertEqual(416, indicator.T.loc['2019-11-14'].max() / 1000000)
        self.assertEqual(225, indicator.T.loc['2020-11-19'].max() / 1000000)

    def test_nonoperating_income_expense_amazon(self):
        wmt_indicators = get_data_new_for(['NASDAQ:AMZN'])
        indicator = wmt_indicators.sel(field='nonoperating_income_expense').to_pandas()
        print_normed(indicator.T, 'NASDAQ:AMZN')

        self.assertEqual(-294, indicator.T.loc['2017-02-10'].max() / 1000000)
        self.assertEqual(-301, indicator.T.loc['2018-02-02'].max() / 1000000)
        self.assertEqual(-565, indicator.T.loc['2020-01-31'].max() / 1000000)

    def test_all_indicators_list(self):
        import qnt.data    as qndata
        import qnt.data.secgov_indicators_new
        all_indicators = qnt.data.get_all_indicators()

        self.assertEqual(
            ['total_revenue',
             'liabilities',
             'cash_and_cash_equivalent',
             'assets',
             'equity',
             'income_before_taxes',
             'income_before_income_taxes',
             'operating_income',
             'interest_net',
             'depreciation_and_amortization',
             'income_interest',
             'interest_expense',
             'interest_expense_debt',
             'interest_expense_capital_lease',
             'interest_income_expense_net',
             'nonoperating_income_expense',
             'other_nonoperating_income_expense',
             'ebitda_use_income_before_taxes',
             'ebitda_use_operating_income',
             'losses_on_extinguishment_of_debt']

            , all_indicators)

    # def test_ebitda_use_income_before_taxes(self):
    #     wmt_indicators = get_data_wmt_new()
    #     indicator = wmt_indicators.sel(field='ebitda_use_income_before_taxes').to_pandas()
    #     print_normed(indicator.T)
    #
    #     # http://s2.q4cdn.com/056532643/files/doc_financials/2017/Annual/WMT_2017_AR-(1).pdf  page 42
    #     self.assertEqual(9173, indicator.T.loc['2015-04-01'].max() / 1000000)
    #     self.assertEqual(9454, indicator.T.loc['2016-03-30 '].max() / 1000000)
    #     self.assertEqual(10080, indicator.T.loc['2017-03-31 '].max() / 1000000)


def get_default_WMT_total_revenue():
    dir = os.path.abspath(os.curdir)
    with open(dir + '/data/fundamental_NYSE_WMT_total_revenue.pkl', 'rb') as input:
        total_revenue_default = pickle.load(input)
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
    import qnt.data    as qndata
    import datetime    as dt

    import qnt.data.secgov_indicators

    pd.set_option('display.max_rows', None)

    def get_data_filter(data, assets):
        filler = data.sel(asset=assets)
        return filler

    import qnt.data    as qndata
    assets = qndata.load_assets(min_date="2010-01-01", max_date="2021-06-25")

    WMT = {}
    for a in assets:
        if a['id'] == 'NYSE:WMT':
            WMT = a

    data = qndata.load_data(min_date="2010-01-01", max_date="2021-06-25",
                            dims=("time", "field", "asset"),
                            assets=['NYSE:WMT'],
                            forward_order=True)

    data_lbls = ['total_revenue',
                 'total_revenue_qf', 'total_revenue_af', 'liabilities', 'liabilities_curr', 'debt_lt', 'debt_st',
                 'liabilities_calculated']
    fun_indicators = qnt.data.secgov_load_indicators([WMT], time_coord=data.time, standard_indicators=data_lbls)

    # def save_object(obj, filename):
    #     with open(filename, 'wb') as output:  # Overwrites any existing file.
    #         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    #
    # save_object(fun_indicators, 'fundamental_NYSE_WMT_total_revenue.pkl')

    NYSE_WMT = get_data_filter(fun_indicators, ['NYSE:WMT'])

    q = NYSE_WMT.sel(field='total_revenue').to_pandas()
    return NYSE_WMT


def test_all():
    import pandas as pd
    import qnt.data    as qndata
    import datetime    as dt

    import qnt.data.secgov_indicators

    pd.set_option('display.max_rows', None)
    assets = qndata.load_assets(min_date="2010-01-01", max_date="2021-06-25")

    WMT = {}
    for a in assets:
        if a['id'] == 'NYSE:WMT':
            WMT = a

    data = qndata.load_data(min_date="2010-01-01", max_date="2021-07-28",
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


def get_data_wmt_new():
    import qnt.data    as qndata
    import qnt.data.secgov_indicators_new

    market_data = qndata.load_data(min_date="2010-01-01", max_date="2021-07-28",
                                   dims=("time", "field", "asset"),
                                   assets=['NYSE:WMT'],
                                   forward_order=True)

    indicators_names = ['total_revenue',
                        'liabilities',
                        'cash_and_cash_equivalent',
                        'assets',
                        'equity',
                        'income_before_taxes',
                        'operating_income',
                        'interest_net',
                        'depreciation_and_amortization',
                        'income_interest',
                        'interest_expense',
                        'nonoperating_income_expense',
                        'losses_on_extinguishment_of_debt',
                        'ebitda_use_operating_income',
                        'ebitda_use_income_before_taxes',
                        ]
    #
    # indicators_names = ['depreciation_and_amortization', 'ebitda_use_operating_income',
    #                     'losses_on_extinguishment_of_debt', 'income_interest', 'nonoperating_income_expense']

    # indicators_names = ['InterestExpense']

    indicators = qnt.data.load_indicators_for(market_data, indicators_names)

    return indicators


def get_data_new_for(asset_names):
    import qnt.data    as qndata
    import qnt.data.secgov_indicators_new

    market_data = qndata.load_data(min_date="2010-01-01", max_date="2021-07-28",
                                   dims=("time", "field", "asset"),
                                   assets=asset_names,
                                   forward_order=True)

    indicators_names = ['total_revenue',
                        'liabilities',
                        'cash_and_cash_equivalent',
                        'assets',
                        'equity',
                        'income_before_taxes',
                        'operating_income',
                        'interest_net',
                        'depreciation_and_amortization',
                        'income_interest',
                        'interest_expense',
                        'nonoperating_income_expense',
                        'losses_on_extinguishment_of_debt',
                        'ebitda_use_operating_income',
                        'ebitda_use_income_before_taxes',
                        'other_nonoperating_income_expense',
                        'income_before_income_taxes',
                        'interest_income_expense_net',
                        'income_before_taxes'
                        ]

    # indicators_names = ['depreciation_and_amortization', 'ebitda_use_operating_income',
    #                     'losses_on_extinguishment_of_debt', 'income_interest', 'nonoperating_income_expense']
    #
    # indicators_names = ['ebitda_use_operating_income', 'operating_income', 'depreciation_and_amortization', 'income_before_income_taxes']
    # indicators_names = ['nonoperating_income_expense']
    # indicators_names = ['other_nonoperating_income_expense']
    # indicators_names = ['income_before_taxes']

    indicators = qnt.data.load_indicators_for(market_data, indicators_names)

    return indicators


if __name__ == '__main__':
    unittest.main()
