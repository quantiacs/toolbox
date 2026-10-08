import os
import shutil
import unittest
from pathlib import Path

import pandas as pd

os.environ["API_KEY"] = "default"
import qnt.data as qndata

pd.set_option("display.max_rows", None)


def clean_cache():
    root = Path(__file__).parent
    for folder in ("data-cache", "id-transaction"):
        path = root / folder
        if path.is_dir():
            shutil.rmtree(path)
    translation = root / "id-translation.csv"
    if translation.exists():
        translation.unlink()
    for gz in root.rglob("*.gz"):
        gz.unlink()


class TestMarketData(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        clean_cache()

    def _assert_structure(self, data, *, assets_len=None, assets_set=None):
        self.assertIsNotNone(data)
        self.assertTrue(data.time.size > 0)
        self.assertTrue(data.field.size > 0)
        self.assertTrue(data.asset.size > 0)
        if assets_len is not None:
            self.assertEqual(assets_len, len(data.asset.values))
        if assets_set is not None:
            self.assertEqual(assets_set, set(data.asset.values))

    def test_spx_data_various_dates(self):
        for min_date in ("2020-01-01", "2024-01-01"):
            with self.subTest(min_date=min_date):
                data = qndata.stocks.load_spx_data(min_date=min_date)
                self._assert_structure(data, assets_len=808)

    def test_spx_data_dims_and_order(self):
        custom = qndata.stocks.load_spx_data(
            min_date="2024-01-01",
            dims=("time", "field", "asset"),
            forward_order=True,
        )
        full = qndata.stocks.load_spx_data(min_date="2020-01-01")
        self._assert_structure(custom)
        self._assert_structure(full, assets_len=808)

    def test_spx_filtered_assets(self):
        assets = ["NYS:WMT", "NAS:DOCU"]
        data = qndata.stocks.load_spx_data(
            min_date="2010-01-01",
            max_date="2021-06-25",
            dims=("time", "field", "asset"),
            assets=assets,
            forward_order=True,
        )
        self._assert_structure(data, assets_set=set(assets))

    def test_raw_data_single_asset(self):
        assets = ["NASDAQ:ASRT"]
        data = qndata.stocks.load_data(
            min_date="2010-01-01",
            max_date="2021-09-28",
            dims=("time", "field", "asset"),
            assets=assets,
            forward_order=True,
        )
        self._assert_structure(data, assets_set=set(assets))

    def test_double_load(self):
        assets = ["NASDAQ:ASRT"]
        data = qndata.stocks.load_data(
            min_date="2010-01-01",
            max_date="2021-09-28",
            dims=("time", "field", "asset"),
            assets=assets,
            forward_order=True,
        )
        assets_spx = ["NYS:WMT", "NAS:DOCU"]
        data_spx = qndata.stocks.load_spx_data(
            min_date="2010-01-01",
            max_date="2021-06-25",
            dims=("time", "field", "asset"),
            assets=assets_spx,
            forward_order=True,
        )
        self._assert_structure(data, assets_set=set(assets))
        self._assert_structure(data_spx, assets_set=set(assets_spx))


if __name__ == "__main__":
    unittest.main()
