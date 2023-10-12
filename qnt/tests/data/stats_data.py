import pandas as pd
import xarray as xr
import numpy as np

import json
import os


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


def get_base_df():
    # def get_base_df():
    #     tail = 100
    #     futures = qndata.cryptofutures_load_data(tail=tail, dims=('time', 'field', 'asset'))
    #     filler = futures.sel(asset=["BTC"])
    #     display(filler.sel(field="close"))
    #     prices_pandas = filler.sel(field="close").to_pandas()
    #     prices_pandas["open"] = range(1, prices_pandas.shape[0] + 1, 1)
    #     prices_pandas["close"] = range(2, prices_pandas.shape[0] + 2, 1)
    #     prices_pandas["low"] = prices_pandas["open"]
    #     prices_pandas["high"] = prices_pandas["close"]
    #     prices_pandas["vol"] = 1000
    #     prices_pandas["divs"] = 0
    #     prices_pandas["split"] = 0
    #     prices_pandas["split_cumprod"] = 0
    #     prices_pandas["is_liquid"] = 1
    #     del prices_pandas["BTC"]
    #
    #     result = prices_pandas.to_json(orient="table")
    #
    #     display(result)
    #
    # get_base_df()

    try_result = {"schema": {"fields": [{"name": "time", "type": "datetime"}, {"name": "open", "type": "integer"},
                                        {"name": "close", "type": "integer"}, {"name": "low", "type": "integer"},
                                        {"name": "high", "type": "integer"}, {"name": "vol", "type": "integer"},
                                        {"name": "divs", "type": "integer"}, {"name": "split", "type": "integer"},
                                        {"name": "split_cumprod", "type": "integer"},
                                        {"name": "is_liquid", "type": "integer"}], "primaryKey": ["time"],
                             "pandas_version": "0.20.0"}, "data": [
        {"time": "2021-01-30T00:00:00.000Z", "open": 1, "close": 2, "low": 1, "high": 2, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-01-31T00:00:00.000Z", "open": 2, "close": 3, "low": 2, "high": 3, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-01T00:00:00.000Z", "open": 3, "close": 4, "low": 3, "high": 4, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-02T00:00:00.000Z", "open": 4, "close": 5, "low": 4, "high": 5, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-03T00:00:00.000Z", "open": 5, "close": 6, "low": 5, "high": 6, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-04T00:00:00.000Z", "open": 6, "close": 7, "low": 6, "high": 7, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-05T00:00:00.000Z", "open": 7, "close": 8, "low": 7, "high": 8, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-06T00:00:00.000Z", "open": 8, "close": 9, "low": 8, "high": 9, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-07T00:00:00.000Z", "open": 9, "close": 10, "low": 9, "high": 10, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-08T00:00:00.000Z", "open": 10, "close": 11, "low": 10, "high": 11, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-09T00:00:00.000Z", "open": 11, "close": 12, "low": 11, "high": 12, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-10T00:00:00.000Z", "open": 12, "close": 13, "low": 12, "high": 13, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-11T00:00:00.000Z", "open": 13, "close": 14, "low": 13, "high": 14, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-12T00:00:00.000Z", "open": 14, "close": 15, "low": 14, "high": 15, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-13T00:00:00.000Z", "open": 15, "close": 16, "low": 15, "high": 16, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-14T00:00:00.000Z", "open": 16, "close": 17, "low": 16, "high": 17, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-15T00:00:00.000Z", "open": 17, "close": 18, "low": 17, "high": 18, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-16T00:00:00.000Z", "open": 18, "close": 19, "low": 18, "high": 19, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-17T00:00:00.000Z", "open": 19, "close": 20, "low": 19, "high": 20, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-18T00:00:00.000Z", "open": 20, "close": 21, "low": 20, "high": 21, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-19T00:00:00.000Z", "open": 21, "close": 22, "low": 21, "high": 22, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-20T00:00:00.000Z", "open": 22, "close": 23, "low": 22, "high": 23, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-21T00:00:00.000Z", "open": 23, "close": 24, "low": 23, "high": 24, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-22T00:00:00.000Z", "open": 24, "close": 25, "low": 24, "high": 25, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-23T00:00:00.000Z", "open": 25, "close": 26, "low": 25, "high": 26, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-24T00:00:00.000Z", "open": 26, "close": 27, "low": 26, "high": 27, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-25T00:00:00.000Z", "open": 27, "close": 28, "low": 27, "high": 28, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-26T00:00:00.000Z", "open": 28, "close": 29, "low": 28, "high": 29, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-27T00:00:00.000Z", "open": 29, "close": 30, "low": 29, "high": 30, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-28T00:00:00.000Z", "open": 30, "close": 31, "low": 30, "high": 31, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-01T00:00:00.000Z", "open": 31, "close": 32, "low": 31, "high": 32, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-02T00:00:00.000Z", "open": 32, "close": 33, "low": 32, "high": 33, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-03T00:00:00.000Z", "open": 33, "close": 34, "low": 33, "high": 34, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-04T00:00:00.000Z", "open": 34, "close": 35, "low": 34, "high": 35, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-05T00:00:00.000Z", "open": 35, "close": 36, "low": 35, "high": 36, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-06T00:00:00.000Z", "open": 36, "close": 37, "low": 36, "high": 37, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-07T00:00:00.000Z", "open": 37, "close": 38, "low": 37, "high": 38, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-08T00:00:00.000Z", "open": 38, "close": 39, "low": 38, "high": 39, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-09T00:00:00.000Z", "open": 39, "close": 40, "low": 39, "high": 40, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-10T00:00:00.000Z", "open": 40, "close": 41, "low": 40, "high": 41, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-11T00:00:00.000Z", "open": 41, "close": 42, "low": 41, "high": 42, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-12T00:00:00.000Z", "open": 42, "close": 43, "low": 42, "high": 43, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-13T00:00:00.000Z", "open": 43, "close": 44, "low": 43, "high": 44, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-14T00:00:00.000Z", "open": 44, "close": 45, "low": 44, "high": 45, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-15T00:00:00.000Z", "open": 45, "close": 46, "low": 45, "high": 46, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-16T00:00:00.000Z", "open": 46, "close": 47, "low": 46, "high": 47, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-17T00:00:00.000Z", "open": 47, "close": 48, "low": 47, "high": 48, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-18T00:00:00.000Z", "open": 48, "close": 49, "low": 48, "high": 49, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-19T00:00:00.000Z", "open": 49, "close": 50, "low": 49, "high": 50, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-20T00:00:00.000Z", "open": 50, "close": 51, "low": 50, "high": 51, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-21T00:00:00.000Z", "open": 51, "close": 52, "low": 51, "high": 52, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-22T00:00:00.000Z", "open": 52, "close": 53, "low": 52, "high": 53, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-23T00:00:00.000Z", "open": 53, "close": 54, "low": 53, "high": 54, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-24T00:00:00.000Z", "open": 54, "close": 55, "low": 54, "high": 55, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-25T00:00:00.000Z", "open": 55, "close": 56, "low": 55, "high": 56, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-26T00:00:00.000Z", "open": 56, "close": 57, "low": 56, "high": 57, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-27T00:00:00.000Z", "open": 57, "close": 58, "low": 57, "high": 58, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-28T00:00:00.000Z", "open": 58, "close": 59, "low": 58, "high": 59, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-29T00:00:00.000Z", "open": 59, "close": 60, "low": 59, "high": 60, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-30T00:00:00.000Z", "open": 60, "close": 61, "low": 60, "high": 61, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-03-31T00:00:00.000Z", "open": 61, "close": 62, "low": 61, "high": 62, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-01T00:00:00.000Z", "open": 62, "close": 63, "low": 62, "high": 63, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-02T00:00:00.000Z", "open": 63, "close": 64, "low": 63, "high": 64, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-03T00:00:00.000Z", "open": 64, "close": 65, "low": 64, "high": 65, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-04T00:00:00.000Z", "open": 65, "close": 66, "low": 65, "high": 66, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-05T00:00:00.000Z", "open": 66, "close": 67, "low": 66, "high": 67, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-06T00:00:00.000Z", "open": 67, "close": 68, "low": 67, "high": 68, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-07T00:00:00.000Z", "open": 68, "close": 69, "low": 68, "high": 69, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-08T00:00:00.000Z", "open": 69, "close": 70, "low": 69, "high": 70, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-09T00:00:00.000Z", "open": 70, "close": 71, "low": 70, "high": 71, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-10T00:00:00.000Z", "open": 71, "close": 72, "low": 71, "high": 72, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-11T00:00:00.000Z", "open": 72, "close": 73, "low": 72, "high": 73, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-12T00:00:00.000Z", "open": 73, "close": 74, "low": 73, "high": 74, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-13T00:00:00.000Z", "open": 74, "close": 75, "low": 74, "high": 75, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-14T00:00:00.000Z", "open": 75, "close": 76, "low": 75, "high": 76, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-15T00:00:00.000Z", "open": 76, "close": 77, "low": 76, "high": 77, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-16T00:00:00.000Z", "open": 77, "close": 78, "low": 77, "high": 78, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-17T00:00:00.000Z", "open": 78, "close": 79, "low": 78, "high": 79, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-18T00:00:00.000Z", "open": 79, "close": 80, "low": 79, "high": 80, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-19T00:00:00.000Z", "open": 80, "close": 81, "low": 80, "high": 81, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-20T00:00:00.000Z", "open": 81, "close": 82, "low": 81, "high": 82, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-21T00:00:00.000Z", "open": 82, "close": 83, "low": 82, "high": 83, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-22T00:00:00.000Z", "open": 83, "close": 84, "low": 83, "high": 84, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-23T00:00:00.000Z", "open": 84, "close": 85, "low": 84, "high": 85, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-24T00:00:00.000Z", "open": 85, "close": 86, "low": 85, "high": 86, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-25T00:00:00.000Z", "open": 86, "close": 87, "low": 86, "high": 87, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-26T00:00:00.000Z", "open": 87, "close": 88, "low": 87, "high": 88, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-27T00:00:00.000Z", "open": 88, "close": 89, "low": 88, "high": 89, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-28T00:00:00.000Z", "open": 89, "close": 90, "low": 89, "high": 90, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-29T00:00:00.000Z", "open": 90, "close": 91, "low": 90, "high": 91, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-04-30T00:00:00.000Z", "open": 91, "close": 92, "low": 91, "high": 92, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-01T00:00:00.000Z", "open": 92, "close": 93, "low": 92, "high": 93, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-02T00:00:00.000Z", "open": 93, "close": 94, "low": 93, "high": 94, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-03T00:00:00.000Z", "open": 94, "close": 95, "low": 94, "high": 95, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-04T00:00:00.000Z", "open": 95, "close": 96, "low": 95, "high": 96, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-05T00:00:00.000Z", "open": 96, "close": 97, "low": 96, "high": 97, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-06T00:00:00.000Z", "open": 97, "close": 98, "low": 97, "high": 98, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-07T00:00:00.000Z", "open": 98, "close": 99, "low": 98, "high": 99, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-08T00:00:00.000Z", "open": 99, "close": 100, "low": 99, "high": 100, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-05-09T00:00:00.000Z", "open": 100, "close": 101, "low": 100, "high": 101, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1}
    ]}
    return get_xr(try_result)


def get_xr(dataframe, asset='BTC'):
    columns = [Dimensions.TIME, f.OPEN, f.CLOSE, f.LOW, f.HIGH, f.VOL, f.DIVS, f.SPLIT, f.SPLIT_CUMPROD,
               f.IS_LIQUID]
    rows = dataframe['data']
    for r in rows:
        r['time'] = np.datetime64(r['time'])
    pandas = pd.DataFrame(columns=columns, data=rows)
    pandas.set_index(Dimensions.TIME, inplace=True)
    prices_array = pandas.to_xarray().to_array(Dimensions.FIELD)
    prices_array.name = "BTC_TEST"
    prices_array_r = xr.concat([prices_array], pd.Index([asset], name='asset'))
    return prices_array_r


def get_cripto_futures():
    data = {"schema": {"fields": [{"name": "time", "type": "datetime"}, {"name": "open", "type": "number"},
                                  {"name": "high", "type": "number"}, {"name": "low", "type": "number"},
                                  {"name": "close", "type": "number"}, {"name": "vol", "type": "number"},
                                  {"name": "oi", "type": "number"}, {"name": "roll", "type": "number"}],
                       "primaryKey": ["time"], "pandas_version": "0.20.0"}, "data": [
        {"time": "2015-01-01T00:00:00.000Z", "open": 317.99, "high": 322.9, "low": 313.81, "close": 315.53,
         "vol": 6171.18437141, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-02T00:00:00.000Z", "open": 315.23, "high": 316.74, "low": 313.28, "close": 315.3,
         "vol": 3831.42367708, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-03T00:00:00.000Z", "open": 315.3, "high": 315.99, "low": 290.11, "close": 291.6,
         "vol": 35459.08541468, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-04T00:00:00.000Z", "open": 291.58, "high": 292.76, "low": 255.03, "close": 264.0,
         "vol": 80739.9239175501, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-05T00:00:00.000Z", "open": 264.0, "high": 279.5, "low": 258.06, "close": 271.01,
         "vol": 46778.64444364, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-06T00:00:00.000Z", "open": 271.42, "high": 290.45, "low": 270.31, "close": 288.5,
         "vol": 34401.84031803, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-07T00:00:00.000Z", "open": 288.5, "high": 303.33, "low": 284.0, "close": 296.46,
         "vol": 31792.93136109, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-08T00:00:00.000Z", "open": 296.64, "high": 300.0, "low": 282.01, "close": 292.54,
         "vol": 29220.16267501, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-09T00:00:00.000Z", "open": 292.07, "high": 297.32, "low": 280.4, "close": 288.39,
         "vol": 30306.8163401, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-10T00:00:00.000Z", "open": 287.96, "high": 292.1, "low": 275.0, "close": 279.27,
         "vol": 24537.28646315, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-11T00:00:00.000Z", "open": 279.27, "high": 282.86, "low": 265.83, "close": 268.79,
         "vol": 21336.14751629, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-12T00:00:00.000Z", "open": 269.0, "high": 274.44, "low": 266.24, "close": 271.52,
         "vol": 19616.29323384, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-13T00:00:00.000Z", "open": 271.59, "high": 272.1, "low": 226.1, "close": 237.0,
         "vol": 124936.68148417, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-14T00:00:00.000Z", "open": 237.0, "high": 238.5, "low": 166.45, "close": 184.42,
         "vol": 262217.28712903, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-15T00:00:00.000Z", "open": 184.42, "high": 230.74, "low": 172.51, "close": 207.1,
         "vol": 188628.351801161, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-16T00:00:00.000Z", "open": 207.08, "high": 223.1, "low": 197.72, "close": 208.0,
         "vol": 76421.45686631, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-17T00:00:00.000Z", "open": 208.66, "high": 214.0, "low": 192.2, "close": 198.54,
         "vol": 49021.04357206, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-18T00:00:00.000Z", "open": 198.51, "high": 222.0, "low": 194.0, "close": 209.48,
         "vol": 64624.2123226601, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-19T00:00:00.000Z", "open": 208.71, "high": 219.95, "low": 207.0, "close": 216.61,
         "vol": 29561.1490178, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-20T00:00:00.000Z", "open": 216.9, "high": 219.25, "low": 203.7, "close": 211.7,
         "vol": 51975.30220128, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-21T00:00:00.000Z", "open": 211.7, "high": 230.0, "low": 210.0, "close": 226.8,
         "vol": 49012.6668899102, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-22T00:00:00.000Z", "open": 227.3, "high": 242.0, "low": 223.3, "close": 232.0,
         "vol": 61684.2434872101, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-23T00:00:00.000Z", "open": 231.94, "high": 236.98, "low": 225.13, "close": 234.81,
         "vol": 38202.17008334, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-24T00:00:00.000Z", "open": 234.84, "high": 248.3, "low": 229.03, "close": 245.31,
         "vol": 36868.59603572, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-25T00:00:00.000Z", "open": 245.5, "high": 259.39, "low": 245.0, "close": 251.49,
         "vol": 44769.63441487, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-26T00:00:00.000Z", "open": 251.49, "high": 315.0, "low": 251.48, "close": 275.0,
         "vol": 204464.37147452, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-27T00:00:00.000Z", "open": 275.0, "high": 280.37, "low": 250.0, "close": 262.66,
         "vol": 84561.4809323701, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-28T00:00:00.000Z", "open": 262.59, "high": 269.97, "low": 231.01, "close": 234.43,
         "vol": 74471.5383570102, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-29T00:00:00.000Z", "open": 234.58, "high": 239.33, "low": 220.0, "close": 234.2,
         "vol": 65789.0114474101, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-30T00:00:00.000Z", "open": 234.21, "high": 243.89, "low": 224.22, "close": 226.89,
         "vol": 49257.65112261, "oi": 0.0, "roll": 0.0},
        {"time": "2015-01-31T00:00:00.000Z", "open": 227.16, "high": 234.03, "low": 221.21, "close": 222.51,
         "vol": 35898.89517609, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-01T00:00:00.000Z", "open": 222.69, "high": 233.32, "low": 211.02, "close": 233.32,
         "vol": 52264.1066262701, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-02T00:00:00.000Z", "open": 233.32, "high": 233.32, "low": 222.5, "close": 227.36,
         "vol": 36724.05784711, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-03T00:00:00.000Z", "open": 227.27, "high": 248.42, "low": 222.66, "close": 226.89,
         "vol": 82346.9360704001, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-04T00:00:00.000Z", "open": 226.89, "high": 233.0, "low": 220.23, "close": 225.33,
         "vol": 43369.2554184101, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-05T00:00:00.000Z", "open": 225.4, "high": 228.8, "low": 210.12, "close": 217.7,
         "vol": 38043.50041893, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-06T00:00:00.000Z", "open": 217.7, "high": 225.88, "low": 215.0, "close": 223.68,
         "vol": 34235.36650102, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-07T00:00:00.000Z", "open": 223.8, "high": 239.78, "low": 222.66, "close": 228.0,
         "vol": 35130.0069874501, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-08T00:00:00.000Z", "open": 227.98, "high": 232.9, "low": 221.1, "close": 222.84,
         "vol": 26284.1508215, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-09T00:00:00.000Z", "open": 222.85, "high": 225.98, "low": 215.33, "close": 221.52,
         "vol": 45715.65736835, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-10T00:00:00.000Z", "open": 221.72, "high": 223.4, "low": 214.0, "close": 219.91,
         "vol": 29016.80877379, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-11T00:00:00.000Z", "open": 219.91, "high": 224.4, "low": 218.1, "close": 219.42,
         "vol": 18910.15901615, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-12T00:00:00.000Z", "open": 219.42, "high": 223.17, "low": 217.8, "close": 222.19,
         "vol": 18635.75906448, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-13T00:00:00.000Z", "open": 222.19, "high": 242.5, "low": 221.46, "close": 235.39,
         "vol": 65286.6208776002, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-14T00:00:00.000Z", "open": 235.39, "high": 259.0, "low": 234.87, "close": 255.3,
         "vol": 71256.3625073302, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-15T00:00:00.000Z", "open": 255.71, "high": 268.54, "low": 228.2, "close": 232.32,
         "vol": 104833.96162536, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-16T00:00:00.000Z", "open": 232.08, "high": 243.65, "low": 228.62, "close": 235.08,
         "vol": 46343.1342315701, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-17T00:00:00.000Z", "open": 236.58, "high": 246.28, "low": 231.5, "close": 241.37,
         "vol": 40442.8875830701, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-18T00:00:00.000Z", "open": 241.46, "high": 244.99, "low": 231.01, "close": 232.6,
         "vol": 37661.9929609501, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-19T00:00:00.000Z", "open": 232.53, "high": 243.42, "low": 231.93, "close": 241.26,
         "vol": 22046.89021568, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-20T00:00:00.000Z", "open": 241.29, "high": 248.98, "low": 238.95, "close": 244.9,
         "vol": 29808.28226303, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-21T00:00:00.000Z", "open": 244.89, "high": 247.73, "low": 243.28, "close": 244.84,
         "vol": 16745.82891697, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-22T00:00:00.000Z", "open": 244.72, "high": 247.78, "low": 232.1, "close": 236.41,
         "vol": 32624.4828957601, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-23T00:00:00.000Z", "open": 236.35, "high": 241.0, "low": 232.61, "close": 240.16,
         "vol": 23028.08714831, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-24T00:00:00.000Z", "open": 240.16, "high": 240.99, "low": 236.7, "close": 239.28,
         "vol": 16653.4756026, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-25T00:00:00.000Z", "open": 239.33, "high": 240.83, "low": 235.53, "close": 237.99,
         "vol": 13578.03837276, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-26T00:00:00.000Z", "open": 237.99, "high": 238.33, "low": 233.62, "close": 236.76,
         "vol": 18511.05608906, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-27T00:00:00.000Z", "open": 236.71, "high": 262.8, "low": 236.64, "close": 257.0,
         "vol": 66929.7166407301, "oi": 0.0, "roll": 0.0},
        {"time": "2015-02-28T00:00:00.000Z", "open": 257.0, "high": 258.38, "low": 251.25, "close": 255.9,
         "vol": 17162.73067631, "oi": 0.0, "roll": 0.0},
        {"time": "2015-03-01T00:00:00.000Z", "open": 255.89, "high": 266.91, "low": 245.65, "close": 262.0,
         "vol": 39152.4134677802, "oi": 0.0, "roll": 0.0}]}

    return get_xr(data)


def get_xr_with_NaN():
    try_result = {"schema": {"fields": [{"name": "time", "type": "datetime"}, {"name": "open", "type": "integer"},
                                        {"name": "close", "type": "integer"}, {"name": "low", "type": "integer"},
                                        {"name": "high", "type": "integer"}, {"name": "vol", "type": "integer"},
                                        {"name": "divs", "type": "integer"}, {"name": "split", "type": "integer"},
                                        {"name": "split_cumprod", "type": "integer"},
                                        {"name": "is_liquid", "type": "integer"}], "primaryKey": ["time"],
                             "pandas_version": "0.20.0"}, "data": [
        {"time": "2021-01-30T00:00:00.000Z", "open": 1, "close": 2, "low": 1, "high": 2, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-01-31T00:00:00.000Z", "open": 2, "close": 3, "low": 2, "high": 3, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-01T00:00:00.000Z", "open": 3, "close": 4, "low": 3, "high": 4, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-02T00:00:00.000Z", "open": 4, "close": 5, "low": 4, "high": 5, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-03T00:00:00.000Z", "open": 5, "close": 6, "low": 5, "high": 6, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-04T00:00:00.000Z", "open": 6, "close": 7, "low": 6, "high": 7, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-05T00:00:00.000Z", "open": 7, "close": 8, "low": 7, "high": 8, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-06T00:00:00.000Z", "open": 8, "close": 9, "low": 8, "high": 9, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-07T00:00:00.000Z", "open": 9, "close": 10, "low": 9, "high": 10, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-08T00:00:00.000Z", "open": 10, "close": 11, "low": 10, "high": 11, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-09T00:00:00.000Z", "open": 11, "close": 12, "low": 11, "high": 12, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-10T00:00:00.000Z", "open": 12, "close": 13, "low": 12, "high": 13, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-11T00:00:00.000Z", "open": 13, "close": 14, "low": 13, "high": 14, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-12T00:00:00.000Z", "open": 14, "close": 15, "low": 14, "high": 15, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-13T00:00:00.000Z", "open": 15, "close": 16, "low": 15, "high": 16, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-14T00:00:00.000Z", "open": None, "close": None, "low": 16, "high": 17, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-15T00:00:00.000Z", "open": np.nan, "close": np.nan, "low": 17, "high": 18, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-16T00:00:00.000Z", "open": 18, "close": 19, "low": 18, "high": 19, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-17T00:00:00.000Z", "open": 19, "close": 20, "low": 19, "high": 20, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
        {"time": "2021-02-18T00:00:00.000Z", "open": None, "close": None, "low": 20, "high": 21, "vol": 1000, "divs": 0,
         "split": 0, "split_cumprod": 0, "is_liquid": 1},
    ]}

    btc = get_xr(try_result, asset='BTC')
    eth = get_xr(try_result, asset='ETH')

    combined = xr.concat([btc, eth], dim='asset')

    return combined