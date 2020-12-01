import talib
import xarray as xr
import numpy as np
from qnt.data import ds, f
import typing as tp
import itertools
import pandas as pd


class MaType:
    """
    Moving Average types
    """
    SMA = 0
    EMA = 1
    WMA = 2
    DEMA = 3
    TEMA = 4
    TRIMA = 5
    KAMA = 6
    MAMA = 7
    T3 = 8


def AD(data: xr.DataArray) -> xr.DataArray:
    """
    Chaikin A/D Line (Volume Indicators)
    Inputs:
        data: ['high', 'low', 'close', 'volume']
    Outputs:
        double series
    """
    return multiple_series_call(talib.AD, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE, f.VOL])


def ADOSC(data: xr.DataArray, fastperiod: int = 3, slowperiod: int = 10) -> xr.DataArray:
    """
    Chaikin A/D Oscillator (Volume Indicators)
    Inputs:
        data: ['high', 'low', 'close', 'volume']
    Parameters:
        fastperiod: 3
        slowperiod: 10
    Outputs:
        double series
    """
    return multiple_series_call(talib.ADOSC, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE, f.VOL],
                                [fastperiod, slowperiod])


def ADX(data: xr.DataArray, timeperiod: int = 14):
    """
    Average Directional Movement Index (Momentum Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.ADX, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def ADXR(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
    Average Directional Movement Index Rating (Momentum Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.ADXR, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def APO(data: xr.DataArray, fastperiod: int = 14, slowperiod: int = 26, matype: int = 0) -> xr.DataArray:
    """

    Absolute Price Oscillator (Momentum Indicators)
    Inputs:
        data: (any price series)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 see the MaType
    Outputs:
        double series
    """
    return single_series_call(talib.APO, data, ds.TIME, [fastperiod, slowperiod, matype])


def AROON(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
    Aroon (Momentum Indicators)
    Inputs:
        data: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        DataArray with the dimension 'field' dimension: ['down', 'up']
    """
    return multiple_series_call(talib.AROON, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW], [timeperiod],
                                ds.FIELD, ['down', 'up'])


def AROONOSC(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
    Aroon Oscillator (Momentum Indicators)
    Inputs:
        data: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
         double series
    """
    return multiple_series_call(talib.AROONOSC, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW], [timeperiod])


def ATR(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
    Average True Range (Volatility Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.ATR, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def BBANDS(data: xr.DataArray, timeperiod: int = 14, nbdevup: int = 2, nbdevdn: int = 2,
           matype: int = 0) -> xr.DataArray:
    """
    Bollinger Bands (Overlap Studies)
    Inputs:
        data: price series
    Parameters:
        timeperiod: 5
        nbdevup: 2
        nbdevdn: 2
        matype: 0 see the MaType
    Outputs:
        DataArray with a new dimension 'bbands': 'upper', 'middle', 'lower'
    """
    return single_series_call(talib.BBANDS, data, ds.TIME, [timeperiod, nbdevup, nbdevdn, matype],
                              'bbands', ['upper', 'middle', 'lower'])


def BETA(first: xr.DataArray, second: xr.DataArray, timeperiod: int = 5) -> xr.DataArray:
    """
        The Beta 'algorithm' is a measure of a stocks volatility vs from index. The stock prices
        are given in 'first' and the index prices are give in 'second'.
        The algorithm is to calculate the change between prices in both vectors
        and then 'plot' these changes are points in the Euclidean plane. The x value of the point
        is market return and the y value is the security return. The beta value is the slope of a
        linear regression through these points. A beta of 1 is simple the line y=x, so the stock
        varies percisely with the market. A beta of less than one means the stock varies less than
        the market and a beta of more than one means the stock varies more than market. A related
        value is the Alpha value (see TA_ALPHA) which is the Y-intercept of the same linear regression.
    Inputs:
        first: price series, if it contains 'asset' dimension, than result will contain dimension 'first' with related assets.
        second: price series, if it contains 'asset' dimension, than result will contain dimension 'second' with related assets.
    Parameters:
        timeperiod: 5
    Outputs:
        double series
    """
    return cross_series_call(talib.BETA, first, second, [timeperiod])


def BOP(data: xr.DataArray) -> xr.DataArray:
    """
    Balance Of Power (Momentum Indicators)
    Inputs:
        data: ['open', 'high', 'low', 'close']
    Outputs:
        double series
    """
    return multiple_series_call(talib.BOP, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE])


def CCI(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
     Commodity Channel Index (Momentum Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.CCI, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def CDL2CROWS(data: xr.DataArray) -> xr.DataArray:
    """
    Two Crows (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL2CROWS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDL3BLACKCROWS(data: xr.DataArray) -> xr.DataArray:
    """
    Three Black Crows (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL3BLACKCROWS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDL3INSIDE(data: xr.DataArray) -> xr.DataArray:
    """
    Three Inside Up/Down (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL3INSIDE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDL3LINESTRIKE(data: xr.DataArray) -> xr.DataArray:
    """
    Three-Line Strike  (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL3LINESTRIKE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDL3OUTSIDE(data: xr.DataArray) -> xr.DataArray:
    """
    Three Outside Up/Down (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL3OUTSIDE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDL3STARSINSOUTH(data: xr.DataArray) -> xr.DataArray:
    """
    Three Stars In The South (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL3STARSINSOUTH, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDL3WHITESOLDIERS(data: xr.DataArray) -> xr.DataArray:
    """
    Three Advancing White Soldiers (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDL3WHITESOLDIERS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLABANDONEDBABY(data: xr.DataArray, penetration: float = 0.3) -> xr.DataArray:
    """
    Abandoned Baby (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLABANDONEDBABY, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLADVANCEBLOCK(data: xr.DataArray) -> xr.DataArray:
    """
    Advance Block (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLADVANCEBLOCK, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLBELTHOLD(data: xr.DataArray) -> xr.DataArray:
    """
    Belt-hold (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLBELTHOLD, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLBREAKAWAY(data: xr.DataArray) -> xr.DataArray:
    """
    Breakaway (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLBREAKAWAY, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLCLOSINGMARUBOZU(data: xr.DataArray) -> xr.DataArray:
    """
    Closing Marubozu (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLCLOSINGMARUBOZU, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLCONCEALBABYSWALL(data: xr.DataArray) -> xr.DataArray:
    """
    Concealing Baby Swallow (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLCONCEALBABYSWALL, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLCOUNTERATTACK(data: xr.DataArray) -> xr.DataArray:
    """
     Counterattack (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLCOUNTERATTACK, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLDARKCLOUDCOVER(data: xr.DataArray, penetration: float = 0.5) -> xr.DataArray:
    """
    Dark Cloud Cover (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLDARKCLOUDCOVER, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLDOJI(data: xr.DataArray) -> xr.DataArray:
    """
    Doji (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLDOJI, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLDOJISTAR(data: xr.DataArray) -> xr.DataArray:
    """
    Doji Star (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLDOJISTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLDRAGONFLYDOJI(data: xr.DataArray) -> xr.DataArray:
    """
    Dragonfly Doji (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLDRAGONFLYDOJI, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLENGULFING(data: xr.DataArray) -> xr.DataArray:
    """
    Engulfing Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLENGULFING, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLEVENINGDOJISTAR(data: xr.DataArray, penetration: float = 0.3) -> xr.DataArray:
    """
    Evening Doji Star (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLEVENINGDOJISTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLEVENINGSTAR(data: xr.DataArray, penetration: float = 0.3) -> xr.DataArray:
    """
    Evening Star (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLEVENINGSTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLGAPSIDESIDEWHITE(data: xr.DataArray) -> xr.DataArray:
    """
    Up/Down-gap side-by-side white lines (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLGAPSIDESIDEWHITE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLGRAVESTONEDOJI(data: xr.DataArray) -> xr.DataArray:
    """
    Gravestone Doji (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLGRAVESTONEDOJI, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHAMMER(data: xr.DataArray) -> xr.DataArray:
    """
    Hammer (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHAMMER, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHANGINGMAN(data: xr.DataArray) -> xr.DataArray:
    """
    Hanging Man (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHANGINGMAN, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHARAMI(data: xr.DataArray) -> xr.DataArray:
    """
    Harami Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHARAMI, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHARAMICROSS(data: xr.DataArray) -> xr.DataArray:
    """
    Harami Cross Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHARAMICROSS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHIGHWAVE(data: xr.DataArray) -> xr.DataArray:
    """
    High-Wave Candle (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHIGHWAVE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHIKKAKE(data: xr.DataArray) -> xr.DataArray:
    """
    Hikkake Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHIKKAKE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHIKKAKEMOD(data: xr.DataArray) -> xr.DataArray:
    """
    Modified Hikkake Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHIKKAKEMOD, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLHOMINGPIGEON(data: xr.DataArray) -> xr.DataArray:
    """
    Homing Pigeon (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLHOMINGPIGEON, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLIDENTICAL3CROWS(data: xr.DataArray) -> xr.DataArray:
    """
    Identical Three Crows (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLIDENTICAL3CROWS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLINNECK(data: xr.DataArray) -> xr.DataArray:
    """
    In-Neck Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLINNECK, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLINVERTEDHAMMER(data: xr.DataArray) -> xr.DataArray:
    """
    Inverted Hammer (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLINVERTEDHAMMER, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLKICKING(data: xr.DataArray) -> xr.DataArray:
    """
    Kicking (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLKICKING, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLKICKINGBYLENGTH(data: xr.DataArray) -> xr.DataArray:
    """
    Kicking - bull/bear determined by the longer marubozu (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLKICKINGBYLENGTH, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLLADDERBOTTOM(data: xr.DataArray) -> xr.DataArray:
    """
    Ladder Bottom (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLLADDERBOTTOM, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLLONGLEGGEDDOJI(data: xr.DataArray) -> xr.DataArray:
    """
    Long Legged Doji (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLLONGLEGGEDDOJI, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLLONGLINE(data: xr.DataArray) -> xr.DataArray:
    """
    Long Line Candle (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLLONGLINE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLMARUBOZU(data: xr.DataArray) -> xr.DataArray:
    """
    Marubozu (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLMARUBOZU, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLMATCHINGLOW(data: xr.DataArray) -> xr.DataArray:
    """
    Matching Low (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLMATCHINGLOW, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLMATHOLD(data: xr.DataArray, penetration: float = 0.5) -> xr.DataArray:
    """
    Mat Hold (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLMATHOLD, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLMORNINGDOJISTAR(data: xr.DataArray, penetration: float = 0.3) -> xr.DataArray:
    """
    Morning Doji Star (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLMORNINGDOJISTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLMORNINGSTAR(data: xr.DataArray, penetration: float = 0.3) -> xr.DataArray:
    """
    Morning Star (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLMORNINGSTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                [penetration], result_divider=100)


def CDLONNECK(data: xr.DataArray) -> xr.DataArray:
    """
    On-Neck Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLONNECK, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLPIERCING(data: xr.DataArray) -> xr.DataArray:
    """
    Piercing Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLPIERCING, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLRICKSHAWMAN(data: xr.DataArray) -> xr.DataArray:
    """
    Rickshaw Man (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLRICKSHAWMAN, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLRISEFALL3METHODS(data: xr.DataArray) -> xr.DataArray:
    """
    Rising/Falling Three Methods (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLRISEFALL3METHODS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLSEPARATINGLINES(data: xr.DataArray) -> xr.DataArray:
    """
    Separating Lines (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLSEPARATINGLINES, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLSHOOTINGSTAR(data: xr.DataArray) -> xr.DataArray:
    """
    Shooting Star (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLSHOOTINGSTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLSHORTLINE(data: xr.DataArray) -> xr.DataArray:
    """
    Short Line Candle (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLSHORTLINE, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLSPINNINGTOP(data: xr.DataArray) -> xr.DataArray:
    """
    Spinning Top (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLSPINNINGTOP, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLSTALLEDPATTERN(data: xr.DataArray) -> xr.DataArray:
    """
    Stalled Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLSTALLEDPATTERN, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLSTICKSANDWICH(data: xr.DataArray) -> xr.DataArray:
    """
    Stick Sandwich (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLSTICKSANDWICH, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLTAKURI(data: xr.DataArray) -> xr.DataArray:
    """
    Takuri (Dragonfly Doji with very long lower shadow) (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLTAKURI, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLTASUKIGAP(data: xr.DataArray) -> xr.DataArray:
    """
    Tasuki Gap (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLTASUKIGAP, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLTHRUSTING(data: xr.DataArray) -> xr.DataArray:
    """
    Thrusting Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLTHRUSTING, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLTRISTAR(data: xr.DataArray) -> xr.DataArray:
    """
    Tristar Pattern (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLTRISTAR, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLUNIQUE3RIVER(data: xr.DataArray) -> xr.DataArray:
    """
    Unique 3 River (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLUNIQUE3RIVER, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLUPSIDEGAP2CROWS(data: xr.DataArray) -> xr.DataArray:
    """
    Upside Gap Two Crows (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLUPSIDEGAP2CROWS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CDLXSIDEGAP3METHODS(data: xr.DataArray) -> xr.DataArray:
    """
    Upside/Downside Gap Three Methods (Pattern Recognition)
    Inputs:
        data:['open', 'high', 'low', 'close']
    Outputs:
        double series (values are -1, 0 or 1)
    """
    return multiple_series_call(talib.CDLXSIDEGAP3METHODS, data, ds.TIME, ds.FIELD, [f.OPEN, f.HIGH, f.LOW, f.CLOSE],
                                result_divider=100)


def CMO(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
    Chande Momentum Oscillator (Momentum Indicators)
    Inputs:
        data: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
         double series
    """
    return single_series_call(talib.CMO, data, ds.TIME, [timeperiod])


def CORREL(first: xr.DataArray, second: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Pearson's Correlation Coefficient (r) (Statistic Functions)
    Inputs:
        first: price series, if it contains 'asset' dimension, than result will contain dimension 'first' with related assets.
        second: price series, if it contains 'asset' dimension, than result will contain dimension 'second' with related assets.
    Parameters:
        timeperiod: 30
    Outputs:
        double series
    """
    return cross_series_call(talib.CORREL, first, second, [timeperiod])


def DEMA(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Double Exponential Moving Average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 30
    """
    return single_series_call(talib.DEMA, data, ds.TIME, [timeperiod])


def DX(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Directional Movement Index (Momentum Indicators)
        Input:
            data:  ['high', 'low', 'close']
        Parameters:
            timeperiod: 14
    """
    return multiple_series_call(talib.DX, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def EMA(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Exponential Moving Average
        Input:
            data: time series
        Parameters:
            timeperiod: 30
    """
    return single_series_call(talib.EMA, data, ds.TIME, [timeperiod])


def HT_DCPERIOD(data: xr.DataArray) -> xr.DataArray:
    """
        Hilbert Transform - Dominant Cycle Period (Cycle Indicators)
        Input:
            data: time series
    """
    return single_series_call(talib.HT_DCPERIOD, data, ds.TIME)


def HT_DCPHASE(data: xr.DataArray) -> xr.DataArray:
    """
        Hilbert Transform - Dominant Cycle Phase (Cycle Indicators)
        Input:
            data: time series
    """
    return single_series_call(talib.HT_DCPHASE, data, ds.TIME)


def HT_PHASOR(data: xr.DataArray) -> xr.DataArray:
    """
        Hilbert Transform - Phasor Components (Cycle Indicators)
        Input:
            data: time series
        Output:
            DataArray with a new dimension: 'ht_phasor' (coord:['inphase', 'quadrature'])
    """
    return single_series_call(talib.HT_PHASOR, data, ds.TIME, result_dim='ht_phasor',
                              result_coord=['inphase', 'quadrature'])


def HT_SINE(data: xr.DataArray) -> xr.DataArray:
    """
        Hilbert Transform - SineWave (Cycle Indicators)
        Input:
            data: time series
        Output:
            DataArray with a new dimension: 'ht_sine' (coord:['sine', 'leadsine'])
    """
    return single_series_call(talib.HT_SINE, data, ds.TIME, result_dim='ht_sine', result_coord=['sine', 'leadsine'])


def HT_TRENDLINE(data: xr.DataArray) -> xr.DataArray:
    """
        Hilbert Transform - Instantaneous Trendline (Overlap Studies)
        Input:
            data: time series
        Output:
            time series
    """
    return single_series_call(talib.HT_TRENDLINE, data, ds.TIME)


def HT_TRENDMODE(data: xr.DataArray) -> xr.DataArray:
    """
        Hilbert Transform - Trend vs Cycle Mode (Cycle Indicators)
        Input:
            data: time series
        Output:
            time series (values are 0 or 1)
    """
    return single_series_call(talib.HT_TRENDMODE, data, ds.TIME)


def KAMA(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Kaufman Adaptive Moving Average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 30
    """
    return single_series_call(talib.KAMA, data, ds.TIME, [timeperiod])


def LINEARREG(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Linear Regression (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
    """
    return single_series_call(talib.LINEARREG, data, ds.TIME, [timeperiod])


def LINEARREG_ANGLE(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Linear Regression Angle (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
    """
    return single_series_call(talib.LINEARREG_ANGLE, data, ds.TIME, [timeperiod])


def LINEARREG_INTERCEPT(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Linear Regression Intercept (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
    """
    return single_series_call(talib.LINEARREG_INTERCEPT, data, ds.TIME, [timeperiod])


def LINEARREG_SLOPE(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Linear Regression Slope (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
    """
    return single_series_call(talib.LINEARREG_SLOPE, data, ds.TIME, [timeperiod])


def MA(data: xr.DataArray, timeperiod: int = 30, matype: int = 0) -> xr.DataArray:
    """
        Moving average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 30
            matype: 0 see MaType
    """
    return single_series_call(talib.MA, data, ds.TIME, [timeperiod, matype])


def MACD(data: xr.DataArray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> xr.DataArray:
    """
        Moving Average Convergence/Divergence (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            fastperiod: 12
            slowperiod: 26
            signalperiod: 9
        Output:
            DataArray with a new dimension: 'macd' (coord:['macd', 'signal', 'hist'])
    """
    return single_series_call(talib.MACD, data, ds.TIME, [fastperiod, slowperiod, signalperiod],
                              'macd', ['macd', 'signal', 'hist'])


def MACDEXT(data: xr.DataArray, fastperiod: int = 12, fastmatype: int = 0, slowperiod: int = 26, slowmatype: int = 0,
            signalperiod: int = 9, signalmatype: int = 0) -> xr.DataArray:
    """
        MACD with controllable MA type (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            fastperiod: 12 (see MaType)
            fastmatype: 0
            slowperiod: 26
            slowmatype: 0
            signalperiod: 9
            signalmatype: 0
        Output:
            DataArray with a new dimension: 'macd' (coord:['macd', 'signal', 'hist'])
    """
    return single_series_call(talib.MACDEXT, data, ds.TIME,
                              [fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype],
                              'macd', ['macd', 'signal', 'hist'])


def MACDFIX(data: xr.DataArray, signalperiod: int = 9) -> xr.DataArray:
    """
        Moving Average Convergence/Divergence Fix 12/26 (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            fastperiod: 12
            slowperiod: 26
            signalperiod: 9
        Output:
            DataArray with a new dimension: 'macd' (coord:['macd', 'signal', 'hist'])
    """
    return single_series_call(talib.MACDFIX, data, ds.TIME, [signalperiod],
                              'macd', ['macd', 'signal', 'hist'])


def MAMA(data: xr.DataArray, fastlimit: float = 0.5, slowlimit: float = 0.05) -> xr.DataArray:
    """
        MESA Adaptive Moving Average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            fastlimit: 0.5
            slowlimit: 0.05
        Output:
            DataArray with a new dimension: 'mama' (coord:['mama', 'fama'])
    """
    return single_series_call(talib.MAMA, data, ds.TIME, [fastlimit, slowlimit],
                              'mama', ['mama', 'fama'])


def MFI(data: xr.DataArray, timeperiod: int = 14):
    """
    Money Flow Index (Momentum Indicators)
    Inputs:
        data: ['high', 'low', 'close', 'volume']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.MFI, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE, f.VOL], [timeperiod])


def MINUS_DI(data: xr.DataArray, timeperiod: int = 14):
    """
    Minus Directional Indicator (Momentum Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.MINUS_DI, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def MINUS_DM(data: xr.DataArray, timeperiod: int = 14):
    """
    Minus Directional Movement (Momentum Indicators)
    Inputs:
        data: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.MINUS_DM, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW], [timeperiod])


def MOM(data: xr.DataArray, timeperiod: int = 10):
    """
    Momentum (Momentum Indicators)
    Inputs:
        data: time series
    Parameters:
        timeperiod: 10
    Outputs:
        double series
    """
    return single_series_call(talib.MOM, data, ds.TIME, [timeperiod])


def NATR(data: xr.DataArray, timeperiod: int = 14):
    """
    Normalized Average True Range (Volatility Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.NATR, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def OBV(data: xr.DataArray, price_field: str = f.CLOSE):
    """
    Normalized Average True Range (Volatility Indicators)
    Inputs:
        data: ['close', 'volume']
        price_field: 'close'
    Outputs:
        double series
    """
    return multiple_series_call(talib.OBV, data, ds.TIME, ds.FIELD, [price_field, f.VOL])


def PLUS_DI(data: xr.DataArray, timeperiod: int = 14):
    """
    Plus Directional Indicator (Momentum Indicators)
    Inputs:
        data: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.PLUS_DI, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def PLUS_DM(data: xr.DataArray, timeperiod: int = 14):
    """
    Plus Directional Movement (Momentum Indicators)
    Inputs:
        data: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        double series
    """
    return multiple_series_call(talib.PLUS_DM, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW], [timeperiod])


def PPO(data: xr.DataArray, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> xr.DataArray:
    """
        Percentage Price Oscillator (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            fastperiod: 12
            slowperiod: 26
            matype: 0 (see MaType)
        Output:
            double series
    """
    return single_series_call(talib.PPO, data, ds.TIME, [fastperiod, slowperiod, matype])


def ROCR(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
         Rate of change ratio: (real/prevPrice) (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.ROCR, data, ds.TIME, [timeperiod])


def ROCR100(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Rate of change ratio 100 scale: (real/prevPrice)*100 (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.ROCR100, data, ds.TIME, [timeperiod])


def ROCP(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
         Rate of change Percentage: (real-prevPrice)/prevPrice (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.ROCP, data, ds.TIME, [timeperiod])


def ROC(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
         Rate of change : ((real/prevPrice)-1)*100 (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.ROC, data, ds.TIME, [timeperiod])


def RSI(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Relative Strength Index (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.RSI, data, ds.TIME, [timeperiod])


def SAR(data: xr.DataArray, acceleration: float = 0.02, maximum: float = 0.2) -> xr.DataArray:
    """
        Parabolic SAR (Overlap Studies)
        Input:
            data:  ['high', 'low']
        Parameters:
            acceleration: 0.02
            maximum: 0.2
        Output:
            double series
    """
    return multiple_series_call(talib.SAR, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW], [acceleration, maximum])


def SAREXT(data: xr.DataArray, startvalue: float = 0, offsetonreverse: float = 0,
           accelerationinitlong: float = 0.02, accelerationlong: float = 0.02, accelerationmaxlong: float = 0.2,
           accelerationinitshort: float = 0.02, accelerationshort: float = 0.02,
           accelerationmaxshort: float = 0.2) -> xr.DataArray:
    """
        Parabolic SAR - Extended (Overlap Studies) // look broken
        Input:
            data:  ['high', 'low']
        Parameters:
            startvalue: 0
            offsetonreverse: 0
            accelerationinitlong: 0.02
            accelerationlong: 0.02
            accelerationmaxlong: 0.2
            accelerationinitshort: 0.02
            accelerationshort: 0.02
            accelerationmaxshort: 0.2
        Output:
            double series
    """
    return multiple_series_call(talib.SAREXT, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW],
                                [
                                    startvalue, offsetonreverse,
                                    accelerationinitlong, accelerationlong, accelerationmaxlong,
                                    accelerationinitshort, accelerationshort, accelerationmaxshort
                                ])


def SMA(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Simple Moving Average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.SMA, data, ds.TIME, [timeperiod])


def STDDEV(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Standard Deviation (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.STDDEV, data, ds.TIME, [timeperiod])


def STOCH(data: xr.DataArray, fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0,
          slowd_period: int = 3, slowd_matype: int = 0) -> xr.DataArray:
    """
        Stochastic (Momentum Indicators)
        Input:
            data: [high, low, close]
        Parameters:
            fastk_period: 5
            slowk_period: 3
            slowk_matype: 0
            slowd_period: 3
            slowd_matype: 0
        Output:
            DataArray with the dimension 'field' (coord:['slowk', 'slowd'])
    """
    return multiple_series_call(talib.STOCH, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE],
                                [fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype],
                                ds.FIELD, ['slowk', 'slowd'])


def STOCHF(data: xr.DataArray, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> xr.DataArray:
    """
        Stochastic (Momentum Indicators)
        Input:
            data: [high, low, close]
        Parameters:
            fastk_period: 5
            fastd_period: 3
            fastd_matype: 0
        Output:
            DataArray with the dimension 'field' (coord:['fastk', 'fastd'])
    """
    return multiple_series_call(talib.STOCHF, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE],
                                [fastk_period, fastd_period, fastd_matype],
                                ds.FIELD, ['fastk', 'fastd'])


def STOCHRSI(data: xr.DataArray, timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3,
             fastd_matype: int = 0) -> xr.DataArray:
    """
        Stochastic Relative Strength Index (Momentum Indicators)
        Input:
            data: [high, low, close]
        Parameters:
            timeperiod: 14
            fastk_period: 5
            fastd_period: 3
            fastd_matype: 0
        Output:
            DataArray with a new dimension: 'stochrsi' (coord:['fastk', 'fastd'])
    """
    return single_series_call(talib.STOCHRSI, data, ds.TIME,
                              [timeperiod, fastk_period, fastd_period, fastd_matype],
                              'stochf', ['fastk', 'fastd'])


def T3(data: xr.DataArray, timeperiod: int = 5, vfactor: float = 0.7) -> xr.DataArray:
    """
        Triple Exponential Moving Average (T3) (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 5
            vfactor: 0.7
        Output:
            double series
    """
    return single_series_call(talib.T3, data, ds.TIME, [timeperiod, vfactor])


def TEMA(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Triple Exponential Moving Average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 30
        Output:
            double series
    """
    return single_series_call(talib.TEMA, data, ds.TIME, [timeperiod])


def TRANGE(data: xr.DataArray) -> xr.DataArray:
    """
        True Range (Volatility Indicators)
        Input:
            data:  ['high', 'low', 'close']
        Output:
            double series
    """
    return multiple_series_call(talib.TRANGE, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE])


def TRIMA(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Triangular Moving Average (Overlap Studies)
        Input:
            data: time series
        Parameters:
            timeperiod: 30
        Output:
            double series
    """
    return single_series_call(talib.TRIMA, data, ds.TIME, [timeperiod])


def TRIX(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (Momentum Indicators)
        Input:
            data: time series
        Parameters:
            timeperiod: 30
        Output:
            double series
    """
    return single_series_call(talib.TRIX, data, ds.TIME, [timeperiod])


def TSF(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
        Time Series Forecast (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 14
        Output:
            double series
    """
    return single_series_call(talib.TSF, data, ds.TIME, [timeperiod])


def TYPPRICE(data: xr.DataArray) -> xr.DataArray:
    """
        Typical Price (Price Transform)
        Input:
            data:  ['high', 'low', 'close']
        Output:
            double series
    """
    return multiple_series_call(talib.TYPPRICE, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE])


def ULTOSC(data: xr.DataArray, timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28) -> xr.DataArray:
    """
        Ultimate Oscillator (Momentum Indicators)
        Input:
            data:  ['high', 'low', 'close']
        Parameters:
            timeperiod1: 7
            timeperiod2: 14
            timeperiod3: 28
        Output:
            double series
    """
    return multiple_series_call(talib.ULTOSC, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE],
                                [timeperiod1, timeperiod2, timeperiod3])


def VAR(data: xr.DataArray, timeperiod: int = 5) -> xr.DataArray:
    """
        Variance (Statistic Functions)
        Input:
            data: time series
        Parameters:
            timeperiod: 5
        Output:
            double series
    """
    return single_series_call(talib.VAR, data, ds.TIME, [timeperiod])


def WCLPRICE(data: xr.DataArray) -> xr.DataArray:
    """
        Weighted Close Price (Price Transform)
        Input:
            data: time series
        Output:
            double series
    """
    return multiple_series_call(talib.WCLPRICE, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE])


def WILLR(data: xr.DataArray, timeperiod: int = 14) -> xr.DataArray:
    """
         Williams' %R (Momentum Indicators)
        Parameters:
            timeperiod: 14
        Input:
            data: ['high', 'low', 'close']
        Output:
            double series
    """
    return multiple_series_call(talib.WILLR, data, ds.TIME, ds.FIELD, [f.HIGH, f.LOW, f.CLOSE], [timeperiod])


def WMA(data: xr.DataArray, timeperiod: int = 30) -> xr.DataArray:
    """
        Weighted Moving Average (Overlap Studies)
        Parameters:
            timeperiod: 30
        Input:
            data: time series
        Output:
            double series
    """
    return single_series_call(talib.WMA, data, ds.TIME, [timeperiod])


def multiple_series_call(origin_func,
                         series: xr.DataArray, sequence_dim: str, field_dim: str, fields: tp.List[str],
                         args: list = [],
                         result_dim: tp.Optional[str] = None, result_coord: tp.Optional[tp.List[str]] = None,
                         result_divider: tp.Union[float, int] = 1):
    other_dims = [d for d in series.dims if d != sequence_dim and d != field_dim]
    other_coords = [series.coords[d].values for d in other_dims]

    result = series[{field_dim: 0}].copy(True)
    if result_dim is not None:
        result = xr.concat([result] * len(result_coord), pd.Index(result_coord, name=result_dim))
    result.loc[:] = np.nan

    for vals in itertools.product(*other_coords):
        selector = dict(zip(other_dims, vals))
        series_slice = series.loc[selector].dropna(sequence_dim)

        if len(series_slice.coords[sequence_dim]) > 0:
            cur_args = [series_slice.loc[{field_dim: f}].values for f in fields]
            cur_args = cur_args + list(args)

            result_slice = origin_func(*cur_args)
            if result_divider != 1:
                result_slice = result_slice / result_divider
            selector[sequence_dim] = series_slice.coords[sequence_dim]

            if result_dim is None:
                result.loc[selector] = result_slice
            else:
                for i in range(len(result_coord)):
                    selector[result_dim] = result_coord[i]
                    result.loc[selector] = result_slice[i]

    return result


def single_series_call(origin_func, series: xr.DataArray, sequence_dim: str, args: list = [],
                       result_dim: tp.Optional[str] = None, result_coord: tp.Optional[tp.List[str]] = None,
                       result_divider: tp.Union[float, int] = 1):
    other_dims = [d for d in series.dims if d != sequence_dim]
    other_coords = [series.coords[d].values for d in other_dims]

    result = series.copy(True)
    if result_dim is not None:
        result = xr.concat([result] * len(result_coord), pd.Index(result_coord, name=result_dim))
    result.loc[:] = np.nan

    for vals in itertools.product(*other_coords):
        selector = dict(zip(other_dims, vals))
        series_slice = series.loc[selector]
        series_slice = series_slice.dropna(sequence_dim)
        if len(series_slice) > 0:
            result_slice = origin_func(series_slice.values, *args)
            if result_divider != 1:
                result_slice = result_slice / result_divider
            selector[sequence_dim] = series_slice.coords[sequence_dim]
            if result_dim is None:
                result.loc[selector] = result_slice
            else:
                for i in range(len(result_coord)):
                    selector[result_dim] = result_coord[i]
                    result.loc[selector] = result_slice[i]

    return result


def cross_series_call(orig_func, first: xr.DataArray, second: xr.DataArray, args: list = None):
    if args is None:
        args = []

    result = first.copy(True)
    if ds.ASSET in first.dims:
        result = result.rename({ds.ASSET: 'first'})
    if ds.ASSET in second.dims:
        result = xr.concat([result] * len(second.coords[ds.ASSET]),
                           pd.Index(second.coords[ds.ASSET].values, name='second'))
    result.loc[:] = np.nan
    other_dims = [d for d in second.dims if d != ds.TIME and d != ds.ASSET]
    other_coords = [first.coords[d].values for d in other_dims]
    first_selectors = None
    if ds.ASSET in first.dims:
        first_selectors = first.coords[ds.ASSET].values
    else:
        first_selectors = (None,)
    second_selectors = None
    if ds.ASSET in second.dims:
        second_selectors = second.coords[ds.ASSET].values
    else:
        second_selectors = (None,)
    for vals in itertools.product(first_selectors, second_selectors, *other_coords):

        selector = dict(zip(other_dims, vals[2:]))

        first_selector = selector.copy()
        second_selector = selector.copy()
        result_selector = selector.copy()

        if vals[0] is not None:
            first_selector[ds.ASSET] = vals[0]
            result_selector['first'] = vals[0]

        if vals[1] is not None:
            second_selector[ds.ASSET] = vals[1]
            result_selector['second'] = vals[1]

        first_slice = first.loc[first_selector].dropna(ds.TIME)
        second_slice = second.loc[second_selector].dropna(ds.TIME)

        time_intersection = np.intersect1d(first_slice.coords[ds.TIME].values, second_slice.coords[ds.TIME].values)

        if len(time_intersection) < 1:
            continue

        first_input = first_slice.loc[{ds.TIME: time_intersection}]
        second_input = second_slice.loc[{ds.TIME: time_intersection}]
        result_selector[ds.TIME] = time_intersection

        result_slice = orig_func(first_input.values, second_input.values, *args)
        result.loc[result_selector] = result_slice
    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from qnt.data import load_data, load_assets

    assets = load_assets()
    ids = [i['id'] for i in assets[0:10]]

    data = load_data(assets=ids, dims=(ds.TIME, ds.ASSET, ds.FIELD), forward_order=True)

    ad = AD(data)
    # plt.plot(data.coords[ds.TIME].values, ad.sel(asset = ids[1]).values)
    # plt.show()

    adocs = ADOSC(data, 12, 40)
    # plt.plot(data.coords[ds.TIME].values, adocs.sel(asset = ids[1]).values)
    # plt.show()

    adx = ADX(data, 14)
    # plt.plot(data.coords[ds.TIME].values, adx.sel(asset = ids[1]).values)
    # plt.show()

    adxr = ADXR(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, adx.sel(asset = ids[0]).values, 'b',
    #     data.coords[ds.TIME].values, adxr.sel(asset = ids[0]).values, 'r',
    # )
    # plt.show()

    apo = APO(data, 14, 26, 1)
    # plt.plot(data.coords[ds.TIME].values, apo.sel(asset = ids[0], field='close').values)
    # plt.show()

    aroon = AROON(data, 50)
    # plt.plot(
    #     data.coords[ds.TIME].values, aroon.sel(asset = ids[0], aroon='down').values, 'b',
    #     data.coords[ds.TIME].values, aroon.sel(asset = ids[0], aroon='up').values, 'r'
    # )
    # plt.show()

    aroonocs = AROONOSC(data, 14)
    # plt.plot(data.coords[ds.TIME].values, aroonocs.sel(asset=ids[0]).values, 'b')
    # plt.show()

    atr = ATR(data, 14)
    # plt.plot(data.coords[ds.TIME].values, atr.sel(asset=ids[0]).values, 'b')
    # plt.show()

    bbands = BBANDS(data, 14, 2, 2, 0)
    # plt.plot(
    #     data.coords[ds.TIME].values, bbands.sel(asset = ids[0], bbands='upper').values, 'b',
    #     data.coords[ds.TIME].values, bbands.sel(asset = ids[0], bbands='lower').values, 'r',
    #     data.coords[ds.TIME].values, bbands.sel(asset = ids[0], bbands='middle').values, 'g',
    # )
    # plt.show()

    prices = data.loc[{ds.FIELD: [f.OPEN, f.LOW, f.HIGH, f.CLOSE]}]
    index_prices = (prices * data.loc[{ds.FIELD: f.VOL}]).sum(ds.ASSET)
    beta = BETA(prices, index_prices, 5)
    # plt.plot(data.coords[ds.TIME].values, beta.sel(first=ids[2], field='close').values, 'b')
    # plt.show()

    bop = BOP(data)
    # plt.plot(data.coords[ds.TIME].values, bop.sel(asset=ids[2]).values, 'b')
    # plt.show()

    # cci = CCI(data)
    # plt.plot(data.coords[ds.TIME].values, cci.sel(asset=ids[2]).values, 'b')
    # plt.show()

    cdl2crows = CDL2CROWS(data)
    # s = cdl2crows.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdl3blackcrows = CDL3BLACKCROWS(data)
    # s = cdl3blackcrows.sel(asset=ids[2]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdl3inside = CDL3INSIDE(data)
    # s = cdl3inside.sel(asset=ids[2]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdl3linestrike = CDL3LINESTRIKE(data)
    # s = cdl3linestrike.sel(asset=ids[2]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdl3outside = CDL3OUTSIDE(data)
    # s = cdl3outside.sel(asset=ids[2]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdl3starinsouth = CDL3STARSINSOUTH(data)
    # s = cdl3starinsouth.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdl3whitesoldiers = CDL3WHITESOLDIERS(data)
    # s = cdl3whitesoldiers.sel(asset=ids[3]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlabandonedbaby = CDLABANDONEDBABY(data, 0.3)
    # s = cdlabandonedbaby.sel(asset=ids[3]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdladvanceblock = CDLADVANCEBLOCK(data)
    # s = cdladvanceblock.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot( s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlbelthold = CDLBELTHOLD(data)
    # s = cdlbelthold.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlbreakaway = CDLBREAKAWAY(data)
    # s = cdlbreakaway.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlclosingmarubozu = CDLCLOSINGMARUBOZU(data)
    # s = cdlclosingmarubozu.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlconcealbabyswallow = CDLCONCEALBABYSWALL(data)
    # s = cdlconcealbabyswallow.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlcounterattack = CDLCOUNTERATTACK(data)
    # s = cdlcounterattack.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdldarccloudcover = CDLDARKCLOUDCOVER(data, 0.5)
    # s = cdldarccloudcover.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdldoji = CDLDOJI(data)
    # s = cdldoji.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdldojistar = CDLDOJISTAR(data)
    # s = cdldojistar.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdldragonflydoji = CDLDRAGONFLYDOJI(data)
    # s = cdldragonflydoji.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlengulfing = CDLENGULFING(data)
    # s = cdlengulfing.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdleveningdojistar = CDLEVENINGDOJISTAR(data, 0.3)
    # s = cdleveningdojistar.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdleveningstar = CDLEVENINGSTAR(data, 0.3)
    # s = cdleveningstar.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlgapsidesidewhite = CDLGAPSIDESIDEWHITE(data)
    # s = cdlgapsidesidewhite.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlgravestonedoji = CDLGRAVESTONEDOJI(data)
    # s = cdlgravestonedoji.sel(asset=ids[9]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlhammer = CDLHAMMER(data)
    # s = cdlhammer.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlhangingman = CDLHANGINGMAN(data)
    # s = cdlhangingman.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlharami = CDLHARAMI(data)
    # s = cdlharami.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlharamicross = CDLHARAMICROSS(data)
    # s = cdlharamicross.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlhighwave = CDLHIGHWAVE(data)
    # s = cdlhighwave.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlhikkake = CDLHIKKAKE(data)
    # s = cdlhikkake.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlhikkakemod = CDLHIKKAKEMOD(data)
    # s = cdlhikkakemod.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    dlhomingpigeon = CDLHOMINGPIGEON(data)
    # s = dlhomingpigeon.sel(asset=ids[1]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlidentical3crows = CDLIDENTICAL3CROWS(data)
    # s = cdlidentical3crows.sel(asset=ids[3]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlinneck = CDLINNECK(data)
    # s = cdlinneck.sel(asset=ids[3]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    dlinvertedhammer = CDLINVERTEDHAMMER(data)
    # s = dlinvertedhammer.sel(asset=ids[3]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlkicking = CDLKICKING(data)
    # s = cdlkicking.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlkickingbylength = CDLKICKINGBYLENGTH(data)
    # s = cdlkickingbylength.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlladderbottom = CDLLADDERBOTTOM(data)
    # s = cdlladderbottom.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdllongleggeddoji = CDLLONGLEGGEDDOJI(data)
    # s = cdllongleggeddoji.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdllongline = CDLLONGLINE(data)
    # s = cdllongline.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlmarubozu = CDLMARUBOZU(data)
    # s = cdlmarubozu.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlmatchinglow = CDLMATCHINGLOW(data)
    # s = cdlmatchinglow.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlmathold = CDLMATHOLD(data, 0.5)
    # s = cdlmathold.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlmorningdojistar = CDLMORNINGDOJISTAR(data, 0.3)
    # s = cdlmorningdojistar.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlmorningstar = CDLMORNINGSTAR(data, 0.3)
    # s = cdlmorningstar.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlonneck = CDLONNECK(data)
    # s = cdlonneck.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlpiercing = CDLPIERCING(data)
    # s = cdlpiercing.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlrickshawman = CDLRICKSHAWMAN(data)
    # s = cdlrickshawman.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlrisefall3methods = CDLRISEFALL3METHODS(data)
    # s = cdlrisefall3methods.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlseparatinglines = CDLSEPARATINGLINES(data)
    # s = cdlseparatinglines.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlshootingstar = CDLSHOOTINGSTAR(data)
    # s = cdlshootingstar.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlshortline = CDLSHORTLINE(data)
    # s = cdlshortline.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlspinningtop = CDLSPINNINGTOP(data)
    # s = cdlspinningtop.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlstalledpattern = CDLSTALLEDPATTERN(data)
    # s = cdlstalledpattern.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlsticksandwich = CDLSTICKSANDWICH(data)
    # s = cdlsticksandwich.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdltakuri = CDLTAKURI(data)
    # s = cdltakuri.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdltasukigap = CDLTASUKIGAP(data)
    # s = cdltasukigap.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlthrusting = CDLTHRUSTING(data)
    # s = cdlthrusting.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdltristar = CDLTRISTAR(data)
    # s = cdltristar.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlunique3river = CDLUNIQUE3RIVER(data)
    # s = cdlunique3river.sel(asset=ids[0]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlupsidegap2crows = CDLUPSIDEGAP2CROWS(data)
    # s = cdlupsidegap2crows.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cdlxsidegap3methods = CDLXSIDEGAP3METHODS(data)
    # s = cdlxsidegap3methods.sel(asset=ids[5]).dropna(ds.TIME)
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    cmo = CMO(data, 14)
    # s = cmo.sel(asset=ids[5], field='close').dropna('time')
    # plt.plot(s.coords[ds.TIME].values, s.values, 'b')
    # plt.show()

    correl = CORREL(prices, prices, 30)
    # plt.plot(data.coords[ds.TIME].values, correl.sel(first=ids[1], second=ids[4], field='close').values, 'b')
    # plt.show()

    dema = DEMA(data, 200)
    # plt.plot(
    #     data.coords[ds.TIME].values, data.sel(asset = ids[0], field='open').values, 'b',
    #     data.coords[ds.TIME].values, dema.sel(asset = ids[0], field='open').values , 'r',
    # )
    # plt.show()

    dx = DX(data, 14)
    # plt.plot(data.coords[ds.TIME].values, dx.sel(asset = ids[0]).values , 'r')
    # plt.show()

    ema = EMA(data, 200)
    # plt.plot(
    #     data.coords[ds.TIME].values, data.sel(asset = ids[0], field='open').values, 'b',
    #     data.coords[ds.TIME].values, ema.sel(asset = ids[0], field='open').values , 'r',
    # )
    # plt.show()

    ht_dcperiod = HT_DCPERIOD(data)
    # plt.plot(data.coords[ds.TIME].values, ht_dcperiod.sel(asset = ids[0], field='close').values , 'r')
    # plt.show()

    ht_dcphase = HT_DCPHASE(data)
    # plt.plot(data.coords[ds.TIME].values, ht_dcphase.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    ht_phasor = HT_PHASOR(data)
    # plt.plot(
    #     data.coords[ds.TIME].values, ht_phasor.sel(asset = ids[2], field='close', ht_phasor='inphase').values , 'r',
    #     data.coords[ds.TIME].values, ht_phasor.sel(asset = ids[2], field='close', ht_phasor='quadrature').values , 'b'
    # )
    # plt.show()

    ht_sine = HT_SINE(data)
    # plt.plot(
    #     data.coords[ds.TIME].values, ht_sine.sel(asset = ids[2], field='close', ht_sine='sine').values , 'r',
    #     data.coords[ds.TIME].values, ht_sine.sel(asset = ids[2], field='close', ht_sine='leadsine').values , 'b'
    # )
    # plt.show()

    ht_trendline = HT_TRENDLINE(data)
    # plt.plot(data.coords[ds.TIME].values, ht_trendline.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    ht_trendmode = HT_TRENDMODE(data)
    # plt.plot(data.coords[ds.TIME].values, ht_trendmode.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    kama = KAMA(data, 200)
    # plt.plot(
    #     data.coords[ds.TIME].values, data.sel(asset = ids[0], field='close').values, 'b',
    #     data.coords[ds.TIME].values, kama.sel(asset = ids[0], field='close').values , 'r',
    # )
    # plt.show()

    linearreg = LINEARREG(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, data.sel(asset = ids[0], field='close').values, 'b',
    #     data.coords[ds.TIME].values, linearreg.sel(asset = ids[0], field='close').values , 'r',
    # )
    # plt.show()

    linearreg_angle = LINEARREG_ANGLE(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, linearreg_angle.sel(asset = ids[0], field='close').values , 'r',
    # )
    # plt.show()

    linearreg_intercept = LINEARREG_INTERCEPT(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, linearreg_intercept.sel(asset = ids[0], field='close').values , 'r',
    # )
    # plt.show()

    linearreg_slope = LINEARREG_SLOPE(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, linearreg_slope.sel(asset=ids[0], field='close').values, 'r',
    # )
    # plt.show()

    ma = MA(data, 100, 1)
    # plt.plot(
    #     data.coords[ds.TIME].values, data.sel(asset = ids[0], field='close').values, 'b',
    #     data.coords[ds.TIME].values, ma.sel(asset = ids[0], field='close').values , 'r',
    # )
    # plt.show()

    macd = MACD(data, 12, 26, 9)
    # plt.plot(
    #     data.coords[ds.TIME].values, macd.sel(asset = ids[2], field='close', macd='macd').values , 'r',
    #     data.coords[ds.TIME].values, macd.sel(asset = ids[2], field='close', macd='signal').values , 'g',
    #     data.coords[ds.TIME].values, macd.sel(asset = ids[2], field='close', macd='hist').values , 'b'
    # )
    # plt.show()

    macdext = MACDEXT(data, 12, 1, 26, 1, 9, 1)
    # plt.plot(
    #     data.coords[ds.TIME].values, macdext.sel(asset = ids[2], field='close', macd='macd').values , 'r',
    #     data.coords[ds.TIME].values, macdext.sel(asset = ids[2], field='close', macd='signal').values , 'g',
    #     data.coords[ds.TIME].values, macdext.sel(asset = ids[2], field='close', macd='hist').values , 'b'
    # )
    # plt.show()

    macdfix = MACDFIX(data, 9)
    # plt.plot(
    #     data.coords[ds.TIME].values, macdfix.sel(asset = ids[2], field='close', macd='macd').values , 'r',
    #     data.coords[ds.TIME].values, macdfix.sel(asset = ids[2], field='close', macd='signal').values , 'g',
    #     data.coords[ds.TIME].values, macdfix.sel(asset = ids[2], field='close', macd='hist').values , 'b'
    # )
    # plt.show()

    mama = MAMA(data, 0.5, 0.05)
    # plt.plot(
    #     data.coords[ds.TIME].values, mama.sel(asset = ids[2], field='close', mama='mama').values , 'r',
    #     data.coords[ds.TIME].values, mama.sel(asset = ids[2], field='close', mama='fama').values , 'g',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    mfi = MFI(data, 14)
    # plt.plot(data.coords[ds.TIME].values, mfi.sel(asset = ids[2]).values , 'r')
    # plt.show()

    minus_di = MINUS_DI(data, 14)
    # plt.plot(data.coords[ds.TIME].values, minus_di.sel(asset = ids[2]).values , 'r')
    # plt.show()

    minus_dm = MINUS_DM(data, 14)
    # plt.plot(data.coords[ds.TIME].values, minus_dm.sel(asset = ids[2]).values , 'r')
    # plt.show()

    mom = MOM(data, 10)
    # plt.plot(data.coords[ds.TIME].values, mom.sel(asset = ids[0], field='close').values , 'r')
    # plt.show()

    natr = NATR(data, 14)
    # plt.plot(data.coords[ds.TIME].values, natr.sel(asset = ids[0]).values , 'r')
    # plt.show()

    obv = OBV(data, f.CLOSE)
    # plt.plot(data.coords[ds.TIME].values, obv.sel(asset = ids[0]).values , 'r')
    # plt.show()

    plus_di = PLUS_DI(data, 14)
    # plt.plot(data.coords[ds.TIME].values, plus_di.sel(asset = ids[2]).values , 'r')
    # plt.show()

    plus_dm = PLUS_DM(data, 14)
    # plt.plot(data.coords[ds.TIME].values, plus_dm.sel(asset = ids[2]).values , 'r')
    # plt.show()

    ppo = PPO(data, 12, 26, 0)
    # plt.plot(data.coords[ds.TIME].values, ppo.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    rocr = ROCR(data, 12)
    # plt.plot(data.coords[ds.TIME].values, rocr.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    rocp = ROCP(data, 12)
    # plt.plot(data.coords[ds.TIME].values, rocp.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    roc = ROC(data, 12)
    # plt.plot(data.coords[ds.TIME].values, roc.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    rocr100 = ROCR100(data, 12)
    # plt.plot(data.coords[ds.TIME].values, rocr100.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    rsi = RSI(data, 12)
    # plt.plot(data.coords[ds.TIME].values, rsi.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    sar = SAR(data, 0.02, 0.2)
    # plt.plot(
    #     data.coords[ds.TIME].values, sar.sel(asset = ids[2]).values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    sarext = SAREXT(data)
    # plt.plot(
    #     data.coords[ds.TIME].values, sarext.sel(asset = ids[2]).values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    sma = SMA(data, 20)
    # plt.plot(
    #     data.coords[ds.TIME].values, sma.sel(asset = ids[2], field='close').values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    stddev = STDDEV(data, 20)
    # plt.plot(
    #     data.coords[ds.TIME].values, stddev.sel(asset=ids[2], field='close').values, 'r'
    # )
    # plt.show()

    stoch = STOCH(data, 5, 3, 1, 3, 1)
    # plt.plot(
    #     data.coords[ds.TIME].values, stoch.sel(asset = ids[2], stoch='slowk').values , 'r',
    #     data.coords[ds.TIME].values, stoch.sel(asset = ids[2], stoch='slowd').values , 'g'
    # )
    # plt.show()

    stochf = STOCHF(data, 5, 3, 1)
    # plt.plot(
    #     data.coords[ds.TIME].values, stochf.sel(asset = ids[2], stochf='fastk').values , 'r',
    #     data.coords[ds.TIME].values, stochf.sel(asset = ids[2], stochf='fastd').values , 'g'
    # )
    # plt.show()

    stochrsi = STOCHRSI(data.loc[{ds.FIELD: f.CLOSE}], 14, 5, 3, 1)
    # plt.plot(
    #     data.coords[ds.TIME].values, stochrsi.sel(asset = ids[2], stochf='fastk').values , 'r',
    #     data.coords[ds.TIME].values, stochrsi.sel(asset = ids[2], stochf='fastd').values , 'g'
    # )
    # plt.show()

    t3 = T3(data, 5, 0.7)
    # plt.plot(
    #     data.coords[ds.TIME].values, t3.sel(asset = ids[2], field='close').values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    tema = TEMA(data, 30)
    # plt.plot(
    #     data.coords[ds.TIME].values, tema.sel(asset = ids[2], field='close').values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    trange = TRANGE(data)
    # plt.plot(data.coords[ds.TIME].values, trange.sel(asset = ids[2]).values , 'r')
    # plt.show()

    trima = TRIMA(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, trima.sel(asset = ids[2], field='close').values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    trix = TRIX(data)
    # plt.plot(data.coords[ds.TIME].values, trix.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    tsf = TSF(data, 14)
    # plt.plot(
    #     data.coords[ds.TIME].values, tsf.sel(asset = ids[2], field='close').values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    typprice = TYPPRICE(data)
    # plt.plot(
    #     data.coords[ds.TIME].values, typprice.sel(asset = ids[2]).values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    ultosc = ULTOSC(data)
    # plt.plot(data.coords[ds.TIME].values, ultosc.sel(asset = ids[2]).values , 'r')
    # plt.show()

    var = VAR(data)
    # plt.plot(data.coords[ds.TIME].values, var.sel(asset = ids[2], field='close').values , 'r')
    # plt.show()

    wclprice = WCLPRICE(data)
    # plt.plot(
    #     data.coords[ds.TIME].values, wclprice.sel(asset = ids[2]).values , 'r',
    #     data.coords[ds.TIME].values, data.sel(asset = ids[2], field='close').values , 'b'
    # )
    # plt.show()

    willr = WILLR(data)
    # plt.plot(data.coords[ds.TIME].values, willr.sel(asset = ids[2]).values , 'r')
    # plt.show()

    wma = WMA(data, 14)
    plt.plot(
        data.coords[ds.TIME].values, wma.sel(asset=ids[2], field='close').values, 'r',
        data.coords[ds.TIME].values, data.sel(asset=ids[2], field='close').values, 'b'
    )
    plt.show()
