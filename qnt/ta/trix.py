from qnt.ta.roc import roc
from qnt.ta.ema import tema
from qnt.ta.ndadapter import NdType
import typing as tp


def trix(series: NdType, periods: tp.Any = 18) -> NdType:
    ma = tema(series, periods)
    return roc(ma, 1)
