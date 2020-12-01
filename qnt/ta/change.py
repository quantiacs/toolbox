import numpy as np
import numba as nb
import qnt.ta.ndadapter as nda
from qnt.ta.shift import shift, shift_np_1d


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def change_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    shifted = shift_np_1d(series, periods)
    return series - shifted


def change(series: np.ndarray, periods: int = 1) -> np.ndarray:
    shifted = shift(series, periods)
    return series - shifted
