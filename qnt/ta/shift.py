import numpy as np
import numba as nb
import qnt.ta.ndadapter as nda
from qnt.log import log_info, log_err


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def shift_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods + 1,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            tail[idx] = series[i]
            if not_nan_cnt >= periods:
                result[i] = tail[idx - periods]
            not_nan_cnt += 1
    return result


def shift(series: nda.NdType, periods: int = 1) -> nda.NdType:
    return nda.nd_universal_adapter(shift_np_1d, (series,), (periods,))


if __name__ == "__main__":
    arr = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 9, 0], np.double)
    sh = shift(arr, 2)
    log_info(sh)
