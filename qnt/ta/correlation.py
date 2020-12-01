import numpy as np
import numba as nb
import qnt.ta.ndadapter as nda
from qnt.ta.roc import roc


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def variance_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            tail[idx] = series[i]
            if not_nan_cnt >= periods:
                result[i] = np.var(tail)
            not_nan_cnt += 1
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.int64), nopython=True)
def std_np_1d(series: np.ndarray, periods: int) -> np.ndarray:
    tail = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(series.shape, np.nan, dtype=np.double)
    for i in range(series.shape[0]):
        if not np.isnan(series[i]):
            idx = not_nan_cnt % tail.shape[0]
            tail[idx] = series[i]
            if not_nan_cnt >= periods:
                result[i] = np.std(tail)
            not_nan_cnt += 1
    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.int64), nopython=True)
def covariance_np_1d(x: np.ndarray, y: np.ndarray, periods: int) -> np.ndarray:
    tail_x = np.empty((periods,), dtype=np.double)
    tail_y = np.empty((periods,), dtype=np.double)
    not_nan_cnt = 0
    result = np.full(x.shape, np.nan, dtype=np.double)
    for i in range(x.shape[0]):
        if not np.isnan(x[i]) and not np.isnan(y[i]):
            idx = not_nan_cnt % tail_x.shape[0]
            tail_x[idx] = x[i]
            tail_y[idx] = y[i]
            if not_nan_cnt >= periods:
                mx = np.mean(tail_x)
                vx = (tail_x - mx)
                my = np.mean(tail_y)
                vy = (tail_y - my)
                cov = np.mean(vx * vy)
                result[i] = cov
            not_nan_cnt += 1
    return result


def variance(series: nda.NdType, periods: int = 1) -> nda.NdType:
    return nda.nd_universal_adapter(variance_np_1d, (series,), (periods,))


def std(series: nda.NdType, periods: int = 1) -> nda.NdType:
    return nda.nd_universal_adapter(std_np_1d, (series,), (periods,))


def covariance(x: nda.NdType, y: nda.NdType, periods: int = 1) -> nda.NdType:
    return nda.nd_universal_adapter(covariance_np_1d, (x, y), (periods,))


def beta(price_x, price_y, periods=252):
    rx = roc(price_x, 1) / 100
    ry = roc(price_y, 1) / 100
    return covariance(rx, ry, periods) / variance(ry, periods)


def correlation(price_x, price_y, periods=252):
    rx = roc(price_x, 1) / 100
    ry = roc(price_y, 1) / 100
    # print(rx)
    # print(ry)
    return covariance(rx, ry, periods) / (std(ry, periods) * std(rx, periods))


if __name__ == '__main__':
    d1_array = np.array([0.1, 1, 2, 3, 4, np.nan, 5, np.nan, 6, 7], np.double)
    d2_array = np.array([0.1, 3, 5, 9, 12, np.nan, 15, np.nan, 18, 21], np.double)
    d1_result = correlation(d1_array, d2_array, 4)
    print("d1_array:\n", d1_array, '\n')
    print("d2_array:\n", d2_array, '\n')
    print('d1_result:\n', d1_result)
    print('---')
