import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import qnt.filter as qnfilter

pytestmark = pytest.mark.usefixtures("quiet_tracking")

DAYS = pd.date_range("2024-01-01", periods=5, freq="D")
ASSETS = ["a", "b", "c", "d", "e"]


def make_data(times: pd.DatetimeIndex = DAYS, assets: list[str] = ASSETS) -> xr.DataArray:
    return xr.DataArray(
        np.ones((1, len(times), len(assets))),
        dims=["field", "time", "asset"],
        coords={"field": ["close"], "time": times, "asset": assets},
    )


def criterion_of(
    rows: npt.ArrayLike, times: pd.DatetimeIndex | None = None, assets: list[str] | None = None
) -> xr.DataArray:
    values = np.asarray(rows, dtype=np.float64)
    return xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={
            "time": times if times is not None else DAYS[: values.shape[0]],
            "asset": assets if assets is not None else ASSETS[: values.shape[1]],
        },
    )


def slow_layout(criterion: xr.DataArray) -> xr.DataArray:
    return criterion.expand_dims(extra=[0])


def expected_marks(
    rows: npt.NDArray[np.float64], top: int, ascending: bool
) -> npt.NDArray[np.float64]:
    marks = np.zeros_like(rows)
    for day, row in enumerate(rows):
        order = np.argsort(row if ascending else -row, kind="stable")
        marks[day, order[:top]] = 1.0
    return marks


@pytest.fixture(scope="module")
def distinct_rows() -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(7)
    return np.array([rng.permutation(5) + rng.random(5) * 0.1 for _ in range(5)])


@pytest.mark.parametrize("top", [1, 3])
@pytest.mark.parametrize("ascending", [True, False])
def test_slow_path_marks_top_assets_like_numpy_argsort(
    distinct_rows: npt.NDArray[np.float64], top: int, ascending: bool
) -> None:
    criterion = slow_layout(criterion_of(distinct_rows))
    result = qnfilter.rank_assets_by(make_data(), criterion, top, ascending)

    assert result.dims == ("time", "asset")
    assert_array_equal(result.values, expected_marks(distinct_rows, top, ascending))


@pytest.mark.parametrize(
    ("rows", "top", "ascending", "expected"),
    [
        ([[1, 1, 2, 3, 4]], 2, True, [[1, 1, 0, 0, 0]]),
        ([[1, 1, 2, 3, 4]], 1, True, [[0, 0, 0, 0, 0]]),
        ([[3, 3, 2, 1, 0]], 2, False, [[1, 1, 0, 0, 0]]),
        ([[5, 5, 5, 1, 9]], 2, True, [[0, 0, 0, 1, 0]]),
        ([[5, 5, 5, 1, 9]], 4, True, [[1, 1, 1, 1, 0]]),
    ],
)
def test_slow_path_ties_use_average_rank(
    rows: list[list[float]], top: int, ascending: bool, expected: list[list[float]]
) -> None:
    criterion = slow_layout(criterion_of(rows))

    result = qnfilter.rank_assets_by(make_data(DAYS[:1]), criterion, top, ascending)

    assert_array_equal(result.values, np.array(expected, dtype=np.float64))


def test_slow_path_leaves_unlisted_days_unmarked() -> None:
    rows = [[1.0, 2.0, 3.0, 4.0, 5.0]]
    criterion = slow_layout(criterion_of(rows, times=DAYS[1:2]))

    result = qnfilter.rank_assets_by(make_data(), criterion, 2, True)

    expected = np.zeros((5, 5))
    expected[1, :2] = 1.0
    assert_array_equal(result.values, expected)


def test_rank_assets_by_subset_of_assets_and_days_keeps_data_layout() -> None:
    rows = [[9.0, 1.0, 5.0], [2.0, 8.0, 4.0]]
    criterion = criterion_of(rows, times=DAYS[1:3], assets=["e", "c", "a"])

    result = qnfilter.rank_assets_by(make_data(), criterion, 1, True)

    expected = np.zeros((5, 5))
    expected[1, 2] = 1.0
    expected[2, 4] = 1.0
    assert result.dims == ("time", "asset")
    assert_array_equal(result.values, expected)


@pytest.mark.parametrize("metric", ["zzz", "avg", "max", "sum"])
def test_calc_rolling_metric_rejects_unknown_metric(metric: str) -> None:
    close = criterion_of(np.arange(10.0).reshape(5, 2))

    with pytest.raises(ValueError, match=f"Unsupported metric: {metric}"):
        qnfilter.calc_rolling_metric(close, 2, metric)


def test_calc_rolling_metric_known_metrics_match_numpy() -> None:
    close = criterion_of(np.array([[1.0], [3.0], [8.0], [4.0]]))

    mean = qnfilter.calc_rolling_metric(close, 2, "mean").values[:, 0]
    std = qnfilter.calc_rolling_metric(close, 2, "std").values[:, 0]

    assert np.isnan(mean[0]) and np.isnan(std[0])
    assert_array_equal(mean[1:], [2.0, 5.5, 6.0])
    expected_std = [np.std(window) for window in ([1, 3], [3, 8], [8, 4])]
    assert_array_equal(std[1:], expected_std)
