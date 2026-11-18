from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objs as go
import xarray as xr


if TYPE_CHECKING:
    from collections.abc import Mapping

FIELDS = ["open", "low", "high", "close", "vol", "divs", "split", "split_cumprod", "is_liquid"]

Floats: TypeAlias = npt.NDArray[np.float64]


class _ToNumpy(Protocol):
    def to_numpy(self) -> Floats: ...


def values_of(frame: _ToNumpy) -> Floats:
    return frame.to_numpy()


def plot_field(figure: go.Figure, *path: str | int) -> object:
    """Reads a nested value of the figure dictionary, e.g. ("layout", "yaxis", "range")."""
    node: object = figure.to_dict()
    for key in path:
        node = cast("Mapping[str | int, object]", node)[key]
    return node


def plot_floats(figure: go.Figure, *path: str | int) -> Floats:
    return np.asarray(plot_field(figure, *path), dtype=np.float64)


def trace_floats(figure: go.Figure, key: str, index: int = 0) -> Floats:
    """Numeric trace data such as "x" or "y" (the dictionary form stores arrays base64-encoded)."""
    return np.asarray(figure.data[index][key], dtype=np.float64)


def float_series(values: Floats, index: pd.DatetimeIndex) -> "pd.Series[float]":
    return pd.Series(values, index=index)


def time_index(arr: xr.DataArray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(arr.coords["time"].values)


def renamed_assets(arr: xr.DataArray, names: list[str]) -> xr.DataArray:
    result = arr.copy()
    result.coords["asset"] = names
    return result


def trailing_mean(values: Floats, window: int, min_periods: int) -> Floats:
    """Rolling mean along axis 0 that skips NaN and needs min_periods valid points."""
    result = np.full(values.shape, np.nan)
    for i in range(values.shape[0]):
        part = values[max(0, i - window + 1) : i + 1]
        valid = np.isfinite(part)
        count = valid.sum(axis=0)
        total = np.where(valid, part, 0.0).sum(axis=0)
        result[i] = np.where(count >= min_periods, total / np.maximum(count, 1), np.nan)
    return result


def trailing_std(values: Floats, window: int, min_periods: int) -> Floats:
    """Rolling population standard deviation along axis 0 that skips NaN."""
    result = np.full(values.shape, np.nan)
    for i in range(values.shape[0]):
        part = values[max(0, i - window + 1) : i + 1]
        valid = np.isfinite(part)
        count = np.maximum(valid.sum(axis=0), 1)
        mean = np.where(valid, part, 0.0).sum(axis=0) / count
        spread = np.where(valid, (part - mean) ** 2, 0.0).sum(axis=0) / count
        result[i] = np.where(valid.sum(axis=0) >= min_periods, np.sqrt(spread), np.nan)
    return result


def business_days(n_days: int, start: str = "2019-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n_days, name="time")


def make_data(
    n_assets: int = 8,
    n_days: int = 300,
    seed: int = 11,
    name: str = "stocks",
    start: str = "2019-01-01",
) -> xr.DataArray:
    """Synthetic market data with dims (time, field, asset); asset 0 is illiquid at the start."""
    rng = np.random.default_rng(seed)
    times = business_days(n_days, start)
    assets = [f"NYS:A{i}" for i in range(n_assets)]
    drift = rng.normal(0.0004, 0.0003, n_assets)
    vol = rng.uniform(0.008, 0.02, n_assets)
    log_ret = rng.normal(drift, vol, (n_days, n_assets))
    close: Floats = 50.0 * np.exp(np.cumsum(log_ret, axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.002, (n_days, n_assets)))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, (n_days, n_assets)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, (n_days, n_assets)))
    volume = rng.uniform(1e5, 1e6, (n_days, n_assets))
    divs = np.zeros((n_days, n_assets))
    split = np.ones((n_days, n_assets))
    split_cumprod = np.ones((n_days, n_assets))
    is_liquid = np.ones((n_days, n_assets))
    is_liquid[: n_days // 5, 0] = 0.0
    layers = [open_, low, high, close, volume, divs, split, split_cumprod, is_liquid]
    values = np.stack(layers, axis=1)
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={"time": times, "field": FIELDS, "asset": assets},
        name=name,
    )


def make_weights(data: xr.DataArray, seed: int = 5) -> xr.DataArray:
    """Random long/short weights over (time, asset), not normalized, with some NaN."""
    rng = np.random.default_rng(seed)
    n_time = data.sizes["time"]
    n_asset = data.sizes["asset"]
    values = rng.normal(0.0, 1.0, (n_time, n_asset))
    values[rng.random((n_time, n_asset)) < 0.05] = np.nan
    return xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={"time": data.coords["time"].values, "asset": data.coords["asset"].values},
    )


def field_values(arr: xr.DataArray, field: str) -> Floats:
    return arr.sel(field=field).values
