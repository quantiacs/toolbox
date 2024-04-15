import numpy as np
import xarray as xr
import qnt.ta as qnta
import qnt.stats as qnstats


def rank_assets_by(data, criterion, top_assets, ascending):
    """
    Rank assets based on a specified criterion. Returns a DataArray where top ranked assets are marked with a '1'.

    Args:
    data (xarray.Dataset): The dataset containing asset data.
    criterion (xarray.DataArray): The data based on which assets are ranked.
    top_assets (int): Number of top assets to select.
    ascending (bool): True for ascending order, False for descending order.
    """
    volatility_ranks = xr.DataArray(
        np.zeros_like(data.sel(field='close').values),
        dims=['time', 'asset'],
        coords={'time': data.coords['time'], 'asset': data.coords['asset']}
    )

    for time in criterion.coords['time'].values:
        daily_vol = criterion.sel(time=time)
        ranks = (daily_vol if ascending else -daily_vol).rank('asset')
        top_assets_indices = ranks.where(ranks <= top_assets, drop=True).asset.values
        volatility_ranks.loc[dict(time=time, asset=top_assets_indices)] = 1

    return volatility_ranks.fillna(0)


def calc_rolling_metric(condition, rolling_window, metric="std"):
    """
    Compute a rolling metric (standard deviation or mean) over a specified window for a given condition.

    Args:
    condition (xarray.DataArray): Data over which the metric is computed.
    rolling_window (int): Window size for the rolling computation.
    metric (str): Type of metric to compute ('std' for standard deviation, 'mean' for average).

    Raises:
    ValueError: If an unsupported metric is specified.
    """
    if metric == "std":
        return condition.rolling({"time": rolling_window}).std()
    elif metric == "mean":
        return condition.rolling({"time": rolling_window}).mean()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def filter_volatility(data, rolling_window, top_assets, metric="std", ascending=True):
    """
    Filter and rank assets based on volatility over a rolling window.

    Args:
    data (xarray.Dataset): The dataset containing asset data.
    rolling_window (int): Window size for the rolling volatility computation.
    top_assets (int): Number of top assets to select.
    metric (str): Volatility metric to use ('std' for standard deviation).
    ascending (bool): Rank order, True for lowest first.
    """
    prices = data.sel(field='close')
    daily_returns = prices.diff('time') / prices.shift(time=1)
    rolling_volatility = calc_rolling_metric(daily_returns, rolling_window, metric)
    volatility_ranks = rank_assets_by(data, rolling_volatility, top_assets, ascending)

    return volatility_ranks


def filter_sharpe_ratio(data, weights, top_assets):
    """
    Filter and rank assets based on the Sharpe Ratio.

    Args:
    data (xarray.Dataset): The dataset containing asset data.
    weights (xarray.DataArray): Weights to apply for each asset.
    top_assets (int): Number of top assets to select.
    """
    stats_per_asset = qnstats.calc_stat(data, weights, per_asset=True)
    sharpe_ratio = stats_per_asset.sel(field="sharpe_ratio")
    sharpe_ratio_ranks = rank_assets_by(data, sharpe_ratio, top_assets, ascending=False)

    return sharpe_ratio_ranks


def filter_volatility_rolling(data, weights, top_assets, rolling_window, metric="std", ascending=True):
    """
    Filter and rank assets based on their volatility, calculated over a rolling window.

    Args:
    data (xarray.Dataset): The dataset containing asset data.
    weights (xarray.DataArray): Weights to apply for each asset.
    top_assets (int): Number of top assets to select.
    rolling_window (int): Window size for the rolling computation.
    metric (str): Volatility metric to use ('std' for standard deviation).
    ascending (bool): Rank order, True for lowest first.
    """
    stats_per_asset = qnstats.calc_stat(data, weights, per_asset=True)
    volatility = stats_per_asset.sel(field="volatility")
    volatility = calc_rolling_metric(volatility, rolling_window, metric)
    volatility_ranks = rank_assets_by(data=data, criterion=volatility, top_assets=top_assets, ascending=ascending)

    return volatility_ranks


def filter_by_normalized_atr(data, top_assets, ma_period=90, ascending=True):
    """
    Filter and rank assets based on their Normalized Average True Range (NATR) over a specified moving average period.
    NATR is calculated as the Average True Range (ATR) divided by the close price, multiplied by 100, which normalizes the volatility measure across different price levels of assets.

    Args:
    data (xarray.Dataset): The dataset containing asset data.
    top_assets (int): Number of top assets to select based on their NATR.
    ma_period (int): The period over which the moving average of the ATR is computed.
    ascending (bool): If True, ranks the assets with the lowest NATR first, suitable for selecting less volatile assets. If False, ranks with higher NATR first, indicating higher volatility.

    Returns:
    xarray.DataArray: A DataArray where top-ranked assets based on NATR are marked with a '1', and others with '0'.
    """
    high = data.sel(field='high')
    low = data.sel(field='low')
    close = data.sel(field='close')

    # Calculating ATR and then Normalized ATR
    atr = qnta.atr(high=high, low=low, close=close, ma=ma_period)
    natr = 100 * (atr / close)  # Normalized ATR

    # Ranking assets based on NATR
    natr_ranks = rank_assets_by(data=data, criterion=natr, top_assets=top_assets, ascending=ascending)

    return natr_ranks
