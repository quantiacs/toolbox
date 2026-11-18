import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.data.common as qndc
import qnt.exits as qnexits
import qnt.exposure as qnexp
import qnt.filter as qnfilter
import qnt.forward_looking as qnfl
import qnt.neutralization as qnneu
import qnt.pd_data as qnpd
import qnt.state as qnstate
import qnt.stats as qnstats
import qnt.stepper as qnstep
import qnt.ta as qnta
from qnt.tests.hl_server import FakeServer, stock_server
from qnt.tests.hl_support import (
    Floats,
    business_days,
    make_data,
    make_weights,
    trailing_std,
    values_of,
)

pytestmark = pytest.mark.usefixtures("quiet_tracking")


def book(rows: list[list[float]], assets: list[str] | None = None) -> xr.DataArray:
    names = assets if assets is not None else [f"a{i}" for i in range(len(rows[0]))]
    return xr.DataArray(
        np.array(rows, dtype=np.float64),
        dims=["time", "asset"],
        coords={"time": business_days(len(rows)), "asset": names},
    )


def test_neutralize_by_market_sector_and_industry() -> None:
    weights = book([[0.4, 0.2, -0.1, 0.3], [0.1, 0.1, 0.1, 0.1]], ["w", "x", "y", "z"])
    market = qnneu.neutralize(weights, [])
    assert_allclose(market.values.sum(axis=1), 0.0, atol=1e-15)
    assert_allclose(market.values[0], [0.2, 0.0, -0.3, 0.1])

    catalog = [
        {"id": "w", "sector": "tech", "industry": "chips"},
        {"id": "x", "sector": "tech", "industry": "software"},
        {"id": "y", "sector": "energy", "industry": "oil"},
        {"id": "z", "sector": "energy", "industry": "oil"},
        {"id": "other", "sector": "tech", "industry": "chips"},
    ]
    by_sector = qnneu.neutralize(weights, catalog, "sector")
    assert_allclose(by_sector.values[0], [0.1, -0.1, -0.2, 0.2])
    assert_allclose(by_sector.values[1], 0.0, atol=1e-15)
    by_industry = qnneu.neutralize(weights, catalog, "industry")
    assert_allclose(by_industry.values[0], [0.0, 0.0, -0.2, 0.2])

    with pytest.warns(UserWarning, match="has no specification"):
        partial = qnneu.neutralize(weights, catalog[:2], "sector")
    assert_allclose(partial.values[0], [0.1, -0.1, -0.2, 0.2])
    with pytest.raises(Exception, match="No such group 'country'"):
        qnneu.neutralize(weights, catalog, "country")


def test_exposure_limiters() -> None:
    diversified = book([[1.0 / 40] * 40])
    concentrated = book([[1.0] + [0.0] * 39])
    mixed = qnexp.mix_weights(concentrated, diversified)
    exposure = qnstats.calc_exposure(mixed)
    assert_allclose(exposure.values[0, 0], 0.049, rtol=1e-12)
    assert_allclose(mixed.values.sum(), 1.0, rtol=1e-12)
    assert_allclose(exposure.values[0, 1:], 0.990625 / 40.625, rtol=1e-12)

    cut = qnexp.cut_big_positions(book([[0.5, -0.5, 0.01, 0.0]]))
    assert_allclose(cut.values[0], [0.049, -0.049, 0.01, 0.0])

    scaled = qnexp.normalize_by_max_exposure(book([[0.5, -0.25], [0.05, 0.02]]), 0.1)
    assert_allclose(scaled.values, [[0.1, -0.05], [0.05, 0.02]])

    days = book([[0.5, 0.5], [0.01, 0.01], [0.9, 0.1], [0.0, 0.0]])
    kept = qnexp.drop_bad_days(days, 0.6)
    assert_allclose(kept.values, [[0.5, 0.5], [0.01, 0.01], [0.0, 0.0], [0.0, 0.0]])


def ranks(values: Floats, axis: int = -1) -> npt.NDArray[np.intp]:
    return np.argsort(np.argsort(values, axis=axis), axis=axis) + 1


def test_rank_filters_mark_top_assets_each_day() -> None:
    data = make_data(n_assets=6, n_days=60, name="stocks", seed=9)
    close = data.sel(field="close")
    criterion = close.copy(data=np.tile(np.array([5.0, 1.0, 4.0, 2.0, 6.0, 3.0]), (60, 1)))
    lowest = qnfilter.rank_assets_by(data, criterion, 2, ascending=True)
    assert_array_equal(lowest.values[0], [0, 1, 0, 1, 0, 0])
    highest = qnfilter.rank_assets_by(data, criterion, 2, ascending=False)
    assert_array_equal(highest.values[-1], [1, 0, 0, 0, 1, 0])

    with pytest.raises(ValueError, match="Unsupported metric: median"):
        qnfilter.calc_rolling_metric(close, 5, "median")
    assert_allclose(
        qnfilter.calc_rolling_metric(close, 5, "mean").values[4],
        close.values[:5].mean(axis=0),
    )

    quiet = qnfilter.filter_volatility(data, 10, 2)
    prices = close.values
    returns = np.vstack([np.full((1, 6), np.nan), prices[1:] / prices[:-1] - 1.0])
    vol = trailing_std(returns, 10, 10)
    last_two = ranks(vol[-1]) <= 2
    assert_array_equal(quiet.values[-1], last_two.astype(float))
    assert quiet.values[:10].sum() == 0.0
    assert_array_equal(quiet.values.sum(axis=1)[11:], 2.0)
    wild = qnfilter.filter_volatility(data, 10, 2, ascending=False)
    assert_array_equal(wild.values[-1] + quiet.values[-1] <= 1, True)

    natr = qnfilter.filter_by_normalized_atr(data, 3, ma_period=14)
    atr = qnta.atr(data.sel(field="high"), data.sel(field="low"), close, 14)
    expected = ranks(100 * atr.values / prices, axis=1) <= 3
    assert_array_equal(natr.values[20:], expected[20:].astype(float))


def test_stat_based_filters_follow_per_asset_statistics() -> None:
    data = make_data(n_assets=5, n_days=80, name="futures", seed=13)
    weights = make_weights(data, seed=2) * 0 + 0.2
    by_sharpe = qnfilter.filter_sharpe_ratio(data, weights, 2)
    stat = qnstats.calc_stat(data, weights, per_asset=True)
    sharpe = stat.sel(field="sharpe_ratio").values
    assert_array_equal(by_sharpe.values.sum(axis=1)[40:], 2.0)
    best = np.argsort(-sharpe[-1])[:2]
    assert set(np.flatnonzero(by_sharpe.values[-1])) == set(best.tolist())

    rolling = qnfilter.filter_volatility_rolling(data, weights, 2, 5)
    vol = stat.sel(field="volatility").values
    smoothed = trailing_std(vol, 5, 5)
    low = np.argsort(smoothed[-1])[:2]
    assert set(np.flatnonzero(rolling.values[-1])) == set(low.tolist())


def exit_data(
    closes: list[float], opens: list[float], assets: tuple[str, ...] = ("long", "short", "flat")
) -> xr.DataArray:
    layers = np.array([[opens, closes], [opens, closes]], dtype=np.float64)
    return xr.DataArray(
        layers.transpose(0, 1, 2),
        dims=["time", "field", "asset"],
        coords={
            "time": business_days(2),
            "field": ["open", "close"],
            "asset": list(assets),
        },
    )


def flags(array: xr.DataArray) -> list[int]:
    return [int(v) for v in array.values]


def test_exit_rules_by_percentage_and_atr() -> None:
    weights = book([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]], ["long", "short", "flat"])
    entry = xr.DataArray(
        [100.0, 100.0, np.nan], dims=["asset"], coords={"asset": ["long", "short", "flat"]}
    )
    atr = xr.DataArray([2.0, 2.0, 2.0], dims=["asset"], coords={"asset": ["long", "short", "flat"]})

    calm = exit_data([105.0, 95.0, 50.0], [104.0, 96.0, 50.0])
    assert flags(qnexits.take_profit_long_percentage(calm, weights, entry, 10)) == [1, 1, 1]
    assert flags(qnexits.stop_loss_long_percentage(calm, weights, entry, 10)) == [1, 1, 1]
    assert flags(qnexits.take_profit_short_percentage(calm, weights, entry, 10)) == [1, 1, 1]
    assert flags(qnexits.stop_loss_short_percentage(calm, weights, entry, 10)) == [1, 1, 1]

    spike = exit_data([115.0, 115.0, 50.0], [114.0, 114.0, 50.0])
    assert flags(qnexits.take_profit_long_percentage(spike, weights, entry, 10)) == [0, 1, 1]
    assert flags(qnexits.stop_loss_short_percentage(spike, weights, entry, 10)) == [1, 0, 1]
    crash = exit_data([85.0, 85.0, 50.0], [86.0, 86.0, 50.0])
    assert flags(qnexits.stop_loss_long_percentage(crash, weights, entry, 10)) == [0, 1, 1]
    assert flags(qnexits.take_profit_short_percentage(crash, weights, entry, 10)) == [1, 0, 1]

    assert flags(qnexits.take_profit_long_atr(spike, weights, entry, atr, 5)) == [0, 1, 1]
    assert flags(qnexits.stop_loss_short_atr(spike, weights, entry, atr, 5)) == [1, 0, 1]
    assert flags(qnexits.stop_loss_long_atr(crash, weights, entry, atr, 5)) == [0, 1, 1]
    assert flags(qnexits.take_profit_short_atr(crash, weights, entry, atr, 5)) == [1, 0, 1]
    assert flags(qnexits.take_profit_long_atr(calm, weights, entry, atr, 5)) == [1, 1, 1]


def test_open_price_tracking_and_max_holding_period() -> None:
    assets = ["new", "kept", "closed", "flipped"]
    weights = book([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, -1.0]], assets)
    data = exit_data([0.0] * 4, [10.0, 20.0, 30.0, 40.0], tuple(assets))
    previous = xr.DataArray([np.nan, 15.0, 25.0, 35.0], dims=["asset"], coords={"asset": assets})
    state = {"open_price": previous}
    result = qnexits.update_open_price(data, weights, state)
    assert_allclose(result.values, [10.0, 15.0, np.nan, 40.0])
    assert state["open_price"] is result

    longs = book([[1.0, 0.0]] * 2, ["l", "s"])
    shorts = book([[0.0, -1.0]] * 2, ["l", "s"])
    long_state = {
        "holding_time": xr.DataArray([0.0, 0.0], dims=["asset"], coords={"asset": ["l", "s"]})
    }
    seen = [flags(qnexits.max_hold_long(longs, long_state, 3)) for _ in range(5)]
    assert seen == [[1, 1], [1, 1], [0, 1], [1, 1], [1, 1]]
    short_state = {
        "holding_time": xr.DataArray([0.0, 0.0], dims=["asset"], coords={"asset": ["l", "s"]})
    }
    kept = [flags(qnexits.max_hold_short(shorts, short_state, 2)) for _ in range(4)]
    assert kept == [[1, 1], [1, 0], [1, 1], [1, 1]]

    flip = book([[1.0, 0.0], [-1.0, 0.0]], ["l", "s"])
    flip_state = {
        "holding_time": xr.DataArray([2.0, 0.0], dims=["asset"], coords={"asset": ["l", "s"]})
    }
    assert flags(qnexits.max_hold_short(flip, flip_state, 5)) == [1, 1]
    assert_allclose(flip_state["holding_time"].values, [1.0, 0.0])


@pytest.mark.xfail(strict=True, reason="TypeError: stepper subscripts the method data.sel[...]")
def test_stepper_runs_strategy_day_by_day_and_normalizes() -> None:
    data = make_data(n_assets=4, n_days=12, name="stocks", seed=3).transpose(
        "field", "time", "asset"
    )
    lengths: list[int] = []
    init_sizes: list[int | None] = []

    def init(history: xr.DataArray | None) -> None:
        init_sizes.append(None if history is None else history.sizes["time"])

    def step(history: xr.DataArray) -> xr.DataArray:
        lengths.append(history.sizes["time"])
        last_close = history.sel(field="close").isel(time=-1)
        result = last_close / last_close.max()
        return result.where(result <= 0.99)

    strategy = qnstep.StrategyAdapter(init=init, step=step, init_data_length=3)
    history = qnstep.test_strategy(data, strategy)
    assert history.dims == ("time", "asset")
    assert history.shape == (9, 4)
    assert lengths == list(range(4, 13))
    assert init_sizes == [9]
    values = history.values
    assert np.isfinite(values).all()
    assert (np.abs(values).sum(axis=1) <= 1.0 + 1e-12).all()
    assert np.abs(values).sum(axis=1).max() == pytest.approx(1.0)
    assert (values == 0.0).any()
    assert_array_equal(history.coords["time"].values, data.coords["time"].values[3:])

    plain = qnstep.calc_step_by_step(data, step=step)
    assert plain.shape == (12, 4)

    backward = data.sortby("time", ascending=False)
    reverse = qnstep.test_strategy(backward, qnstep.StrategyAdapter(step=step))
    assert reverse.shape == (12, 4)
    assert reverse.coords["time"].values[0] < reverse.coords["time"].values[-1]

    equal_weights = qnstep.SimpleStrategy().step(data)
    assert equal_weights.dims == ("asset",)
    assert equal_weights.shape == (4,)
    assert_allclose(equal_weights.values, [0.25] * 4)


def test_state_roundtrip_and_failures(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    qnstate.write(None)
    assert not (sandbox / "state.out.pickle.gz").exists()
    payload = {"counter": 3, "weights": [0.1, 0.2]}
    qnstate.write(payload)
    assert qnstate.read(str(sandbox / "state.out.pickle.gz")) == payload
    assert "State saved." in capsys.readouterr().out

    assert qnstate.read(str(sandbox / "missing.pickle.gz")) is None
    assert "Can't load state." in capsys.readouterr().err
    (sandbox / "state.in.pickle.gz").write_bytes(gzip.compress(pickle.dumps([1, 2])))
    assert qnstate.read() == [1, 2]


def test_forward_looking_detection() -> None:
    data = make_data(n_assets=3, n_days=300, name="stocks", seed=2)
    close = data.sel(field="close")

    def honest(history: xr.DataArray) -> xr.DataArray:
        prices = history.sel(field="close")
        return prices / prices.rolling(time=5).mean() - 1

    def peeking(history: xr.DataArray) -> xr.DataArray:
        prices = history.sel(field="close")
        return (prices.shift(time=-1) / prices - 1).fillna(0)

    whole = honest(data)
    assert qnfl.check_forward_looking(honest(data.isel(time=slice(0, 200))), whole) is False
    assert qnfl.check_forward_looking(peeking(data.isel(time=slice(0, 200))), peeking(data)) is True
    returned = qnfl.calc_output_and_check_forward_looking(data, honest)
    assert_allclose(returned.values, whole.values, equal_nan=True)
    assert (
        qnfl.calc_output_and_check_forward_looking(data, peeking).values.shape == close.shape
    )


def test_forward_looking_with_date_limited_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    data = make_data(n_assets=2, n_days=60, name="stocks", seed=6)
    today = pd.Timestamp.today().date()
    last = pd.Timestamp(data.coords["time"].values[-1]).date()
    monkeypatch.setattr(qnfl, "FORWARD_LOOKING_TEST_OFFSET", (today - last).days + 20)
    monkeypatch.setattr(qndc, "MAX_DATE_LIMIT", None)
    monkeypatch.setattr(qndc, "MAX_DATETIME_LIMIT", None)

    def peeking_strategy() -> xr.DataArray:
        window = data
        if qndc.MAX_DATE_LIMIT is not None:
            window = data.sel(time=slice(None, str(qndc.MAX_DATE_LIMIT)))
        prices = window.sel(field="close")
        return (prices.shift(time=-1) / prices - 1).fillna(0)

    output = qnfl.load_data_calc_output_and_check_forward_looking(peeking_strategy)
    assert output.sizes["time"] == 60
    assert qndc.MAX_DATE_LIMIT is None


def test_pandas_view_of_market_data(server: FakeServer) -> None:
    stock_server(server)
    frames = qnpd.load_data(["NYS:AAA", "NAS:BBB"], min_date="2021-01-06", max_date="2021-02-05")
    assert sorted(frames) == sorted(
        ["open", "low", "high", "close", "vol", "divs", "split", "split_cumprod", "is_liquid"]
    )
    close = frames["close"]
    assert list(close.columns) == ["NAS:BBB", "NYS:AAA"]
    assert close.index[0] < close.index[-1]
    assert close.shape == (23, 2)
    raw = stock_server(server).sel(field="close", asset="s2").values
    assert_allclose(values_of(close)[:, 0], raw[2:25])
    assert list(make_data(n_assets=2, n_days=3).coords["asset"].values) == ["NYS:A0", "NYS:A1"]


def reference_mix(primary: Floats, secondary: Floats, max_weight: float) -> tuple[Floats, float]:
    p_sum = float(np.abs(primary).sum())
    s_sum = float(np.abs(secondary).sum())
    p_max = float(np.abs(primary).max()) / p_sum
    s_max = float(np.abs(secondary).max()) / s_sum
    k = p_sum * (p_max - max_weight) / (s_sum * (max_weight - s_max))
    k = k if k > 0 else 0.0
    mix = primary + secondary * k
    total = float(np.abs(mix).sum())
    return mix / (total if total > 1 else 1.0), k


@pytest.mark.parametrize(
    ("primary", "secondary", "expected_k"),
    [
        ([0.225, 0.175, 0.1, 0, 0], [0.2] * 5, 0.0),
        ([0.7, 0.3, 0, 0, 0], [0.2] * 5, 2 / 3),
        ([0.7, 0.3, 0, 0, 0], [0.3, 0.3, 0.4, 0, 0], 2.0),
    ],
)
def test_mix_weights_follows_formula_in_every_regime(
    primary: list[float], secondary: list[float], expected_k: float
) -> None:
    mixed = qnexp.mix_weights(book([primary]), book([secondary]), 0.5)
    expected, k = reference_mix(np.array(primary), np.array(secondary), 0.5)
    assert_allclose(k, expected_k, atol=1e-12)
    assert_allclose(mixed.values[0], expected, atol=1e-12)


def test_mix_weights_fills_missing_assets_with_zero() -> None:
    primary = book([[0.7, 0.3]], ["a", "b"])
    secondary = book([[0.2] * 5], ["b", "c", "d", "e", "f"])
    mixed = qnexp.mix_weights(primary, secondary, 0.5)
    expected, _ = reference_mix(
        np.array([0.7, 0.3, 0, 0, 0, 0]), np.array([0, 0.2, 0.2, 0.2, 0.2, 0.2]), 0.5
    )
    assert list(mixed.coords["asset"].values) == ["a", "b", "c", "d", "e", "f"]
    assert_allclose(mixed.values[0], expected, atol=1e-12)


def test_exposure_limiters_default_limits_and_boundaries() -> None:
    days = book([[1.0 / 40] * 40, [1.0] + [0.0] * 39])
    kept = qnexp.drop_bad_days(days)
    assert_allclose(kept.values[0], 1.0 / 40)
    assert_array_equal(kept.values[1], 0.0)

    on_limit = qnexp.drop_bad_days(book([[0.5, 0.5]]), 0.5)
    assert_array_equal(on_limit.values, [[0.0, 0.0]])

    scaled = qnexp.normalize_by_max_exposure(book([[0.5, -0.25], [0.05, 0.02]]))
    assert_allclose(scaled.values, [[0.1, -0.05], [0.05, 0.02]])


def test_neutralize_handles_single_unspecified_asset_and_equal_group_labels() -> None:
    weights = book([[0.4, 0.2, 0.3]], ["w", "x", "y"])
    catalog = [
        {"id": "w", "sector": "tech"},
        {"id": "x", "sector": "tech"},
    ]
    with pytest.warns(UserWarning, match="has no specification"):
        result = qnneu.neutralize(weights, catalog, "sector")
    assert_allclose(result.values[0], [0.1, -0.1, 0.0], atol=1e-15)

    stem = "te"
    separate_labels = [
        {"id": "w", "sector": f"{stem}ch"},
        {"id": "x", "sector": f"{stem}ch"},
        {"id": "y", "sector": f"{stem}ch"},
    ]
    assert separate_labels[0]["sector"] is not separate_labels[1]["sector"]
    grouped = qnneu.neutralize(weights, separate_labels, "sector")
    assert_allclose(grouped.values[0], [0.4 - 0.3, 0.2 - 0.3, 0.0], atol=1e-15)

    market_name = "".join(["mar", "ket"])
    assert_allclose(
        qnneu.neutralize(weights, [], market_name).values,
        qnneu.neutralize(weights, []).values,
    )
    with pytest.raises(Exception, match="No such group 'zone'"):
        qnneu.neutralize(weights, catalog, "zone")


def test_volatility_filter_ranks_match_returns_on_every_day() -> None:
    data = make_data(n_assets=6, n_days=60, name="stocks", seed=9)
    prices = data.sel(field="close").values
    returns = np.vstack([np.full((1, 6), np.nan), prices[1:] / prices[:-1] - 1.0])
    vol = trailing_std(returns, 10, 10)
    for ascending in (True, False):
        chosen = qnfilter.filter_volatility(data, 10, 2, ascending=ascending).values
        for day in range(11, 60):
            order = vol[day] if ascending else -vol[day]
            assert_array_equal(chosen[day], (ranks(order) <= 2).astype(float))


def test_normalized_atr_filter_default_period_and_order() -> None:
    data = make_data(n_assets=6, n_days=150, name="stocks", seed=9)
    close = data.sel(field="close")
    atr = qnta.atr(data.sel(field="high"), data.sel(field="low"), close, 90)
    natr = 100 * atr.values / close.values
    for ascending in (True, False):
        chosen = qnfilter.filter_by_normalized_atr(data, 3, ascending=ascending).values
        expected = ranks(natr if ascending else -natr, axis=1) <= 3
        assert_array_equal(chosen[95:], expected[95:].astype(float))


def high_low_close(seed: int, n_days: int) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, 2)), axis=0)
    high = close * (1 + np.abs(rng.normal(0, 0.01, (n_days, 2))))
    low = close * (1 - np.abs(rng.normal(0, 0.01, (n_days, 2))))
    return xr.DataArray(
        np.stack([high, low, close], axis=1),
        dims=["time", "field", "asset"],
        coords={
            "time": business_days(n_days),
            "field": ["high", "low", "close"],
            "asset": ["a0", "a1"],
        },
    )


def test_volatility_filter_uses_previous_close_as_return_base() -> None:
    data = high_low_close(seed=4, n_days=40)
    prices = data.sel(field="close").values
    returns = np.vstack([np.full((1, 2), np.nan), (prices[1:] - prices[:-1]) / prices[:-1]])
    vol = trailing_std(returns, 5, 5)
    chosen = qnfilter.filter_volatility(data, 5, 1).values
    for day in range(6, 40):
        expected = np.zeros(2)
        expected[np.argmin(vol[day])] = 1.0
        assert_array_equal(chosen[day], expected)


def test_normalized_atr_filter_default_period_separates_neighbouring_periods() -> None:
    data = high_low_close(seed=6, n_days=200)
    close = data.sel(field="close")
    atr = qnta.atr(data.sel(field="high"), data.sel(field="low"), close, 90)
    natr = atr.values / close.values
    chosen = qnfilter.filter_by_normalized_atr(data, 1).values
    for day in range(95, 200):
        expected = np.zeros(2)
        expected[np.argmin(natr[day])] = 1.0
        assert_array_equal(chosen[day], expected)


def test_rolling_metric_accepts_equal_but_distinct_metric_names() -> None:
    close = make_data(n_assets=3, n_days=30, name="stocks", seed=1).sel(field="close")
    std_name = "".join(["s", "td"])
    mean_name = "".join(["me", "an"])
    assert_allclose(
        qnfilter.calc_rolling_metric(close, 5, std_name).values,
        close.rolling({"time": 5}).std().values,
        equal_nan=True,
    )
    assert_allclose(
        qnfilter.calc_rolling_metric(close, 5, mean_name).values,
        close.rolling({"time": 5}).mean().values,
        equal_nan=True,
    )


def test_neutralize_does_not_warn_when_every_asset_is_specified() -> None:
    weights = book([[0.4, 0.2]], ["w", "x"])
    catalog = [{"id": "w", "sector": "tech"}, {"id": "x", "sector": "energy"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = qnneu.neutralize(weights, catalog, "sector")
    assert_allclose(result.values[0], [0.0, 0.0], atol=1e-15)
