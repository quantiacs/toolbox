import json
import os
import urllib.request
from types import TracebackType

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from typing_extensions import Self

import qnt.output as qnout
import qnt.stats as qnstats
from qnt.tests.hl_support import (
    Floats,
    business_days,
    field_values,
    make_data,
    make_weights,
    trailing_mean,
    trailing_std,
)

pytestmark = pytest.mark.usefixtures("quiet_tracking")

PPY = 252


def returns_series(n: int = 300, seed: int = 3) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(0.0006, 0.01, n),
        dims=["time"],
        coords={"time": business_days(n)},
    )


def test_performance_metrics_match_independent_numpy_formulas() -> None:
    rr = returns_series()
    returns = rr.values
    n = len(returns)

    equity = qnstats.calc_equity(rr)
    expected_equity = np.cumprod(1.0 + returns)
    assert_allclose(equity.values, expected_equity, rtol=1e-12)

    underwater = qnstats.calc_underwater(equity)
    expected_underwater = expected_equity / np.maximum.accumulate(expected_equity) - 1.0
    assert_allclose(underwater.values, expected_underwater, atol=1e-14)
    drawdown = qnstats.calc_max_drawdown(underwater)
    assert_allclose(drawdown.values, np.minimum.accumulate(expected_underwater), atol=1e-14)

    # volatility is the population standard deviation (ddof=0), not the sample one
    vol = qnstats.calc_volatility(rr)
    assert_allclose(vol.values[1:], trailing_std(returns, n, 2)[1:], rtol=1e-10)
    assert np.isnan(vol.values[0])
    vol_window = qnstats.calc_volatility(rr, max_periods=20, min_periods=5)
    assert_allclose(vol_window.values[4:], trailing_std(returns, 20, 5)[4:], rtol=1e-10)
    vol_ann = qnstats.calc_volatility_annualized(rr, points_per_year=PPY)
    assert_allclose(vol_ann.values[1:], vol.values[1:] * np.sqrt(PPY), rtol=1e-12)

    log_mean = trailing_mean(np.log1p(returns), n, 1)
    mean_daily = qnstats.calc_mean_return(rr, points_per_year=PPY)
    assert_allclose(mean_daily.values, np.expm1(log_mean), rtol=1e-10)
    mean_ann = qnstats.calc_mean_return_annualized(rr, points_per_year=PPY)
    assert_allclose(mean_ann.values, (1 + np.expm1(log_mean)) ** PPY - 1, rtol=1e-9)
    simple_ann = qnstats.calc_simple_mean_return_annualized(rr, points_per_year=PPY)
    assert_allclose(simple_ann.values, trailing_mean(returns, n, 1) * PPY, rtol=1e-10)

    geometric = qnstats.calc_sharpe_ratio_annualized(rr, points_per_year=PPY)
    arithmetic = qnstats.calc_sharpe_ratio_annualized(
        rr, points_per_year=PPY, mean_estimator="arithmetic"
    )
    assert_allclose(
        geometric.values[1:],
        (mean_ann.values / vol_ann.values)[1:],
        rtol=1e-10,
    )
    assert_allclose(
        arithmetic.values[1:],
        (simple_ann.values / vol_ann.values)[1:],
        rtol=1e-10,
    )
    assert_allclose(geometric.values[-1], 1.872220371444398, rtol=1e-9)
    assert_allclose(arithmetic.values[-1], 1.7179201004914173, rtol=1e-9)
    with pytest.raises(ValueError, match="Unknown mean_estimator='median'"):
        qnstats.calc_sharpe_ratio_annualized(rr, points_per_year=PPY, mean_estimator="median")


def test_defaults_match_explicit_arguments() -> None:
    rr = returns_series()
    ppy = qnstats.calc_avg_points_per_year(rr)
    assert ppy != PPY
    cases = [
        (qnstats.calc_volatility, 2),
        (qnstats.calc_volatility_annualized, 2),
        (qnstats.calc_sharpe_ratio_annualized, 2),
        (qnstats.calc_mean_return, 1),
        (qnstats.calc_mean_return_annualized, 1),
        (qnstats.calc_simple_mean_return_annualized, 1),
    ]
    for func, min_periods in cases:
        assert_allclose(
            func(rr).values,
            func(rr, max_periods=len(rr.time), min_periods=min_periods).values,
            equal_nan=True,
        )
    for func in (
        qnstats.calc_volatility_annualized,
        qnstats.calc_sharpe_ratio_annualized,
        qnstats.calc_mean_return_annualized,
        qnstats.calc_simple_mean_return_annualized,
    ):
        assert_allclose(
            func(rr).values,
            func(rr, points_per_year=ppy).values,
            equal_nan=True,
        )


def test_windowed_metrics_respect_max_periods() -> None:
    rr = returns_series()
    returns = rr.values
    vol = trailing_std(returns, 20, 5) * np.sqrt(PPY)
    log_mean = trailing_mean(np.log1p(returns), 20, 5)
    mean_ann = (1 + np.expm1(log_mean)) ** PPY - 1
    simple_ann = trailing_mean(returns, 20, 5) * PPY
    assert_allclose(
        qnstats.calc_volatility_annualized(
                rr, max_periods=20, min_periods=5, points_per_year=PPY
            ).values[4:],
        vol[4:],
        rtol=1e-10,
    )
    assert_allclose(
        qnstats.calc_mean_return(rr, max_periods=20, min_periods=5, points_per_year=PPY).values[4:],
        np.expm1(log_mean)[4:],
        rtol=1e-10,
    )
    assert_allclose(
        qnstats.calc_mean_return_annualized(
                rr, max_periods=20, min_periods=5, points_per_year=PPY
            ).values[4:],
        mean_ann[4:],
        rtol=1e-9,
    )
    assert_allclose(
        qnstats.calc_simple_mean_return_annualized(
                rr, max_periods=20, min_periods=5, points_per_year=PPY
            ).values[4:],
        simple_ann[4:],
        rtol=1e-10,
    )
    assert_allclose(
        qnstats.calc_sharpe_ratio_annualized(
                rr, max_periods=20, min_periods=5, points_per_year=PPY
            ).values[4:],
        (mean_ann / vol)[4:],
        rtol=1e-9,
    )


def running_mean_skipping_nan(values: Floats) -> Floats:
    running_sum: Floats = np.nancumsum(values)
    running_count: Floats = np.cumsum(np.isfinite(values))
    return running_sum / running_count


def test_means_skip_nan_returns() -> None:
    rr = returns_series()
    returns = rr.values.copy()
    returns[[5, 40, 41, 200]] = np.nan
    gappy = xr.DataArray(returns, dims=["time"], coords={"time": business_days(len(returns))})

    assert_allclose(
        qnstats.calc_mean_return(gappy, points_per_year=PPY).values,
        np.expm1(running_mean_skipping_nan(np.log1p(returns))),
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
    )
    assert_allclose(
        qnstats.calc_simple_mean_return_annualized(gappy, points_per_year=PPY).values,
        running_mean_skipping_nan(returns) * PPY,
        rtol=1e-10,
        atol=1e-14,
        equal_nan=True,
    )


def small_book(rows: list[list[float]]) -> xr.DataArray:
    return xr.DataArray(
        np.array(rows, dtype=float),
        dims=["time", "asset"],
        coords={"time": business_days(len(rows), "2020-01-01"), "asset": ["NYS:A0", "NYS:A1"]},
    )


def test_bias_instruments_turnover_holding_time_on_hand_made_books() -> None:
    assert_allclose(
        qnstats.calc_bias(small_book([[0.5, 0.5], [0.5, -0.5], [0.0, 0.0]])).values,
        [1.0, 0.0, 0.0],
    )
    assert_allclose(
        qnstats.calc_instruments(small_book([[0.5, 0.0], [0.0, 0.0], [0.0, 0.5]])).values,
        [1.0, 1.0, 2.0],
    )

    book = small_book([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    data = make_data(n_assets=2, n_days=4, name="futures", start="2020-01-01")
    flat = data.copy(data=np.full(data.shape, 10.0))
    equity = xr.ones_like(book.isel(asset=0, drop=True))
    turnover = qnstats.calc_avg_turnover(book, equity, flat, points_per_year=PPY)
    assert_allclose(turnover.values, [0.0, 0.5, 1.0 / 3.0, 0.75])

    held = small_book(
        [[0, 0], [1, 0], [1, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 0]],
    )
    per_asset = qnstats.calc_avg_holding_time(held, per_asset=True, points_per_year=PPY)
    assert_array_equal(np.isnan(per_asset.values).sum(axis=1), [2, 2, 2, 2, 1, 1, 1, 0])
    assert_allclose(per_asset.values[4:, 0], 2.0)
    assert_allclose(per_asset.values[-1], [2.0, 2.0])
    total = qnstats.calc_avg_holding_time(held, points_per_year=PPY)
    assert_allclose(total.values[4:], 2.0)
    assert np.isnan(total.values[:4]).all()


def reference_relative_return(data: xr.DataArray, weights: xr.DataArray, fract: float) -> Floats:
    """Independent re-implementation of the daily rebalancing simulation (no divs, no roll)."""
    open_ = field_values(data, "open")
    high = field_values(data, "high")
    low = field_values(data, "low")
    close = field_values(data, "close")
    n_days = open_.shape[0]

    prev_close = np.vstack([np.full((1, open_.shape[1]), np.nan), close[:-1]])
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(prev_close - low)]
    )
    slippage = trailing_mean(true_range, 14, 14) * fract

    target = np.vstack([np.zeros((1, open_.shape[1])), np.nan_to_num(weights.values)[:-1]])
    target = np.nan_to_num(target * open_ / prev_close)
    target = target / np.maximum(np.abs(target).sum(axis=1), 1.0)[:, None]

    tradeable = np.isfinite(slippage)
    shares = np.zeros(open_.shape[1])
    equity_after = 1.0
    previous_tonight = 1.0
    returns = np.zeros(n_days)
    for day in range(n_days):
        if day > 0:
            before = equity_after + float(((open_[day] - open_[day - 1]) * shares).sum())
        else:
            before = 1.0
        free_cash = max(1.0, np.abs(target[day]).sum()) - np.abs(target[day]).sum()
        operable_weight = np.abs(target[day][tradeable[day]]).sum() + free_cash
        locked_value = float((open_[day][~tradeable[day]] * np.abs(shares[~tradeable[day]])).sum())
        operable = before - locked_value
        new_shares = shares.copy()
        if operable_weight < 1e-7:
            equity_after = before
        else:
            new_shares[tradeable[day]] = (
                operable
                * target[day][tradeable[day]]
                / (operable_weight * open_[day][tradeable[day]])
            )
            traded = np.abs(new_shares - shares)[tradeable[day]]
            equity_after = before - float((slippage[day][tradeable[day]] * traded).sum())
        tonight = equity_after + float(((close[day] - open_[day]) * new_shares).sum())
        returns[day] = tonight / previous_tonight - 1.0 if day > 0 else 0.0
        previous_tonight = tonight
        shares = new_shares
        shares[~tradeable[day]] = 0.0
    return returns


def test_relative_return_matches_independent_simulation() -> None:
    data = make_data(n_assets=5, n_days=90, name="futures", seed=21)
    weights = qnout.normalize(make_weights(data, seed=8))
    expected = reference_relative_return(data, weights, 0.03)
    actual = qnstats.calc_relative_return(data, weights, 0.03, 0.03)
    assert_allclose(actual.values, expected, atol=1e-12)
    assert_array_equal(actual.values[:14], 0.0)
    assert np.abs(expected[15:]).sum() > 0.05
    assert_allclose(actual.values[-1], expected[-1], atol=1e-12)


def test_single_asset_per_asset_equals_global_calculation() -> None:
    data = make_data(n_assets=2, n_days=90, name="futures", seed=4)
    solo = data.sel(asset=["NYS:A1"])
    weights = qnout.normalize(make_weights(solo, seed=2) * 0.4)
    total = qnstats.calc_relative_return(solo, weights, 0.03, 0.03)
    per_asset = qnstats.calc_relative_return(solo, weights, 0.03, 0.03, per_asset=True)
    assert per_asset.dims == ("time", "asset")
    assert_allclose(per_asset.values[:, 0], total.values, atol=1e-12)


def test_slippage_matches_average_true_range() -> None:
    data = make_data(n_assets=3, n_days=60, name="futures")
    close = field_values(data, "close")
    high = field_values(data, "high")
    low = field_values(data, "low")
    prev = np.vstack([np.full((1, 3), np.nan), close[:-1]])
    true_range = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(prev - low)])
    expected = trailing_mean(true_range, 14, 14) * 0.07
    actual = qnstats.calc_slippage(data, fract=0.07).values
    assert_allclose(actual, expected, rtol=1e-12)
    assert np.isnan(actual[:14]).all()
    assert_allclose(qnstats.calc_slippage(data).values, expected / 0.07 * 0.04, rtol=1e-12)


def test_liquidity_and_missed_dates_helpers() -> None:
    data = make_data(n_assets=3, n_days=30, name="futures")
    weights = qnout.normalize(make_weights(data))
    non_liquid = qnstats.calc_non_liquid(data, weights)
    assert list(non_liquid.coords["asset"].values) == ["NYS:A0"]
    assert non_liquid.sizes["time"] == 6
    assert str(non_liquid.coords["time"].values[-1])[:10] == "2019-01-08"

    kept = [i for i in range(30) if i not in (4, 20)]
    missed = qnstats.find_missed_dates(weights.isel(time=kept), data)
    assert_array_equal(missed, data.coords["time"].values[[4, 20]])
    starts_later = weights.isel(time=slice(10, None))
    assert qnstats.find_missed_dates(starts_later, data).size == 0


@pytest.mark.parametrize(
    ("name", "n_days", "expected"),
    [
        ("crypto", 100, 8766),
        ("crypto_daily", 100, 365),
        ("stocks_nasdaq100", 100, 251),
        ("futures", 100, 251),
        ("unknown", 300, 261),
        ("stocks", 300, 261),
    ],
)
def test_points_per_year_by_name_and_by_timestamps(name: str, n_days: int, expected: int) -> None:
    data = make_data(n_assets=2, n_days=n_days, name=name)
    assert qnstats.calc_avg_points_per_year(data) == expected


def test_points_per_year_for_hourly_series() -> None:
    hours = pd.date_range("2021-01-01", periods=500, freq="h", name="time")
    series = xr.DataArray(np.zeros(500), dims=["time"], coords={"time": hours})
    assert qnstats.calc_avg_points_per_year(series) == 8766
    assert qnstats.calc_points_per_day(8766) == 24
    assert qnstats.calc_points_per_day(251) == 1


@pytest.mark.parametrize(
    ("kind", "period", "start", "slippage", "exposure"),
    [
        ("stocks_s&p500", 4536, "2006-01-01", 0.05, 0.1),
        ("stocks_nasdaq100", 3528, "2006-01-01", 0.05, 0.1),
        ("stocks", 1512, "2015-01-01", 0.05, 1.0),
        ("stocks_long", 1512, "2015-01-01", 0.05, 1.0),
        ("futures", 3528, "2006-01-01", 0.04, 1.0),
        ("crypto_futures", 1764, "2014-01-01", 0.04, 1.0),
        ("crypto_daily", 1764, "2016-01-01", 0.04, 1.0),
        ("crypto", 60000, "2014-01-01", 0.05, 1.0),
    ],
)
def test_competition_defaults(
    kind: str,
    period: int,
    start: str,
    slippage: float,
    exposure: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in [k for k in os.environ if k.startswith(("IS_", "SD_", "SL_", "ME_"))]:
        monkeypatch.delenv(key)
    data = make_data(n_assets=2, n_days=10, name=kind)
    assert qnstats.get_default_is_period_for_type(kind) == period
    assert qnstats.get_default_is_period(data) == period
    assert qnstats.get_default_is_start_date_for_type(kind) == start
    assert qnstats.get_default_slippage(data) == slippage
    assert qnstats.get_default_max_exposure(kind) == exposure


def test_competition_defaults_unknown_kind_and_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_data(n_assets=2, n_days=300, name="mystery")
    assert qnstats.get_default_is_period_for_type("mystery") is None
    assert qnstats.get_default_is_start_date_for_type("mystery") is None
    assert qnstats.get_default_slippage(data) == 0.05
    assert qnstats.get_default_is_period(data) == 261 * 5
    monkeypatch.setenv("IS_STOCKS", "77")
    monkeypatch.setenv("SD_FUTURES", "2010-05-05")
    monkeypatch.setenv("ME_STOCKS_NASDAQ100", "0.25")
    assert qnstats.get_default_is_period_for_type("stocks") == 77
    assert qnstats.get_default_is_start_date_for_type("futures") == "2010-05-05"
    assert qnstats.get_default_max_exposure("stocks_nasdaq100") == 0.25


def test_exposure_checks(capsys: pytest.CaptureFixture[str]) -> None:
    spread = xr.DataArray(
        np.full((300, 40), 1.0 / 40),
        dims=["time", "asset"],
        coords={"time": business_days(300), "asset": [f"s{i}" for i in range(40)]},
    )
    assert qnstats.check_exposure(spread)
    assert "Ok. The exposure check succeed." in capsys.readouterr().out
    assert_allclose(qnstats.calc_exposure(spread).values, 1.0 / 40)

    concentrated = spread.copy(deep=True)
    concentrated[:, 0] = 0.0847826087
    assert not qnstats.check_exposure(concentrated)
    report = capsys.readouterr()
    assert "WARNING! The exposure check failed." in report.err
    assert "Hard limit check:  Ok." in report.out
    assert "Days check:  Failed." in report.out
    exposure = qnstats.calc_exposure(concentrated).values
    assert_allclose(exposure[0, 0], 0.08, rtol=1e-6)
    assert_allclose(exposure.sum(axis=1), 1.0)

    empty = spread * 0
    assert_allclose(qnstats.calc_exposure(empty).values, 0.0)


def test_sector_distribution_groups_assets_and_cash(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog: list[dict[str, object]] = [
        {"id": "x1", "sector": "Tech"},
        {"id": "x2", "sector": "Tech"},
        {"id": "x3", "sector": "Energy"},
        {"id": "x4", "sector": "n/a"},
    ]

    def fake_list(**_kwargs: object) -> list[dict[str, object]]:
        return catalog

    monkeypatch.setattr(qnstats, "stocks_load_list", fake_list)
    weights = xr.DataArray(
        [[0.2, 0.3, -0.1, 0.1, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0]],
        dims=["time", "asset"],
        coords={"time": business_days(2), "asset": ["x1", "x2", "x3", "x4", "zz"]},
    )
    result = qnstats.calc_sector_distribution(weights, kind="stocks")
    assert list(result.coords["sector"].values) == ["Energy", "Other", "Tech", "Cash"]
    assert_allclose(result.values, [[0.1, 0.1, 0.5, 0.3], [0.0, 0.0, 0.5, 0.5]], atol=1e-12)
    unknown = qnstats.calc_sector_distribution(weights, kind="whatever")
    assert list(unknown.coords["sector"].values) == ["Other", "Cash"]
    assert_allclose(unknown.values[:, 0], [0.7, 0.5])


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def install_fake_server(
    monkeypatch: pytest.MonkeyPatch, submissions: list[dict[str, object]]
) -> list[tuple[str, int | None]]:
    calls: list[tuple[str, int | None]] = []

    def fake_urlopen(url: str, payload: bytes | None = None) -> FakeResponse:
        if payload is None:
            calls.append((url, None))
            return FakeResponse(submissions)
        body = json.loads(payload)
        ids = body["submission_ids"]
        calls.append((url, len(ids)))
        return FakeResponse([{"id": i, "cofactor": 0.95, "sharpe_ratio": 1.5} for i in ids])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def test_correlation_merges_submissions_in_chunks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SUBMISSION_ID", raising=False)
    submissions: list[dict[str, object]] = [
        {"id": i, "name": f"s{i}", "my": i % 2 == 0, "template": i % 3 == 0} for i in range(150)
    ]
    calls = install_fake_server(monkeypatch, submissions)
    result = qnstats.calc_correlation(returns_series(30), False, "stocks")
    assert [c[1] for c in calls] == [None, 100, 50]
    assert "competitionType=stocks" in calls[0][0]
    assert len(result) == 150
    assert result[5] == {
        "id": 5, "name": "s5", "my": False, "template": False, "cofactor": 0.95, "sharpe_ratio": 1.5,
    }  # fmt: skip

    data = make_data(n_assets=4, n_days=30, name="stocks")
    weights = qnout.normalize(make_weights(data))
    qnstats.check_correlation(weights, data, print_stack_trace=False)
    report = capsys.readouterr()
    assert "WARNING! This strategy correlates with other strategies" in report.err
    assert (
        "The number of systems with a larger Sharpe ratio and correlation larger than 0.9: 150"
        in report.out
    )
    assert "My correlated submissions" in report.out
    assert "Correlated examples" in report.out


def test_correlation_errors_and_submission_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_args: object) -> FakeResponse:
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", broken)
    monkeypatch.delenv("SUBMISSION_ID", raising=False)
    assert qnstats.calc_correlation(returns_series(30)) == []
    with pytest.raises(OSError, match="offline"):
        qnstats.calc_correlation(returns_series(30), suppress_exception=False)
    assert "WARNING! Can't calculate correlation." in capsys.readouterr().err

    monkeypatch.setenv("SUBMISSION_ID", "42")
    assert qnstats.calc_correlation(returns_series(30), suppress_exception=False) == []
    assert "correlation check disabled" in capsys.readouterr().out
