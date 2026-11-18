import gzip
import pickle
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, TypeVar, cast

import dash
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.backtester as qnbt
import qnt.data as qndata
import qnt.output as qnout
import qnt.state as qnstate
import qnt.stats as qnstats
from qnt.tests.hl_support import field_values, make_data, make_weights

_F = TypeVar("_F")
_S = TypeVar("_S")
BacktestOutput: TypeAlias = (
    xr.DataArray | tuple[xr.DataArray, _S | list[_S] | list[_S | None] | None]
)

pytestmark = pytest.mark.usefixtures("quiet_tracking")

STAT_FIELDS = [
    "equity",
    "relative_return",
    "volatility",
    "underwater",
    "max_drawdown",
    "sharpe_ratio",
    "mean_return",
    "bias",
    "instruments",
    "avg_turnover",
    "avg_holding_time",
]

FUTURES_LAST_ROW = [
    0.599628328188458,
    -0.015999891522856524,
    0.11526643798985542,
    -0.40633929362909027,
    -0.40633929362909027,
    -3.1158108131719833,
    -0.3591484138846094,
    -0.46190239731322585,
    8.0,
    1.4192315361246652,
    1.404514341003508,
]
NASDAQ_LAST_ROW = [
    0.7643716430183453,
    -0.006936435034855304,
    0.05769243193607766,
    -0.2406771460257462,
    -0.2406771460257462,
    -3.613232329366962,
    -0.2084561602312388,
    -0.5143386812058672,
    8.0,
    0.7189345977948798,
    1.6166248288822724,
]


@dataclass(frozen=True)
class Scenario:
    data: xr.DataArray
    raw: xr.DataArray
    output: xr.DataArray
    stat: xr.DataArray


def build_scenario(kind: str) -> Scenario:
    data = make_data(name=kind)
    raw = make_weights(data)
    output = qnout.clean(raw, data, kind, debug=False)
    return Scenario(data, raw, output, qnstats.calc_stat(data, output))


@pytest.fixture(scope="module")
def futures() -> Scenario:
    return build_scenario("futures")


@pytest.fixture(scope="module")
def nasdaq() -> Scenario:
    return build_scenario("stocks_nasdaq100")


def test_clean_futures_normalizes_and_zeroes_illiquid(futures: Scenario) -> None:
    out = futures.output
    values = out.values
    assert out.dims == ("time", "asset")
    assert out.shape == (300, 8)
    assert_array_equal(out.coords["time"].values, futures.data.coords["time"].values)
    assert np.isfinite(values).all()
    assert_allclose(np.abs(values).sum(axis=1), 1.0, rtol=1e-12)
    assert_array_equal(values[:60, 0], 0.0)
    assert qnstats.calc_non_liquid(futures.data, out).sizes["time"] == 0
    assert_allclose(
        values[0],
        [
            0.0,
            0.0,
            -0.076372256828,
            0.129288701824,
            0.349339148358,
            0.033735185146,
            -0.169941405371,
            -0.241323302473,
        ],
        atol=1e-11,
    )
    assert_allclose(float(out.sum()), 7.824175166571813, rtol=1e-10)
    assert_allclose(float(np.abs(values).max()), 0.5222935454731128, rtol=1e-10)


def test_clean_nasdaq_caps_positions_without_renormalizing(nasdaq: Scenario) -> None:
    values = nasdaq.output.values
    assert_allclose(np.abs(values).max(), 0.1, rtol=1e-12)
    total = np.abs(values).sum(axis=1)
    assert_allclose([total.min(), total.max()], [0.38712332194715704, 0.7665549629012955])
    assert_allclose(values[0, 3:5], [0.1, 0.1])
    assert_allclose(values[0, 6:], [-0.1, -0.1])


@pytest.mark.parametrize("scenario_name", ["futures", "nasdaq"])
def test_calc_stat_last_row_golden(scenario_name: str, futures: Scenario, nasdaq: Scenario) -> None:
    scenario = futures if scenario_name == "futures" else nasdaq
    expected = FUTURES_LAST_ROW if scenario_name == "futures" else NASDAQ_LAST_ROW
    stat = scenario.stat
    assert stat.dims == ("time", "field")
    assert list(stat.coords["field"].values) == STAT_FIELDS
    assert_array_equal(stat.coords["time"].values, scenario.data.coords["time"].values)
    assert_allclose(stat.values[-1], expected, rtol=1e-9)
    sharpe = field_values(stat, "sharpe_ratio")
    assert np.isnan(sharpe[:2]).all()
    assert_allclose(
        qnout.calc_sharpe_ratio_for_check(scenario.data, scenario.output, None, False),
        sharpe[-1],
        rtol=1e-12,
    )


def test_calc_stat_internal_consistency(futures: Scenario) -> None:
    stat = futures.stat
    rr = field_values(stat, "relative_return")
    equity = field_values(stat, "equity")
    underwater = field_values(stat, "underwater")
    drawdown = field_values(stat, "max_drawdown")
    assert_allclose(equity, np.cumprod(1.0 + rr), rtol=1e-12)
    assert_allclose(underwater, equity / np.maximum.accumulate(equity) - 1.0, atol=1e-12)
    assert_allclose(drawdown, np.minimum.accumulate(underwater), atol=1e-12)
    assert rr[0] == 0.0
    assert_allclose(field_values(stat, "instruments")[[0, 1, 100]], [6.0, 7.0, 8.0])
    assert_allclose(field_values(stat, "avg_turnover")[[0, 1]], [0.0, 0.5])


def test_slippage_is_a_cost(futures: Scenario, nasdaq: Scenario) -> None:
    for scenario, free, charged in (
        (futures, 0.8535874543843159, FUTURES_LAST_ROW[0]),
        (nasdaq, 0.9561873918975299, NASDAQ_LAST_ROW[0]),
    ):
        rr_free = qnstats.calc_relative_return(scenario.data, scenario.output, 0.0, 0.0)
        rr_charged = qnstats.calc_relative_return(scenario.data, scenario.output)
        assert_allclose(float((1 + rr_free).prod()), free, rtol=1e-10)
        assert_allclose(float((1 + rr_charged).prod()), charged, rtol=1e-10)
    values = qnstats.calc_slippage(futures.data).values
    assert np.isnan(values[:14]).all()
    assert_allclose(
        values[14],
        [
            0.052023682388,
            0.042942311411,
            0.045798266171,
            0.047228450857,
            0.042946380234,
            0.036114650998,
            0.03194695985,
            0.040322780001,
        ],
        rtol=1e-9,
    )


def test_per_asset_stat_layout(futures: Scenario) -> None:
    per_asset = qnstats.calc_stat(futures.data, futures.output, per_asset=True)
    assert per_asset.dims == ("time", "field", "asset")
    assert per_asset.shape == (300, 11, 8)
    assert np.isfinite(per_asset.sel(field="relative_return").values[-1]).all()
    assert_allclose(
        per_asset.sel(field="bias").values[100], futures.output.values[100]
    )


@dataclass
class FakeMarket:
    data: xr.DataArray
    requests: list[dict[str, object]]


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> FakeMarket:
    request.getfixturevalue("sandbox")
    data = make_data(name="futures")
    requests: list[dict[str, object]] = []

    def fake_load(competition_type: str, **kwargs: object) -> xr.DataArray:
        requests.append({"competition_type": competition_type, **kwargs})
        assets = kwargs.get("assets")
        if assets is None:
            return data
        return data.sel(asset=assets)

    monkeypatch.setattr(qndata, "load_data_by_type", fake_load)
    return FakeMarket(data, requests)


def momentum(data: qnbt.DataSet) -> xr.DataArray:
    assert isinstance(data, xr.DataArray)
    close = data.sel(field="close")
    ratio = close.isel(time=-1) / close.isel(time=-21) - 1
    return ratio.where(data.sel(field="is_liquid").isel(time=-1) > 0).fillna(0)


def counting_momentum(data: qnbt.DataSet, state: int | None) -> tuple[xr.DataArray, int]:
    return momentum(data), (0 if state is None else state) + 1


def test_backtest_end_to_end_stateless(market: FakeMarket, sandbox: Path) -> None:
    result: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
        start_date="2019-06-03",
        end_date="2020-02-24",
        build_plots=False,
    )
    assert isinstance(result, xr.DataArray)
    assert result.name == "futures"
    assert result.shape == (191, 8)
    assert str(result.coords["time"].values[0])[:10] == "2019-06-03"
    assert str(result.coords["time"].values[-1])[:10] == "2020-02-24"
    values = result.values
    assert_allclose(
        values[0],
        [
            0.081819990638,
            0.092885724853,
            -0.0215390115,
            -0.06018269708,
            0.013037607709,
            -0.034680798231,
            -0.001741324053,
            -0.014729125735,
        ],
        atol=1e-11,
    )
    assert_allclose(values[50, :3], [-0.1484792299, -0.121074358438, 0.061246134968], atol=1e-11)
    assert_allclose(values[-1, 4], -0.068117572117, atol=1e-11)
    assert_allclose(values.sum(), 18.306864475137928, rtol=1e-10)
    assert_allclose(np.abs(values).sum(), 93.915839240282, rtol=1e-10)

    stat = qnstats.calc_stat(market.data, result)
    assert_allclose(
        stat.values[-1],
        [
            1.0701092084712562,
            0.0006849833202653421,
            0.05818206246595051,
            -0.01831908924623815,
            -0.021351448799546535,
            1.6674688462661262,
            0.09701677657348218,
            0.4530482950617865,
            8.0,
            0.13306041431634644,
            7.295373495127145,
        ],
        rtol=1e-9,
    )

    with gzip.open(sandbox / "fractions.nc.gz", "rb") as stream:
        stored = xr.open_dataarray(stream.read(), cache=False, decode_times=True).compute()
    assert list(stored.coords["asset"].values) == sorted(market.data.coords["asset"].values)
    assert_allclose(stored.values, values, atol=1e-12)
    assert not (sandbox / "state.out.pickle.gz").exists()
    cleanup_requests = [r for r in market.requests if "assets" in r]
    assert cleanup_requests[0]["tail"] == 60


def test_backtest_stateful_final_and_collected_states(market: FakeMarket, sandbox: Path) -> None:
    outcome: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=counting_momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
        start_date="2019-06-03",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
    )
    assert isinstance(outcome, tuple)
    result, state = outcome
    assert result.shape == (191, 8)
    assert state == 191
    assert qnstate.read(str(sandbox / "state.out.pickle.gz")) == 191

    collected = qnbt.backtest(
        competition_type="futures",
        strategy=counting_momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
        start_date="2019-12-02",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
        collect_all_states=True,
    )
    assert isinstance(collected, tuple)
    window_result, states = collected
    assert states == list(range(1, window_result.shape[0] + 1))


def test_backtest_step_leaves_skipped_days_flat(market: FakeMarket) -> None:
    result: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
        start_date="2019-06-03",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
        step=5,
    )
    assert isinstance(result, xr.DataArray)
    traded_days = np.abs(result.values).sum(axis=1) > 0
    assert result.shape == (191, 8)
    assert traded_days.sum() == 39
    assert_array_equal(np.flatnonzero(traded_days)[:4], [0, 5, 10, 15])


def test_run_iterations_window_step_and_validation(market: FakeMarket) -> None:
    times = market.data.coords["time"].values
    start = np.datetime64("2019-06-03")
    lengths: list[int] = []

    def recording(window: qnbt.DataSet, _state: int | None) -> xr.DataArray:
        assert isinstance(window, xr.DataArray)
        lengths.append(window.sizes["time"])
        return momentum(window)

    outcome = qnbt.run_iterations(
        times, market.data, qnbt.standard_window, start, 60, recording, 7, False
    )
    assert outcome is not None
    output, state = outcome
    assert state is None
    assert output is not None
    assert output.shape == (28, 8)
    run_days = times[times >= start][::7]
    expected = [
        int(((times >= day - np.timedelta64(60, "D")) & (times <= day)).sum()) for day in run_days
    ]
    assert lengths == expected

    def wrong_dims(window: qnbt.DataSet, _state: int | None) -> xr.DataArray:
        assert isinstance(window, xr.DataArray)
        return window

    assert (
        qnbt.run_iterations(
            times, market.data, qnbt.standard_window, start, 60, wrong_dims, 1, False
        )
        is None
    )


def test_unpack_result_and_loader_edge_cases() -> None:
    output = xr.DataArray([1.0], dims=["asset"], coords={"asset": ["a"]})
    assert qnbt.unpack_result(output)[1] is None
    assert qnbt.unpack_result((output, "state"))[1] == "state"
    with pytest.raises(IndexError):
        qnbt.unpack_result(cast("tuple[xr.DataArray, None]", ()))
    with pytest.raises(AttributeError):
        qnbt.extract_time_series({"a": output})
    with pytest.raises(AttributeError):
        qnbt.standard_window({"a": output}, np.datetime64("2020-01-01"), 10)


def tail_loader(market: FakeMarket) -> Callable[[int], xr.DataArray]:
    last = market.data.coords["time"].values[-1]

    def load(tail: int) -> xr.DataArray:
        return market.data.sel(time=slice(last - np.timedelta64(tail, "D"), None))

    return load


def mean_return_model(data: qnbt.DataSet) -> float:
    assert isinstance(data, xr.DataArray)
    close = data.sel(field="close")
    return float((close.isel(time=-1) / close.isel(time=0) - 1).mean())


def trend_follower(model: float, data: qnbt.DataSet) -> xr.DataArray:
    assert isinstance(data, xr.DataArray)
    close = data.sel(field="close")
    trend = (close / close.shift(time=5) - 1).fillna(0) * (1 if model >= 0 else -1)
    return trend * data.sel(field="is_liquid")


def counting_trend_follower(
    model: float, data: qnbt.DataSet, state: int | None
) -> tuple[xr.DataArray, int]:
    return trend_follower(model, data), (0 if state is None else state) + 1


@pytest.mark.parametrize("predict_each_day", [False, True])
def test_backtest_ml_walk_forward(
    market: FakeMarket, sandbox: Path, predict_each_day: bool
) -> None:
    result: BacktestOutput[int] | None = qnbt.backtest_ml(
        train=mean_return_model,
        predict=trend_follower,
        competition_type="futures",
        load_data=tail_loader(market),
        train_period=60,
        retrain_interval=30,
        lookback_period=20,
        start_date="2019-09-02",
        end_date="2020-02-24",
        predict_each_day=predict_each_day,
        build_plots=False,
    )
    assert isinstance(result, xr.DataArray)
    assert result.name == "futures"
    assert result.shape == (126, 8)
    assert str(result.coords["time"].values[0])[:10] == "2019-09-02"
    values = result.values
    assert_allclose(
        values[0],
        [-0.0231595047, 0.0176298818, -0.0174220644, -0.0027681225, 0.0622234237,
         -0.0171663818, -0.0165180921, 0.0285008629],
        atol=1e-9,
    )  # fmt: skip
    assert_allclose(values.sum(), 2.841252492895272, rtol=1e-9)
    assert_allclose(np.abs(values).sum(), 26.88792543765484, rtol=1e-9)
    assert (sandbox / "fractions.nc.gz").exists()


def test_backtest_ml_collects_states_per_retrain_batch(market: FakeMarket, sandbox: Path) -> None:
    outcome = qnbt.backtest_ml(
        train=mean_return_model,
        predict=counting_trend_follower,
        competition_type="futures",
        load_data=tail_loader(market),
        train_period=60,
        retrain_interval=30,
        lookback_period=20,
        start_date="2019-09-02",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
        collect_all_states=True,
    )
    assert isinstance(outcome, tuple)
    result, states = outcome
    assert result.shape == (126, 8)
    assert states == [1, 2, 3, 4, 5, 6]
    saved = qnstate.read(str(sandbox / "state.out.pickle.gz"))
    assert isinstance(saved, tuple)
    assert saved[2] == 6


def test_submission_mode_runs_only_the_last_pass(
    market: FakeMarket, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    stateless: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
    )
    assert isinstance(stateless, xr.DataArray)
    assert stateless.shape == (1, 8)
    assert stateless.coords["time"].values[0] == market.data.coords["time"].values[-1]
    assert_allclose(np.abs(stateless.values).sum(), 0.3423072777065316, rtol=1e-9)

    with gzip.open(sandbox / "state.in.pickle.gz", "wb") as stream:
        pickle.dump(41, stream)
    stateful = qnbt.backtest(
        competition_type="futures",
        strategy=counting_momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
    )
    assert isinstance(stateful, tuple)
    assert stateful[1] == 42
    assert qnstate.read(str(sandbox / "state.out.pickle.gz")) == 42

    (sandbox / "state.in.pickle.gz").unlink()
    ml_output: BacktestOutput[int] | None = qnbt.backtest_ml(
        train=mean_return_model,
        predict=trend_follower,
        competition_type="futures",
        load_data=tail_loader(market),
        train_period=60,
        lookback_period=20,
    )
    assert isinstance(ml_output, xr.DataArray)
    assert ml_output.shape == (15, 8)


def test_precomputed_time_series_and_custom_window(market: FakeMarket) -> None:
    times = market.data.coords["time"].values
    data, series = qnbt.extract_time_series((market.data, times[:10]))
    assert data is market.data
    assert_array_equal(series, times[:10])

    def momentum_of_recent_window(window: qnbt.DataSet) -> xr.DataArray:
        assert isinstance(window, xr.DataArray)
        return momentum(window.isel(time=slice(-21, None)))

    def short_window(data: qnbt.DataSet, max_date: np.datetime64, _lookback: int) -> qnbt.DataSet:
        assert isinstance(data, xr.DataArray)
        return data.sel(time=slice(max_date - np.timedelta64(30, "D"), max_date))

    result: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum_of_recent_window,
        load_data=lambda _tail: market.data,
        window=short_window,
        lookback_period=60,
        start_date="2019-12-02",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == (61, 8)


class DashRecorder:
    def __init__(self) -> None:
        self.callbacks: list[object] = []
        self.started = False

    def callback(self, _output: object, _inputs: object, /) -> Callable[[_F], _F]:
        def register(function: _F) -> _F:
            self.callbacks.append(function)
            return function

        return register

    def run(self, *, dev_tools_hot_reload: bool) -> None:
        assert not dev_tools_hot_reload
        self.started = True


class NoTimer:
    def __init__(self, interval: float, function: Callable[[], object]) -> None:
        self.interval = interval
        self.function = function

    def start(self) -> None:
        return None


class TableCallback(Protocol):
    def __call__(
        self, asset: str, row_offset: int, column_offset: int = 0
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]: ...


def test_backtest_builds_dash_report_with_working_callbacks(
    market: FakeMarket, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = DashRecorder()

    def fake_app(_name: str) -> DashRecorder:
        return recorder

    monkeypatch.setattr(dash, "Dash", fake_app)
    monkeypatch.setattr(threading, "Timer", NoTimer)
    result: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
        start_date="2019-12-02",
        end_date="2020-02-24",
        build_plots=True,
    )
    assert isinstance(result, xr.DataArray)
    assert recorder.started
    assert len(recorder.callbacks) == 3
    on_asset = cast("Callable[[str], tuple[object, ...]]", recorder.callbacks[0])
    on_output = cast("TableCallback", recorder.callbacks[1])
    on_stats = cast("TableCallback", recorder.callbacks[2])

    overall = on_asset("")
    assert len(overall) == 5
    assert overall[4] == 0
    assert all(item is not None for item in overall[:4])
    assert len(on_asset("NYS:A1")) == 5

    columns, rows = on_output("", 0, 0)
    assert [c["id"] for c in columns] == ["time", *sorted(result.coords["asset"].values)]
    assert len(rows) == 10
    assert rows[0]["time"] == "2019-12-02T00"
    assert rows[0]["NYS:A0"] == f"{result.values[0, 0]:.6f}"
    single_columns, single_rows = on_output("NYS:A3", 5, 0)
    assert [c["id"] for c in single_columns] == ["time", "NYS:A3"]
    assert len(single_rows) == 10

    stat_columns, stat_rows = on_stats("", 0)
    assert stat_columns[0]["id"] == "time"
    assert "sharpe_ratio" in [c["id"] for c in stat_columns]
    assert len(stat_rows) == 10
    assert len(on_stats("NYS:A2", 3)[1]) == 10


def test_jupyter_report_prints_output_and_stats_tables(
    market: FakeMarket, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NONINTERACT", "1")
    data = market.data
    weights = qnout.clean(make_weights(data), data, "futures", debug=False)
    qnbt.build_plots_jupyter(
        weights,
        qnstats.calc_stat(data, weights),
        qnstats.calc_stat(data, weights, per_asset=True),
    )
    printed = capsys.readouterr().out
    assert "Build plots..." in printed
    assert "Output:" in printed
    assert "Stats:" in printed
    assert "[10 rows x 11 columns]" in printed


def test_analyze_results_skips_empty_output(
    market: FakeMarket, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = xr.DataArray(
        np.zeros((0, 8)),
        dims=["time", "asset"],
        coords={
            "time": np.array([], dtype="datetime64[ns]"),
            "asset": market.data.coords["asset"].values,
        },
    )
    qnbt.analyze_results(empty, market.data, "futures", build_plots=False)
    assert "ERROR! Output is empty!" in capsys.readouterr().err


def test_default_loader_and_empty_full_history(market: FakeMarket) -> None:
    result: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        lookback_period=60,
        start_date="2019-12-02",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == (61, 8)
    assert {r["competition_type"] for r in market.requests} == {"futures"}
    assert any(r.get("tail") == 60 for r in market.requests)

    recent = tail_loader(market)

    def shrinking(tail: int) -> xr.DataArray:
        if tail <= 100:
            return recent(tail)
        return recent(tail).sel(time=slice("2100-01-01", None))

    outcome: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        load_data=shrinking,
        lookback_period=60,
        start_date="2019-12-02",
        end_date="2020-02-24",
        build_plots=False,
    )
    assert outcome is None


def test_multi_pass_mode_in_a_submission_runs_every_pass(
    market: FakeMarket, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "9")
    monkeypatch.setenv("MULTI_PASS_MODE_ENABLED", "1")
    result: BacktestOutput[int] | None = qnbt.backtest(
        competition_type="futures",
        strategy=momentum,
        load_data=lambda _tail: market.data,
        lookback_period=60,
        start_date="2019-12-02",
        end_date="2020-02-24",
        build_plots=False,
        analyze=False,
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == (61, 8)


def test_backtest_ml_requires_prediction_for_the_last_date(market: FakeMarket) -> None:
    def truncated(model: float, data: qnbt.DataSet) -> xr.DataArray:
        return trend_follower(model, data).isel(time=slice(0, -1))

    with pytest.raises(UnboundLocalError):
        qnbt.backtest_ml(
            train=mean_return_model,
            predict=truncated,
            competition_type="futures",
            load_data=tail_loader(market),
            train_period=60,
            lookback_period=20,
        )


def test_backtest_ml_reuses_the_persisted_model_in_submissions(
    market: FakeMarket, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    trained: list[int] = []

    def counting_train(data: qnbt.DataSet) -> float:
        trained.append(1)
        return mean_return_model(data)

    def run() -> None:
        qnbt.backtest_ml(
            train=counting_train,
            predict=trend_follower,
            competition_type="futures",
            load_data=tail_loader(market),
            train_period=60,
            lookback_period=20,
        )

    run()
    assert len(trained) == 1
    (sandbox / "state.out.pickle.gz").rename(sandbox / "state.in.pickle.gz")
    run()
    assert len(trained) == 1
    monkeypatch.setenv("IN_STATE_PATH", str(sandbox / "missing.pickle.gz"))
    run()
    assert len(trained) == 2
