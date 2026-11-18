import gzip
import inspect
import pickle
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

import dash
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.backtester as qnbt
import qnt.data as qndata
import qnt.data.common as qndc
import qnt.output as qnout
import qnt.state as qnstate
import qnt.stats as qnstats
from qnt.tests.hl_support import make_data, make_weights

pytestmark = pytest.mark.usefixtures("quiet_tracking")

Times: TypeAlias = npt.NDArray[np.datetime64]
Span: TypeAlias = tuple[np.datetime64, np.datetime64]
MaxDate: TypeAlias = object
_S = TypeVar("_S")
_F = TypeVar("_F")
BacktestOutput: TypeAlias = (
    xr.DataArray | tuple[xr.DataArray, _S | list[_S] | list[_S | None] | None]
)

START = "2019-12-02"
END = "2020-02-21"
DAYS_START_TO_END = 81


@dataclass
class DashCalls:
    apps: list[object] = field(default_factory=list)

    def make_app(self, _name: str) -> "FakeDashApp":
        return FakeDashApp(self)


class FakeDashApp:
    def __init__(self, calls: DashCalls) -> None:
        self.calls = calls
        self.layout: object = None

    def callback(self, _output: object, _inputs: object, /) -> Callable[[_F], _F]:
        return lambda function: function

    def run(self, *, dev_tools_hot_reload: bool) -> None:
        assert not dev_tools_hot_reload
        self.calls.apps.append(self)


def _no_browser(*_args: object, **_kwargs: object) -> bool:
    return True


@pytest.fixture(autouse=True)
def dash_calls(monkeypatch: pytest.MonkeyPatch) -> DashCalls:
    calls = DashCalls()
    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(webbrowser, name, _no_browser)
    monkeypatch.setattr(dash, "Dash", calls.make_app)
    return calls


@pytest.fixture(scope="module", autouse=True)
def warm_numba(request: pytest.FixtureRequest) -> None:
    request.getfixturevalue("quiet_tracking")
    data = make_data(name="futures")
    output = qnout.clean(make_weights(data), data, "futures", debug=False)
    qnstats.calc_stat(data, output)
    qnstats.calc_stat(data, output, per_asset=True)


class NoTimer:
    def __init__(self, interval: float, function: Callable[[], object]) -> None:
        self.interval = interval
        self.function = function

    def start(self) -> None:
        return None


@dataclass
class TailLoader:
    """Loads the last `tail` calendar days up to the maximum date that the backtester has set."""

    data: xr.DataArray
    tails: list[int] = field(default_factory=list)

    def __call__(self, tail: int) -> xr.DataArray:
        self.tails.append(tail)
        limit = qndc.MAX_DATE_LIMIT
        part = (
            self.data
            if limit is None
            else self.data.sel(time=slice(None, np.datetime64(limit)))
        )
        last = part.coords["time"].values[-1]
        return part.sel(time=slice(last - np.timedelta64(tail, "D"), None))


@dataclass
class Market:
    data: xr.DataArray
    times: Times
    requests: list[dict[str, object]]
    loader: TailLoader

    def until(self, day: str | np.datetime64) -> Times:
        return self.times[self.times <= np.datetime64(day)]


@pytest.fixture
def market(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Market:
    request.getfixturevalue("sandbox")
    data = make_data(name="futures")
    requests: list[dict[str, object]] = []

    def fake_load(competition_type: str, **kwargs: object) -> xr.DataArray:
        requests.append({"competition_type": competition_type, **kwargs})
        limit = qndc.MAX_DATE_LIMIT
        part = data if limit is None else data.sel(time=slice(None, np.datetime64(limit)))
        assets = kwargs.get("assets")
        if assets is None:
            return part
        return part.sel(asset=assets)

    monkeypatch.setattr(qndata, "load_data_by_type", fake_load)
    return Market(data, data.coords["time"].values, requests, TailLoader(data))


@pytest.fixture
def limits(monkeypatch: pytest.MonkeyPatch) -> list[MaxDate]:
    calls: list[MaxDate] = []
    original = qndc.set_max_datetime

    def recording(dt: str | np.datetime64 | None) -> None:
        calls.append(dt)
        original(dt)

    monkeypatch.setattr(qndc, "set_max_datetime", recording)
    return calls


class ReadCounter:
    def __init__(self) -> None:
        self.count = 0


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> ReadCounter:
    counter = ReadCounter()
    original = qnstate.read

    def counting(path: str | None = None) -> object:
        counter.count += 1
        return original(path)

    monkeypatch.setattr(qnstate, "read", counting)
    return counter


def span_of(data: qnbt.DataSet) -> Span:
    assert isinstance(data, xr.DataArray)
    times = data.coords["time"].values
    return times[0], times[-1]


def expected_span(times: Times, end: np.datetime64, tail: int) -> Span:
    inside = times[(times >= end - np.timedelta64(tail, "D")) & (times <= end)]
    return inside[0], inside[-1]


def batches(days: Times, interval: int) -> list[Span]:
    plan: list[Span] = []
    i = 0
    while i < len(days):
        bound = days[i] + np.timedelta64(max(interval - 1, 0), "D")
        j = int(np.flatnonzero(days <= bound)[-1])
        plan.append((days[i], days[j]))
        i = j + 1
    return plan


def signal(data: qnbt.DataSet) -> xr.DataArray:
    assert isinstance(data, xr.DataArray)
    close = data.sel(field="close")
    return (close / close.shift(time=5) - 1).fillna(0) * data.sel(field="is_liquid")


def normalized_signal(market: Market, days: Times) -> npt.NDArray[np.float64]:
    raw = np.asarray(signal(market.data).sel(time=days).values, dtype=np.float64)
    return np.asarray(
        raw / np.maximum(1.0, np.abs(raw).sum(axis=1, keepdims=True)), dtype=np.float64
    )


@dataclass(frozen=True)
class PredictCall:
    model: np.datetime64 | None
    span: Span
    state: int | None


@dataclass
class Spy:
    trains: list[Span] = field(default_factory=list)
    predicts: list[PredictCall] = field(default_factory=list)

    def train(self, data: qnbt.DataSet) -> np.datetime64:
        window = span_of(data)
        self.trains.append(window)
        return window[1]


StatelessPredict: TypeAlias = Callable[[np.datetime64, qnbt.DataSet], xr.DataArray]
StatefulPredict: TypeAlias = Callable[
    [np.datetime64, qnbt.DataSet, int | None], tuple[xr.DataArray, int]
]


def stateless_predict(spy: Spy) -> StatelessPredict:
    def predict(model: np.datetime64, data: qnbt.DataSet) -> xr.DataArray:
        spy.predicts.append(PredictCall(model, span_of(data), None))
        return signal(data)

    return predict


def stateful_predict(spy: Spy) -> StatefulPredict:
    def predict(
        model: np.datetime64, data: qnbt.DataSet, state: int | None
    ) -> tuple[xr.DataArray, int]:
        spy.predicts.append(PredictCall(model, span_of(data), state))
        return signal(data), (0 if state is None else state) + 1

    return predict


def four_argument_predict(spy: Spy) -> StatefulPredict:
    def predict(
        model: np.datetime64, data: qnbt.DataSet, state: int | None = None, scale: float = 1.0
    ) -> tuple[xr.DataArray, int]:
        spy.predicts.append(PredictCall(model, span_of(data), state))
        return signal(data) * scale, (0 if state is None else state) + 1

    return predict


def run_ml(
    market: Market,
    spy: Spy,
    predict: StatelessPredict | StatefulPredict,
    *,
    train_period: int = 60,
    retrain_interval: int = 30,
    predict_each_day: bool = False,
    retrain_interval_after_submit: int | None = None,
    lookback_period: int = 20,
    test_period: int = 365 * 15,
    start_date: str | None = START,
    end_date: str | None = END,
    collect_all_states: bool = False,
    analyze: bool = False,
    build_plots: bool = False,
) -> BacktestOutput[int] | None:
    return qnbt.backtest_ml(
        train=spy.train,
        predict=predict,
        competition_type="futures",
        load_data=market.loader,
        train_period=train_period,
        retrain_interval=retrain_interval,
        predict_each_day=predict_each_day,
        retrain_interval_after_submit=retrain_interval_after_submit,
        lookback_period=lookback_period,
        test_period=test_period,
        start_date=start_date,
        end_date=end_date,
        analyze=analyze,
        build_plots=build_plots,
        collect_all_states=collect_all_states,
    )


def flat_strategy(data: qnbt.DataSet) -> xr.DataArray:
    return signal(data).isel(time=-1)


def varargs_strategy(*data: qnbt.DataSet) -> xr.DataArray:
    return signal(data[0]).isel(time=-1)


def counting_strategy(data: qnbt.DataSet, state: int | None) -> tuple[xr.DataArray, int]:
    return signal(data).isel(time=-1), (0 if state is None else state) + 1


def extra_argument_strategy(
    data: qnbt.DataSet, state: int | None = None, scale: float = 1.0
) -> tuple[xr.DataArray, int]:
    return signal(data).isel(time=-1) * scale, (0 if state is None else state) + 1


def run_bt(
    market: Market,
    strategy: Callable[[qnbt.DataSet], xr.DataArray]
    | Callable[[qnbt.DataSet, int | None], tuple[xr.DataArray, int]],
    *,
    load_data: Callable[[int], xr.DataArray] | None = None,
    lookback_period: int = 60,
    test_period: int = 365 * 15,
    start_date: str | None = START,
    end_date: str | None = END,
    collect_all_states: bool = False,
    analyze: bool = False,
    build_plots: bool = False,
) -> BacktestOutput[int] | None:
    return qnbt.backtest(
        competition_type="futures",
        strategy=strategy,
        load_data=market.loader if load_data is None else load_data,
        lookback_period=lookback_period,
        test_period=test_period,
        start_date=start_date,
        end_date=end_date,
        analyze=analyze,
        build_plots=build_plots,
        collect_all_states=collect_all_states,
    )


def read_fractions(sandbox: Path) -> xr.DataArray:
    with gzip.open(sandbox / "fractions.nc.gz", "rb") as stream:
        return xr.open_dataarray(stream.read(), cache=False, decode_times=True).compute()


def read_pickle(path: Path) -> object:
    return qnstate.read(str(path))


def write_pickle(path: Path, value: object) -> None:
    with gzip.open(path, "wb") as stream:
        pickle.dump(value, stream)


@dataclass
class AnalyzeCall:
    output: xr.DataArray
    data: xr.DataArray
    kind: str
    build_plots: bool
    start: np.datetime64 | str | None
    check_correlation: bool


@pytest.fixture
def analyzed(monkeypatch: pytest.MonkeyPatch) -> list[AnalyzeCall]:
    calls: list[AnalyzeCall] = []

    def recording(
        output: xr.DataArray,
        data: xr.DataArray,
        kind: str,
        build_plots: bool,
        start: np.datetime64 | str | None = None,
        check_correlation: bool = True,
    ) -> None:
        calls.append(AnalyzeCall(output, data, kind, build_plots, start, check_correlation))

    monkeypatch.setattr(qnbt, "analyze_results", recording)
    return calls


# ---------------------------------------------------------------- defaults


SIGNATURE_DEFAULTS: list[tuple[str, str, object]] = [
    ("backtest_ml", "train_period", 4 * 365),
    ("backtest_ml", "retrain_interval", 365),
    ("backtest_ml", "predict_each_day", False),
    ("backtest_ml", "retrain_interval_after_submit", None),
    ("backtest_ml", "load_data", None),
    ("backtest_ml", "lookback_period", 365),
    ("backtest_ml", "test_period", 365 * 15),
    ("backtest_ml", "start_date", None),
    ("backtest_ml", "end_date", None),
    ("backtest_ml", "window", None),
    ("backtest_ml", "analyze", True),
    ("backtest_ml", "build_plots", True),
    ("backtest_ml", "collect_all_states", False),
    ("backtest_ml", "check_correlation", False),
    ("backtest", "load_data", None),
    ("backtest", "lookback_period", 365),
    ("backtest", "test_period", 365 * 15),
    ("backtest", "start_date", None),
    ("backtest", "end_date", None),
    ("backtest", "window", None),
    ("backtest", "step", 1),
    ("backtest", "analyze", True),
    ("backtest", "build_plots", True),
    ("backtest", "collect_all_states", False),
    ("backtest", "check_correlation", False),
    ("analyze_results", "start", None),
]


@pytest.mark.parametrize(
    ("function", "name", "expected"),
    SIGNATURE_DEFAULTS,
    ids=[f"{f}.{n}" for f, n, _ in SIGNATURE_DEFAULTS],
)
def test_documented_defaults(function: str, name: str, expected: object) -> None:
    default = inspect.signature(getattr(qnbt, function)).parameters[name].default
    assert default == expected
    assert type(default) is type(expected)


def test_analyze_results_check_correlation_default_differs_characterization() -> None:
    """The docstrings of backtest/backtest_ml say False, analyze_results itself defaults to True."""
    parameters = inspect.signature(qnbt.analyze_results).parameters
    assert parameters["check_correlation"].default is True


def whole_history_loader(market: Market, tails: list[int]) -> Callable[[int], xr.DataArray]:
    def load(tail: int) -> xr.DataArray:
        tails.append(tail)
        return market.data

    return load


def test_backtest_ml_defaults_reach_the_loader_and_the_analysis(
    market: Market, limits: list[MaxDate], analyzed: list[AnalyzeCall]
) -> None:
    spy = Spy()
    tails: list[int] = []
    outcome = qnbt.backtest_ml(
        train=spy.train,
        predict=stateful_predict(spy),
        competition_type="futures",
        load_data=whole_history_loader(market, tails),
        end_date=END,
    )
    assert tails == [1460, 1460, 5475 + 1460 + 365, 5475]
    assert limits[1] == np.datetime64("2020-02-21") - np.timedelta64(5474, "D")
    assert len(analyzed) == 1
    call = analyzed[0]
    assert (call.kind, call.build_plots, call.check_correlation) == ("futures", True, False)
    assert call.start == limits[1]
    assert isinstance(outcome, tuple)
    plan = batches(market.times, 365)
    assert plan != batches(market.times, 366)
    assert outcome[1] == len(plan)
    assert [t[1] for t in spy.trains[2:]] == [begin for begin, _ in plan]


def test_backtest_defaults_reach_the_loader_and_the_analysis(
    market: Market, limits: list[MaxDate], analyzed: list[AnalyzeCall]
) -> None:
    tails: list[int] = []
    outcome = qnbt.backtest(
        competition_type="futures",
        strategy=counting_strategy,
        load_data=whole_history_loader(market, tails),
        end_date=END,
    )
    assert tails == [365, 365, 5475 + 365]
    assert limits[1] == np.datetime64("2020-02-21") - np.timedelta64(5474, "D")
    assert len(analyzed) == 1
    call = analyzed[0]
    assert (call.kind, call.build_plots, call.check_correlation) == ("futures", True, False)
    assert isinstance(outcome, tuple)
    assert outcome[1] == len(market.times)


# ------------------------------------------------- backtest_ml, full backtest


@pytest.mark.parametrize("predict_each_day", [False, True])
def test_backtest_ml_trains_and_predicts_on_exact_windows(
    market: Market, limits: list[MaxDate], predict_each_day: bool
) -> None:
    spy = Spy()
    result = run_ml(market, spy, stateless_predict(spy), predict_each_day=predict_each_day)
    assert isinstance(result, xr.DataArray)

    last_all = market.times[-1]
    until_end = market.until(END)
    until_start = market.until(START)
    days = until_end[until_end >= np.datetime64(START)]
    plan = batches(days, 30)
    assert len(plan) == 3

    assert spy.trains == [
        expected_span(market.times, last_all, 60),
        expected_span(until_start, until_start[-1], 60),
        *[expected_span(until_end, begin, 60) for begin, _ in plan],
    ]
    expected_predicts = [
        PredictCall(last_all, expected_span(market.times, last_all, 20), None),
        PredictCall(until_start[-1], expected_span(until_start, until_start[-1], 20), None),
    ]
    for begin, end in plan:
        if predict_each_day:
            expected_predicts += [
                PredictCall(begin, expected_span(until_end, day, 20), None)
                for day in days[(days >= begin) & (days <= end)]
            ]
        else:
            expected_predicts.append(PredictCall(begin, expected_span(until_end, end, 50), None))
    assert spy.predicts == expected_predicts

    assert_array_equal(result.coords["time"].values, days)
    assert_allclose(result.values, normalized_signal(market, days), atol=1e-12)
    assert result.name == "futures"
    assert limits == [END, np.datetime64(START), END, None]
    assert qndc.MAX_DATE_LIMIT is None


def test_backtest_ml_start_date_sets_the_test_period_in_calendar_days(market: Market) -> None:
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy))
    assert market.loader.tails == [
        60,
        60,
        DAYS_START_TO_END + 60 + 20,
        DAYS_START_TO_END,
    ]


def test_backtest_ml_without_start_date_counts_back_from_the_last_date(
    market: Market, limits: list[MaxDate]
) -> None:
    spy = Spy()
    result = run_ml(market, spy, stateless_predict(spy), start_date=None, test_period=40)
    assert isinstance(result, xr.DataArray)
    assert limits[0] == END
    assert limits[1] == np.datetime64("2020-01-13")
    assert limits[2] == END
    assert limits[3] is None
    assert market.loader.tails == [60, 60, 40 + 60 + 20, 40]
    assert result.coords["time"].values[0] == np.datetime64("2020-01-13")
    assert spy.trains[1] == expected_span(
        market.until("2020-01-13"), np.datetime64("2020-01-13"), 60
    )


def test_backtest_ml_analysis_data_starts_sixty_days_before_the_first_test_day(
    market: Market,
) -> None:
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy))
    requested = [r["min_date"] for r in market.requests if "min_date" in r]
    assert requested == ["2019-10-03"]


@pytest.mark.parametrize(
    ("predict_each_day", "collect"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_backtest_ml_threads_the_state_through_every_prediction(
    market: Market, predict_each_day: bool, collect: bool
) -> None:
    spy = Spy()
    outcome = run_ml(
        market,
        spy,
        stateful_predict(spy),
        predict_each_day=predict_each_day,
        collect_all_states=collect,
    )
    assert isinstance(outcome, tuple)
    result, state = outcome
    days = market.until(END)
    days = days[days >= np.datetime64(START)]
    calls = len(days) if predict_each_day else 3
    loop_states = [call.state for call in spy.predicts[2:]]
    assert loop_states == [None, *range(1, calls)]
    assert len(spy.predicts) == calls + 2
    assert state == (list(range(1, calls + 1)) if collect else calls)
    assert result.shape == (len(days), 8)


def test_backtest_ml_retrain_interval_one_trains_every_day(market: Market) -> None:
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy), retrain_interval=1, start_date="2020-02-03")
    until_end = market.until(END)
    days = until_end[until_end >= np.datetime64("2020-02-03")]
    assert len(days) == 15
    assert [t[1] for t in spy.trains[2:]] == list(days)
    assert [p.span for p in spy.predicts[2:]] == [expected_span(until_end, d, 21) for d in days]


def test_backtest_ml_retrain_interval_zero_works_like_one_characterization(
    market: Market,
) -> None:
    spy = Spy()
    result = run_ml(
        market, spy, stateless_predict(spy), retrain_interval=0, start_date="2020-02-03"
    )
    assert isinstance(result, xr.DataArray)
    until_end = market.until(END)
    days = until_end[until_end >= np.datetime64("2020-02-03")]
    assert [t[1] for t in spy.trains[2:]] == list(days)
    assert [p.span for p in spy.predicts[2:]] == [expected_span(until_end, d, 20) for d in days]
    assert_array_equal(result.coords["time"].values, days)


def test_backtest_ml_after_submit_interval_is_ignored_outside_submissions(
    market: Market,
) -> None:
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy), retrain_interval_after_submit=7)
    assert len(spy.trains) == 2 + 3


def test_backtest_ml_negative_test_period_characterization(market: Market) -> None:
    """A start date after the last date gives a negative test period; the loader is asked for it."""
    spy = Spy()
    with pytest.raises(IndexError):
        run_ml(market, spy, stateless_predict(spy), start_date="2020-03-02")
    assert market.loader.tails[2:] == [-10 + 60 + 20, -10]


def test_backtest_ml_flags_control_the_analysis(
    market: Market, analyzed: list[AnalyzeCall]
) -> None:
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy), analyze=False)
    assert analyzed == []
    run_ml(market, spy, stateless_predict(spy), analyze=True, build_plots=False)
    assert len(analyzed) == 1
    assert analyzed[0].build_plots is False
    assert analyzed[0].start == np.datetime64(START)
    assert analyzed[0].output.name == "futures"


# ------------------------------------------------------------ submission mode


@dataclass(frozen=True)
class SubmissionCase:
    stateful: bool
    interval: int
    saved_age: int | None
    reads: int
    trained: bool
    written: bool
    saved_model: bool = True
    saved_created: bool = True
    saved_state: int | None = None


SUBMISSION_CASES = {
    "no_saved_state": SubmissionCase(False, 365, None, reads=1, trained=True, written=True),
    "young_model_is_reused": SubmissionCase(False, 365, 10, reads=1, trained=False, written=False),
    "age_equal_to_interval": SubmissionCase(False, 365, 365, reads=1, trained=True, written=True),
    "age_above_interval": SubmissionCase(False, 30, 31, reads=1, trained=True, written=True),
    "interval_one_reads_nothing": SubmissionCase(False, 1, 0, reads=0, trained=True, written=False),
    "interval_two_young": SubmissionCase(False, 2, 1, reads=1, trained=False, written=False),
    "interval_two_boundary": SubmissionCase(False, 2, 2, reads=1, trained=True, written=True),
    "interval_zero_reads_nothing": SubmissionCase(
        False, 0, 1, reads=0, trained=True, written=False
    ),
    "stateful_interval_one": SubmissionCase(
        True, 1, 1, reads=1, trained=True, written=True, saved_state=41
    ),
    "stateful_young_model": SubmissionCase(
        True, 365, 10, reads=1, trained=False, written=True, saved_state=41
    ),
    "saved_model_missing": SubmissionCase(
        False, 365, 10, reads=1, trained=True, written=True, saved_model=False
    ),
    "saved_creation_date_missing": SubmissionCase(
        False, 365, 10, reads=1, trained=True, written=True, saved_created=False
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=pytest.mark.xfail(strict=True, reason="TypeError: None + timedelta when the saved model has no creation date"),
        )
        if name == "saved_creation_date_missing"
        else case
        for name, case in SUBMISSION_CASES.items()
    ],
    ids=list(SUBMISSION_CASES),
)
def test_backtest_ml_submission_state_machine(
    market: Market,
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    reads: ReadCounter,
    case: SubmissionCase,
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    last = market.times[-1]
    old_model = market.times[10]
    created = last - np.timedelta64(case.saved_age or 0, "D")
    if case.saved_age is not None:
        write_pickle(
            sandbox / "state.in.pickle.gz",
            (
                created if case.saved_created else None,
                old_model if case.saved_model else None,
                case.saved_state,
            ),
        )
    spy = Spy()
    predict = stateful_predict(spy) if case.stateful else stateless_predict(spy)
    outcome = run_ml(
        market,
        spy,
        predict,
        train_period=60,
        lookback_period=20,
        retrain_interval=case.interval,
        start_date=None,
        end_date=None,
    )

    assert reads.count == case.reads
    assert market.loader.tails == [60]
    assert spy.trains == ([expected_span(market.times, last, 60)] if case.trained else [])
    model = last if case.trained else old_model
    received = case.saved_state if case.stateful else None
    assert spy.predicts == [PredictCall(model, expected_span(market.times, last, 20), received)]

    fractions = read_fractions(sandbox)
    assert list(fractions.coords["time"].values) == [last]
    assert_allclose(
        fractions.values[0],
        normalized_signal(market, market.times[-1:])[0],
        atol=1e-12,
    )
    out_path = sandbox / "state.out.pickle.gz"
    new_state = (0 if case.saved_state is None else case.saved_state) + 1
    if not case.written:
        assert not out_path.exists()
    else:
        assert read_pickle(out_path) == (
            last if case.trained or not case.saved_created else created,
            model,
            new_state if case.stateful else None,
        )
    if case.stateful:
        assert isinstance(outcome, tuple)
        assert outcome[1] == new_state
    else:
        assert isinstance(outcome, xr.DataArray)
        assert outcome.coords["time"].values[-1] == last


@pytest.mark.parametrize("collect", [False, True])
def test_backtest_ml_submission_returns_the_state_or_a_list_of_it(
    market: Market, monkeypatch: pytest.MonkeyPatch, collect: bool
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    spy = Spy()
    outcome = run_ml(
        market,
        spy,
        stateful_predict(spy),
        start_date=None,
        end_date=None,
        collect_all_states=collect,
    )
    assert isinstance(outcome, tuple)
    assert outcome[1] == ([1] if collect else 1)


def test_backtest_ml_submission_uses_the_interval_after_submit(
    market: Market, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    last = market.times[-1]
    write_pickle(
        sandbox / "state.in.pickle.gz", (last - np.timedelta64(5, "D"), market.times[10], None)
    )
    spy = Spy()
    run_ml(
        market,
        spy,
        stateless_predict(spy),
        retrain_interval=365,
        retrain_interval_after_submit=3,
        start_date=None,
        end_date=None,
    )
    assert len(spy.trains) == 1


def test_backtest_ml_reads_no_state_outside_submissions(market: Market, reads: ReadCounter) -> None:
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy), retrain_interval=365)
    assert reads.count == 0


def test_backtest_ml_submission_writes_the_row_of_the_last_date(
    market: Market, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    spy = Spy()
    run_ml(market, spy, stateless_predict(spy), start_date=None, end_date=None)
    assert read_fractions(sandbox).shape == (1, 8)
    assert spy.predicts[0].span[1] == market.times[-1]


# ----------------------------------------------------------------- backtest


def test_backtest_without_start_date_counts_back_from_the_last_date(
    market: Market, limits: list[MaxDate]
) -> None:
    result = run_bt(market, flat_strategy, start_date=None, test_period=40)
    assert isinstance(result, xr.DataArray)
    assert limits == [END, np.datetime64("2020-01-13"), END, None]
    assert market.loader.tails == [60, 60, 40 + 60]
    assert result.coords["time"].values[0] == np.datetime64("2020-01-13")


def test_backtest_start_date_sets_the_test_period_in_calendar_days(market: Market) -> None:
    result = run_bt(market, flat_strategy)
    assert isinstance(result, xr.DataArray)
    assert market.loader.tails == [60, 60, DAYS_START_TO_END + 60]
    until_end = market.until(END)
    assert_array_equal(result.coords["time"].values, until_end[until_end >= np.datetime64(START)])


def test_backtest_analysis_data_starts_sixty_days_before_the_first_loaded_day(
    market: Market,
) -> None:
    run_bt(market, flat_strategy)
    requested = [r["min_date"] for r in market.requests if "min_date" in r]
    assert requested == ["2019-08-04"]


def test_backtest_negative_test_period_characterization(market: Market) -> None:
    """A start date after the last date gives a negative test period; nothing is iterated."""
    outcome = run_bt(market, flat_strategy, start_date="2020-03-02")
    assert market.loader.tails[2] == -10 + 60
    assert outcome is None or isinstance(outcome, xr.DataArray)


def test_backtest_runs_on_a_series_of_exactly_one_day(market: Market) -> None:
    def one_day(tail: int) -> xr.DataArray:
        part = market.loader(tail)
        if tail <= 60:
            return part
        return part.sel(time=[part.coords["time"].values[-1]])

    result = run_bt(market, flat_strategy, load_data=one_day)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (1, 8)
    assert result.coords["time"].values[0] == np.datetime64(END)


def test_backtest_reports_an_empty_series(
    market: Market, capsys: pytest.CaptureFixture[str]
) -> None:
    def empty(tail: int) -> xr.DataArray:
        part = market.loader(tail)
        return part if tail <= 60 else part.sel(time=slice("2100-01-01", None))

    assert run_bt(market, flat_strategy, load_data=empty) is None
    assert "Time series is empty" in capsys.readouterr().err


def test_backtest_flags_control_the_analysis(market: Market, analyzed: list[AnalyzeCall]) -> None:
    run_bt(market, flat_strategy, analyze=False)
    assert analyzed == []
    run_bt(market, flat_strategy, analyze=True)
    assert len(analyzed) == 1
    assert analyzed[0].start == np.datetime64(START)
    assert analyzed[0].build_plots is False


def test_backtest_extra_argument_strategy_receives_the_state(market: Market) -> None:
    outcome = qnbt.backtest(
        competition_type="futures",
        strategy=extra_argument_strategy,
        load_data=market.loader,
        lookback_period=60,
        start_date="2020-02-03",
        end_date=END,
        analyze=False,
    )
    assert isinstance(outcome, tuple)
    assert outcome[1] == 15


def test_backtest_strategy_with_star_args_is_stateless(market: Market) -> None:
    result = run_bt(market, varargs_strategy)
    assert isinstance(result, xr.DataArray)
    assert result.shape[0] == len(market.until(END)[market.until(END) >= np.datetime64(START)])


AnyStrategy: TypeAlias = (
    Callable[[qnbt.DataSet], xr.DataArray]
    | Callable[[qnbt.DataSet, int | None], tuple[xr.DataArray, int]]
)


@dataclass(frozen=True)
class BacktestSubmissionCase:
    strategy: AnyStrategy
    multi_pass: bool
    reads: int
    kind: str


BACKTEST_SUBMISSION_CASES = {
    "stateless": BacktestSubmissionCase(flat_strategy, False, 0, "array"),
    "star_args": BacktestSubmissionCase(varargs_strategy, False, 0, "array"),
    "stateful": BacktestSubmissionCase(counting_strategy, False, 1, "pair"),
    "stateful_multi_pass": BacktestSubmissionCase(counting_strategy, True, 0, "full"),
}


@pytest.mark.parametrize(
    "case", BACKTEST_SUBMISSION_CASES.values(), ids=list(BACKTEST_SUBMISSION_CASES)
)
def test_backtest_submission_reads_the_state_only_for_stateful_single_pass(
    market: Market,
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
    reads: ReadCounter,
    case: BacktestSubmissionCase,
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    if case.multi_pass:
        monkeypatch.setenv("MULTI_PASS_MODE_ENABLED", "1")
    write_pickle(sandbox / "state.in.pickle.gz", 41)
    outcome = run_bt(market, case.strategy, collect_all_states=True)
    assert reads.count == case.reads
    if case.kind == "array":
        assert isinstance(outcome, xr.DataArray)
        assert outcome.shape == (1, 8)
    elif case.kind == "pair":
        assert isinstance(outcome, tuple)
        assert outcome[0].shape == (1, 8)
        assert outcome[1] == [42]
    else:
        assert isinstance(outcome, tuple)
        assert outcome[0].shape[0] > 1
        assert outcome[1] == list(range(1, outcome[0].shape[0] + 1))


def test_backtest_reads_no_state_outside_submissions(market: Market, reads: ReadCounter) -> None:
    run_bt(market, counting_strategy)
    assert reads.count == 0


def test_backtest_warns_about_a_mismatch_of_data_and_competition_types(
    market: Market, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SUBMISSION_ID", "5")
    same = market.data
    run_bt(market, flat_strategy, load_data=lambda _tail: same)
    assert "WARNING!" not in capsys.readouterr().err
    other = market.data.rename("stocks_nasdaq100")
    run_bt(market, flat_strategy, load_data=lambda _tail: other)
    assert (
        "WARNING! The data type and the competition type are mismatched. "
        "Data type: stocks_nasdaq100, competition type: futures"
    ) in capsys.readouterr().err


def test_predict_with_extra_argument_receives_the_state(market: Market) -> None:
    spy = Spy()
    outcome = run_ml(market, spy, four_argument_predict(spy), start_date="2020-02-03")
    assert isinstance(outcome, tuple)
    assert outcome[1] == len(spy.predicts) - 2


# -------------------------------------------------------------------- Dash


def test_dash_is_started_only_for_plots_outside_submissions(
    market: Market,
    dash_calls: DashCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[xr.DataArray, xr.DataArray, xr.DataArray]] = []

    def recording(
        output: xr.DataArray, stat_global: xr.DataArray, stat_per_asset: xr.DataArray
    ) -> None:
        built.append((output, stat_global, stat_per_asset))

    monkeypatch.setattr(qnbt, "build_plots_dash", recording)
    spy = Spy()
    run_bt(market, flat_strategy, analyze=True, build_plots=False)
    run_ml(market, spy, stateless_predict(spy), analyze=True, build_plots=False)
    assert built == []
    monkeypatch.setenv("SUBMISSION_ID", "5")
    run_bt(market, flat_strategy, analyze=True, build_plots=True)
    run_ml(market, spy, stateless_predict(spy), analyze=True, build_plots=True)
    assert built == []
    assert dash_calls.apps == []

    monkeypatch.delenv("SUBMISSION_ID")
    run_bt(market, flat_strategy, analyze=True, build_plots=True)
    run_ml(market, spy, stateless_predict(spy), analyze=True, build_plots=True)
    assert len(built) == 2


def test_backtest_with_plots_starts_the_dash_server_once(
    market: Market, dash_calls: DashCalls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(threading, "Timer", NoTimer)
    run_bt(market, flat_strategy, analyze=True, build_plots=True)
    assert len(dash_calls.apps) == 1
    run_bt(market, flat_strategy, analyze=True, build_plots=False)
    assert len(dash_calls.apps) == 1


# --------------------------------------------------------- unpack and helpers


def test_unpack_result_shapes_of_the_tuple(capsys: pytest.CaptureFixture[str]) -> None:
    output = xr.DataArray([1.0], dims=["asset"], coords={"asset": ["a"]})
    assert qnbt.unpack_result(cast("tuple[xr.DataArray, None]", (output,))) == (output, None)
    three = cast("tuple[xr.DataArray, str]", (output, "s", "extra"))
    assert qnbt.unpack_result(three) == (output, "s")
    assert capsys.readouterr().err == ""
    with pytest.raises(IndexError):
        qnbt.unpack_result(cast("tuple[xr.DataArray, None]", ()))
    assert "ERROR! The result tuple is empty." in capsys.readouterr().err


def test_unpack_result_reports_a_missing_output(capsys: pytest.CaptureFixture[str]) -> None:
    output = xr.DataArray([1.0], dims=["asset"], coords={"asset": ["a"]})
    qnbt.unpack_result(output)
    assert "Strategy output is None" not in capsys.readouterr().err
    missing: tuple[object, object] = qnbt.unpack_result(cast("xr.DataArray", None))
    assert missing == (None, None)
    assert "ERROR! Strategy output is None!" in capsys.readouterr().err
    kept: tuple[object, object] = qnbt.unpack_result(cast("tuple[xr.DataArray, int]", (None, 3)))
    assert kept == (None, 3)
    assert "ERROR! Strategy output is None!" in capsys.readouterr().err


# ----------------------------------------------------------- analyze_results


class Reached(Exception):
    pass


@dataclass
class Scenario:
    data: xr.DataArray
    output: xr.DataArray


@pytest.fixture
def scenario(market: Market) -> Scenario:
    output = qnout.clean(make_weights(market.data), market.data, "futures", debug=False)
    return Scenario(market.data, output)


def test_analyze_results_guard_counts_assets_exactly(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def stop(*_args: object, **_kwargs: object) -> None:
        raise Reached

    monkeypatch.setattr(qnout, "check", stop)
    single = scenario.output.sel(asset=[scenario.output.coords["asset"].values[0]])
    with pytest.raises(Reached):
        qnbt.analyze_results(single, scenario.data, "futures", build_plots=False)
    none = scenario.output.sel(asset=[])
    qnbt.analyze_results(none, scenario.data, "futures", build_plots=False)
    assert "ERROR! Output is empty!" in capsys.readouterr().err


@pytest.mark.parametrize("start", ["2019-06-03", None])
def test_analyze_results_passes_start_to_align_unchanged(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch, start: np.datetime64 | str | None
) -> None:
    checked: list[bool] = []
    starts: list[object] = []

    def check(*_args: object, check_correlation: bool, **_kwargs: object) -> None:
        checked.append(check_correlation)

    original = qnout.align

    def align(
        output: xr.DataArray,
        time_coord: xr.DataArray,
        start: str | None = None,
        end: str | None = None,
    ) -> xr.DataArray:
        starts.append(start)
        return original(output, time_coord, start, end)

    monkeypatch.setattr(qnout, "check", check)
    monkeypatch.setattr(qnout, "align", align)
    qnbt.analyze_results(scenario.output, scenario.data, "futures", build_plots=False, start=start)
    assert checked == [True]
    assert starts == [start]


def test_analyze_results_cuts_statistics_at_the_first_output_day(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[xr.DataArray, xr.DataArray, xr.DataArray]] = []

    def recording(
        output: xr.DataArray, stat_global: xr.DataArray, stat_per_asset: xr.DataArray
    ) -> None:
        built.append((output, stat_global, stat_per_asset))

    monkeypatch.setattr(qnout, "check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qnbt, "is_notebook", lambda: False)
    monkeypatch.setattr(qnbt, "build_plots_dash", recording)
    qnbt.analyze_results(
        scenario.output, scenario.data, "futures", build_plots=True, start="2019-06-03"
    )
    output, stat_global, stat_per_asset = built[0]
    first = np.datetime64("2019-06-03")
    expected_days = scenario.data.coords["time"].values[scenario.data.coords["time"].values >= first]
    assert_array_equal(output.coords["time"].values, expected_days)
    assert_array_equal(stat_global.coords["time"].values, expected_days)
    assert_array_equal(stat_per_asset.coords["time"].values, expected_days)


def test_analyze_results_without_plots_builds_nothing(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(qnout, "check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(qnbt, "build_plots_dash", lambda *_args: calls.append("dash"))
    monkeypatch.setattr(qnbt, "build_plots_jupyter", lambda *_args: calls.append("jupyter"))
    qnbt.analyze_results(scenario.output, scenario.data, "futures", build_plots=False)
    assert calls == []
    monkeypatch.setattr(qnbt, "is_notebook", lambda: True)
    qnbt.analyze_results(scenario.output, scenario.data, "futures", build_plots=True)
    assert calls == ["jupyter"]
