import itertools
import queue
import random
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import FunctionType, SimpleNamespace, TracebackType
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict, TypeVar, cast

import dash
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.backtester as qnbt
import qnt.graph as qngraph
import qnt.log as qnlog
import qnt.optimizer as qno
import qnt.output as qnout
import qnt.stats as qns
import qnt.ta as qnta
from qnt.tests.hl_support import make_data, make_weights

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

pytestmark = pytest.mark.usefixtures("quiet_tracking")

Floats: TypeAlias = npt.NDArray[np.float64]
_F = TypeVar("_F")
Args: TypeAlias = dict[str, object]
Stats: TypeAlias = Mapping[str, float | Floats]
StatsFunction: TypeAlias = Callable[[xr.DataArray, FunctionType, int], Stats]
StatsToWeight: TypeAlias = Callable[[Stats | None], float]
WorkerOutput: TypeAlias = tuple[Args, Stats | None, float | None, Exception | None]
ChartRow: TypeAlias = dict[str, object]


class TargetCallable(Protocol):
    def __call__(self, **args: object) -> Stats | None: ...


class Iteration(TypedDict):
    args: Args
    result: Stats | None
    weight: float | None
    exception: Exception | None


class OptimizationResult(TypedDict):
    iterations: list[Iteration]
    best_iteration: Iteration


WAIT = 2.0
NOTHING = object()
WORKER_IDLE = 10.0
GRID_X = [1, 2, 3]
GRID_Y = [0, 1]
GRID = [{"x": x, "y": y} for x in GRID_X for y in GRID_Y]


@dataclass
class UiCalls:
    timers: list["FakeTimer"] = field(default_factory=list)
    apps: list[object] = field(default_factory=list)
    opened: list[tuple[str, int]] = field(default_factory=list)

    def make_app(self, _name: str) -> "FakeDashApp":
        return FakeDashApp(self)


class FakeDashApp:
    def __init__(self, calls: UiCalls) -> None:
        self.calls = calls
        self.layout: object = None

    def callback(self, *_args: object) -> Callable[[_F], _F]:
        return lambda function: function

    def run(self, *, dev_tools_hot_reload: bool) -> None:
        assert not dev_tools_hot_reload
        self.calls.apps.append(self)


class FakeTimer:
    def __init__(self, interval: float, function: Callable[[], object]) -> None:
        self.interval = interval
        self.function = function
        self.started = False

    def start(self) -> None:
        self.started = True


@pytest.fixture(autouse=True)
def ui_calls(monkeypatch: pytest.MonkeyPatch) -> UiCalls:
    calls = UiCalls()

    def open_url(url: str, new: int = 0, _autoraise: bool = True) -> bool:
        calls.opened.append((url, new))
        return True

    def timer_factory(interval: float, function: Callable[[], object]) -> FakeTimer:
        timer = FakeTimer(interval, function)
        calls.timers.append(timer)
        return timer

    monkeypatch.setattr(webbrowser, "open", open_url)
    monkeypatch.setattr(webbrowser, "open_new", open_url)
    monkeypatch.setattr(webbrowser, "open_new_tab", open_url)
    monkeypatch.setattr(dash, "Dash", calls.make_app)
    monkeypatch.setattr(threading, "Timer", timer_factory)
    return calls


@pytest.fixture(scope="module", autouse=True)
def warm_numba(request: pytest.FixtureRequest) -> None:
    request.getfixturevalue("quiet_tracking")
    data = make_data(n_assets=3, n_days=60, name="futures")
    output = qnout.clean(make_weights(data), data, "futures", debug=False)
    qns.calc_stat(data, output)


@dataclass
class LogCalls:
    errors: list[tuple[object, ...]] = field(default_factory=list)

    def log_err(self, *args: object) -> None:
        self.errors.append(args)


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> LogCalls:
    calls = LogCalls()
    monkeypatch.setattr(qnlog, "log_err", calls.log_err)
    return calls


class Registry:
    def __init__(self) -> None:
        self.queues: list[FakeQueue] = []
        self.processes: list[FakeProcess] = []
        self.bars: list[FakeBar] = []
        self.by_thread: dict[int, FakeProcess] = {}

    def make_queue(self, maxsize: int = 0) -> "FakeQueue":
        fake = FakeQueue(self, maxsize)
        self.queues.append(fake)
        return fake

    def make_process(
        self, name: str, target: Callable[..., object], args: tuple[object, ...]
    ) -> "FakeProcess":
        fake = FakeProcess(self, name, target, args)
        self.processes.append(fake)
        return fake

    def make_bar(self, **kwargs: object) -> "FakeBar":
        bar = FakeBar(kwargs)
        self.bars.append(bar)
        return bar


class FakeQueue:
    def __init__(self, registry: Registry, maxsize: int) -> None:
        self.maxsize = maxsize
        self.registry = registry
        self.items: queue.Queue[object] = queue.Queue(maxsize)

    def put(self, item: object) -> None:
        try:
            self.items.put(item, timeout=WAIT)
        except queue.Full as exc:
            raise AssertionError("deadlock") from exc

    def empty(self) -> bool:
        return self.items.empty()

    def get(self) -> object:
        started = time.monotonic()
        while True:
            try:
                item = self.items.get(timeout=0.01)
            except queue.Empty:
                item = NOTHING
            if item is not NOTHING:
                return item
            process = self.registry.by_thread.get(threading.get_ident())
            if process is None:
                if time.monotonic() - started > WAIT:
                    raise AssertionError("deadlock")
            elif process.terminated or time.monotonic() - started > WORKER_IDLE:
                raise SystemExit


class FakeProcess:
    def __init__(
        self,
        registry: Registry,
        name: str,
        target: Callable[..., object],
        args: tuple[object, ...],
    ) -> None:
        self.registry = registry
        self.name = name
        self.target = target
        self.args = args
        self.daemon = False
        self.started = False
        self.terminated = False
        self.joined = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        self.registry.by_thread[threading.get_ident()] = self
        try:
            self.target(*self.args)
        except SystemExit:
            return

    def start(self) -> None:
        self.started = True
        self.thread.start()

    def terminate(self) -> None:
        self.terminated = True

    def join(self) -> None:
        self.joined = True
        self.thread.join(WAIT)
        assert not self.thread.is_alive(), "deadlock"


class FakeBar:
    def __init__(self, kwargs: dict[str, object]) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []

    def __enter__(self) -> "FakeBar":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def update(self, value: int) -> None:
        self.updates.append(value)

    @property
    def value(self) -> int:
        return self.updates[-1] if self.updates else 0

    @property
    def max_value(self) -> object:
        return self.kwargs.get("max_value")


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[Registry]:
    registry = Registry()
    monkeypatch.setattr(
        qno,
        "multiprocessing",
        SimpleNamespace(Queue=registry.make_queue, Process=registry.make_process),
    )
    monkeypatch.setattr(qno, "progressbar", SimpleNamespace(ProgressBar=registry.make_bar))
    yield registry
    for process in registry.processes:
        process.terminated = True
    for process in registry.processes:
        if process.started:
            process.thread.join(WAIT)


def int_arg(args: Mapping[str, object], name: str) -> int:
    value = args[name]
    assert isinstance(value, int)
    return value


class ScoreTarget:
    """sharpe_ratio = sign * (10 x + y); raises ValueError for x == fail_x."""

    def __init__(self, sign: float = 1.0, fail_x: int | None = None) -> None:
        self.sign = sign
        self.fail_x = fail_x
        self.calls: list[Args] = []

    def __call__(self, **args: object) -> Stats | None:
        self.calls.append(args)
        x = int_arg(args, "x")
        y = int_arg(args, "y")
        if x == self.fail_x:
            raise ValueError("bad x " + str(x))
        return {"sharpe_ratio": self.sign * (x * 10 + y), "mean_return": float(y)}


class DistanceTarget:
    """sharpe_ratio = |x - 2|: equal for x = 1 and x = 3 and for every y."""

    def __call__(self, **args: object) -> Stats | None:
        return {"sharpe_ratio": float(abs(int_arg(args, "x") - 2))}


def sharpe_weight(stat: Stats | None) -> float:
    assert stat is not None
    return float(stat["sharpe_ratio"])


def grid_generator() -> qno.IterWithLen:
    return qno.full_range_args_generator(x=GRID_X, y=GRID_Y)


def arg_key(iteration: Iteration) -> tuple[int, int]:
    return int_arg(iteration["args"], "x"), int_arg(iteration["args"], "y")


# ---------------------------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fakes")
def test_optimize_single_worker_keeps_generator_order() -> None:
    target = ScoreTarget()
    result = qno.optimize(target, grid_generator(), sharpe_weight, 1)

    iterations = result["iterations"]
    assert [i["args"] for i in iterations] == GRID
    assert [i["result"] for i in iterations] == [
        {"sharpe_ratio": float(x * 10 + y), "mean_return": float(y)} for x in GRID_X for y in GRID_Y
    ]
    assert [i["weight"] for i in iterations] == [float(x * 10 + y) for x in GRID_X for y in GRID_Y]
    assert [i["exception"] for i in iterations] == [None] * 6
    assert all(set(i) == {"args", "result", "weight", "exception"} for i in iterations)
    assert target.calls == GRID
    best = result["best_iteration"]
    assert best is iterations[-1]
    assert best["args"] == {"x": 3, "y": 1}
    assert best["weight"] == 31.0


@pytest.mark.usefixtures("fakes")
def test_optimize_three_workers_returns_the_same_set() -> None:
    result = qno.optimize(ScoreTarget(), grid_generator(), sharpe_weight, 3)

    iterations = sorted(result["iterations"], key=arg_key)
    assert [i["args"] for i in iterations] == GRID
    assert [i["weight"] for i in iterations] == [float(x * 10 + y) for x in GRID_X for y in GRID_Y]
    assert [i["result"] for i in iterations] == [
        {"sharpe_ratio": float(x * 10 + y), "mean_return": float(y)} for x in GRID_X for y in GRID_Y
    ]
    assert len(result["iterations"]) == 6
    assert result["best_iteration"]["args"] == {"x": 3, "y": 1}


@pytest.mark.usefixtures("fakes")
def test_optimize_best_is_the_first_of_equal_weights() -> None:
    result = qno.optimize(DistanceTarget(), grid_generator(), sharpe_weight, 1)

    weights = [i["weight"] for i in result["iterations"]]
    assert weights == [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert result["best_iteration"] is result["iterations"][0]
    assert result["best_iteration"]["args"] == {"x": 1, "y": 0}


@pytest.mark.usefixtures("fakes", "logs")
@pytest.mark.xfail(strict=True, reason="TypeError: optimize compares None weights of failed iterations in max()")
def test_optimize_failed_iteration_is_never_the_best() -> None:
    result = qno.optimize(ScoreTarget(sign=-1.0, fail_x=1), grid_generator(), sharpe_weight, 1)

    failed = result["iterations"][:2]
    assert [i["args"] for i in failed] == [{"x": 1, "y": 0}, {"x": 1, "y": 1}]
    assert all(i["weight"] is None for i in failed)
    assert result["best_iteration"] is result["iterations"][2]
    assert result["best_iteration"]["args"] == {"x": 2, "y": 0}
    assert result["best_iteration"]["weight"] == -20.0


@pytest.mark.parametrize("workers", [1, 3])
def test_optimize_queue_sizes(fakes: Registry, workers: int) -> None:
    qno.optimize(ScoreTarget(), grid_generator(), sharpe_weight, workers)

    assert len(fakes.queues) == 2
    assert fakes.queues[0].maxsize == workers * 2
    assert fakes.queues[1].maxsize == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_optimize_starts_and_stops_daemon_processes(fakes: Registry, workers: int) -> None:
    target = ScoreTarget()
    qno.optimize(target, grid_generator(), sharpe_weight, workers)

    assert len(fakes.processes) == workers
    assert [p.name for p in fakes.processes] == [
        "optimize_worker#" + str(i) for i in range(workers)
    ]
    for process in fakes.processes:
        assert process.daemon is True
        assert process.target is qno.optimize_worker_target
        assert process.args == (target, sharpe_weight, fakes.queues[0], fakes.queues[1])
        assert process.started
        assert process.terminated
        assert process.joined


def test_optimize_progress_bar(fakes: Registry) -> None:
    qno.optimize(ScoreTarget(), grid_generator(), sharpe_weight, 1)

    assert len(fakes.bars) == 1
    assert fakes.bars[0].kwargs == {"max_value": 6, "poll_interval": 1}
    assert fakes.bars[0].updates == [0, 1, 2, 3, 4, 5, 6]


def test_optimize_progress_bar_with_three_workers(fakes: Registry) -> None:
    qno.optimize(
        ScoreTarget(), qno.full_range_args_generator(x=[1, 2, 3, 4], y=[0, 1]), sharpe_weight, 3
    )

    assert fakes.bars[0].kwargs == {"max_value": 8, "poll_interval": 1}
    assert fakes.bars[0].updates == list(range(9))


@pytest.mark.usefixtures("fakes")
def test_optimize_generator_without_len_characterization() -> None:
    """A generator without len -> TypeError about '<' and NoneType after all arguments are queued."""
    generator = iter([{"x": 1, "y": 0}, {"x": 2, "y": 0}])
    with pytest.raises(
        TypeError, match="'<' not supported between instances of 'int' and 'NoneType'"
    ):
        qno.optimize(ScoreTarget(), generator, sharpe_weight, 1)


@pytest.mark.usefixtures("fakes")
def test_optimize_empty_generator_characterization() -> None:
    """No arguments at all -> TypeError from max() of an empty sequence of iterations."""
    generator = qno.full_range_args_generator(x=[], y=[0])
    with pytest.raises(TypeError):
        qno.optimize(ScoreTarget(), generator, sharpe_weight, 1)


# ---------------------------------------------------------------------------------------------
# optimize_worker_target
# ---------------------------------------------------------------------------------------------


class ScriptedInQueue:
    def __init__(self, items: list[Args]) -> None:
        self.items = list(items)

    def get(self) -> Args:
        if not self.items:
            raise SystemExit
        return self.items.pop(0)


class ListOutQueue:
    def __init__(self) -> None:
        self.items: list[WorkerOutput] = []

    def put(self, item: WorkerOutput) -> None:
        self.items.append(item)


def run_worker(
    target: TargetCallable, to_weight: StatsToWeight, items: list[Args]
) -> list[WorkerOutput]:
    out = ListOutQueue()
    with pytest.raises(SystemExit):
        qno.optimize_worker_target(
            target,
            to_weight,
            cast("Queue[Args]", ScriptedInQueue(items)),
            cast("Queue[WorkerOutput]", out),
        )
    return out.items


def test_worker_target_success(logs: LogCalls) -> None:
    target = ScoreTarget()
    items = run_worker(target, sharpe_weight, [{"x": 1, "y": 1}, {"x": 2, "y": 0}])

    assert items == [
        ({"x": 1, "y": 1}, {"sharpe_ratio": 11.0, "mean_return": 1.0}, 11.0, None),
        ({"x": 2, "y": 0}, {"sharpe_ratio": 20.0, "mean_return": 0.0}, 20.0, None),
    ]
    assert target.calls == [{"x": 1, "y": 1}, {"x": 2, "y": 0}]
    assert logs.errors == []


# ---------------------------------------------------------------------------------------------
# TargetFunction
# ---------------------------------------------------------------------------------------------

StrategyOutput: TypeAlias = tuple[int, int, int, int]


def make_strategy() -> FunctionType:
    def strategy(
        data: xr.DataArray, alpha: int = 1, beta: int = 2, gamma: int = 3
    ) -> StrategyOutput:
        del data
        scaled = alpha * 100 + beta * 10 + gamma
        return alpha, beta, gamma, scaled

    return strategy


@dataclass
class StatsSpy:
    calls: list[tuple[xr.DataArray, FunctionType, int]] = field(default_factory=list)
    outputs: list[object] = field(default_factory=list)
    defaults: list[tuple[object, ...] | None] = field(default_factory=list)

    def __call__(self, data: xr.DataArray, output: FunctionType, lookback: int) -> Stats:
        self.calls.append((data, output, lookback))
        self.defaults.append(output.__defaults__)
        self.outputs.append(output(data))
        return {"sharpe_ratio": float(len(self.calls))}


def test_target_function_overrides_defaults_by_name() -> None:
    data = make_data(n_assets=2, n_days=5, name="futures")
    strategy = make_strategy()
    spy = StatsSpy()

    result = qno.TargetFunction(data, strategy, spy, 7)(beta=20, unknown=5)

    assert result == {"sharpe_ratio": 1.0}
    assert len(spy.calls) == 1
    got_data, got_output, got_lookback = spy.calls[0]
    assert got_data is data
    assert got_output is strategy
    assert got_lookback == 7
    assert spy.defaults == [(1, 20, 3)]
    assert spy.outputs == [(1, 20, 3, 1 * 100 + 20 * 10 + 3)]


def test_target_function_overrides_every_argument() -> None:
    data = make_data(n_assets=2, n_days=5, name="futures")
    spy = StatsSpy()

    qno.TargetFunction(data, make_strategy(), spy, 365)(alpha=7, beta=8, gamma=9)

    assert spy.outputs == [(7, 8, 9, 789)]


def test_target_function_without_arguments_keeps_defaults() -> None:
    data = make_data(n_assets=2, n_days=5, name="futures")
    spy = StatsSpy()

    qno.TargetFunction(data, make_strategy(), spy, 365)()

    assert spy.outputs == [(1, 2, 3, 123)]
    assert spy.calls[0][2] == 365


def test_target_function_passes_stats_arguments_positionally() -> None:
    data = make_data(n_assets=2, n_days=5, name="futures")
    received: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def stats(*args: object, **kwargs: object) -> Stats:
        received.append((args, kwargs))
        return {"sharpe_ratio": 2.5}

    strategy = make_strategy()
    result = qno.TargetFunction(data, strategy, cast("StatsFunction", stats), 11)(alpha=4)

    assert result == {"sharpe_ratio": 2.5}
    assert received == [((data, strategy, 11), {})]


def test_target_function_changes_user_strategy_defaults_characterization() -> None:
    """copy.deepcopy returns the same function, so a call rewrites __defaults__ of the user's strategy."""
    data = make_data(n_assets=2, n_days=5, name="futures")
    strategy = make_strategy()
    spy = StatsSpy()
    target = qno.TargetFunction(data, strategy, spy, 365)

    target(alpha=10)
    target(beta=5)

    assert spy.outputs == [(10, 2, 3, 1023), (10, 5, 3, 1053)]
    assert strategy.__defaults__ == (10, 5, 3)


# ---------------------------------------------------------------------------------------------
# argument generators
# ---------------------------------------------------------------------------------------------


def test_full_range_generator_is_the_cartesian_product() -> None:
    generator = qno.full_range_args_generator(a=range(1, 4), b=["x", "y"])

    assert len(generator) == 6
    expected = [{"a": a, "b": b} for a, b in itertools.product(range(1, 4), ["x", "y"])]
    assert list(generator) == expected
    assert [list(item) for item in expected] == [["a", "b"]] * 6


def test_full_range_generator_three_arguments() -> None:
    generator = qno.full_range_args_generator(a=[1, 2], b=["p"], c=range(3))

    assert len(generator) == 6
    assert list(generator) == [{"a": a, "b": "p", "c": c} for a in (1, 2) for c in range(3)]


def test_full_range_generator_single_range() -> None:
    generator = qno.full_range_args_generator(a=range(5, 9))

    assert len(generator) == 4
    assert list(generator) == [{"a": 5}, {"a": 6}, {"a": 7}, {"a": 8}]


def test_full_range_generator_empty_range() -> None:
    generator = qno.full_range_args_generator(a=range(1, 4), b=[])

    assert len(generator) == 0
    assert list(generator) == []


def test_full_range_generator_without_arguments_characterization() -> None:
    """No ranges -> TypeError (reduce without an initial value), the empty product is one empty combination."""
    with pytest.raises(TypeError):
        qno.full_range_args_generator()


def test_random_range_generator_reproduces_the_random_stream() -> None:
    ranges: dict[str, list[object]] = {
        "a": list(range(10, 20)),
        "b": ["p", "q", "r"],
        "c": list(range(0, 1000, 7)),
    }
    random.seed(1234)
    generator = qno.random_range_args_generator(25, **ranges)
    assert len(generator) == 25
    actual = list(generator)

    random.seed(1234)
    expected: list[Args] = []
    for _ in range(25):
        item: Args = {}
        for name, values in ranges.items():
            item[name] = values[random.randrange(len(values))]
        expected.append(item)

    assert actual == expected
    assert all(list(item) == ["a", "b", "c"] for item in actual)


def test_random_range_generator_reaches_every_value() -> None:
    random.seed(5)
    items = list(qno.random_range_args_generator(300, k=range(3)))

    assert len(items) == 300
    assert {int_arg(item, "k") for item in items} == {0, 1, 2}


def test_random_range_generator_zero_iterations() -> None:
    generator = qno.random_range_args_generator(0, k=range(3))

    assert len(generator) == 0
    assert list(generator) == []


def test_iter_with_len() -> None:
    wrapped = qno.IterWithLen((i * i for i in range(4)), 4)

    assert len(wrapped) == 4
    assert list(wrapped) == [0, 1, 4, 9]
    assert list(wrapped) == []


def test_iter_with_len_reports_the_given_length() -> None:
    assert len(qno.IterWithLen([1, 2, 3], 17)) == 17


# ---------------------------------------------------------------------------------------------
# standard_stats_to_weight
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1.5, -2.25, 0.0, 1e9])
def test_stats_to_weight_returns_the_sharpe_ratio(value: float) -> None:
    assert qno.standard_stats_to_weight({"sharpe_ratio": value}) == value


def test_stats_to_weight_ignores_other_keys() -> None:
    assert qno.standard_stats_to_weight({"mean_return": 9.0, "sharpe_ratio": 0.5}) == 0.5


def test_stats_to_weight_missing_key_is_minus_infinity() -> None:
    assert qno.standard_stats_to_weight({"mean_return": 9.0}) == float("-inf")
    assert qno.standard_stats_to_weight({}) == float("-inf")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_stats_to_weight_not_finite_is_minus_infinity(value: float) -> None:
    assert qno.standard_stats_to_weight({"sharpe_ratio": value}) == float("-inf")


def test_stats_to_weight_none_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
        qno.standard_stats_to_weight(None)


# ---------------------------------------------------------------------------------------------
# standard_stats_function
# ---------------------------------------------------------------------------------------------

STAT_FIELDS = ["alpha", "beta", "gamma"]


def make_stat() -> xr.DataArray:
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    values = np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [1.5, -2.0, 7.25]])
    return xr.DataArray(
        values, dims=["time", "field"], coords={"time": times, "field": STAT_FIELDS}
    )


@dataclass
class CalcStatSpy:
    calls: list[tuple[xr.DataArray, xr.DataArray]] = field(default_factory=list)
    log_flags: list[tuple[bool, bool]] = field(default_factory=list)

    def __call__(self, data: xr.DataArray, output: xr.DataArray) -> xr.DataArray:
        self.calls.append((data, output))
        self.log_flags.append((qnlog.info_enabled, qnlog.err_enabled))
        return make_stat()


@pytest.fixture
def calc_stat_spy(monkeypatch: pytest.MonkeyPatch) -> CalcStatSpy:
    spy = CalcStatSpy()
    monkeypatch.setattr(qns, "calc_stat", spy)
    return spy


@dataclass
class BacktestSpy:
    output: xr.DataArray
    result_is_tuple: bool = True
    calls: list[dict[str, object]] = field(default_factory=list)
    log_flags: list[tuple[bool, bool]] = field(default_factory=list)

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        self.log_flags.append((qnlog.info_enabled, qnlog.err_enabled))
        if self.result_is_tuple:
            return self.output, "state"
        return self.output


def simple_data() -> xr.DataArray:
    return make_data(n_assets=3, n_days=60, name="futures")


def plain_strategy_for(data: xr.DataArray) -> FunctionType:
    weights = make_weights(data)

    def strategy(_data: xr.DataArray) -> xr.DataArray:
        return weights

    return strategy


def test_standard_stats_function_returns_last_row_as_floats(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    data = simple_data()
    strategy = plain_strategy_for(data)

    result = qno.standard_stats_function(data, strategy)

    assert result == {"alpha": 1.5, "beta": -2.0, "gamma": 7.25}
    assert all(type(k) is str and type(v) is float for k, v in result.items())
    assert len(calc_stat_spy.calls) == 1
    assert calc_stat_spy.calls[0][0] is data


def test_standard_stats_function_cuts_output_at_default_start(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    data = simple_data()
    weights = make_weights(data)
    times = data.coords["time"].values

    qno.standard_stats_function(data, plain_strategy_for(data))

    received = calc_stat_spy.calls[0][1]
    expected_times = times[times >= np.datetime64("2019-02-05")]
    assert expected_times[0] == np.datetime64("2019-02-05")
    assert_array_equal(received.coords["time"].values, expected_times)
    assert_array_equal(
        received.values, weights.values[times >= np.datetime64("2019-02-05")]
    )


def test_standard_stats_function_start_on_weekend_takes_next_business_day(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-09")
    data = simple_data()
    times = data.coords["time"].values

    qno.standard_stats_function(data, plain_strategy_for(data))

    received = calc_stat_spy.calls[0][1].coords["time"].values
    assert received[0] == np.datetime64("2019-02-11")
    assert_array_equal(received, times[times >= np.datetime64("2019-02-11")])


def test_standard_stats_function_default_start_before_data_uses_first_date(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2006-01-01")
    data = simple_data()
    weights = make_weights(data)

    qno.standard_stats_function(data, plain_strategy_for(data))

    received = calc_stat_spy.calls[0][1]
    assert_array_equal(received.coords["time"].values, data.coords["time"].values)
    assert_array_equal(received.values, weights.values)


def test_standard_stats_function_default_start_after_data_gives_empty_cut(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-03-04")
    data = simple_data()
    times = data.coords["time"].values

    qno.standard_stats_function(data, plain_strategy_for(data))

    assert_array_equal(
        calc_stat_spy.calls[0][1].coords["time"].values, times[times >= np.datetime64("2019-03-04")]
    )


def test_standard_stats_function_unknown_type_starts_at_first_date(
    calc_stat_spy: CalcStatSpy,
) -> None:
    data = make_data(n_assets=3, n_days=20, name="unknown_type")

    qno.standard_stats_function(data, plain_strategy_for(data))

    assert_array_equal(calc_stat_spy.calls[0][1].coords["time"].values, data.coords["time"].values)


def test_standard_stats_function_disables_logs_only_inside(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    monkeypatch.setattr(qnlog, "info_enabled", True)
    monkeypatch.setattr(qnlog, "err_enabled", True)
    data = simple_data()
    weights = make_weights(data)
    strategy_flags: list[tuple[bool, bool]] = []

    def strategy(_data: xr.DataArray) -> xr.DataArray:
        strategy_flags.append((qnlog.info_enabled, qnlog.err_enabled))
        return weights

    qno.standard_stats_function(data, strategy)

    assert strategy_flags == [(False, False)]
    assert calc_stat_spy.log_flags == [(False, False)]
    assert qnlog.info_enabled is True
    assert qnlog.err_enabled is True


def test_standard_stats_function_state_strategy_uses_the_backtester(
    monkeypatch: pytest.MonkeyPatch, calc_stat_spy: CalcStatSpy
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    data = simple_data()
    weights = make_weights(data)
    times = data.coords["time"].values
    backtest = BacktestSpy(weights)
    monkeypatch.setattr(qnbt, "backtest", backtest)

    def strategy(_data: xr.DataArray, state: object) -> xr.DataArray:
        raise AssertionError("the strategy must be run by the backtester", state)

    result = qno.standard_stats_function(data, strategy, 30)

    assert result == {"alpha": 1.5, "beta": -2.0, "gamma": 7.25}
    assert backtest.calls == [
        {
            "competition_type": "futures",
            "lookback_period": 30,
            "start_date": "2019-02-05",
            "strategy": strategy,
            "analyze": False,
            "build_plots": False,
            "collect_all_states": False,
        }
    ]
    assert backtest.log_flags == [(False, False)]
    assert len(calc_stat_spy.calls) == 1
    got_data, got_output = calc_stat_spy.calls[0]
    assert got_data is data
    assert_array_equal(got_output.coords["time"].values, times[times >= np.datetime64("2019-02-05")])
    assert_array_equal(
        got_output.values, weights.values[times >= np.datetime64("2019-02-05")]
    )


@pytest.mark.usefixtures("calc_stat_spy")
def test_standard_stats_function_state_strategy_start_date_before_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2006-01-01")
    data = simple_data()
    backtest = BacktestSpy(make_weights(data))
    monkeypatch.setattr(qnbt, "backtest", backtest)

    def strategy(_data: xr.DataArray, state: object) -> xr.DataArray:
        raise AssertionError("the strategy must be run by the backtester", state)

    qno.standard_stats_function(data, strategy)

    assert backtest.calls[0]["start_date"] == "2019-01-01"


@pytest.mark.usefixtures("calc_stat_spy")
def test_standard_stats_function_default_lookback_is_365(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    data = simple_data()
    backtest = BacktestSpy(make_weights(data))
    monkeypatch.setattr(qnbt, "backtest", backtest)

    def strategy(_data: xr.DataArray, state: object) -> xr.DataArray:
        raise AssertionError("the strategy must be run by the backtester", state)

    default = qno.standard_stats_function(data, strategy)
    explicit = qno.standard_stats_function(data, strategy, 365)

    assert default == explicit
    assert [call["lookback_period"] for call in backtest.calls] == [365, 365]


@pytest.mark.usefixtures("calc_stat_spy")
def test_standard_stats_function_plain_strategy_does_not_use_the_backtester(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    data = simple_data()
    backtest = BacktestSpy(make_weights(data))
    monkeypatch.setattr(qnbt, "backtest", backtest)
    seen: list[xr.DataArray] = []
    weights = make_weights(data)

    def strategy(arg: xr.DataArray) -> xr.DataArray:
        seen.append(arg)
        return weights

    qno.standard_stats_function(data, strategy)

    assert backtest.calls == []
    assert len(seen) == 1
    assert seen[0] is data


def test_standard_stats_function_with_real_calc_stat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SD_FUTURES", "2019-02-05")
    data = simple_data()
    output = qnout.clean(make_weights(data), data, "futures", debug=False)
    times = data.coords["time"].values

    def strategy(_data: xr.DataArray) -> xr.DataArray:
        return output

    result = qno.standard_stats_function(data, strategy)

    expected = qns.calc_stat(data, output.sel(time=slice("2019-02-05", None)))
    last = expected.isel(time=-1)
    names = [str(name) for name in last.coords["field"].values]
    assert times[times >= np.datetime64("2019-02-05")][0] == np.datetime64("2019-02-05")
    assert list(result) == names
    assert all(type(value) is float for value in result.values())
    assert_allclose(list(result.values()), last.values, rtol=1e-12, equal_nan=True)
    assert np.isfinite(result["sharpe_ratio"])


# ---------------------------------------------------------------------------------------------
# fast_stats_function
# ---------------------------------------------------------------------------------------------


def fast_stats_inputs() -> tuple[xr.DataArray, xr.DataArray]:
    data = make_data(n_assets=3, n_days=45, name="futures")
    close_index = list(data.coords["field"].values).index("close")
    values = data.values.copy()
    values[30, close_index, :] = np.nan
    values[37, close_index, 1] = np.nan
    data = data.copy(data=values)
    rng = np.random.default_rng(21)
    weights = rng.normal(0.0, 2.0, (45, 3)) * rng.choice([-1.0, 1.0, 0.5], (45, 3))
    output = xr.DataArray(
        weights,
        dims=["time", "asset"],
        coords={"time": data.coords["time"].values, "asset": data.coords["asset"].values},
    )
    return data, output


def reference_fast_stats(data: xr.DataArray, output: xr.DataArray) -> tuple[float, float, float]:
    close_full = data.sel(field="close").values
    atr_full = qnta.atr(
            data.sel(field="high"), data.sel(field="low"), data.sel(field="close"), 14
        ).values
    out_full = output.values
    nan_row = np.full((1, close_full.shape[1]), np.nan)
    kept = np.flatnonzero(~np.all(np.isnan(close_full), axis=1))
    close = np.where(np.isnan(close_full[kept]), 0.0, close_full[kept])
    prev_close = np.vstack([nan_row, close[:-1]])
    out = out_full[kept]
    prev_out = np.vstack([nan_row, out_full[:-1]])[kept]
    atr = atr_full[kept]
    with np.errstate(all="ignore"):
        returns = (close - prev_close) * prev_out / prev_close
        slippage = np.abs(out / close - prev_out / prev_close) * 0.05 * atr
        total = np.nansum((returns - slippage) * np.abs(out), axis=1)
        total = np.where(np.isfinite(total), total, np.nan)
        mean_return = float(np.exp(np.nanmean(np.log(1 + total))) - 1)
        volatility = float(np.nanstd(total))
    return mean_return / volatility, mean_return, volatility


def test_fast_stats_function_matches_the_definition() -> None:
    data, output = fast_stats_inputs()

    result = qno.fast_stats_function(data, output)

    sharpe, mean_return, volatility = reference_fast_stats(data, output)
    assert set(result) == {"sharpe_ratio", "mean_return", "volatility"}
    assert_allclose(result["sharpe_ratio"], sharpe, rtol=1e-12)
    assert_allclose(result["mean_return"], mean_return, rtol=1e-12)
    assert_allclose(result["volatility"], volatility, rtol=1e-12)
    assert volatility > 0.0
    assert mean_return != 0.0


def test_fast_stats_function_reference_is_not_degenerate() -> None:
    data, _output = fast_stats_inputs()
    close_full = data.sel(field="close").values
    atr = qnta.atr(
            data.sel(field="high"), data.sel(field="low"), data.sel(field="close"), 14
        ).values

    assert np.all(np.isnan(close_full[30]))
    assert np.isnan(close_full[37, 1])
    assert np.isfinite(close_full[37, [0, 2]]).all()
    assert np.isfinite(atr[20:30]).all()
    assert np.isfinite(atr[38:]).any()


# ---------------------------------------------------------------------------------------------
# prepare_data_for_chart
# ---------------------------------------------------------------------------------------------

ALL_STATS = [
    "sharpe_ratio",
    "max_drawdown",
    "mean_return",
    "volatility",
    "equity",
    "avg_turnover",
    "avg_holding_time",
]


def make_iteration(args: Args, result: Stats | None) -> Iteration:
    return {"args": args, "result": result, "weight": None, "exception": None}


def test_prepare_data_orders_stat_fields_and_skips_unknown() -> None:
    best = make_iteration(
        {"a": 1, "b": 2},
        {"volatility": 0.3, "extra": 9.0, "sharpe_ratio": 1.5, "equity": 2.0, "mean_return": 0.1},
    )
    other = make_iteration(
        {"a": 3, "b": 4},
        {"volatility": 0.6, "sharpe_ratio": 0.5, "equity": 3.0, "mean_return": 0.2, "extra": 8.0},
    )
    failed = make_iteration({"a": 5, "b": 6}, None)

    data, field_list = qno.prepare_data_for_chart(
        {"iterations": [other, failed, best], "best_iteration": best}
    )

    assert field_list == ["a", "b", "sharpe_ratio", "mean_return", "volatility", "equity"]
    assert data == [
        {"a": 3, "b": 4, "sharpe_ratio": 0.5, "mean_return": 0.2, "volatility": 0.6, "equity": 3.0},
        {"a": 1, "b": 2, "sharpe_ratio": 1.5, "mean_return": 0.1, "volatility": 0.3, "equity": 2.0},
    ]
    assert [list(row) for row in data] == [field_list, field_list]


def test_prepare_data_knows_all_seven_statistics_in_order() -> None:
    reverse = {name: float(i) for i, name in enumerate(reversed(ALL_STATS))}
    best = make_iteration({"p": 7}, reverse)

    data, field_list = qno.prepare_data_for_chart({"iterations": [best], "best_iteration": best})

    assert field_list == ["p", *ALL_STATS]
    assert data == [{"p": 7, **reverse}]
    assert list(data[0]) == field_list


def test_prepare_data_only_statistics_present_in_the_best_result() -> None:
    best = make_iteration({"p": 1, "q": 2}, {"sharpe_ratio": 1.0, "max_drawdown": -0.2})

    data, field_list = qno.prepare_data_for_chart({"iterations": [best], "best_iteration": best})

    assert field_list == ["p", "q", "sharpe_ratio", "max_drawdown"]
    assert data == [{"p": 1, "q": 2, "sharpe_ratio": 1.0, "max_drawdown": -0.2}]


def test_prepare_data_best_without_result_raises() -> None:
    best = make_iteration({"p": 1}, None)

    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'keys'"):
        qno.prepare_data_for_chart({"iterations": [best], "best_iteration": best})


# ---------------------------------------------------------------------------------------------
# build_plotly_scatter3d
# ---------------------------------------------------------------------------------------------

CHART_FIELDS = ["a", "b", "c", "d", "e"]
CHART_ROWS: list[ChartRow] = [
    {"a": 1.0, "b": 10.0, "c": 100.0, "d": 0.5, "e": 7.0},
    {"a": 2.0, "b": 20.0, "c": 200.0, "d": 1.5, "e": 8.0},
    {"a": 3.0, "b": 30.0, "c": 300.0, "d": 2.5, "e": 9.0},
    {"a": 4.0, "b": 40.0, "c": 400.0, "d": 3.5, "e": 6.0},
]


def test_scatter3d_marks_the_selected_point() -> None:
    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", 2)

    marker = fig.data[0].marker
    assert tuple(marker.size) == (15, 15, 25, 15)
    assert tuple(marker.symbol) == ("circle", "circle", "diamond", "circle")
    assert marker.opacity == 1
    assert fig.layout.clickmode == "event"


def test_scatter3d_without_selection_uses_default_markers() -> None:
    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", None)

    marker = fig.data[0].marker
    assert tuple(marker.size) == (15, 15, 15, 15)
    assert tuple(marker.symbol) == ("circle",) * 4
    assert marker.opacity == 1


def test_scatter3d_selects_the_first_point() -> None:
    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", 0)

    marker = fig.data[0].marker
    assert tuple(marker.size) == (25, 15, 15, 15)
    assert tuple(marker.symbol) == ("diamond", "circle", "circle", "circle")


def test_scatter3d_axes_and_color_come_from_the_arguments() -> None:
    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", None)

    trace = fig.data[0]
    assert_array_equal(trace.x, [10.0, 20.0, 30.0, 40.0])
    assert_array_equal(trace.y, [100.0, 200.0, 300.0, 400.0])
    assert_array_equal(trace.z, [0.5, 1.5, 2.5, 3.5])
    assert_array_equal(trace.marker.color, [1.0, 2.0, 3.0, 4.0])


def test_scatter3d_hover_format_for_every_field() -> None:
    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", None)

    template = fig.data[0].hovertemplate
    assert template.count(":.2f") == len(CHART_FIELDS)
    for name in CHART_FIELDS:
        assert name + "=" in template


def test_scatter3d_scene_camera() -> None:
    camera = {"eye": {"x": 1.5, "y": 0.5, "z": 0.25}}

    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", None, camera)

    eye = fig.layout.scene.camera.eye
    assert (eye.x, eye.y, eye.z) == (1.5, 0.5, 0.25)


def test_scatter3d_without_scene_camera_leaves_it_unset() -> None:
    fig = qno.build_plotly_scatter3d(CHART_ROWS, CHART_FIELDS, "b", "c", "d", "a", None)

    assert fig.layout.scene.camera.to_plotly_json() == {}


# ---------------------------------------------------------------------------------------------
# build_plot, build_plot_dash, optimize_strategy
# ---------------------------------------------------------------------------------------------


def make_results() -> OptimizationResult:
    best: Iteration = {
        "args": {"x": 1, "y": 2},
        "result": {"sharpe_ratio": 1.5, "mean_return": 0.1},
        "weight": 1.5,
        "exception": None,
    }
    return {"iterations": [best], "best_iteration": best}


@pytest.mark.parametrize("notebook", [True, False])
def test_build_plot_chooses_the_viewer(monkeypatch: pytest.MonkeyPatch, notebook: bool) -> None:
    calls: list[tuple[str, OptimizationResult]] = []
    results = make_results()
    monkeypatch.setattr(qngraph, "is_notebook", lambda: notebook)
    monkeypatch.setattr(qno, "build_plot_jupyter", lambda r: calls.append(("jupyter", r)))
    monkeypatch.setattr(qno, "build_plot_dash", lambda r: calls.append(("dash", r)))

    qno.build_plot(results)

    assert calls == [("jupyter" if notebook else "dash", results)]


def test_build_plot_dash_opens_the_browser_and_runs_the_app(
    monkeypatch: pytest.MonkeyPatch, ui_calls: UiCalls
) -> None:
    monkeypatch.setenv("PORT", "8123")

    qno.build_plot_dash(make_results())

    assert len(ui_calls.apps) == 1
    assert len(ui_calls.timers) == 1
    timer = ui_calls.timers[0]
    assert timer.interval == 1
    assert timer.started
    assert ui_calls.opened == []
    timer.function()
    assert ui_calls.opened == [("http://127.0.0.1:8123", 2)]


@dataclass
class OptimizeSpy:
    result: OptimizationResult
    calls: list[tuple[TargetCallable, object, StatsToWeight, int]] = field(
        default_factory=list
    )

    def __call__(
        self,
        target_function: TargetCallable,
        argument_generator: object,
        result_to_weight: StatsToWeight,
        workers: int = 1,
    ) -> OptimizationResult:
        self.calls.append((target_function, argument_generator, result_to_weight, workers))
        return self.result


def test_optimize_strategy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = OptimizeSpy(make_results())
    monkeypatch.setattr(qno, "optimize", spy)
    data = make_data(n_assets=2, n_days=5, name="futures")
    strategy = make_strategy()
    generator = grid_generator()

    result = qno.optimize_strategy(data, strategy, generator)

    assert result is spy.result
    assert len(spy.calls) == 1
    target, got_generator, to_weight, workers = spy.calls[0]
    assert isinstance(target, qno.TargetFunction)
    assert target.data is data
    assert target.output_func is strategy
    assert target.stats_func is qno.standard_stats_function
    assert target.lookback_period == 365
    assert got_generator is generator
    assert to_weight is qno.standard_stats_to_weight
    assert workers == 1


def test_optimize_strategy_passes_explicit_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = OptimizeSpy(make_results())
    monkeypatch.setattr(qno, "optimize", spy)
    data = make_data(n_assets=2, n_days=5, name="futures")
    strategy = make_strategy()
    generator = grid_generator()
    stats = StatsSpy()

    result = qno.optimize_strategy(data, strategy, generator, stats, sharpe_weight, 17, 4)

    assert result is spy.result
    target, got_generator, to_weight, workers = spy.calls[0]
    assert isinstance(target, qno.TargetFunction)
    assert target.data is data
    assert target.output_func is strategy
    assert target.stats_func is stats
    assert target.lookback_period == 17
    assert got_generator is generator
    assert to_weight is sharpe_weight
    assert workers == 4


def test_optimize_defaults_to_one_worker(fakes: Registry) -> None:
    qno.optimize(ScoreTarget(), grid_generator(), sharpe_weight)

    assert len(fakes.processes) == 1
    assert fakes.processes[0].name == "optimize_worker#0"
    assert [q.maxsize for q in fakes.queues] == [2, 0]


def test_optimize_waits_for_more_than_small_cached_integers(fakes: Registry) -> None:
    generator = qno.full_range_args_generator(x=range(300), y=[0])

    result = qno.optimize(ScoreTarget(), generator, sharpe_weight, 1)

    assert len(result["iterations"]) == 300
    assert fakes.bars[0].updates == list(range(301))
    assert result["best_iteration"]["args"] == {"x": 299, "y": 0}
    assert result["best_iteration"]["weight"] == 2990.0


def test_prepare_data_args_in_another_order_characterization() -> None:
    """Args of a non-best iteration are matched to the best one by position, not by name."""
    best = make_iteration({"a": 1, "b": 2}, {"sharpe_ratio": 1.0})
    other = make_iteration({"b": 5, "a": 6}, {"sharpe_ratio": 0.5})

    data, _fields = qno.prepare_data_for_chart(
        {"iterations": [other, best], "best_iteration": best}
    )

    assert data[0] == {"a": 5, "b": 6, "sharpe_ratio": 0.5}
