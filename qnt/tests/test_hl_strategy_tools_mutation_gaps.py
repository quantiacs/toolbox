import datetime
import gzip
import pickle
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TypeAlias

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.data.common as qndc
import qnt.exits as qnexits
import qnt.exposure as qnexp
import qnt.forward_looking as qnfl
import qnt.neutralization as qnneu
import qnt.pd_data as qnpd
import qnt.state as qnstate
import qnt.stepper as qnstep
from qnt.tests.hl_support import FIELDS, Floats, business_days, make_data

pytestmark = pytest.mark.usefixtures("quiet_tracking")

PctRule: TypeAlias = Callable[[xr.DataArray, xr.DataArray, xr.DataArray, float], xr.DataArray]
AtrRule: TypeAlias = Callable[
    [xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, float], xr.DataArray
]


def vec(values: Sequence[float], assets: Sequence[str]) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=np.float64), dims=["asset"], coords={"asset": list(assets)}
    )


def grid(rows: Sequence[Sequence[float]], assets: Sequence[str]) -> xr.DataArray:
    return xr.DataArray(
        np.array(rows, dtype=np.float64),
        dims=["time", "asset"],
        coords={"time": business_days(len(rows)), "asset": list(assets)},
    )


def flags(array: xr.DataArray) -> list[int]:
    return [int(v) for v in array.values]


def spy_calls(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def spy(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(module, name, spy)
    return calls


# ---------------------------------------------------------------- exits


POSITIONS = (1.0, -1.0, 0.0, 0.4, -0.4)
GARBAGE_HIGH = 1e6
GARBAGE_LOW = 1e-3


@dataclass(frozen=True)
class ExitCase:
    data: xr.DataArray
    weights: xr.DataArray
    entry: xr.DataArray
    expected: list[int]
    atr_values: list[float]


def exit_case(
    entry: float, levels: tuple[float, float], atr: float, sign: int, keep_below: bool, upper: bool
) -> ExitCase:
    """Positions x last closes around both thresholds; earlier rows hold decoy values."""
    around = {entry}
    for level in levels:
        around.update({level - 1.0, level - 0.1, level, level + 0.1, level + 1.0})
    closes = sorted(around)
    scenarios = [(p, c, entry) for p in POSITIONS for c in closes]
    scenarios += [(p, entry, float("nan")) for p in POSITIONS]
    names = [f"x{i}" for i in range(len(scenarios))]
    pos = np.array([s[0] for s in scenarios])
    last_close = np.array([s[1] for s in scenarios])
    weights = np.stack([-pos, -2.0 * pos, pos])
    close_rows = np.stack(
        [np.full_like(pos, GARBAGE_HIGH), np.full_like(pos, GARBAGE_LOW), last_close]
    )
    open_rows = np.stack(
        [np.full_like(pos, GARBAGE_LOW), np.full_like(pos, GARBAGE_HIGH), last_close + 7]
    )
    layers = np.stack([open_rows, close_rows], axis=1)
    data = xr.DataArray(
        layers,
        dims=["time", "field", "asset"],
        coords={"time": business_days(3), "field": ["open", "close"], "asset": names},
    )
    threshold = levels[1] if upper else levels[0]
    expected: list[int] = []
    for p, c, e in scenarios:
        if np.isnan(e) or p * sign <= 0:
            expected.append(1)
        else:
            expected.append(int(c < threshold if keep_below else c > threshold))
    return ExitCase(
        data,
        xr.DataArray(
            weights, dims=["time", "asset"], coords={"time": business_days(3), "asset": names}
        ),
        vec([s[2] for s in scenarios], names),
        expected,
        [atr] * len(names),
    )


PCT_RULES = [
    pytest.param(qnexits.take_profit_long_percentage, 1, True, True, id="take_profit_long"),
    pytest.param(qnexits.take_profit_short_percentage, -1, False, False, id="take_profit_short"),
    pytest.param(qnexits.stop_loss_long_percentage, 1, False, False, id="stop_loss_long"),
    pytest.param(qnexits.stop_loss_short_percentage, -1, True, True, id="stop_loss_short"),
]
ATR_RULES = [
    pytest.param(qnexits.take_profit_long_atr, 1, True, True, id="take_profit_long"),
    pytest.param(qnexits.take_profit_short_atr, -1, False, False, id="take_profit_short"),
    pytest.param(qnexits.stop_loss_long_atr, 1, False, False, id="stop_loss_long"),
    pytest.param(qnexits.stop_loss_short_atr, -1, True, True, id="stop_loss_short"),
]


@pytest.mark.parametrize("entry_price", [80.0, 40.0])
@pytest.mark.parametrize("rule, sign, keep_below, upper", PCT_RULES)
def test_percentage_exits_match_formula_at_and_around_thresholds(
    rule: PctRule, sign: int, keep_below: bool, upper: bool, entry_price: float
) -> None:
    percent = {80.0: 25.0, 40.0: 50.0}[entry_price]
    levels = (entry_price * (100.0 - percent) / 100.0, entry_price * (100.0 + percent) / 100.0)
    case = exit_case(entry_price, levels, 0.0, sign, keep_below, upper)
    result = rule(case.data, case.weights, case.entry, percent)
    assert flags(result) == case.expected
    assert 0 in case.expected
    assert list(result.coords["asset"].values) == list(case.weights.coords["asset"].values)


@pytest.mark.parametrize("config", [(80.0, 4.0, 2.5), (40.0, 8.0, 0.5)])
@pytest.mark.parametrize("rule, sign, keep_below, upper", ATR_RULES)
def test_atr_exits_match_formula_at_and_around_thresholds(
    rule: AtrRule, sign: int, keep_below: bool, upper: bool, config: tuple[float, float, float]
) -> None:
    entry_price, atr, amount = config
    levels = (entry_price - amount * atr, entry_price + amount * atr)
    case = exit_case(entry_price, levels, atr, sign, keep_below, upper)
    names = list(case.weights.coords["asset"].values)
    result = rule(case.data, case.weights, case.entry, vec(case.atr_values, names), amount)
    assert flags(result) == case.expected
    assert 0 in case.expected


def test_update_open_price_handles_flat_flip_and_unchanged_positions() -> None:
    names = [f"a{i}" for i in range(10)]
    # (weights two steps ago, previous, current, last open, stored open price)
    rows = [
        (1.0, 1.0, 0.0, 11.0, 50.0),
        (1.0, 0.0, 1.0, 12.0, 51.0),
        (0.0, 0.0, 0.0, 13.0, 52.0),
        (1.0, 1.0, -1.0, 14.0, 53.0),
        (1.0, 1.0, 0.5, 15.0, 54.0),
        (-1.0, -1.0, -0.25, 16.0, 55.0),
        (-1.0, 1.0, 1.0, 17.0, 56.0),
        (1.0, -1.0, -1.0, 18.0, 57.0),
        (1.0, 1.0, 1.0, 19.0, float("nan")),
        (0.0, -0.5, 0.0, 20.0, 59.0),
    ]
    expected = [np.nan, 12.0, np.nan, 14.0, 54.0, 55.0, 56.0, 57.0, np.nan, np.nan]
    last_open = np.array([r[3] for r in rows])
    opens = np.stack([np.full(10, GARBAGE_HIGH), np.full(10, GARBAGE_LOW), last_open])
    closes = np.stack([np.full(10, 901.0), np.full(10, 902.0), np.full(10, 903.0)])
    data = xr.DataArray(
        np.stack([opens, closes], axis=1),
        dims=["time", "field", "asset"],
        coords={"time": business_days(3), "field": ["open", "close"], "asset": names},
    )
    weights = grid([[r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]], names)
    state = {"open_price": vec([r[4] for r in rows], names)}
    result = qnexits.update_open_price(data, weights, state)
    assert_allclose(result.values, expected, equal_nan=True)
    assert state["open_price"] is result


HOLD_SEQUENCES: dict[str, list[float]] = {
    "steady_long": [1.0] * 10,
    "steady_short": [-1.0] * 10,
    "gaps": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    "divisible_flips": [0.5, -0.25, -0.25, 0.5, 0.5, -0.125, 0.25, -1.0, 1.0, 0.5],
    "halving_flips": [1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 1.0, -0.5],
    "tiny_flips": [1e-200, -1e-200] * 5,
    "irregular": [0.3, 0.3, -0.7, -0.7, 0.7, 0.0, 0.2, -0.3, -0.3, 0.9],
    "fractional_flips": [-0.3, 0.7, -0.3, 0.7, 0.7, -0.4, -0.4, 0.4, 0.0, 0.0],
    "flat": [0.0] * 10,
}


def reference_hold(
    stored: float, current: float, previous: float, max_period: float, side: int
) -> tuple[float, int]:
    on_side = current * side > 0
    grown = 0.0 if stored >= max_period else stored + 1.0
    held = grown if on_side else stored
    if current == 0:
        held = 0.0
    if current * previous < 0:
        held = 1.0
    keep = int(held < max_period) if on_side else 1
    return held, keep


@pytest.mark.parametrize("max_period", [3, 1, 2.5, 0])
@pytest.mark.parametrize("side", [1, -1])
def test_max_hold_follows_state_machine_for_every_asset(side: int, max_period: float) -> None:
    names = list(HOLD_SEQUENCES)
    series = np.array([HOLD_SEQUENCES[n] for n in names]).T
    stored = {"steady_long": 2.0, "steady_short": 3.0, "gaps": 5.0}
    state = {"holding_time": vec(list(stored.values()), list(stored))}
    held = {n: stored.get(n, 0.0) for n in names}
    rule = qnexits.max_hold_long if side == 1 else qnexits.max_hold_short
    for t in range(2, series.shape[0]):
        keep = rule(grid(series[t - 2 : t + 1].tolist(), names), state, max_period)
        steps = [
            reference_hold(held[n], series[t, i], series[t - 1, i], max_period, side)
            for i, n in enumerate(names)
        ]
        held = {n: steps[i][0] for i, n in enumerate(names)}
        assert flags(keep) == [s[1] for s in steps]
        assert_allclose(state["holding_time"].values, [s[0] for s in steps])
        assert list(state["holding_time"].coords["asset"].values) == names


def test_max_hold_exits_exactly_when_period_is_reached() -> None:
    long_only = grid([[1.0]] * 3, ["a"])
    state = {"holding_time": vec([0.0], ["a"])}
    kept = [flags(qnexits.max_hold_long(long_only, state, 2))[0] for _ in range(5)]
    assert kept == [1, 0, 1, 1, 0]
    reset = {"holding_time": vec([2.0], ["a"])}
    assert flags(qnexits.max_hold_long(long_only, reset, 2)) == [1]
    assert_allclose(reset["holding_time"].values, [0.0])
    over = {"holding_time": vec([7.0], ["a"])}
    assert flags(qnexits.max_hold_long(long_only, over, 2)) == [1]
    assert_allclose(over["holding_time"].values, [0.0])


# ---------------------------------------------------------------- exposure


def reference_mix(
    primary: Sequence[float], secondary: Sequence[float], max_weight: float
) -> Floats:
    first = np.array(primary)
    second = np.array(secondary)
    k = (
        np.abs(first).sum()
        * (np.abs(first).max() / np.abs(first).sum() - max_weight)
        / (np.abs(second).sum() * (max_weight - np.abs(second).max() / np.abs(second).sum()))
    )
    mix = first + second * max(k, 0.0)
    return mix / max(np.abs(mix).sum(), 1.0)


@pytest.mark.parametrize(
    ("primary", "secondary"),
    [
        ([0.7, 0.3, 0.0, 0.0, 0.0], [0.2] * 5),
        ([0.1] * 10, [0.2] * 5),
        ([0.5, 0.5], [0.9, 0.1]),
        ([0.5, 0.5, 0.0, 0.0], [0.25] * 4),
    ],
)
@pytest.mark.parametrize("max_weight", [0.049, 0.3])
def test_mix_weights_default_equals_explicit_limit_and_reference(
    primary: list[float], secondary: list[float], max_weight: float
) -> None:
    width = max(len(primary), len(secondary))
    names = [f"a{i}" for i in range(width)]
    first = grid([primary + [0.0] * (width - len(primary))], names)
    second = grid([secondary + [0.0] * (width - len(secondary))], names)
    padded_first = primary + [0.0] * (width - len(primary))
    padded_second = secondary + [0.0] * (width - len(secondary))
    mixed = qnexp.mix_weights(first, second, max_weight)
    assert_allclose(mixed.values[0], reference_mix(padded_first, padded_second, max_weight))
    if max_weight == 0.049:
        assert_array_equal(qnexp.mix_weights(first, second).values, mixed.values)


def test_drop_bad_days_default_limit_and_exact_boundary() -> None:
    names = [f"a{i}" for i in range(21)]
    rows = [[1.0 / 21] * 21, [1.0 / 20] * 20 + [0.0], [0.5] + [0.0] * 20]
    weights = grid(rows, names)
    default = qnexp.drop_bad_days(weights).values
    assert_array_equal(default, qnexp.drop_bad_days(weights, 0.049).values)
    assert_allclose(default[0], 1.0 / 21)
    assert_array_equal(default[1:], 0.0)

    quarters = grid([[0.25] * 4, [0.25] * 4], list("abcd"))
    assert_array_equal(qnexp.drop_bad_days(quarters, 0.25).values, 0.0)
    assert_array_equal(qnexp.drop_bad_days(quarters, 0.2500001).values, 0.25)
    assert_array_equal(qnexp.drop_bad_days(quarters, 0.2499999).values, 0.0)


def test_cut_big_positions_default_limit_and_boundary() -> None:
    weights = grid([[0.049, 0.05, -0.05, -0.049, np.nan, 0.01, 0.0, 2.0]], list("abcdefgh"))
    expected = [[0.049, 0.049, -0.049, -0.049, np.nan, 0.01, 0.0, 0.049]]
    default = qnexp.cut_big_positions(weights).values
    assert_allclose(default, expected, equal_nan=True)
    assert_array_equal(default, qnexp.cut_big_positions(weights, 0.049).values, strict=False)
    wide = qnexp.cut_big_positions(weights, 0.5).values
    assert_allclose(wide, [[0.049, 0.05, -0.05, -0.049, np.nan, 0.01, 0.0, 0.5]], equal_nan=True)


def test_normalize_by_max_exposure_default_limit_and_boundary() -> None:
    weights = grid([[0.5, -0.25], [0.1, 0.05], [0.1000000001, 0.0], [-0.3, 0.1]], ["a", "b"])
    default = qnexp.normalize_by_max_exposure(weights).values
    assert_allclose(default[0], [0.1, -0.05])
    assert_array_equal(default[1], [0.1, 0.05])
    assert_allclose(default[2], [0.1, 0.0], atol=1e-12)
    assert_allclose(default[3], [-0.1, 0.1 / 3])
    assert_array_equal(default, qnexp.normalize_by_max_exposure(weights, 0.1).values)
    loose = qnexp.normalize_by_max_exposure(weights, 0.25).values
    assert_allclose(loose[0], [0.25, -0.125])
    assert_array_equal(loose[1], [0.1, 0.05])


# ---------------------------------------------------------------- neutralization


def test_neutralize_unknown_group_message_and_default_group() -> None:
    weights = grid([[0.4, 0.2, -0.1, 0.3]], list("wxyz"))
    with pytest.raises(Exception, match="No such group") as error:
        qnneu.neutralize(weights, [], "country")
    assert (
        str(error.value) == "No such group 'country'. Use 'market', 'sector' or 'industry' instead."
    )
    default = qnneu.neutralize(weights, [])
    assert_allclose(default.values, [[0.2, 0.0, -0.3, 0.1]])
    assert_array_equal(default.values, qnneu.neutralize(weights, [], "market").values)
    assert_allclose(weights.values, [[0.4, 0.2, -0.1, 0.3]])


def test_neutralize_sector_with_three_groups_and_missing_assets() -> None:
    weights = grid([[0.6, 0.2, 0.1, 0.5, -0.4, 0.9]], list("abcdef"))
    catalog = [
        {"id": "a", "sector": "s1"},
        {"id": "b", "sector": "s1"},
        {"id": "c", "sector": "s2"},
        {"id": "d", "sector": "s2"},
        {"id": "e", "sector": "s3"},
        {"id": "f", "sector": "s3"},
    ]
    result = qnneu.neutralize(weights, catalog, "sector").values
    assert_allclose(result, [[0.2, -0.2, -0.2, 0.2, -0.65, 0.65]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        partial = qnneu.neutralize(weights, catalog[:3], "sector").values
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert str(caught[0].message) == (
        "Some stocks has no specification. "
        "Perhaps you are using illiquid instruments or outdated assets data."
    )
    assert Path(caught[0].filename).name == "neutralization.py"
    mean_unknown = (0.5 - 0.4 + 0.9) / 3
    assert_allclose(
        partial,
        [[0.2, -0.2, 0.0, 0.5 - mean_unknown, -0.4 - mean_unknown, 0.9 - mean_unknown]],
    )


def test_neutralize_industry_uses_industry_key_and_ignores_unlisted_assets() -> None:
    weights = grid([[0.1, 0.5, 0.9]], list("abc"))
    catalog = [
        {"id": "a", "sector": "x", "industry": "i1"},
        {"id": "b", "sector": "x", "industry": "i2"},
        {"id": "c", "sector": "y", "industry": "i1"},
        {"id": "zzz", "sector": "y", "industry": "i2"},
    ]
    assert_allclose(
        qnneu.neutralize(weights, catalog, "industry").values, [[-0.4, 0.0, 0.4]]
    )
    assert_allclose(qnneu.neutralize(weights, catalog, "sector").values, [[-0.2, 0.2, 0.0]])


# ---------------------------------------------------------------- stepper


def stepper_data(n_days: int, n_assets: int = 3) -> xr.DataArray:
    return make_data(n_assets=n_assets, n_days=n_days, seed=4).transpose("field", "time", "asset")


def empty_non_liquid(data: xr.DataArray, history: xr.DataArray) -> xr.DataArray:
    del data, history
    return xr.DataArray(
        np.zeros((0, 1)), dims=["time", "asset"], coords={"time": [], "asset": ["a"]}
    )


def constant_step(history: xr.DataArray) -> xr.DataArray:
    return vec([0.1] * history.sizes["asset"], list(history.coords["asset"].values))


def with_times(data: xr.DataArray, days: Sequence[int]) -> xr.DataArray:
    index = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=d) for d in days])
    return data.assign_coords(time=index)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("init_length", [0, 1, 3])
def test_stepper_pieces_and_output_axis(
    monkeypatch: pytest.MonkeyPatch, backward: bool, init_length: int
) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    n_days = 9
    data = stepper_data(n_days)
    if backward:
        data = data.sortby("time", ascending=False)
    times = data.coords["time"].values
    point_count = n_days - init_length
    pieces: list[list[np.datetime64]] = []
    init_calls: list[list[np.datetime64] | None] = []

    def init(history: xr.DataArray | None) -> None:
        init_calls.append(None if history is None else list(history.coords["time"].values))

    def step(history: xr.DataArray) -> xr.DataArray:
        pieces.append(list(history.coords["time"].values))
        return constant_step(history)

    strategy = qnstep.StrategyAdapter(init=init, step=step, init_data_length=init_length)
    result = qnstep.test_strategy(data, strategy)
    if backward:
        expected_pieces = [list(times[point_count - 1 - s :]) for s in range(point_count)]
        expected_axis = list(times[:point_count][::-1])
    else:
        expected_pieces = [list(times[: init_length + s + 1]) for s in range(point_count)]
        expected_axis = list(times[init_length : init_length + point_count])
    assert pieces == expected_pieces
    assert init_calls == [None if init_length == 0 else list(times[:point_count])]
    assert list(result.coords["time"].values) == expected_axis
    assert result.shape == (point_count, 3)
    assert_allclose(result.values, 0.1)


@pytest.mark.parametrize(
    ("days", "ascending"),
    [
        pytest.param([150, 300, 118, 200], True, id="middle-values-above-and-below"),
        pytest.param([150, 118, 300, 200], True, id="second-below-first"),
        pytest.param([3, 2, 9, 4], True, id="second-below-first-short"),
        pytest.param([150, 400, 120, 90], False, id="descending-overall"),
        pytest.param([200, 300, 118, 200], False, id="first-equals-last"),
    ],
)
def test_stepper_direction_comes_from_first_and_last_time_only(
    monkeypatch: pytest.MonkeyPatch, days: list[int], ascending: bool
) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    data = with_times(stepper_data(len(days)), days)
    times = data.coords["time"].values
    pieces: list[list[np.datetime64]] = []

    def step(history: xr.DataArray) -> xr.DataArray:
        pieces.append(list(history.coords["time"].values))
        return constant_step(history)

    qnstep.test_strategy(data, qnstep.StrategyAdapter(step=step))
    count = len(days)
    if ascending:
        expected = [list(times[: s + 1]) for s in range(count)]
    else:
        expected = [list(times[count - 1 - s :]) for s in range(count)]
    assert pieces == expected


def test_stepper_negative_init_length_characterization(monkeypatch: pytest.MonkeyPatch) -> None:
    """init_data_length=-2 -> init(None) is called, then the output axis does not fit: ValueError."""
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    init_calls: list[bool] = []

    def init(history: xr.DataArray | None) -> None:
        init_calls.append(history is None)

    strategy = qnstep.StrategyAdapter(init=init, step=constant_step, init_data_length=-2)
    with pytest.raises(ValueError, match="conflicting sizes"):
        qnstep.test_strategy(stepper_data(5), strategy)
    assert init_calls == [True]


def test_stepper_defaults_without_optional_strategy_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    data = stepper_data(6)
    assert qnstep.SimpleStrategy.init_data_length == 0
    assert qnstep.calc_step_by_step is qnstep.test_strategy

    bare = qnstep.test_strategy(data, step=constant_step)
    assert bare.shape == (6, 3)
    assert_array_equal(bare.coords["time"].values, data.coords["time"].values)

    init_calls: list[bool] = []

    def init(history: xr.DataArray | None) -> None:
        init_calls.append(history is None)

    only_init = qnstep.StrategyAdapter(init=init, step=constant_step)
    assert qnstep.test_strategy(data, only_init).shape == (6, 3)
    assert init_calls == [True]

    sized = qnstep.test_strategy(data, step=constant_step, init_data_length=2)
    assert sized.shape == (4, 3)
    assert_array_equal(sized.coords["time"].values, data.coords["time"].values[2:])

    ignored = qnstep.test_strategy(data, only_init, init_data_length=4)
    assert ignored.shape == (6, 3)


@pytest.mark.xfail(strict=True, reason="TypeError: stepper subscripts the method data.sel[...]")
def test_simple_strategy_weights_only_assets_with_first_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    data = stepper_data(4, n_assets=4)
    data.loc[{"field": "open", "time": data.time[0], "asset": "NYS:A1"}] = np.nan
    data.loc[{"field": "open", "time": data.time[1], "asset": "NYS:A2"}] = np.nan
    weights = qnstep.SimpleStrategy().step(data)
    assert list(weights.coords["asset"].values) == ["NYS:A0", "NYS:A2", "NYS:A3"]
    assert_allclose(weights.values, [1 / 3] * 3)

    history = qnstep.test_strategy(data, qnstep.SimpleStrategy())
    assert history.shape == (4, 4)
    expected = [1 / 3, 0.0, 1 / 3, 1 / 3]
    assert_allclose(history.values[0], expected)
    assert_allclose(history.values[3], expected)


def test_stepper_normalizes_each_day_portfolio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    data = stepper_data(6, n_assets=4)
    names = list(data.coords["asset"].values)
    stamp = data.coords["time"].values
    stacked = xr.DataArray(
        [[0.9, 0.9, 0.9, 0.9], [0.1, 0.2, 0.3, 0.2]],
        dims=["time", "asset"],
        coords={"time": [stamp[-1], stamp[0]], "asset": names},
    )
    outputs = [
        vec([np.nan, np.inf, 0.6, 0.6], names),
        vec([0.5, 0.5, 0.0, 0.0], names),
        vec([0.3, 0.2, 0.0, 0.0], names),
        vec([0.8, 0.6, 0.0, 0.0], names),
        vec([1.5, -0.5], [names[1], names[3]]),
        stacked,
    ]
    counter = {"n": 0}

    def step(history: xr.DataArray) -> xr.DataArray:
        del history
        counter["n"] += 1
        return outputs[counter["n"] - 1]

    result = qnstep.test_strategy(data, qnstep.StrategyAdapter(step=step)).values
    expected = [
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0, 0.0],
        [0.3, 0.2, 0.0, 0.0],
        [0.8 / 1.4, 0.6 / 1.4, 0.0, 0.0],
        [0.0, 0.75, 0.0, -0.25],
        [0.9 / 3.6, 0.9 / 3.6, 0.9 / 3.6, 0.9 / 3.6],
    ]
    assert_allclose(result, expected, rtol=1e-12)


def test_stepper_progress_and_completion_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    messages = spy_calls(monkeypatch, qnstep, "log_info")
    ticks = iter([100.0, 101.0, 106.0, 106.25, 107.0, 111.0, 112.5])
    monkeypatch.setattr(qnstep, "time", SimpleNamespace(time=lambda: next(ticks)))
    qnstep.test_strategy(stepper_data(3), qnstep.StrategyAdapter(step=constant_step))
    assert messages == [
        ("Testing started...",),
        ("Testing progress: 2/3 7s",),
        ("Testing complete 12.5s",),
    ]
    assert next(ticks, None) is None


def test_stepper_progress_log_uses_elapsed_differences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qnstep, "calc_non_liquid", empty_non_liquid)
    messages = spy_calls(monkeypatch, qnstep, "log_info")
    ticks = iter([10.0, 20.0, 44.0, 45.0, 50.0, 49.9, 52.5])
    monkeypatch.setattr(qnstep, "time", SimpleNamespace(time=lambda: next(ticks)))
    qnstep.test_strategy(stepper_data(3), qnstep.StrategyAdapter(step=constant_step))
    assert messages == [
        ("Testing started...",),
        ("Testing progress: 1/3 35s",),
        ("Testing complete 42.5s",),
    ]


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_stepper_warns_about_non_liquid_trades(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    messages = spy_calls(monkeypatch, qnstep, "log_info")
    seen: list[tuple[xr.DataArray, xr.DataArray]] = []

    def fake(data: xr.DataArray, history: xr.DataArray) -> xr.DataArray:
        seen.append((data, history))
        return xr.DataArray(
            np.ones((rows, 1)),
            dims=["time", "asset"],
            coords={"time": business_days(rows), "asset": ["a"]},
        )

    monkeypatch.setattr(qnstep, "calc_non_liquid", fake)
    data = stepper_data(4)
    history = qnstep.test_strategy(data, qnstep.StrategyAdapter(step=constant_step))
    assert len(seen) == 1
    assert seen[0][0] is data
    assert seen[0][1] is history
    warned = ("WARNING: Strategy trades non-liquid assets.",) in messages
    assert warned is (rows > 0)


@pytest.mark.xfail(strict=True, reason="TypeError: stepper subscripts the method data.sel[...]")
def test_stepper_with_real_liquidity_check_warns_only_for_illiquid_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = spy_calls(monkeypatch, qnstep, "log_info")
    data = stepper_data(10, n_assets=3)
    qnstep.test_strategy(data, qnstep.SimpleStrategy())
    assert ("WARNING: Strategy trades non-liquid assets.",) in messages

    messages.clear()

    def avoid_illiquid(history: xr.DataArray) -> xr.DataArray:
        return vec([0.5, 0.5], ["NYS:A1", "NYS:A2"]).reindex(
            asset=list(history.coords["asset"].values), fill_value=0.0
        )

    qnstep.test_strategy(data, qnstep.StrategyAdapter(step=avoid_illiquid))
    assert ("WARNING: Strategy trades non-liquid assets.",) not in messages


# ---------------------------------------------------------------- forward_looking


def fl_output(
    rows: Sequence[Sequence[float]], names: Sequence[str], days: Sequence[int]
) -> xr.DataArray:
    index = pd.DatetimeIndex([pd.Timestamp("2021-03-01") + pd.Timedelta(days=d) for d in days])
    return xr.DataArray(
        np.array(rows, dtype=np.float64),
        dims=["time", "asset"],
        coords={"time": index, "asset": list(names)},
    )


def test_forward_looking_constants() -> None:
    assert qnfl.FORWARD_LOOKING_TEST_OFFSET == 182
    assert qnfl.FORWARD_LOOKING_TEST_DELTA == 1e-7


@pytest.mark.parametrize(
    ("shift", "found", "message"),
    [
        (0.0, False, "Ok. There is no forward looking."),
        (5e-8, False, "Ok. There is no forward looking."),
        (1e-7, False, "Ok. There is no forward looking."),
        (2e-7, True, "WARNING: This strategy uses forward looking! Delta = 2e-07"),
        (3e-7, True, "WARNING: This strategy uses forward looking! Delta = 3e-07"),
    ],
)
def test_check_forward_looking_threshold(
    monkeypatch: pytest.MonkeyPatch, shift: float, found: bool, message: str
) -> None:
    messages = spy_calls(monkeypatch, qnfl, "log_info")
    cropped = fl_output([[0.0, 0.5], [0.0, 0.5]], ["a", "b"], [0, 1])
    whole = fl_output([[0.0, 0.5], [shift, 0.5]], ["a", "b"], [0, 1])
    assert qnfl.check_forward_looking(cropped, whole) is found
    assert messages == [(message,)]


@pytest.mark.parametrize("descending", [False, True])
def test_check_forward_looking_compares_only_common_period(descending: bool) -> None:
    short = fl_output([[0.2, 0.3], [0.1, 0.4], [0.5, 0.5]], ["a", "b"], [0, 1, 2])
    long = fl_output(
        [[0.2, 0.3], [0.1, 0.4], [0.5, 0.5], [0.9, 0.0], [0.0, 0.9]], ["a", "b"], [0, 1, 2, 3, 4]
    )
    if descending:
        long = long.isel(time=slice(None, None, -1))
    assert qnfl.check_forward_looking(short, long) is False
    assert qnfl.check_forward_looking(long, short) is False

    changed = fl_output([[0.2, 0.3], [0.1, 0.4], [0.5, 0.0]], ["a", "b"], [0, 1, 2])
    assert qnfl.check_forward_looking(changed, long) is True


def test_check_forward_looking_fills_missing_assets_with_zero() -> None:
    both = fl_output([[0.2, 0.0], [0.1, 0.0]], ["a", "b"], [0, 1])
    only_a = fl_output([[0.2], [0.1]], ["a"], [0, 1])
    assert qnfl.check_forward_looking(only_a, both) is False
    assert qnfl.check_forward_looking(both, only_a) is False

    real_b = fl_output([[0.2, 0.0], [0.1, 0.4]], ["a", "b"], [0, 1])
    assert qnfl.check_forward_looking(only_a, real_b) is True
    assert qnfl.check_forward_looking(real_b, only_a) is True


def test_calc_output_crops_data_by_offset_from_last_date(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = spy_calls(monkeypatch, qnfl, "log_info")
    base = make_data(n_assets=2, n_days=400, seed=2)
    daily = pd.date_range("2020-01-01", periods=400, freq="D", name="time")
    data = base.assign_coords(time=daily)
    data.loc[{"asset": "NYS:A1", "time": daily[:300]}] = np.nan
    data.loc[{"time": daily[10]}] = np.nan
    last_day = daily[-1]
    cutoff = last_day - pd.Timedelta(days=182)
    seen: list[xr.DataArray] = []

    def strategy(history: xr.DataArray) -> xr.DataArray:
        seen.append(history)
        return history.sel(field="close")

    whole = qnfl.calc_output_and_check_forward_looking(data, strategy)
    cropped = seen[0]
    assert seen[1] is data
    assert whole.shape == (400, 2)
    assert pd.Timestamp(cropped.coords["time"].values[-1]) == cutoff
    assert pd.Timestamp(cropped.coords["time"].values[0]) == daily[0]
    assert daily[10] not in pd.DatetimeIndex(cropped.coords["time"].values)
    assert cropped.sizes["time"] == (cutoff - daily[0]).days + 1 - 1
    assert list(cropped.coords["asset"].values) == ["NYS:A0"]
    assert messages[:2] == [
        ("Computing of the cropped output...",),
        ("Computing of the whole output...",),
    ]


def test_calc_output_descending_time_characterization() -> None:
    """Descending time -> else branch slice(last_date, None) -> expected: crop from the other side; actual: KeyError."""
    daily = pd.date_range("2020-01-01", periods=400, freq="D", name="time")
    data = make_data(n_assets=2, n_days=400, seed=2).assign_coords(time=daily)
    data = data.sortby("time", ascending=False)

    def strategy(history: xr.DataArray) -> xr.DataArray:
        return history.sel(field="close")

    with pytest.raises(KeyError):
        qnfl.calc_output_and_check_forward_looking(data, strategy)


def test_calc_output_single_date_uses_descending_branch_and_keeps_the_row() -> None:
    data = make_data(n_assets=2, n_days=1, seed=2)
    seen: list[xr.DataArray] = []

    def strategy(history: xr.DataArray) -> xr.DataArray:
        seen.append(history)
        return fl_output([[0.1, 0.2]], ["NYS:A0", "NYS:A1"], [0])

    qnfl.calc_output_and_check_forward_looking(data, strategy)
    assert seen[0].sizes["time"] == 1
    assert seen[1] is data


def test_load_data_variant_limits_dates_only_during_second_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = spy_calls(monkeypatch, qnfl, "log_info")
    monkeypatch.setattr(qndc, "MAX_DATE_LIMIT", datetime.date(2001, 1, 1))
    monkeypatch.setattr(
        qndc, "MAX_DATETIME_LIMIT", datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)
    )
    observed: list[tuple[object, object]] = []
    values = [[0.1, 0.2], [0.1, 0.2]]

    def strategy() -> xr.DataArray:
        observed.append((qndc.MAX_DATE_LIMIT, qndc.MAX_DATETIME_LIMIT))
        return fl_output(values, ["a", "b"], [0, 1]) * (1 + len(observed))

    before = pd.Timestamp.today().date() - datetime.timedelta(days=182)
    result = qnfl.load_data_calc_output_and_check_forward_looking(strategy)
    after = pd.Timestamp.today().date() - datetime.timedelta(days=182)
    assert observed[0] == (None, None)
    limit_date, limit_time = observed[1]
    assert limit_date in {before, after}
    assert isinstance(limit_date, datetime.date)
    assert limit_time == datetime.datetime.combine(limit_date, datetime.time.min)
    assert qndc.MAX_DATE_LIMIT is None
    assert qndc.MAX_DATETIME_LIMIT is None
    assert_allclose(result.values, np.array(values) * 2)
    assert messages[0] == ("Computing of the whole output...",)
    assert messages[1] == ("Computing of the cropped output...",)
    assert str(messages[2][0]).startswith("WARNING: This strategy uses forward looking! Delta = ")


# ---------------------------------------------------------------- state


def test_state_write_and_read_use_environment_paths(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    infos = spy_calls(monkeypatch, qnstate, "log_info")
    out_path = sandbox / "custom_out.gz"
    in_path = sandbox / "custom_in.gz"
    monkeypatch.setenv("OUT_STATE_PATH", str(out_path))
    monkeypatch.setenv("IN_STATE_PATH", str(in_path))
    qnstate.write({"n": 1})
    assert gzip.decompress(out_path.read_bytes()) == pickle.dumps({"n": 1})
    assert infos == [("State saved.",)]
    assert not (sandbox / "state.out.pickle.gz").exists()

    in_path.write_bytes(gzip.compress(pickle.dumps([4, 5])))
    assert qnstate.read() == [4, 5]
    assert infos[-1] == ("State loaded.",)
    assert len(infos) == 2


def test_state_default_file_names_and_failure_log(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    infos = spy_calls(monkeypatch, qnstate, "log_info")
    errors = spy_calls(monkeypatch, qnstate, "log_err")
    monkeypatch.delenv("OUT_STATE_PATH")
    monkeypatch.delenv("IN_STATE_PATH")
    qnstate.write("saved")
    default_out = sandbox / "state.out.pickle.gz"
    assert gzip.decompress(default_out.read_bytes()) == pickle.dumps("saved")
    (sandbox / "state.in.pickle.gz").write_bytes(gzip.compress(pickle.dumps(("x", 2))))
    assert qnstate.read() == ("x", 2)
    assert ("State saved.",) in infos
    assert ("State loaded.",) in infos

    assert qnstate.read(str(sandbox / "absent.gz")) is None
    assert len(errors) == 1
    assert errors[0][0] == "Can't load state."
    assert isinstance(errors[0][1], FileNotFoundError)
    assert len(errors[0]) == 2


# ---------------------------------------------------------------- pd_data


def test_pd_write_output_passes_time_by_asset_array(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[xr.DataArray] = []
    monkeypatch.setattr(qnpd, "origin_write_output", captured.append)
    index = pd.DatetimeIndex(business_days(3).to_numpy())
    frame = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [-0.1, 0.0, 0.4]}, index=index)
    assert frame.index.name is None
    qnpd.write_output(frame)
    assert frame.index.name == "time"
    assert len(captured) == 1
    array = captured[0]
    assert array.dims == ("asset", "time")
    assert list(array.coords["asset"].values) == ["A", "B"]
    assert_array_equal(array.coords["time"].values, index.to_numpy())
    assert_allclose(array.values, [[0.1, 0.2, 0.3], [-0.1, 0.0, 0.4]])


@pytest.mark.parametrize("explicit", [False, True])
def test_pd_load_data_forwards_arguments_and_builds_frames(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    calls: list[tuple[object, ...]] = []
    source = make_data(n_assets=2, n_days=5, seed=1).transpose("field", "time", "asset")

    def fake(
        assets: list[str] | None, min_date: str, max_date: str | None, dims: tuple[str, ...]
    ) -> xr.DataArray:
        calls.append((assets, min_date, max_date, dims))
        return source

    monkeypatch.setattr(qnpd, "origin_load_data", fake)
    if explicit:
        frames = qnpd.load_data(["NYS:A0"], "2010-05-06", "2011-01-01")
        assert calls == [(["NYS:A0"], "2010-05-06", "2011-01-01", ("field", "time", "asset"))]
    else:
        frames = qnpd.load_data(["NYS:A0"])
        assert calls == [(["NYS:A0"], "2007-01-01", None, ("field", "time", "asset"))]
    assert sorted(frames) == sorted(FIELDS)
    for field, frame in frames.items():
        assert list(frame.columns) == ["NYS:A0", "NYS:A1"]
        assert_array_equal(frame.index.to_numpy(), source.coords["time"].values)
        assert_allclose(frame.to_numpy(), source.sel(field=field).values)
