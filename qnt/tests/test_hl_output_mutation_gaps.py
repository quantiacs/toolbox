import datetime
import gzip
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from pandas.errors import InvalidIndexError

import qnt.data.common as common
import qnt.exposure as qne
import qnt.log as qlog
import qnt.output as qnout
import qnt.stats as qns
from qnt.tests.hl_support import FIELDS, Floats, business_days

pytestmark = pytest.mark.usefixtures("quiet_tracking")

ASSETS = ("A", "B", "C")
GOOD_ROW = [0.5, 0.3, 0.2]
CRYPTODAILY = "".join(("crypto", "daily"))
CRYPTO = "".join(("cry", "pto"))
STOCKS = "".join(("sto", "cks"))
KIND_MISMATCH = "WARNING! The kind of the data and the output are different."
CLEANED_WITH_DATA_KIND = "The output will be cleaned with the data kind."
NOT_ENOUGH_DATA = "WARNING! There are not enough points in the data"
SLIPPAGE_WARNING = "WARNING! There are not enough points in the data for the slippage calculation."
NOT_ENOUGH_OUTPUT = "WARNING! There are not enough points in the output."


class LogSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> None:
        self.calls.append(args)

    @property
    def texts(self) -> list[str]:
        return [" ".join(str(arg) for arg in call) for call in self.calls]


@dataclass
class Logs:
    info: LogSpy
    err: LogSpy


class Recorder:
    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0] if self.results else None


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> Logs:
    spies = Logs(LogSpy(), LogSpy())
    monkeypatch.setattr(qnout, "log_info", spies.info)
    monkeypatch.setattr(qnout, "log_err", spies.err)
    return spies


def spy(monkeypatch: pytest.MonkeyPatch, target: object, name: str, *results: object) -> Recorder:
    recorder = Recorder(*results)
    monkeypatch.setattr(target, name, recorder)
    return recorder


def build_data(
    name: Hashable,
    n_days: int = 6,
    assets: Sequence[str] = ASSETS,
    close: Floats | None = None,
    liquid: Floats | None = None,
    with_liquid: bool = True,
) -> xr.DataArray:
    shape = (n_days, len(assets))
    price = np.full(shape, 100.0) if close is None else close
    layers: dict[str, Floats] = {
        "open": price,
        "low": price,
        "high": price,
        "close": price,
        "vol": np.full(shape, 1e5),
        "divs": np.zeros(shape),
        "split": np.ones(shape),
        "split_cumprod": np.ones(shape),
        "is_liquid": np.ones(shape) if liquid is None else liquid,
    }
    fields = [field for field in FIELDS if with_liquid or field != "is_liquid"]
    values = np.stack([layers[field] for field in fields], axis=1)
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={"time": business_days(n_days), "field": fields, "asset": list(assets)},
        name=name,
    )


def weights_at(
    rows: Sequence[Sequence[float]],
    days: Sequence[int] | None = None,
    assets: Sequence[str] = ASSETS,
) -> xr.DataArray:
    values = np.array(rows, dtype=float)
    index = list(range(len(values))) if days is None else list(days)
    return xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={"time": business_days(max(index) + 1)[index], "asset": list(assets)},
    )


def vector(values: Sequence[float], assets: Sequence[str] = ASSETS) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=float), dims=["asset"], coords={"asset": list(assets)[: len(values)]}
    )


def good_pair(kind: Hashable = "futures", n_days: int = 4) -> tuple[xr.DataArray, xr.DataArray]:
    return weights_at([GOOD_ROW] * n_days), build_data(kind, n_days)


def dates_of(arr: xr.DataArray, count: int) -> xr.DataArray:
    return arr.isel(time=list(range(count)))


# ---------------------------------------------------------------- normalize


def frame(
    values: Sequence[Sequence[float]], times: Sequence[str], assets: Sequence[str]
) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=float),
        dims=["time", "asset"],
        coords={"time": pd.DatetimeIndex(times, name="time"), "asset": list(assets)},
    )


@pytest.mark.parametrize(
    ("times", "assets", "expected"),
    [
        (["2019-01-02", "2019-01-01"], ["A", "B", "C"], [[0.1, 0.1, 0.8], [0.2, 0.3, 0.5]]),
        (["2019-01-01", "2019-01-02"], ["C", "B", "A"], [[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]]),
    ],
)
def test_normalize_sorts_when_time_or_assets_are_not_increasing(
    times: list[str], assets: list[str], expected: list[list[float]]
) -> None:
    first, second = ([0.2, 0.3, 0.5], [0.1, 0.1, 0.8])
    result = qnout.normalize(frame([first, second], times, assets))
    assert list(result.coords["asset"].values) == ["A", "B", "C"]
    assert [str(t)[:10] for t in result.coords["time"].values] == ["2019-01-01", "2019-01-02"]
    assert_allclose(result.values, expected)


@pytest.mark.parametrize(
    ("times", "assets"),
    [
        (["2019-01-01", "2019-01-01", "2019-01-02"], ["A", "B"]),
        (["2019-01-01", "2019-01-02", "2019-01-03"], ["A", "B", "B"]),
    ],
)
def test_normalize_duplicate_labels_characterization(times: list[str], assets: list[str]) -> None:
    arr = frame(np.full((len(times), len(assets)), 0.1).tolist(), times, assets)
    with pytest.raises(InvalidIndexError, match="uniquely valued"):
        qnout.normalize(arr)


def test_normalize_keeps_already_sorted_unique_frame_untouched() -> None:
    arr = frame([[0.2, 0.3], [0.1, 0.1]], ["2019-01-01", "2019-01-02"], ["A", "B"])
    assert_allclose(qnout.normalize(arr).values, [[0.2, 0.3], [0.1, 0.1]])


def test_normalize_per_asset_clips_exactly_at_the_limits() -> None:
    row = [1.0, -1.0, 1.0000001, -1.0000001, 1.5, -1.5, 5.0, -5.0, 0.5, -0.5]
    arr = xr.DataArray(
        np.array([row]),
        dims=["time", "asset"],
        coords={"time": business_days(1), "asset": list("abcdefghij")},
    )
    result = qnout.normalize(arr, per_asset=True).values
    assert_allclose(result, [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.5, -0.5]], atol=0)


def test_normalize_per_asset_replaces_non_finite_values_with_zero() -> None:
    arr = vector([np.nan, np.inf, -np.inf])
    assert_array_equal(qnout.normalize(arr, per_asset=True).values, [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "row",
    [
        [0.5, -0.5, 0.0],
        [0.25, -0.25, 0.0],
        [0.5000005, -0.5, 0.0],
        [0.9, -0.6, 0.0],
        [2.0, -1.0, 1.0],
    ],
)
def test_normalize_divides_each_day_only_when_the_sum_exceeds_one(row: list[float]) -> None:
    other = [0.1, 0.2, 0.3]
    result = qnout.normalize(weights_at([row, other]))
    total = float(np.abs(row).sum())
    assert_allclose(result.values[0], np.array(row) / max(total, 1.0), rtol=1e-14)
    assert_allclose(result.values[1], other)


@pytest.mark.parametrize(
    "values", [[0.6, -0.4], [0.3, 0.2], [0.5000005, -0.5], [1.5, -0.5, 1.0], [3.0, -1.0, np.inf]]
)
def test_normalize_single_day_vector_divides_only_when_sum_exceeds_one(
    values: list[float],
) -> None:
    finite = np.where(np.isfinite(values), values, 0.0)
    expected = finite / max(float(np.abs(finite).sum()), 1.0)
    assert_allclose(qnout.normalize(vector(values)).values, expected, rtol=1e-14)


@pytest.mark.parametrize("per_asset", [False, True])
@pytest.mark.parametrize("with_time", [False, True])
def test_normalize_drops_the_field_coordinate(per_asset: bool, with_time: bool) -> None:
    arr = weights_at([GOOD_ROW]) if with_time else vector(GOOD_ROW)
    result = qnout.normalize(arr.assign_coords(field="close"), per_asset=per_asset)
    assert "field" not in result.coords


# ---------------------------------------------------------------- clean: kinds and logs


def test_clean_logs_every_step_and_tracks_the_event(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    track = spy(monkeypatch, common, "track_event", None)
    spy(monkeypatch, qns, "check_exposure", True)
    output, data = good_pair("stocks_long")
    result = qnout.clean(output, data, "stocks_long")
    assert logs.info.texts == [
        "Output cleaning...",
        "Fix unique timestamps",
        "Forward filling missing prices...",
        "Check liquidity...",
        "Ok.",
        "Check for missed dates...",
        "Ok.",
        "Check positive positions...",
        "Ok.",
        "Check exposure...",
        "Final normalization...",
        "Output cleaning complete.",
    ]
    assert logs.err.calls == []
    assert track.calls == [(("OUTPUT_CLEAN",), {})]
    assert_allclose(result.values, [GOOD_ROW] * 4)


@pytest.mark.parametrize("call", ["default", "explicit_true"])
def test_clean_debug_is_enabled_by_default(call: str, logs: Logs) -> None:
    output, data = good_pair()
    if call == "default":
        qnout.clean(output, data, "futures")
    else:
        qnout.clean(output, data, "futures", True)
    assert logs.info.texts[0] == "Output cleaning..."
    assert logs.info.texts[-1] == "Output cleaning complete."
    assert len(logs.info.calls) == 9


def test_clean_without_debug_logs_nothing(logs: Logs) -> None:
    output, data = good_pair()
    qnout.clean(output, data, "futures", debug=False)
    assert logs.info.calls == []
    assert logs.err.calls == []


def test_clean_reroutes_errors_of_inner_calls_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy(_data: xr.DataArray, portfolio: xr.DataArray) -> xr.DataArray:
        qlog.log_err("inner-problem")
        return portfolio.isel(time=slice(0, 0))

    monkeypatch.setattr(qns, "calc_non_liquid", noisy)
    output, data = good_pair()
    qnout.clean(output, data, "futures", debug=False)
    captured = capsys.readouterr()
    assert "inner-problem" in captured.out
    assert "inner-problem" not in captured.err
    qlog.log_err("after")
    assert "after" in capsys.readouterr().err


def test_clean_uses_the_data_name_when_kind_is_omitted(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    output = weights_at([[0.5, -0.3, 0.2]] * 3)
    data = build_data("stocks_long", 3)
    omitted = qnout.clean(output, data)
    explicit = qnout.clean(output, data, "stocks_long")
    assert logs.err.calls == []
    assert_allclose(omitted.values, explicit.values)
    assert omitted.values.min() == 0.0


MISMATCH_CASES = [
    (CRYPTODAILY, "crypto_daily", False),
    (CRYPTODAILY, "crypto_daily_long", False),
    (CRYPTODAILY, "crypto_daily_long_short", False),
    (CRYPTODAILY, "stocks", True),
    (STOCKS, "crypto_daily", True),
    ("bonds", "crypto_daily", True),
    ("crypto_daily", CRYPTODAILY, True),
    (STOCKS, "stocks", False),
    ("stocks", "futures", True),
    ("futures", "stocks", True),
    ("stocks_long", "stocks", True),
]


def quiet_check_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    spy(monkeypatch, qns, "calc_non_liquid", dates_of(weights_at([GOOD_ROW]), 0))
    spy(monkeypatch, qns, "find_missed_dates", np.array([], dtype="datetime64[ns]"))
    spy(monkeypatch, qns, "check_exposure", True)
    spy(monkeypatch, qns, "check_correlation", None)
    spy(monkeypatch, common, "track_event", None)
    spy(monkeypatch, qnout, "calc_sharpe_ratio_for_check", 1.0)


@pytest.mark.parametrize("function", ["clean", "check"])
@pytest.mark.parametrize(("data_kind", "kind", "warns"), MISMATCH_CASES)
def test_kind_mismatch_is_reported_unless_cryptodaily_alias(
    function: str,
    data_kind: str,
    kind: str,
    warns: bool,
    monkeypatch: pytest.MonkeyPatch,
    logs: Logs,
) -> None:
    quiet_check_stats(monkeypatch)
    spy(monkeypatch, qne, "cut_big_positions", None)
    output, data = weights_at([GOOD_ROW] * 3), build_data(data_kind, 3)
    if function == "clean":
        qnout.clean(output, data, kind, debug=False)
    else:
        qnout.check(output, data, kind, check_correlation=False)
    expected = [
        (KIND_MISMATCH,),
        (f"The kind of the data is {data_kind} and the kind of the output is {kind}",),
        (CLEANED_WITH_DATA_KIND,),
    ]
    assert logs.err.calls == (expected if warns else [])


def test_clean_with_mismatched_kind_still_applies_the_given_kind_characterization(
    logs: Logs,
) -> None:
    # input: data.name == "futures", kind == "stocks_long"; the log says the data kind is used,
    # actual: the given kind is used (negative positions are removed, as for stocks_long)
    output = weights_at([[0.5, -0.3, 0.2]] * 3)
    cleaned = qnout.clean(output, build_data("futures", 3), "stocks_long", debug=False)
    assert logs.err.texts[2] == CLEANED_WITH_DATA_KIND
    assert cleaned.values.min() == 0.0
    kept = qnout.clean(output, build_data("futures", 3), "futures", debug=False)
    assert kept.values.min() < 0.0


# ---------------------------------------------------------------- clean: structure


def test_clean_single_day_output_goes_to_the_last_data_date(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    data = build_data("futures", 4)
    track_events = spy(monkeypatch, common, "track_event", None)
    result = qnout.clean(vector([0.2, 0.3, 0.1]), data, "futures")
    with_scalar_time = qnout.clean(weights_at([GOOD_ROW]).isel(time=0), data, "futures")
    assert result.dims == ("time", "asset")
    assert result.shape == (1, 3)
    assert result.coords["time"].values[0] == data.coords["time"].values[-1]
    assert_allclose(result.values, [[0.2, 0.3, 0.1]])
    assert with_scalar_time.coords["time"].values[0] == data.coords["time"].values[-1]
    assert_allclose(with_scalar_time.values, [GOOD_ROW])
    assert track_events.calls == []
    first_run = logs.info.texts[:8]
    assert first_run == [
        "Output cleaning...",
        "Fix unique timestamps",
        "Check liquidity...",
        "Ok.",
        "Final normalization...",
        "Output cleaning complete.",
        "Output cleaning...",
        "Fix unique timestamps",
    ]


def test_clean_keeps_the_first_of_duplicated_times_and_assets() -> None:
    data = build_data("futures", 3)
    by_time = xr.DataArray(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]),
        dims=["time", "asset"],
        coords={"time": business_days(3)[[0, 1, 1, 2]], "asset": list(ASSETS)},
    )
    cleaned = qnout.clean(by_time, data, "futures", debug=False)
    assert_allclose(cleaned.values, [[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]], atol=0)
    by_asset = xr.DataArray(
        np.array([[0.4, 0.3, 0.2, 0.1]] * 3),
        dims=["time", "asset"],
        coords={"time": business_days(3), "asset": ["A", "B", "B", "C"]},
    )
    cleaned = qnout.clean(by_asset, data, "futures", debug=False)
    assert list(cleaned.coords["asset"].values) == ["A", "B", "C"]
    assert_allclose(cleaned.values, [[0.4, 0.3, 0.1]] * 3, atol=1e-15)


def identity_cut(weights: xr.DataArray, *_rest: object) -> xr.DataArray:
    return weights


FFILL_ROWS = [[0.5, 0.25, 0.25], [0.4, 0.3, 0.3], [0.2, np.nan, 0.4]]


@pytest.mark.parametrize(
    ("kind", "forward_fill"),
    [
        ("futures", True),
        ("stocks", True),
        ("crypto_daily", True),
        ("stocks_nasdaq100", False),
        ("stocks_s&p500", False),
    ],
)
def test_clean_fills_gaps_of_the_price_forward_except_for_index_kinds(
    kind: str, forward_fill: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    spy(monkeypatch, qns, "get_default_max_exposure", 0.5)
    monkeypatch.setattr(qne, "cut_big_positions", identity_cut)
    close = np.full((3, 3), 100.0)
    close[1, 0] = np.nan
    cleaned = qnout.clean(
        weights_at(FFILL_ROWS), build_data(kind, 3, close=close), kind, debug=False
    )
    second = np.array([0.5, 0.3, 0.3]) / 1.1 if forward_fill else [0.0, 0.3, 0.3]
    assert_allclose(cleaned.values, [[0.5, 0.25, 0.25], second, [0.2, 0.0, 0.4]], rtol=1e-14)


def test_clean_flags_an_empty_output_without_looking_for_missed_dates(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    missed = spy(monkeypatch, qns, "find_missed_dates", np.array([], dtype="datetime64[ns]"))
    output, data = good_pair()
    result = qnout.clean(dates_of(output, 0), data, "futures")
    assert result.shape == (0, 3)
    assert missed.calls == []
    assert "WARNING! Output is empty." in logs.info.texts
    assert "Ok." not in logs.info.texts[logs.info.texts.index("Check for missed dates...") :][:2]


def test_clean_with_a_single_output_row_still_looks_for_missed_dates(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    missed = spy(monkeypatch, qns, "find_missed_dates", np.array([], dtype="datetime64[ns]"))
    output, data = good_pair()
    qnout.clean(output.isel(time=[0]), data, "futures")
    assert len(missed.calls) == 1
    assert "WARNING! Output is empty." not in logs.info.texts


# ---------------------------------------------------------------- clean: liquidity and missed dates


def test_clean_zeroes_exactly_the_non_liquid_positions(logs: Logs) -> None:
    liquid = np.ones((4, 3))
    liquid[1, 1] = 0.0
    liquid[2, 2] = -1.0
    output = weights_at([[0.3, 0.3, 0.4]] * 4)
    cleaned = qnout.clean(output, build_data("futures", 4, liquid=liquid), "futures")
    expected = np.array([[0.3, 0.3, 0.4]] * 4)
    expected[1, 1] = 0.0
    assert_allclose(cleaned.values, expected)
    assert logs.info.texts[3:7] == [
        "Check liquidity...",
        "WARNING! Strategy trades non-liquid assets.",
        "Applying liquidity filter...",
        "Ok.",
    ]


def test_clean_does_not_filter_when_all_traded_assets_are_liquid(logs: Logs) -> None:
    liquid = np.ones((4, 3))
    liquid[1, 1] = 0.0
    output = weights_at([[0.5, 0.0, 0.5]] * 4)
    cleaned = qnout.clean(output, build_data("futures", 4, liquid=liquid), "futures")
    assert_allclose(cleaned.values, [[0.5, 0.0, 0.5]] * 4)
    assert not any("non-liquid" in text for text in logs.info.texts)
    assert "Applying liquidity filter..." not in logs.info.texts


def test_clean_without_the_liquidity_field_skips_the_check(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    non_liquid = spy(monkeypatch, qns, "calc_non_liquid", None)
    output = weights_at([GOOD_ROW] * 3)
    qnout.clean(output, build_data("futures", 3, with_liquid=False), "futures")
    assert non_liquid.calls == []
    assert "WARNING! 'is_liquid' field not found in data. Skipping liquidity check." in (
        logs.info.texts
    )
    assert "Check liquidity..." not in logs.info.texts


@pytest.mark.parametrize("missing", [[3, 4], [3]])
def test_clean_adds_zero_rows_for_missed_dates_with_the_last_row_attributes(
    missing: list[int], logs: Logs
) -> None:
    days = [d for d in range(6) if d not in missing]
    tags = [10 + d for d in days]
    output = weights_at([GOOD_ROW] * len(days), days).assign_coords(tag=("time", tags))
    cleaned = qnout.clean(output, build_data("futures", 6), "futures")
    expected = np.array([GOOD_ROW] * 6)
    expected[missing] = 0.0
    assert_allclose(cleaned.values, expected, atol=0)
    tag_of_day = dict(zip(days, tags, strict=True))
    assert_array_equal(cleaned.coords["tag"].values, [tag_of_day.get(d, 15) for d in range(6)])
    assert logs.info.texts[5:9] == [
        "Check for missed dates...",
        "WARNING! Output contains missed dates.",
        "Adding missed dates and setting positions to zero...",
        "Final normalization...",
    ]


@pytest.mark.parametrize("illiquid_value", [0.0, -1.0, np.nan])
def test_clean_drops_assets_that_are_never_liquid_when_dates_are_missed(
    illiquid_value: float,
) -> None:
    liquid = np.ones((5, 3))
    liquid[:, 2] = illiquid_value
    output = weights_at([[0.6, 0.4, 0.0]] * 4, [0, 1, 2, 4])
    cleaned = qnout.clean(output, build_data("futures", 5, liquid=liquid), "futures", debug=False)
    assert list(cleaned.coords["asset"].values) == ["A", "B"]
    expected = np.array([[0.6, 0.4]] * 5)
    expected[3] = 0.0
    assert_allclose(cleaned.values, expected)


def test_clean_keeps_liquid_value_one_when_dates_are_missed() -> None:
    output = weights_at([[0.6, 0.4, 0.0]] * 4, [0, 1, 2, 4])
    cleaned = qnout.clean(output, build_data("futures", 5), "futures", debug=False)
    assert list(cleaned.coords["asset"].values) == ["A", "B", "C"]
    assert cleaned.values[0, 0] == 0.6


def test_clean_fills_missed_dates_without_the_liquidity_field() -> None:
    output = weights_at([GOOD_ROW] * 3, [0, 1, 3])
    data = build_data("futures", 4, with_liquid=False)
    cleaned = qnout.clean(output, data, "futures", debug=False)
    assert_allclose(cleaned.values, [GOOD_ROW, GOOD_ROW, [0, 0, 0], GOOD_ROW], atol=0)


def test_clean_finds_no_missed_dates_for_a_complete_output(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    missed = spy(monkeypatch, qns, "find_missed_dates", np.array([], dtype="datetime64[ns]"))
    output, data = good_pair()
    qnout.clean(output, data, "futures")
    assert len(missed.calls) == 1
    assert logs.info.texts[5:7] == ["Check for missed dates...", "Ok."]
    assert "WARNING! Output contains missed dates." not in logs.info.texts


# ---------------------------------------------------------------- clean: kind rules


EXPOSURE_ROWS = [GOOD_ROW] * 3


@pytest.mark.parametrize("kind", ["stocks", "stocks_long"])
@pytest.mark.parametrize(
    ("verdicts", "scale", "cuts", "drops"),
    [([True], 1.0, 0, 0), ([False, True], 0.5, 1, 0), ([False, False], 0.25, 1, 1)],
)
def test_clean_exposure_escalation_for_stocks(
    kind: str,
    verdicts: list[bool],
    scale: float,
    cuts: int,
    drops: int,
    monkeypatch: pytest.MonkeyPatch,
    logs: Logs,
) -> None:
    exposure = spy(monkeypatch, qns, "check_exposure", *verdicts)
    cut_calls: list[tuple[xr.DataArray, tuple[object, ...]]] = []
    drop_calls: list[tuple[xr.DataArray, tuple[object, ...]]] = []

    def fake_cut(weights: xr.DataArray, *rest: object) -> xr.DataArray:
        cut_calls.append((weights, rest))
        return weights * 0.5

    def fake_drop(weights: xr.DataArray, *rest: object) -> xr.DataArray:
        drop_calls.append((weights, rest))
        return weights * 0.5

    monkeypatch.setattr(qne, "cut_big_positions", fake_cut)
    monkeypatch.setattr(qne, "drop_bad_days", fake_drop)
    cleaned = qnout.clean(weights_at(EXPOSURE_ROWS), build_data(kind, 3), kind)
    assert_allclose(cleaned.values, np.array(EXPOSURE_ROWS) * scale)
    assert len(exposure.calls) == len(verdicts)
    assert len(cut_calls) == cuts
    assert len(drop_calls) == drops
    if cuts:
        assert_allclose(cut_calls[0][0].values, EXPOSURE_ROWS)
        assert cut_calls[0][1] == ()
    if drops:
        assert_allclose(drop_calls[0][0].values, np.array(EXPOSURE_ROWS) * 0.5)
        assert drop_calls[0][1] == ()
        assert_allclose(exposure.calls[1][0][0].values, np.array(EXPOSURE_ROWS) * 0.5)
    texts = logs.info.texts
    assert ("Cutting oversized positions..." in texts) == bool(cuts)
    assert ("Rechecking exposure..." in texts) == bool(cuts)
    assert ("Dropping days with bad exposure..." in texts) == bool(drops)


@pytest.mark.parametrize(
    "kind", ["futures", "bonds", "crypto_daily", "stocks_nasdaq100", "stocks_s&p500"]
)
def test_clean_does_not_check_exposure_for_other_kinds(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    exposure = spy(monkeypatch, qns, "check_exposure", False)
    drops = spy(monkeypatch, qne, "drop_bad_days", None)
    monkeypatch.setattr(qne, "cut_big_positions", identity_cut)
    qnout.clean(weights_at(EXPOSURE_ROWS), build_data(kind, 3), kind)
    assert exposure.calls == []
    assert drops.calls == []
    assert "Check exposure..." not in logs.info.texts


@pytest.mark.parametrize("kind", ["stocks_nasdaq100", "stocks_s&p500"])
def test_clean_index_kinds_normalize_then_cut_with_the_default_limit(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    limit = spy(monkeypatch, qns, "get_default_max_exposure", 0.07)
    cut_calls: list[tuple[xr.DataArray, tuple[object, ...]]] = []

    def fake_cut(weights: xr.DataArray, *rest: object) -> xr.DataArray:
        cut_calls.append((weights, rest))
        return weights

    monkeypatch.setattr(qne, "cut_big_positions", fake_cut)
    row = [0.5, 0.4, 0.3]
    cleaned = qnout.clean(weights_at([row] * 3), build_data(kind, 3), kind)
    expected = np.array([row] * 3) / 1.2
    assert limit.calls == [((kind,), {})]
    assert len(cut_calls) == 1
    assert_allclose(cut_calls[0][0].values, expected, rtol=1e-14)
    assert cut_calls[0][1] == (0.07,)
    assert_allclose(cleaned.values, expected, rtol=1e-14)
    assert (
        "Normalize and cut big positions for index strategies (nasdaq100, s&p500)..."
        in logs.info.texts
    )


@pytest.mark.parametrize("kind", ["stocks", "futures", "stocks_long"])
def test_clean_other_kinds_do_not_cut_as_index_kinds(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    limit = spy(monkeypatch, qns, "get_default_max_exposure", 0.07)
    cut = spy(monkeypatch, qne, "cut_big_positions", None)
    qnout.clean(weights_at(EXPOSURE_ROWS), build_data(kind, 3), kind)
    assert limit.calls == []
    assert cut.calls == []
    assert not any("index strategies" in text for text in logs.info.texts)


NEGATIVE_ROWS = [[0.5, 0.0, 0.3], [0.2, 0.3, 0.5], [0.4, -0.5, 0.1]]


@pytest.mark.parametrize("kind", ["stocks_long", "crypto_daily_long"])
def test_clean_removes_negative_positions_of_long_only_kinds(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    cleaned = qnout.clean(weights_at(NEGATIVE_ROWS), build_data(kind, 3), kind)
    assert_allclose(cleaned.values, [[0.5, 0, 0.3], [0.2, 0.3, 0.5], [0.4, 0, 0.1]])
    assert "Check positive positions..." in logs.info.texts
    assert "WARNING! Output contains negative positions. Clean..." in logs.info.texts


@pytest.mark.parametrize("kind", ["stocks_long", "crypto_daily_long"])
@pytest.mark.parametrize(
    "rows",
    [
        [[0.5, 0.0, 0.3], [0.2, 0.3, 0.5], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]] * 3,
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.25, 0.25, 0.25]],
    ],
)
def test_clean_keeps_non_negative_positions_of_long_only_kinds(
    kind: str, rows: list[list[float]], monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    cleaned = qnout.clean(weights_at(rows), build_data(kind, 3), kind)
    assert_allclose(cleaned.values, rows)
    texts = logs.info.texts
    assert texts[texts.index("Check positive positions...") + 1] == "Ok."
    assert "WARNING! Output contains negative positions. Clean..." not in texts


def test_clean_removes_a_single_tiny_negative_position() -> None:
    rows = [[0.2, 0.3, 0.5], [0.5, 0.5, -1e-9], [0.2, 0.3, 0.5]]
    cleaned = qnout.clean(weights_at(rows), build_data("crypto_daily_long", 3), "crypto_daily_long")
    assert cleaned.values.min() == 0.0
    assert_allclose(cleaned.values[1], [0.5, 0.5, 0.0])


@pytest.mark.parametrize("kind", ["stocks", "futures", "crypto_daily"])
def test_clean_keeps_negative_positions_of_other_kinds(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    cleaned = qnout.clean(weights_at(NEGATIVE_ROWS), build_data(kind, 3), kind)
    assert cleaned.values[2, 1] < 0.0
    assert "Check positive positions..." not in logs.info.texts


CRYPTO_ASSETS = ("BTC", "ETH")


@pytest.mark.parametrize(
    ("row", "only_btc"),
    [
        ([0.5, 0.0], True),
        ([-0.5, 0.0], True),
        ([0.5, -0.4], False),
        ([-0.5, 0.4], False),
        ([0.0, 0.5], False),
        ([0.0, 0.0], False),
        ([0.5, 0.4], False),
    ],
)
def test_clean_crypto_keeps_only_btc(row: list[float], only_btc: bool, logs: Logs) -> None:
    output = weights_at([row] * 3, assets=CRYPTO_ASSETS)
    cleaned = qnout.clean(output, build_data(CRYPTO, 3, CRYPTO_ASSETS), CRYPTO)
    texts = logs.info.texts
    marker = texts.index("Check for BTC-only positions...")
    if only_btc:
        assert texts[marker + 1] == "Ok."
    else:
        assert texts[marker + 1] == "WARNING! Output contains assets other than BTC. Filtering..."
    assert list(cleaned.coords["asset"].values) == (["BTC", "ETH"] if only_btc else ["BTC"])
    assert_allclose(cleaned.values[:, 0], [row[0]] * 3)


@pytest.mark.parametrize("kind", ["bonds", "stocks", "futures", "cryptodaily", "crypto_daily"])
def test_clean_non_crypto_kinds_keep_all_assets(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spy(monkeypatch, qns, "check_exposure", True)
    spy(monkeypatch, qne, "cut_big_positions", None)
    cleaned = qnout.clean(weights_at([[0.5, 0.0, 0.3]] * 3), build_data(kind, 3), kind, debug=True)
    assert list(cleaned.coords["asset"].values) == list(ASSETS)
    assert "Check for BTC-only positions..." not in logs.info.texts


# ---------------------------------------------------------------- check


@dataclass
class CheckEnv:
    non_liquid: Recorder
    missed: Recorder
    exposure: Recorder
    correlation: Recorder
    sharpe: Recorder
    track: Recorder


def check_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    non_liquid_dates: int = 0,
    missed_dates: int = 0,
    exposure: bool = True,
    sharpe: float = 1.0,
) -> CheckEnv:
    flagged = xr.DataArray(
        np.zeros(non_liquid_dates),
        dims=["time"],
        coords={"time": business_days(non_liquid_dates)},
    )
    missed = business_days(missed_dates).to_numpy()
    return CheckEnv(
        spy(monkeypatch, qns, "calc_non_liquid", flagged),
        spy(monkeypatch, qns, "find_missed_dates", missed),
        spy(monkeypatch, qns, "check_exposure", exposure),
        spy(monkeypatch, qns, "check_correlation", None),
        spy(monkeypatch, qnout, "calc_sharpe_ratio_for_check", sharpe),
        spy(monkeypatch, common, "track_event", None),
    )


@pytest.mark.parametrize(
    ("kind", "exposure_lines"), [("futures", []), ("stocks", ["Check exposure..."])]
)
def test_check_reports_ok_for_every_clean_step(
    kind: str,
    exposure_lines: list[str],
    monkeypatch: pytest.MonkeyPatch,
    logs: Logs,
) -> None:
    env = check_env(monkeypatch)
    output, data = good_pair(kind)
    qnout.check(output, data, kind)
    assert logs.info.texts == [
        "Check liquidity...",
        "Ok.",
        "Check missed dates...",
        "Ok.",
        *exposure_lines,
        "Check the sharpe ratio...",
        "Sharpe Ratio = 1.0",
        "Ok.",
        "Check correlation.",
    ]
    assert logs.err.calls == []
    assert env.track.calls == [(("OUTPUT_CHECK",), {})]
    assert len(env.exposure.calls) == len(exposure_lines)


@pytest.mark.parametrize("call", ["default", "explicit_true", "explicit_false"])
def test_check_correlation_is_requested_by_default(
    call: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch)
    output, data = good_pair()
    if call == "default":
        qnout.check(output, data, "futures")
    elif call == "explicit_true":
        qnout.check(output, data, "futures", True)
    else:
        qnout.check(output, data, "futures", check_correlation=False)
    if call == "explicit_false":
        assert env.correlation.calls == []
        assert "Check correlation." not in logs.info.texts
    else:
        assert len(env.correlation.calls) == 1
        args, kwargs = env.correlation.calls[0]
        assert args[0] is output
        assert args[1] is data
        assert kwargs == {"print_stack_trace": False}


def test_check_passes_data_output_kind_and_date_check_to_the_sharpe_calculation(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch, sharpe=1.25)
    output, data = good_pair("stocks")
    qnout.check(output, data, check_correlation=False)
    args, kwargs = env.sharpe.calls[0]
    assert args[0] is data
    assert args[1] is output
    assert args[2] == "stocks"
    assert args[3] is True
    assert kwargs == {}
    assert ("Sharpe Ratio =", 1.25) in logs.info.calls
    assert logs.err.calls == []


@pytest.mark.parametrize(
    ("sharpe", "low"),
    [(0.7, False), (0.7000001, False), (0.6999999, True), (0.0, True), (-1.5, True), (1.0, False)],
)
def test_check_sharpe_threshold(
    sharpe: float, low: bool, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch, sharpe=sharpe)
    output, data = good_pair()
    qnout.check(output, data, "futures", check_correlation=False)
    if low:
        assert logs.err.calls == [
            ("ERROR! The Sharpe Ratio is too low.", sharpe, "<", 0.7),
            ("Improve the strategy and make sure that the in-sample Sharpe Ratio more than 0.7.",),
        ]
        assert logs.info.texts[-1] == f"Sharpe Ratio = {sharpe}"
    else:
        assert logs.err.calls == []
        assert logs.info.texts[-1] == "Ok."


def test_check_flags_non_liquid_trading_with_a_single_date(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch, non_liquid_dates=1)
    output, data = good_pair()
    qnout.check(output, data, "futures", check_correlation=False)
    assert logs.err.calls[:2] == [
        ("ERROR! Strategy trades non-liquid assets.",),
        ("Multiply the output by data.sel(field='is_liquid') or use qnt.output.clean",),
    ]
    assert logs.info.texts[:3] == ["Check liquidity...", "Check missed dates...", "Ok."]


def test_check_skips_the_liquidity_check_without_the_field(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch)
    output = weights_at([GOOD_ROW] * 4)
    qnout.check(output, build_data("futures", 4, with_liquid=False), "futures", False)
    assert env.non_liquid.calls == []
    assert "Check liquidity..." not in logs.info.texts


def test_check_reports_missed_dates(monkeypatch: pytest.MonkeyPatch, logs: Logs) -> None:
    check_env(monkeypatch, missed_dates=1)
    output, data = good_pair()
    qnout.check(output, data, "futures", check_correlation=False)
    assert logs.err.calls == [
        ("ERROR! Some dates were missed)",),
        ("Your strategy dropped some days, your strategy should produce a continuous series.",),
    ]
    assert logs.info.texts[2:5] == [
        "Check missed dates...",
        "Check the sharpe ratio...",
        "Sharpe Ratio = 1.0",
    ]


@pytest.mark.parametrize("rows", [0, 1])
def test_check_distinguishes_empty_output_from_a_single_row(
    rows: int, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch)
    output, data = good_pair()
    qnout.check(output.isel(time=list(range(rows))), data, "futures", check_correlation=False)
    assert len(env.missed.calls) == rows
    assert env.track.calls == [(("OUTPUT_CHECK",), {})]
    empty_warning = ("WARNING! Output is empty.",)
    assert (empty_warning in logs.err.calls) == (rows == 0)
    zero_warning = ("ERROR! Output is empty. All positions are zero.",)
    assert (zero_warning in logs.err.calls) == (rows == 0)
    assert len(env.sharpe.calls) == rows


def test_check_all_zero_output_is_an_error_without_sharpe(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch)
    output, data = good_pair()
    qnout.check(output * 0, data, "futures", check_correlation=False)
    assert logs.err.calls == [("ERROR! Output is empty. All positions are zero.",)]
    assert env.sharpe.calls == []
    assert env.correlation.calls == []


def test_check_single_day_output_uses_the_last_data_date(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch)
    data = build_data("stocks", 4)
    qnout.check(vector(GOOD_ROW), data, "stocks", check_correlation=True)
    data_last = data.coords["time"].values[-1]
    for placed in (env.non_liquid.calls[0][0][1], env.exposure.calls[0][0][0]):
        assert isinstance(placed, xr.DataArray)
        assert list(placed.coords["time"].values) == [data_last]
        assert_allclose(placed.values, [GOOD_ROW])
    assert env.missed.calls == []
    assert env.track.calls == []
    assert env.sharpe.calls == []
    assert env.correlation.calls == []
    assert logs.info.texts == [
        "Check liquidity...",
        "Ok.",
        "Check exposure...",
    ]


def test_check_exposure_failure_advises_more_assets(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch, exposure=False)
    output, data = good_pair("stocks")
    qnout.check(output, data, "stocks", check_correlation=False)
    assert logs.err.calls == [("Use more assets or/and use qnt.output.clean",)]


@pytest.mark.parametrize("kind", ["futures", "bonds", "crypto_daily", "stocks_nasdaq100"])
def test_check_does_not_check_exposure_for_other_kinds(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = check_env(monkeypatch)
    spy(monkeypatch, qns, "get_default_max_exposure", 1.0)
    output, data = good_pair(kind)
    qnout.check(output, data, kind, check_correlation=False)
    assert env.exposure.calls == []


@pytest.mark.parametrize("kind", ["stocks_nasdaq100", "stocks_s&p500"])
@pytest.mark.parametrize(
    ("limit", "too_high"), [(0.5, False), (0.5000001, False), (0.4999, True), (0.0, True)]
)
def test_check_index_kinds_compare_max_exposure_with_the_hard_limit(
    kind: str, limit: float, too_high: bool, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    limits = spy(monkeypatch, qns, "get_default_max_exposure", limit)
    output = weights_at([[0.5, -0.5, 0.0]] * 3)
    qnout.check(output, build_data(kind, 3), kind, check_correlation=False)
    assert limits.calls == [((kind,), {})]
    marker = logs.info.texts.index("Check max exposure for index stocks (nasdaq100, s&p500)…")
    if too_high:
        assert logs.err.calls == [
            ("ERROR! The max exposure is too high.",),
            (f"Max exposure: 0.5 Hard limit: {limit}",),
            ("Use qnt.output.cut_big_positions() or normalize_by_max_exposure() to fix.",),
        ]
    else:
        assert logs.err.calls == []
        assert logs.info.texts[marker + 1] == "Ok."


@pytest.mark.parametrize(
    ("row", "only_btc"),
    [
        ([0.5, 0.0], True),
        ([-0.5, 0.0], True),
        ([0.5, -0.4], False),
        ([-0.5, 0.4], False),
        ([0.0, 0.5], False),
        ([0.0, 0.0], False),
        ([0.5, 0.4], False),
    ],
)
def test_check_crypto_requires_only_btc_positions(
    row: list[float], only_btc: bool, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    output = weights_at([row] * 3, assets=CRYPTO_ASSETS)
    qnout.check(output, build_data(CRYPTO, 3, CRYPTO_ASSETS), CRYPTO, check_correlation=False)
    marker = logs.info.texts.index("Check BTC...")
    btc_errors = [
        ("ERROR! Output contains not only BTC.\n",),
        ("Remove the other assets from the output or use qnt.output.clean",),
    ]
    if only_btc:
        assert logs.info.texts[marker + 1] == "Ok."
        assert btc_errors[0] not in logs.err.calls
    else:
        assert logs.err.calls[:2] == btc_errors


@pytest.mark.parametrize("kind", ["bonds", "stocks", "futures", "cryptodaily", "crypto_daily"])
def test_check_non_crypto_kinds_do_not_check_btc(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    output, data = good_pair(kind)
    qnout.check(output, data, kind, check_correlation=False)
    assert "Check BTC..." not in logs.info.texts
    assert logs.err.calls == []


@pytest.mark.parametrize("kind", ["stocks_long", "crypto_daily_long"])
def test_check_long_only_kinds_flag_negative_positions(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    qnout.check(weights_at(NEGATIVE_ROWS), build_data(kind, 3), kind, check_correlation=False)
    assert logs.err.calls == [
        ("ERROR! Output contains negative positions.",),
        ("Drop all negative positions.",),
    ]
    assert "Check positive positions..." in logs.info.texts


@pytest.mark.parametrize("kind", ["stocks_long", "crypto_daily_long"])
@pytest.mark.parametrize(
    "rows",
    [
        [[0.5, 0.0, 0.3], [0.2, 0.3, 0.5], [0.0, 0.0, 0.0]],
        [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.25, 0.25, 0.25]],
    ],
)
def test_check_long_only_kinds_accept_non_negative_positions(
    kind: str, rows: list[list[float]], monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    qnout.check(weights_at(rows), build_data(kind, 3), kind, check_correlation=False)
    texts = logs.info.texts
    assert texts[texts.index("Check positive positions...") + 1] == "Ok."
    assert logs.err.calls == []


def test_check_flags_a_single_tiny_negative_position(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    rows = [[0.5, 0.5, 0.0], [0.5, 0.5, -1e-9], [0.5, 0.5, 0.0]]
    qnout.check(weights_at(rows), build_data("stocks_long", 3), "stocks_long", False)
    assert ("ERROR! Output contains negative positions.",) in logs.err.calls


@pytest.mark.parametrize("kind", ["stocks", "futures"])
def test_check_other_kinds_do_not_check_positive_positions(
    kind: str, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    check_env(monkeypatch)
    qnout.check(weights_at(NEGATIVE_ROWS), build_data(kind, 3), kind, check_correlation=False)
    assert "Check positive positions..." not in logs.info.texts


def test_check_uses_the_data_name_when_kind_is_omitted(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = check_env(monkeypatch)
    qnout.check(weights_at(NEGATIVE_ROWS), build_data("stocks_long", 3), check_correlation=False)
    assert ("ERROR! Output contains negative positions.",) in logs.err.calls
    assert env.sharpe.calls[0][0][2] == "stocks_long"
    assert not any(call[0] == KIND_MISMATCH for call in logs.err.calls)


def test_check_reports_exceptions_through_the_error_log(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    failure = ValueError("boom")

    def broken(_data: xr.DataArray, _output: xr.DataArray) -> xr.DataArray:
        raise failure

    monkeypatch.setattr(qns, "calc_non_liquid", broken)
    spy(monkeypatch, common, "track_event", None)
    output, data = good_pair()
    qnout.check(output, data, "futures", check_correlation=False)
    assert len(logs.err.calls) == 1
    assert logs.err.calls[0][0] is failure


# ---------------------------------------------------------------- calc_sharpe_ratio_for_check


@dataclass
class SharpeEnv:
    start: Recorder
    returns: Recorder
    sharpe: Recorder


MARKER_RETURNS = xr.DataArray([0.0], dims=["time"], coords={"time": business_days(1)})
SHARPE_SERIES = xr.DataArray([1.0, 2.0, 3.5], dims=["time"], coords={"time": business_days(3)})
DATA_DAYS = 40


def sharpe_env(monkeypatch: pytest.MonkeyPatch, start_date: str) -> SharpeEnv:
    return SharpeEnv(
        spy(monkeypatch, qns, "get_default_is_start_date_for_type", start_date),
        spy(monkeypatch, qns, "calc_relative_return", MARKER_RETURNS),
        spy(monkeypatch, qns, "calc_sharpe_ratio_annualized", SHARPE_SERIES),
    )


def output_from(first_active_day: int) -> xr.DataArray:
    rows = [[0.0, 0.0, 0.0] if d < first_active_day else [0.5, 0.5, 0.0] for d in range(DATA_DAYS)]
    return weights_at(rows)


def day(index: int) -> pd.Timestamp:
    return business_days(DATA_DAYS)[index]


def date_text(stamp: pd.Timestamp, offset_days: int = 0) -> str:
    return (stamp + pd.Timedelta(days=offset_days)).strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    ("kind", "estimator"),
    [
        ("crypto_daily", "arithmetic"),
        (CRYPTODAILY, "arithmetic"),
        ("crypto_daily_long", "arithmetic"),
        ("crypto_daily_long_short", "arithmetic"),
        ("stocks", "geometric"),
        ("crypto", "geometric"),
    ],
)
def test_sharpe_for_check_picks_the_mean_estimator_by_kind(
    kind: str, estimator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = sharpe_env(monkeypatch, "2019-01-02")
    data = build_data(kind, DATA_DAYS)
    result = qnout.calc_sharpe_ratio_for_check(data, output_from(0), kind, check_dates=False)
    assert env.sharpe.calls[0][1] == {"mean_estimator": estimator}
    assert env.sharpe.calls[0][0][0] is MARKER_RETURNS
    assert env.start.calls == [((kind,), {})]
    assert float(result) == 3.5


def test_sharpe_for_check_uses_the_data_name_when_kind_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = sharpe_env(monkeypatch, "2019-01-02")
    data = build_data("crypto_daily", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), check_dates=False)
    assert env.start.calls == [(("crypto_daily",), {})]
    assert env.sharpe.calls[0][1] == {"mean_estimator": "arithmetic"}


def test_sharpe_for_check_slices_from_the_later_of_start_date_and_data_start(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    start_index = 5
    period = f"Period: {date_text(day(5))} - {date_text(day(DATA_DAYS - 1))}"
    env = sharpe_env(monkeypatch, date_text(day(start_index)))
    data = build_data("stocks", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks", check_dates=False)
    sliced = env.returns.calls[0][0][1]
    assert env.returns.calls[0][0][0] is data
    assert isinstance(sliced, xr.DataArray)
    assert [str(t)[:10] for t in sliced.coords["time"].values] == [
        date_text(day(i)) for i in range(start_index, DATA_DAYS)
    ]
    assert period in logs.info.texts


def test_sharpe_for_check_starts_at_the_data_start_when_the_default_date_is_earlier(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    env = sharpe_env(monkeypatch, "2010-01-01")
    data = build_data("stocks", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks", check_dates=False)
    sliced = env.returns.calls[0][0][1]
    assert isinstance(sliced, xr.DataArray)
    assert sliced.sizes["time"] == DATA_DAYS
    assert logs.info.texts == [f"Period: 2019-01-01 - {date_text(day(DATA_DAYS - 1))}"]


@pytest.mark.parametrize(("gap", "warns"), [(9, False), (10, False), (11, True), (12, True)])
def test_sharpe_for_check_warns_when_data_starts_more_than_ten_days_after_the_start_date(
    gap: int, warns: bool, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    start = date_text(day(0), -gap)
    sharpe_env(monkeypatch, start)
    data = build_data("stocks", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks", check_dates=True)
    errors = logs.err.texts
    assert (NOT_ENOUGH_DATA in errors) == warns
    if warns:
        assert errors[:3] == [
            NOT_ENOUGH_DATA,
            f"The first point(2019-01-01) should be earlier than {start}",
            "Load data more historical data.",
        ]
        assert SLIPPAGE_WARNING not in errors
    else:
        assert errors[:2] == [
            SLIPPAGE_WARNING,
            "Add 15 extra data points to the data head (load data more historical data).",
        ]


@pytest.mark.parametrize(("points", "warns"), [(14, True), (15, False), (16, False)])
def test_sharpe_for_check_wants_fifteen_points_before_the_start_date(
    points: int, warns: bool, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    sharpe_env(monkeypatch, date_text(day(points - 1)))
    data = build_data("stocks", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks", check_dates=True)
    assert (SLIPPAGE_WARNING in logs.err.texts) == warns
    assert NOT_ENOUGH_DATA not in logs.err.texts


@pytest.mark.parametrize(("gap", "warns"), [(6, False), (7, False), (8, True), (9, True)])
def test_sharpe_for_check_warns_when_output_starts_more_than_seven_days_after_the_start_date(
    gap: int, warns: bool, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    first_active = 20
    start = date_text(day(first_active), -gap)
    sharpe_env(monkeypatch, start)
    data = build_data("stocks", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(first_active), "stocks", True)
    errors = logs.err.texts
    assert (NOT_ENOUGH_OUTPUT in errors) == warns
    if warns:
        assert errors[-2:] == [
            NOT_ENOUGH_OUTPUT,
            (
                f"The output series should start from {start} or earlier instead of "
                f"{date_text(day(first_active))}"
            ),
        ]


def test_sharpe_for_check_date_checks_are_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    sharpe_env(monkeypatch, date_text(day(0), -11))
    data = build_data("stocks", DATA_DAYS)
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks")
    default_errors = list(logs.err.texts)
    logs.err.calls.clear()
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks", True)
    assert default_errors == logs.err.texts
    assert NOT_ENOUGH_DATA in default_errors
    logs.err.calls.clear()
    qnout.calc_sharpe_ratio_for_check(data, output_from(0), "stocks", False)
    assert logs.err.calls == []


# ---------------------------------------------------------------- write and align


def test_write_stores_the_file_at_the_configured_path_and_reports_it(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    (sandbox / "id-translation.csv").write_text("server_id,user_id\nsrv0,A\nsrv1,B\n")
    track = spy(monkeypatch, common, "track_event", None)
    target = sandbox / "custom-output.nc.gz"
    monkeypatch.setenv("OUTPUT_PATH", str(target))
    qnout.write(weights_at([[0.5, 0.25, 0.25], [0.1, 0.1, 0.1]]))
    assert target.exists()
    assert not (sandbox / "fractions.nc.gz").exists()
    assert logs.info.texts == [f"Write output: {target}"]
    assert track.calls == [(("OUTPUT_WRITE",), {})]
    with gzip.open(target, "rb") as stream:
        stored = xr.open_dataarray(stream.read(), cache=False, decode_times=True).compute()
    assert list(stored.coords["asset"].values) == ["C", "srv0", "srv1"]
    assert_allclose(stored.values, [[0.25, 0.5, 0.25], [0.1, 0.1, 0.1]])


def test_write_defaults_to_fractions_file_in_the_working_directory(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    (sandbox / "id-translation.csv").write_text("server_id,user_id\nsrv0,A\n")
    spy(monkeypatch, common, "track_event", None)
    monkeypatch.delenv("OUTPUT_PATH")
    qnout.write(weights_at([GOOD_ROW]))
    assert (sandbox / "fractions.nc.gz").exists()
    assert logs.info.texts == ["Write output: fractions.nc.gz"]


ALIGN_TIMES = business_days(10)


def align_input() -> xr.DataArray:
    return weights_at([[0.5, 0.25, 0.25]] * 8, list(range(1, 9)))


@pytest.mark.parametrize(
    ("start", "end", "first", "last"),
    [
        (None, None, 0, 9),
        ("2019-01-04", None, 3, 9),
        (None, "2019-01-08", 0, 5),
        ("2019-01-04", "2019-01-08", 3, 5),
        (datetime.date(2019, 1, 4), datetime.date(2019, 1, 9), 3, 6),
    ],
)
def test_align_cuts_the_requested_window_and_zero_fills(
    start: datetime.date | str | None, end: datetime.date | str | None, first: int, last: int
) -> None:
    time_coord = xr.DataArray(ALIGN_TIMES, dims=["time"], coords={"time": ALIGN_TIMES})
    cut = qnout.align(align_input(), time_coord, start, end)
    expected_days = list(range(first, last + 1))
    assert [str(t)[:10] for t in cut.coords["time"].values] == [date_text(day(i)) for i in expected_days]
    values = cut.values
    for position, index in enumerate(expected_days):
        wanted = [0.5, 0.25, 0.25] if 1 <= index <= 8 else [0.0, 0.0, 0.0]
        assert_allclose(values[position], wanted)
