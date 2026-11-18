import json
import math
from io import BytesIO
from typing import TypedDict, cast

import avro.datafile as adf
import avro.io as aio
import avro.schema as avs
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as ply
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.graph as qng
import qnt.output_avro as qnavro
from qnt.tests.hl_support import business_days, plot_field, plot_floats, trace_floats

class Coordinates(TypedDict):
    time: list[str]
    asset: list[str]


class Record(TypedDict):
    coords: Coordinates
    values: list[float]


def decode(schema: avs.Schema, record: bytes) -> Record:
    return cast("Record", aio.DatumReader(schema, schema).read(aio.BinaryDecoder(BytesIO(record))))


def sample_output() -> xr.DataArray:
    values = np.array([[0.25, -0.5, np.nan], [0.0, 0.125, 0.75]], dtype=np.float64)
    return xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={
            "time": pd.DatetimeIndex(["2019-01-01", "2019-01-02"], name="time"),
            "asset": np.array(["NYS:A", "NAS:B", "C"]),
        },
    )


def assert_same_output(actual: xr.DataArray, expected: xr.DataArray) -> None:
    assert actual.dims == ("time", "asset")
    assert_array_equal(
        actual.coords["time"].values.astype("datetime64[D]"), expected.coords["time"].values.astype("datetime64[D]")
    )
    assert list(actual.coords["asset"].values) == list(expected.coords["asset"].values)
    assert_array_equal(actual.values, expected.values)


def test_avro_record_layout_matches_the_schema() -> None:
    output = sample_output()
    record = qnavro.convert_output_to_avro_record(output)
    schema = avs.parse(json.dumps(qnavro.output_schema))
    parsed = decode(schema, record)
    assert aio.validate(schema, parsed)
    assert parsed["coords"] == {
        "time": ["2019-01-01", "2019-01-02"],
        "asset": ["NYS:A", "NAS:B", "C"],
    }
    flat = parsed["values"]
    assert len(flat) == 6
    assert flat[:2] == [0.25, -0.5]
    assert math.isnan(flat[2])
    assert flat[3:] == [0.0, 0.125, 0.75]


def container_file(outputs: list[xr.DataArray]) -> bytes:
    schema = avs.parse(json.dumps(qnavro.output_schema))
    buffer = BytesIO()
    writer = adf.DataFileWriter(buffer, aio.DatumWriter(), schema)
    for output in outputs:
        record = qnavro.convert_output_to_avro_record(output)
        writer.append(decode(schema, record))
    writer.flush()
    return buffer.getvalue()


def test_avro_roundtrip_through_records_and_container_files() -> None:
    output = sample_output()
    record = qnavro.convert_output_to_avro_record(output)
    assert_same_output(qnavro.convert_avro_record_to_output(record), output)

    single = container_file([output])
    assert_same_output(qnavro.avro_file_to_output(single), output)

    second = output * 2
    outputs = qnavro.read_outputs_from_file(container_file([output, second]))
    assert len(outputs) == 2
    assert_same_output(outputs[1], second)
    with pytest.raises(AssertionError):
        qnavro.avro_file_to_output(container_file([output, second]))


def stat_frame() -> xr.DataArray:
    days = business_days(120)
    rng = np.random.default_rng(4)
    equity = np.cumprod(1.0 + rng.normal(0.001, 0.01, 120))
    underwater = equity / np.maximum.accumulate(equity) - 1.0
    sharpe = np.cumsum(rng.normal(0.0, 0.1, 120)) + 1.0
    bias = rng.uniform(-1.0, 1.0, 120)
    fields = ["equity", "underwater", "sharpe_ratio", "bias"]
    return xr.DataArray(
        np.column_stack([equity, underwater, sharpe, bias]),
        dims=["time", "field"],
        coords={"time": days, "field": fields},
    )


def test_major_plots_use_log_equity_and_padded_ranges() -> None:
    stats = stat_frame()
    plots = qng.make_major_plots(stats)
    assert plots is not None
    equity_fig, underwater_fig, sharpe_fig, bias_fig = plots
    assert equity_fig is not None
    assert underwater_fig is not None
    assert sharpe_fig is not None
    assert bias_fig is not None

    equity = stats.values[:, 0]
    low, high = math.log10(equity.min()), math.log10(equity.max())
    span = high - low
    assert plot_field(equity_fig, "layout", "yaxis", "type") == "log"
    assert_allclose(
        plot_floats(equity_fig, "layout", "yaxis", "range"),
        [low - 0.05 * span, high + 0.05 * span],
    )
    assert plot_field(equity_fig, "data", 0, "name") == " PnL (Equity)"
    assert plot_field(equity_fig, "data", 0, "fill") == "tonexty"
    assert_allclose(trace_floats(equity_fig, "y"), equity)

    underwater = stats.values[:, 1]
    assert plot_field(underwater_fig, "data", 0, "line", "color") == "darkred"
    depth = 0.0 - underwater.min()
    assert_allclose(
        plot_floats(underwater_fig, "layout", "yaxis", "range"),
        [underwater.min() - 0.05 * depth, 0.05 * depth],
    )

    tail = stats.values[-60:, 2]
    low_sr, high_sr = tail.min(), tail.max()
    padded = (low_sr - 0.03 * (high_sr - low_sr), high_sr + 0.03 * (high_sr - low_sr))
    width = padded[1] - padded[0]
    assert_allclose(
        plot_floats(sharpe_fig, "layout", "yaxis", "range"),
        [padded[0] - 0.05 * width, padded[1] + 0.05 * width],
    )
    assert plot_field(sharpe_fig, "data", 0, "name") == "Rolling SR"
    assert plot_field(bias_fig, "data", 0, "name") == "Bias Chart"
    assert len(trace_floats(bias_fig, "y")) == 120


def test_filled_plot_and_environment_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    index = list(pd.bdate_range("2020-01-01", periods=5))
    series = pd.Series([1.0, 2.0, 4.0, 3.0, 5.0], index=index)
    figure = qng.make_plot_filled(
        index, series, color="#112233", width=5, name="demo", range_min=0.0, range_max=10.0
    )
    assert figure is not None
    assert plot_field(figure, "data", 0, "line", "color") == "#112233"
    assert plot_field(figure, "data", 0, "line", "width") == 5
    assert plot_field(figure, "data", 0, "name") == "demo"
    assert_allclose(plot_floats(figure, "layout", "yaxis", "range"), [-0.5, 10.5])
    assert plot_field(figure, "layout", "showlegend") is True

    assert not qng.is_notebook()
    monkeypatch.delenv("NONINTERACT", raising=False)
    assert qng.is_interact()
    monkeypatch.setenv("NONINTERACT", "1")
    assert not qng.is_interact()


def test_line_plots_are_sent_to_the_plotly_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[go.Figure] = []

    def capture(figure: go.Figure) -> None:
        shown.append(figure)

    monkeypatch.setattr(ply, "iplot", capture)
    index = list(pd.bdate_range("2020-01-01", periods=4))
    first = pd.Series([1.0, 2.0, 4.0, 3.0], index=index)
    second = pd.Series([2.0, 1.0, 0.5, 0.25], index=index)

    qng.make_plot(index, first, color="#010203", name="one", range_min=0.0, range_max=8.0)
    qng.make_plot_double(index, first, second, name1="a", name2="b", color2="#aabbcc")
    assert len(shown) == 2
    assert plot_field(shown[0], "data", 0, "name") == "one"
    assert_allclose(plot_floats(shown[0], "layout", "yaxis", "range"), [-0.4, 8.4])
    assert [plot_field(shown[1], "data", i, "name") for i in (0, 1)] == ["a", "b"]
    assert plot_field(shown[1], "data", 1, "line", "color") == "#aabbcc"
    assert_allclose(trace_floats(shown[1], "y", 1), [2.0, 1.0, 0.5, 0.25])
