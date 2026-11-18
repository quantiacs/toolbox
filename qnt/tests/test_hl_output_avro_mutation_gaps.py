import json
from io import BytesIO

import avro.datafile as adf
import avro.io as aio
import avro.schema as avs
import numpy as np
import pytest
import xarray as xr

import qnt.output_avro as qnavro

TIMES = ["2020-01-01", "2020-01-02"]
ASSETS = ["A1", "B2", "C3"]


def make_output(values: list[list[float]], times: list[str] = TIMES) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=np.float64),
        dims=["time", "asset"],
        coords={"time": np.array(times, dtype="datetime64[D]"), "asset": np.array(ASSETS)},
    )


FIRST = make_output([[0.5, -1.0, 2.0], [0.0, 0.25, 4.0]])
SECOND = make_output([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ["2021-03-04", "2021-03-05"])


def read_dicts(data: bytes) -> list[dict[str, object]]:
    reader = adf.DataFileReader(BytesIO(data), aio.DatumReader())
    return [dict(item) for item in reader]


def corrupt_record(bad_index: int, bad_value: int) -> bytes:
    """Record whose terminator number `bad_index` (0..2) is replaced by bad_value."""
    bio = BytesIO()
    enc = aio.BinaryEncoder(bio)
    terms = [0, 0, 0]
    terms[bad_index] = bad_value
    enc.write_int(1)
    enc.write_utf8("2020-01-01")
    enc.write_int(terms[0])
    enc.write_int(1)
    enc.write_utf8("A")
    enc.write_int(terms[1])
    enc.write_int(1)
    enc.write(np.array([1.5]).tobytes())
    enc.write_int(terms[2])
    return bio.getvalue()


@pytest.mark.xfail(strict=True, reason="AttributeError: avro.schema has no attribute 'Parse' (removed in avro 1.12)")
def test_raw_writer_accepts_bytes_records() -> None:
    data = qnavro.merge_output_records_to_file([qnavro.convert_output_to_avro_record(FIRST)])
    assert len(read_dicts(data)) == 1


@pytest.mark.xfail(strict=True, reason="AttributeError: avro.schema has no attribute 'Parse' (removed in avro 1.12)")
def test_container_file_holds_all_records_in_order() -> None:
    records = [qnavro.convert_output_to_avro_record(o) for o in (FIRST, SECOND)]
    data = qnavro.merge_output_records_to_file(records)

    expected: list[dict[str, object]] = [
        {
            "coords": {"time": TIMES, "asset": ASSETS},
            "values": [0.5, -1.0, 2.0, 0.0, 0.25, 4.0],
        },
        {
            "coords": {"time": ["2021-03-04", "2021-03-05"], "asset": ASSETS},
            "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
    ]
    assert read_dicts(data) == expected

    outputs = qnavro.read_outputs_from_file(data)
    assert len(outputs) == 2
    assert outputs[0].identical(FIRST)
    assert outputs[1].identical(SECOND)


@pytest.mark.xfail(strict=True, reason="AttributeError: avro.schema has no attribute 'Parse' (removed in avro 1.12)")
def test_file_metadata_carries_the_output_schema() -> None:
    data = qnavro.output_to_avro_file(FIRST)
    reader = adf.DataFileReader(BytesIO(data), aio.DatumReader())
    written = avs.parse(reader.meta["avro.schema"].decode())
    assert written.to_json() == avs.parse(json.dumps(qnavro.output_schema)).to_json()
    assert written.to_json()["name"] == "DataArray"


@pytest.mark.xfail(strict=True, reason="AttributeError: avro.schema has no attribute 'Parse' (removed in avro 1.12)")
def test_avro_file_to_output_requires_exactly_one_record() -> None:
    empty = qnavro.merge_output_records_to_file([])
    assert qnavro.read_outputs_from_file(empty) == []
    with pytest.raises(AssertionError):
        qnavro.avro_file_to_output(empty)

    two = qnavro.merge_output_records_to_file(
        [qnavro.convert_output_to_avro_record(o) for o in (FIRST, SECOND)]
    )
    with pytest.raises(AssertionError):
        qnavro.avro_file_to_output(two)

    one = qnavro.merge_output_records_to_file([qnavro.convert_output_to_avro_record(FIRST)])
    assert qnavro.avro_file_to_output(one).identical(FIRST)


@pytest.mark.xfail(strict=True, reason="AttributeError: avro.schema has no attribute 'Parse' (removed in avro 1.12)")
def test_nan_is_preserved_in_files() -> None:
    nan_out = make_output([[np.nan, 1.0, 2.0], [3.0, np.nan, 5.0]])
    back = qnavro.avro_file_to_output(qnavro.output_to_avro_file(nan_out))
    assert back.identical(nan_out)
    assert np.isnan(back.values[0, 0])


def test_valid_hand_built_record_decodes() -> None:
    out = qnavro.convert_avro_record_to_output(corrupt_record(0, 0))
    assert out.shape == (1, 1)
    assert out.values[0, 0] == 1.5


@pytest.mark.parametrize("bad_index", [0, 1, 2])
@pytest.mark.parametrize("bad_value", [1, -1])
def test_nonzero_block_terminator_is_rejected(bad_index: int, bad_value: int) -> None:
    with pytest.raises(AssertionError):
        qnavro.convert_avro_record_to_output(corrupt_record(bad_index, bad_value))
