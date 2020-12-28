from qnt.log import log_info, log_err

output_schema = {
    "type": "record",
    "name": "DataArray",
    "fields": [
        {
            "name": "coords",
            "type": {
                "name": "Coordinates",
                "type": "record",
                "fields": [
                    {"name": "time", "type": {"type": "array", "items": "string", "doc": "iso dates"}},
                    {"name": "asset", "type": {"type": "array", "items": "string", "doc": "asset ids"}},
                ]
            }
        },
        {"name": "values", "type": {"type": "array", "doc": "C-array, row-major order", "items": "double"}}
    ]
}

# optional dependency:
# conda install conda-forge::python-avro=1.8.*

import avro.io as aio
import avro.datafile as adf
import avro.schema as avs

from io import BytesIO
import numpy as np
import xarray as xr
import pandas as pd
from qnt.data import ds


def convert_output_to_avro_record(output):
    bio = BytesIO()
    encoder = aio.BinaryEncoder(bio)

    write_output_to_encoder(encoder, output)

    return bio.getvalue()


def write_output_to_encoder(encoder, output):
    time = output.coords[ds.TIME].values
    encoder.write_int(len(time))
    for t in time:
        encoder.write_utf8(str(t)[0:10])
    encoder.write_int(0)

    asset = output.coords[ds.ASSET].values.tolist()
    encoder.write_int(len(asset))
    for a in asset:
        encoder.write_utf8(str(a))
    encoder.write_int(0)

    encoder.write_int(len(time) * len(asset))
    encoder.write(output.values.tobytes('C'))
    encoder.write_int(0)


def convert_avro_record_to_output(avr_bytes):
    bio = BytesIO(avr_bytes)
    decoder = aio.BinaryDecoder(bio)

    return read_output_from_decoder(decoder)


def read_output_from_decoder(decoder):
    time_len = decoder.read_int()
    time = []
    for t in range(time_len):
        time.append(decoder.read_utf8())
    assert decoder.read_int() == 0
    time = np.array(time, dtype='datetime64[D]')

    asset_len = decoder.read_int()
    asset = []
    for a in range(asset_len):
        asset.append(decoder.read_utf8())
    assert decoder.read_int() == 0
    asset = np.array(asset)

    data_len = decoder.read_int()
    data = decoder.read(8 * data_len)
    data = np.frombuffer(data, np.float64)
    data = data.reshape([time_len, asset_len], order='C')
    assert decoder.read_int() == 0

    return xr.DataArray(data, dims=[ds.TIME, ds.ASSET], coords={ds.TIME: time, ds.ASSET: asset})


def output_to_avro_file(output):
    return merge_output_records_to_file([convert_output_to_avro_record(output)])


def avro_file_to_output(avf_bytes):
    outputs = read_outputs_from_file(avf_bytes)
    assert len(outputs) == 1
    return outputs[0]


def merge_output_records_to_file(records):
    bio = BytesIO()

    schema = avs.Parse(json.dumps(output_schema))

    writer = aio.DatumWriter()
    writer.write = lambda datum, encoder: encoder.write(datum)

    dw = adf.DataFileWriter(bio, writer, schema)

    for r in records:
        dw.append(r)
    dw.flush()

    return bio.getvalue()


def read_outputs_from_file(avf_bytes):
    bio = BytesIO(avf_bytes)
    reader = aio.DatumReader()
    reader.read = lambda decoder: read_output_from_decoder(decoder)
    dr = adf.DataFileReader(bio, reader)
    return [i for i in dr]


if __name__ == '__main__':
    log_info("testing avro conversions...")
    output = xr.DataArray(
        np.random.rand(2, 2),
        dims=['time', 'asset'],
        coords={
            'time': pd.to_datetime(['2019-01-01', '2019-01-02']),
            'asset': np.array(['ASSET1', '2ASSET'])
        }
    )

    log_info("check conversion to avro record")

    av_record = convert_output_to_avro_record(output)
    import json

    schema = avs.Parse(json.dumps(output_schema))
    reader = aio.DatumReader(schema, schema)
    parsed = reader.read(aio.BinaryDecoder(BytesIO(av_record)))

    assert aio.Validate(schema, parsed)
    assert parsed['coords']['time'] == [str(i)[0:10] for i in output.coords[ds.TIME].values]
    assert parsed['coords']['asset'] == output.coords[ds.ASSET].values.tolist()
    assert parsed['values'] == output.values.reshape([4]).tolist()

    log_info("check conversion from avro record")

    output2 = convert_avro_record_to_output(av_record)

    assert (output.identical(output2))

    log_info("check conversion to avro file")

    avr_file = output_to_avro_file(output)
    avr_file_reader = adf.DataFileReader(BytesIO(avr_file), aio.DatumReader())
    file_parsed = avr_file_reader.__next__()

    assert aio.Validate(schema, file_parsed)
    assert parsed == file_parsed

    log_info("check conversion from avro file")

    output3 = avro_file_to_output(avr_file)

    assert output.identical(output3)
