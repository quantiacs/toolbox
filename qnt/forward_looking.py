from qnt.data import ds
import xarray as xr
import numpy as np
from qnt.data import sort_and_crop_output
import datetime
import qnt.data.common as qdc
from qnt.log import log_info, log_err

FORWARD_LOOKING_TEST_OFFSET = 182
FORWARD_LOOKING_TEST_DELTA = 10 ** -7


def load_data_calc_output_and_check_forward_looking(strategy):
    """
    :param strategy: function with data loading and output calculation
    :return: whole output
    """
    qdc.MAX_DATE_LIMIT = None
    qdc.MAX_DATETIME_LIMIT = None

    log_info("Computing of the whole output...")
    whole_output = strategy()

    last_date = datetime.datetime.now().date()
    last_date = last_date - datetime.timedelta(days=FORWARD_LOOKING_TEST_OFFSET)
    qdc.MAX_DATE_LIMIT = last_date
    qdc.MAX_DATETIME_LIMIT = datetime.datetime.combine(last_date, datetime.time.min)

    log_info("Computing of the cropped output...")
    cropped_output = strategy()

    qdc.MAX_DATE_LIMIT = None
    qdc.MAX_DATETIME_LIMIT = None

    check_forward_looking(cropped_output, whole_output)

    return whole_output


def calc_output_and_check_forward_looking(data, strategy):
    """
    :param data: loaded data xarray
    :param strategy: function, that calculates outputs using provided data
    :return: output
    """
    cropped_data = data

    last_date = data.coords[ds.TIME].values.max()
    last_date = str(last_date)[0:10]
    last_date = datetime.datetime.strptime(last_date, '%Y-%m-%d').date()
    last_date = last_date - datetime.timedelta(days=FORWARD_LOOKING_TEST_OFFSET)
    last_date = str(last_date)

    if data.coords[ds.TIME][0] < data.coords[ds.TIME][-1]:
        cropped_data = cropped_data.loc[{ds.TIME: slice(None, last_date)}]
    else:
        cropped_data = cropped_data.loc[{ds.TIME: slice(last_date, None)}]

    cropped_data = cropped_data.dropna(ds.ASSET, 'all')
    cropped_data = cropped_data.dropna(ds.TIME, 'all')

    log_info("Computing of the cropped output...")
    cropped_output = strategy(cropped_data)
    log_info("Computing of the whole output...")
    whole_output = strategy(data)

    check_forward_looking(cropped_output, whole_output)

    return whole_output


def check_forward_looking(cropped_output, whole_output):
    cropped_output = sort_and_crop_output(cropped_output)
    whole_output = sort_and_crop_output(whole_output)

    max_time = min(cropped_output.coords[ds.TIME].values.max(), whole_output.coords[ds.TIME].values.max())

    cropped_output = cropped_output.loc[:max_time]
    whole_output = whole_output.loc[:max_time]

    cropped_output, whole_output = xr.align(cropped_output, whole_output, join='outer')

    cropped_output = cropped_output.fillna(0)
    whole_output = whole_output.fillna(0)

    diff = whole_output - cropped_output
    # print(diff.where(diff!=0).dropna('time', 'all').dropna('asset','all'))
    delta = abs(diff).max().values
    if delta > FORWARD_LOOKING_TEST_DELTA:
        log_info('WARNING: This strategy uses forward looking! Delta = ' + str(delta))
        return True
    else:
        log_info('Ok. There is no forward looking.')
        return False
