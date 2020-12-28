import xarray as xr
import numpy as np
import pandas as pd
import os
import datetime
import qnt.data.common
import qnt.data
import urllib.parse
import shutil
from qnt.log import log_info, log_err
import subprocess
import io
import qnt.stats
from .data import id_translation as idt
import qnt.output

fractions_fn = "../fractions.nc.gz"
last_data_fn = "../last_data.txt"
html_fn = "../strategy.html"
result_dir = "precheck_results"


def run_init():
    if os.path.exists("init.ipynb"):
        log_info("Run init.ipynb..")
        cmd = "jupyter nbconvert --to html --ExecutePreprocessor.timeout=1800 --execute init.ipynb --stdout "  + \
              "| html2text -utf8"
        # "\\\n 2>&1"
        log_info("cmd:", cmd)
        log_info("output:")
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable='bash')
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            sys.stdout.write(line)
        proc.wait()
        code = proc.returncode
        log_info("return code:", code)


def evaluate_passes(data_type='stocks', passes=3, dates=None):
    in_sample_days = qnt.stats.get_default_is_period_for_type_calendar_days(data_type)

    log_info("Output directory is:", result_dir)
    os.makedirs(result_dir, exist_ok=True)

    log_info("Rm previous results...")
    for i in os.listdir(result_dir):
        fn = result_dir + "/" + i
        if os.path.isfile(fn):
            log_info("rm:", fn)
            os.remove(fn)

    if dates is None:
        log_info("Prepare test dates...")
        log_info(in_sample_days)
        data = qnt.data.load_data_by_type(data_type, tail=in_sample_days)
        if 'is_liquid' in data.field:
            data = data.where(data.sel(field='is_liquid') > 0).dropna('time', 'all')
        data = data.time
        dates = [data.isel(time=-1).values, data.isel(time=1).values] \
                + [data.isel(time=round(len(data) * (i+1)/(passes-1))).values for i in range(passes-2)]
        dates = list(set(dates))
        dates.sort()
        dates = [pd.Timestamp(i).date() for i in dates]

        del data
    else:
        dates = [qnt.data.common.parse_date(d) for d in dates]

    log_info("Dates:", *(i.isoformat() for i in dates))

    i = 0
    for date in dates:
        try:
            os.remove(fractions_fn)
        except FileNotFoundError:
            pass
        try:
            os.remove(last_data_fn)
        except FileNotFoundError:
            pass
        try:
            os.remove(html_fn)
        except FileNotFoundError:
            pass

        log_info("---")
        i += 1
        log_info("pass:", i, "/", len(dates), "max_date:", date.isoformat())

        if data_type == 'stocks' or data_type == 'stocks_long':
            timeout = 30 * 60
        if data_type == 'futures':
            timeout = 10 * 60
        if data_type == 'crypto' or data_type == 'crypto_futures':
            timeout = 5 * 60

        data_url = urllib.parse.urljoin(urllib.parse.urljoin(qnt.data.common.BASE_URL, 'last/'), date.isoformat()) + "/"
        cmd = "DATA_BASE_URL=" + data_url + " \\\n" + \
              "LAST_DATA_PATH=" + last_data_fn + " \\\n" + \
              "OUTPUT_PATH=" + fractions_fn + " \\\n" + \
              "SUBMISSION_ID=-1\\\n" + \
              "jupyter nbconvert --to html --ExecutePreprocessor.timeout=" + str(timeout)+ " --execute strategy.ipynb --output=" + html_fn  # + \
              # "\\\n 2>&1"
        log_info("cmd:", cmd)
        log_info("output:")
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable='bash')
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            sys.stdout.write(line)
        proc.wait()
        code = proc.returncode
        log_info("return code:", code)

        if not os.path.exists(fractions_fn):
            log_err("ERROR! Output is not found.")
        if not os.path.exists(last_data_fn):
            log_err("ERROR! The strategy does not use all data.")
        if not os.path.exists(html_fn):
            log_err("ERROR! Conversion to html failed.")
        if code != 0:
            log_err("ERROR! Return code != 0.")

        if os.path.exists(fractions_fn):
            log_info("Check the output...")
            output = load_output(fractions_fn, date)

            if data_type == 'stocks' or data_type == 'stocks_long':
                qnt.stats.check_exposure(output)

            log_info("Load data...")
            data = qnt.data.load_data_by_type(data_type, assets=output.asset.values.tolist(),
                                              min_date=str(output.time.min().values)[:10], max_date=date)

            if data_type == 'stocks' or data_type == 'stocks_long':
                non_liquid = qnt.stats.calc_non_liquid(data, output)
                if len(non_liquid.time) > 0:
                    log_err("ERROR! The output contains illiquid positions.")

            missed = qnt.stats.find_missed_dates(output, data)
            if len(missed) > 0:
                log_err("ERROR: some dates are missed in the output.", missed)
            else:
                log_info("There are no missed dates.")

            del data

        try:
            shutil.move(fractions_fn, result_dir + "/" + date.isoformat() + ".fractions.nc.gz")
        except FileNotFoundError:
            pass
        try:
            shutil.move(last_data_fn, result_dir + "/" + date.isoformat() + ".last_data.txt")
        except FileNotFoundError:
            pass
        try:
            shutil.move(html_fn, result_dir + "/" + date.isoformat() + ".strategy.html")
        except FileNotFoundError:
            pass

    log_info("---")
    log_info("Evaluation complete.")


def assemble_output(add_mode='all'):
    log_info("Merge outputs...")
    files = os.listdir(result_dir)
    files = [f for f in files if f.endswith(".fractions.nc.gz")]
    files.sort()
    output = None

    if len(files) == 0:
        log_err("ERROR! There are no outputs.")

    for f in files:
        date = f.split(".")[0]
        date = datetime.date.fromisoformat(date)
        fn = result_dir + "/" + f
        _output = load_output(fn, date)
        _output = _output.where(_output.time <= np.datetime64(date)).dropna('time', 'all')
        if len(_output) == 0:
            continue
        if output is None:
            log_info("init output:", fn, str(_output.time.min().values)[:10], str(_output.time.max().values)[:10])
            output = _output
        else:
            if add_mode == 'all':
                _output = _output.where(_output.time > output.time.max()).dropna('time', 'all')
            elif add_mode == 'one':
                _output = _output.where(_output.time == np.datetime64(date)).dropna('time', 'all')
            else:
                raise Exception("wrong add_mode")
            if len(_output) == 0:
                continue
            log_info("add output:", fn, str(_output.time.min().values)[:10], str(_output.time.max().values)[:10])
            output = xr.concat([output, _output], dim="time")
    return output


def load_output(fn, date):
    output = xr.open_dataarray(fn, cache=False)
    output = output.compute()
    if 'time' not in output.coords:
        log_info('append dimension')
        output = xr.concat([output], pd.DatetimeIndex([date], name='time'))
    output.coords['asset'] = [idt.translate_server_id_to_user_id(id) for id in output.asset.values]
    return output


def check_output(output, data_type='stocks'):
    if data_type != 'stocks' and data_type != 'futures' and data_type != 'crypto':
        log_err("Unsupported data_type", data_type)
        return

    in_sample_days = qnt.stats.get_default_is_period_for_type_calendar_days(data_type)
    in_sample_points = qnt.stats.get_default_is_period_for_type(data_type)

    min_date = np.datetime64(datetime.date.today() - datetime.timedelta(days=in_sample_days))
    output_tail = output.where(output.time > min_date).dropna('time', 'all')
    if len(output_tail) < in_sample_points:
        log_err("ERROR! In sample period does not contain enough points. " +
                str(len(output_tail)) + " < " + str(in_sample_points))
    else:
        log_info("Ok. In sample period contains enough points." + str(len(output_tail)) + " >= " + str(in_sample_points))

    log_info()

    log_info("Load data...")
    data = qnt.data.load_data_by_type(data_type, assets=output.asset.values.tolist(), tail=in_sample_days + 60)

    log_info()

    qnt.output.check(output, data)

