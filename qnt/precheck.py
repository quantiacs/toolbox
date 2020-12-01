import xarray as xr
import numpy as np
import pandas as pd
import os
import datetime
import qnt.data.common
import qnt.data
import urllib.parse
import shutil
import sys
import subprocess
import io
import qnt.stats
from .data import id_translation as idt

fractions_fn = "../fractions.nc.gz"
last_data_fn = "../last_data.txt"
html_fn = "../strategy.html"
result_dir = "precheck_results"


def run_init():
    if os.path.exists("init.ipynb"):
        print("Run init.ipynb..")
        cmd = "jupyter nbconvert --to html --ExecutePreprocessor.timeout=1800 --execute init.ipynb --stdout "  + \
              "| html2text -utf8"
        # "\\\n 2>&1"
        print("cmd:", cmd)
        print("output:")
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable='bash')
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            sys.stdout.write(line)
        proc.wait()
        code = proc.returncode
        print("return code:", code)


def evaluate_passes(data_type='stocks', passes=3, dates=None):
    if data_type == 'stocks' or data_type == 'futures':
        in_sample_days = (5 * 366 + 183)
    elif data_type == 'crypto':
        in_sample_days = 7*366
    else:
        print("Unsupported data_type", data_type, file=sys.stderr, flush=True)
        return

    print("Output directory is:", result_dir)
    os.makedirs(result_dir, exist_ok=True)

    print("Rm previous results...")
    for i in os.listdir(result_dir):
        fn = result_dir + "/" + i
        if os.path.isfile(fn):
            print("rm:", fn)
            os.remove(fn)

    if dates is None:
        print("Prepare test dates...")
        print(in_sample_days)
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

    print("Dates:", *(i.isoformat() for i in dates))

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

        print("---")
        i += 1
        print("pass:", i, "/", len(dates), "max_date:", date.isoformat())

        if data_type == 'stocks':
            timeout = 30 * 60
        if data_type == 'futures':
            timeout = 10 * 60
        if data_type == 'crypto':
            timeout = 5 * 60

        data_url = urllib.parse.urljoin(urllib.parse.urljoin(qnt.data.common.BASE_URL, 'last/'), date.isoformat()) + "/"
        cmd = "DATA_BASE_URL=" + data_url + " \\\n" + \
              "LAST_DATA_PATH=" + last_data_fn + " \\\n" + \
              "OUTPUT_PATH=" + fractions_fn + " \\\n" + \
              "jupyter nbconvert --to html --ExecutePreprocessor.timeout=" + str(timeout)+ " --execute strategy.ipynb --output=" + html_fn  # + \
              # "\\\n 2>&1"
        print("cmd:", cmd)
        print("output:")
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable='bash')
        for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
            sys.stdout.write(line)
        proc.wait()
        code = proc.returncode
        print("return code:", code)

        if not os.path.exists(fractions_fn):
            print("ERROR! Output is not found.", file=sys.stderr, flush=True)
        if not os.path.exists(last_data_fn):
            print("ERROR! The strategy does not use all data.", file=sys.stderr, flush=True)
        if not os.path.exists(html_fn):
            print("ERROR! Conversion to html failed.", file=sys.stderr, flush=True)
        if code != 0:
            print("ERROR! Return code != 0.", file=sys.stderr, flush=True)

        if os.path.exists(fractions_fn):
            print("Check the output...")
            output = load_output(fractions_fn, date)

            if data_type == 'stocks':
                qnt.stats.check_exposure(output)

            print("Load data...")
            data = qnt.data.load_data_by_type(data_type, assets=output.asset.values.tolist(),
                                              min_date=str(output.time.min().values)[:10], max_date=date)

            if data_type == 'stocks':
                non_liquid = qnt.stats.calc_non_liquid(data, output)
                if len(non_liquid.time) > 0:
                    print("ERROR! The output contains illiquid positions.", file=sys.stderr, flush=True)

            missed = qnt.stats.find_missed_dates(output, data)
            if len(missed) > 0:
                print("ERROR: some dates are missed in the output.", missed, file=sys.stderr, flush=True)
            else:
                print("There are no missed dates.")

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

    print("---")
    print("Evaluation complete.")


def assemble_output(add_mode='all'):
    print("Merge outputs...")
    files = os.listdir(result_dir)
    files = [f for f in files if f.endswith(".fractions.nc.gz")]
    files.sort()
    output = None

    if len(files) == 0:
        print("ERROR! There are no outputs.", file=sys.stderr, flush=True)

    for f in files:
        date = f.split(".")[0]
        date = datetime.date.fromisoformat(date)
        fn = result_dir + "/" + f
        _output = load_output(fn, date)
        _output = _output.where(_output.time <= np.datetime64(date)).dropna('time', 'all')
        if len(_output) == 0:
            continue
        if output is None:
            print("init output:", fn, str(_output.time.min().values)[:10], str(_output.time.max().values)[:10])
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
            print("add output:", fn, str(_output.time.min().values)[:10], str(_output.time.max().values)[:10])
            output = xr.concat([output, _output], dim="time")
    return output


def load_output(fn, date):
    output = xr.open_dataarray(fn, cache=False)
    output = output.compute()
    if 'time' not in output.coords:
        print('append dimension')
        output = xr.concat([output], pd.DatetimeIndex([date], name='time'))
    output.coords['asset'] = [idt.translate_server_id_to_user_id(id) for id in output.asset.values]
    return output


def check_output(output, data_type='stocks'):
    if data_type != 'stocks' and data_type != 'futures' and data_type != 'crypto':
        print("Unsupported data_type", data_type, file=sys.stderr, flush=True)
        return

    if data_type == 'stocks' or data_type == 'futures':
        in_sample_days = 5 * 366 + 183
        in_sample_points = 5 * 252
    elif data_type == 'crypto':
        in_sample_days = 7 * 366 + 183
        in_sample_points = 60000

    min_date = np.datetime64(datetime.date.today() - datetime.timedelta(days=in_sample_days))
    output_tail = output.where(output.time > min_date).dropna('time', 'all')
    if len(output_tail) < in_sample_points:
        print("ERROR! In sample period does not contain enough points. " +
              str(len(output_tail)) + " < " + str(in_sample_points), file=sys.stderr, flush=True)
    else:
        print("Ok. In sample period contains enough points." + str(len(output_tail)) + " >= " + str(in_sample_points))

    print()

    print("Load data...")
    data = qnt.data.load_data_by_type(data_type, assets=output.asset.values.tolist(), tail=in_sample_days + 31)

    print()

    stat = qnt.stats.calc_stat(data, output)

    sr = stat.sel(field="sharpe_ratio").isel(time=-1).values
    if sr < 1:
        print("ERROR! In sample sharpe ratio is too low. " + str(sr) + " < 1", file=sys.stderr, flush=True)
    else:
        print("Ok. In sample sharpe ratio is enough. " + str(sr) + " >= 1")

    print()

    qnt.stats.check_correlation(output, data)

