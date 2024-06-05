import typing as tp
import xarray as xr
import pandas as pd
import numpy as np
import os, sys
import inspect
import copy

import datetime
import progressbar

import qnt.output as qnout
import qnt.state as qnstate
import qnt.data as qndata
import qnt.data.common as qndc
import qnt.stats as qnstat
from qnt.graph import is_notebook, make_major_plots, is_interact
from qnt.log import log_info, log_err


DataSet = tp.Union[xr.DataArray,dict]


def backtest_ml(
        *,
        train: tp.Callable[[DataSet], tp.Any],
        predict: tp.Union[
            tp.Callable[[tp.Any, DataSet], xr.DataArray],
            tp.Callable[[tp.Any, DataSet, tp.Any], tp.Tuple[xr.DataArray, tp.Any]],
        ],

        train_period: int = 4*365,
        retrain_interval: int = 365,
        predict_each_day: bool = False,
        retrain_interval_after_submit: tp.Union[int, None] = None,

        competition_type: str,
        load_data: tp.Union[tp.Callable[[int], tp.Union[DataSet,tp.Tuple[DataSet,np.ndarray]]],None] = None,
        lookback_period: int = 365,
        test_period: int = 365*15,
        start_date: tp.Union[np.datetime64, str, datetime.datetime, datetime.date, None] = None,
        end_date: tp.Union[np.datetime64, str, datetime.datetime, datetime.date, None] = None,
        window: tp.Union[tp.Callable[[DataSet,np.datetime64,int], DataSet], None] = None,
        analyze: bool = True,
        build_plots: bool = True,
        collect_all_states: bool = False,
        check_correlation: bool = False,
    ):
    """

    :param train: creates and trains model for prediction
    :param predict: predicts price movements and generates outputs
    :param train_period: the data length in trading days for training
    :param retrain_interval: how often to retrain the model(in calendar days)
    :param predict_each_day: perform predict for every day. Set True if you suspect the looking forward
    :param retrain_interval_after_submit:
    :param competition_type: "futures" | "stocks" | "cryptofutures" | "stocks_long" | "crypto" | "crypto_daily"
    :param load_data: data load function, accepts tail arg, returns time series and data
    :param lookback_period: the minimal calendar days period for one prediction
    :param test_period:  test period (calendar days)
    :param start_date: start date for backtesting, overrides test period
    :param end_date: end date for backtesting, by default - now
    :param window: function which isolates data for one prediction or training
    :param analyze: analyze the output and calc stats
    :param build_plots: build plots (require analyze=True)
    :param collect_all_states: collect all states instead of the last one
    :param check_correlation: correlation check
    :return:
    """
    qndc.track_event("ML_BACKTEST")

    if load_data is None:
        load_data = lambda tail: qndata.load_data_by_type(competition_type, tail=tail)

    if window is None:
        window = standard_window
    def copy_window(data, dt, tail):
        return copy.deepcopy(window(data, dt, tail))

    args_count = len(inspect.getfullargspec(predict).args)
    predict_wrap = (lambda m, d, s: predict(m, d)) if args_count < 3 else predict

    log_info("Run the last iteration...")

    data = load_data(max(train_period, lookback_period))
    data, data_ts = extract_time_series(data)

    retrain_interval_cur = retrain_interval_after_submit if is_submitted() else retrain_interval
    if retrain_interval_cur is None:
        retrain_interval_cur = retrain_interval
    created = None
    model = None
    state = None
    if is_submitted() and (args_count > 2 or retrain_interval_cur > 1):
        state = qnstate.read()
        if state is not None:
            created = state[0]
            model = state[1]
            state = state[2]
    need_retrain = model is None or retrain_interval_cur == 1 \
                   or data_ts[-1] >= created + np.timedelta64(retrain_interval_cur, 'D')
    if need_retrain:
        train_data_slice = copy_window(data, data_ts[-1], train_period)
        model = train(train_data_slice)
        created = data_ts[-1]

    test_data_slice = copy_window(data, data_ts[-1], lookback_period)
    output = predict_wrap(model, test_data_slice, state)
    output, state = unpack_result(output)

    if data_ts[-1] in output.time:
        result = output.sel(time=[data_ts[-1]])

    data = qndata.load_data_by_type(competition_type, assets=result.asset.values.tolist(), tail=60)
    result = qnout.clean(result, data, competition_type)

    result.name = competition_type
    qnout.write(result)

    if need_retrain and retrain_interval_cur > 1 or state is not None:
        qnstate.write((created, model, state))

    if is_submitted():
        if state is not None:
            return output, [state] if collect_all_states else state
        else:
            return output

    try:
        print("---")
        qndc.set_max_datetime(end_date)

        last_date = np.datetime64(qndc.parse_date(datetime.date.today()))
        if start_date is None:
            start_date = last_date - np.timedelta64(test_period-1, 'D')
        else:
            start_date = pd.Timestamp(start_date).to_datetime64()
            test_period = (last_date - start_date) // np.timedelta64(1, 'D')

        # ---
        log_info("Run First Iteration...") # to catch most errors
        qndc.set_max_datetime(start_date)
        data = load_data(max(train_period, lookback_period))
        data, data_ts = extract_time_series(data)

        train_data_slice = copy_window(data, data_ts[-1], train_period)
        model = train(train_data_slice)

        test_data_slice = copy_window(data, data_ts[-1], lookback_period)
        output = predict_wrap(model, test_data_slice, state)
        output, state = unpack_result(output)

        # ---
        print("---")
        qndc.set_max_datetime(end_date)
        log_info("Run all iterations...")
        log_info('Load data...')

        train_data = load_data(test_period + train_period + lookback_period)
        train_data, train_ts = extract_time_series(train_data)

        test_data = load_data(test_period)
        test_ts = extract_time_series(test_data)[1]

        log_info('Backtest...')
        outputs = []
        t = test_ts[0]
        state = None
        model = None
        states = []
        with progressbar.ProgressBar(max_value=len(test_ts), poll_interval=1) as p:
            go = True
            while go:
                end_t = t + np.timedelta64(max(retrain_interval - 1, 0), 'D')
                end_t = test_ts[test_ts <= end_t][-1]

                train_data_slice = copy_window(train_data, t, train_period)
                # print("train model t <=", str(t)[:10])
                model = train(train_data_slice)
                # print("predict", str(t)[:10], "<= t <=", str(end_t)[:10])
                if predict_each_day:
                    for test_t in test_ts[np.logical_and(test_ts >= t, test_ts <= end_t)]:
                        test_data_slice = copy_window(train_data, test_t, lookback_period)
                        output = predict_wrap(model, test_data_slice, state)
                        output, state = unpack_result(output)
                        if collect_all_states:
                            states.append(state)
                        if test_t in output.time:
                            output = output.sel(time=[test_t])
                            outputs.append(output)
                            p.update(np.where(test_ts == test_t)[0].item())
                else:
                    test_data_slice = copy_window(train_data, end_t, lookback_period + retrain_interval)
                    output = predict_wrap(model, test_data_slice, state)
                    output, state = unpack_result(output)
                    if collect_all_states:
                        states.append(state)
                    output = output.where(output.time >= t).where(output.time <= end_t).dropna('time', 'all')
                    outputs.append(output)

                p.update(np.where(test_ts == end_t)[0].item())

                next_t = test_ts[test_ts > end_t]
                if len(next_t) > 0:
                    t = next_t[0]
                else:
                    go = False

            result = xr.concat(outputs, dim='time')
            min_date = test_ts[0] - np.timedelta64(60, 'D')
            data = qndata.load_data_by_type(competition_type, min_date=str(min_date)[:10])
            result = qnout.clean(result, data, competition_type)
            result.name = competition_type
            qnout.write(result)
            qnstate.write((t, model, state))
            if analyze:
                log_info("---")
                analyze_results(output=result, data=data, kind=competition_type, build_plots=build_plots,
                                start=start_date,
                                check_correlation=check_correlation)

            if state is None:
                return result
            elif collect_all_states:
                return result, states
            else:
                return result, state
    finally:
        qndc.set_max_datetime(None)


def backtest(
        *,
        competition_type: str,
        strategy:  tp.Union[
            tp.Callable[[DataSet], xr.DataArray],
            tp.Callable[[DataSet, tp.Any], tp.Tuple[xr.DataArray, tp.Any]],
        ],
        load_data: tp.Union[tp.Callable[[int], tp.Union[DataSet,tp.Tuple[DataSet,np.ndarray]]],None] = None,
        lookback_period: int = 365,
        test_period: int = 365*15,
        start_date: tp.Union[np.datetime64, str, datetime.datetime, datetime.date, None] = None,
        end_date: tp.Union[np.datetime64, str, datetime.datetime, datetime.date, None] = None,
        window: tp.Union[tp.Callable[[DataSet,np.datetime64,int], DataSet], None] = None,
        step: int = 1,
        analyze: bool = True,
        build_plots: bool = True,
        collect_all_states: bool = False,
        check_correlation: bool = False,
    ):
    """

    :param competition_type: "futures" | "stocks" | "cryptofutures" | "stocks_long" | "crypto"
    :param load_data: data load function, accepts tail arg, returns time series and data
    :param lookback_period: calendar days period for one iteration
    :param strategy: accepts data, returns weights distribution for the last day
    :param test_period: test period (calendar days)
    :param start_date: start date for backtesting, overrides test period
    :param end_date: end date for backtesting, by default - now
    :param step: step size
    :param window: function which isolates data for one iteration
    :param analyze: analyze the output and calc stats
    :param build_plots: build plots (require analyze=True)
    :param collect_all_states: collect all states instead of the last one
    :param check_correlation: correlation check
    :return:
    """
    qndc.track_event("BACKTEST")

    if window is None:
        window = standard_window

    if load_data is None:
        load_data = lambda tail: qndata.load_data_by_type(competition_type, tail=tail)

    args_count = len(inspect.getfullargspec(strategy).args)
    strategy_wrap = (lambda d, s: strategy(d)) if args_count < 2 else strategy

    # ---
    log_info("Run last pass...")
    log_info("Load data...")
    if is_single_pass():
        checking_start_data_date = pd.to_datetime(
            qnstat.get_default_is_start_date_for_type(competition_type)) - datetime.timedelta(days=lookback_period)
        current_date = pd.to_datetime('today')
        data_days_difference = (current_date - checking_start_data_date).days
        data = load_data(data_days_difference)
        try:
            if data.name == 'stocks' and competition_type != 'stocks' and competition_type != 'stocks_long' \
                    or data.name == 'stocks_nasdaq100' and competition_type != 'stocks_nasdaq100' \
                    or data.name == 'cryptofutures' and competition_type != 'cryptofutures' and competition_type != 'crypto_futures' \
                    or data.name == 'crypto' and competition_type != 'crypto' \
                    or data.name == 'futures' and competition_type != 'futures':
                log_err("WARNING! The data type and the competition type are mismatch.")
        except:
            pass
        data, time_series = extract_time_series(data)

        log_info("Run strategy...")
        state = None
        if is_submitted() and args_count > 1:
            state = qnstate.read()
        result = strategy_wrap(data, state)
        result, state = unpack_result(result)

        log_info("Load data for cleanup...")
        data = qndata.load_data_by_type(competition_type, assets=result.asset.values.tolist(), min_date=(
                pd.to_datetime(qnstat.get_default_is_start_date_for_type(competition_type)) - datetime.timedelta(
            days=60)))
    else:
        data = load_data(lookback_period)
        try:
            if data.name == 'stocks' and competition_type != 'stocks' and competition_type != 'stocks_long' \
                    or data.name == 'stocks_nasdaq100' and competition_type != 'stocks_nasdaq100' \
                    or data.name == 'cryptofutures' and competition_type != 'cryptofutures' and competition_type != 'crypto_futures' \
                    or data.name == 'crypto' and competition_type != 'crypto' \
                    or data.name == 'futures' and competition_type != 'futures':
                log_err("WARNING! The data type and the competition type are mismatch.")
        except:
            pass
        data, time_series = extract_time_series(data)

        log_info("Run strategy...")
        state = None
        if is_submitted() and args_count > 1 and not should_run_iterations():
            state = qnstate.read()
        result = strategy_wrap(data, state)
        result, state = unpack_result(result)

        log_info("Load data for cleanup...")
        data = qndata.load_data_by_type(competition_type, assets=result.asset.values.tolist(), tail=60)

    result = qnout.clean(result, data)
    result.name = competition_type
    log_info("Write result...")
    qnout.write(result)
    qnstate.write(state)

    if is_submitted() and not should_run_iterations():
        if args_count > 1:
            return result, [state] if collect_all_states else state
        else:
            return result
    # ---

    log_info("---")
    try:
        qndc.set_max_datetime(end_date)
        last_date = np.datetime64(qndc.parse_date(datetime.date.today()))
        if start_date is None:
            start_date = last_date - np.timedelta64(test_period-1, 'D')
        else:
            start_date = pd.Timestamp(start_date).to_datetime64()
            test_period = (last_date - start_date) // np.timedelta64(1, 'D')

        # ---

        log_info("Run first pass...")
        qndc.set_max_datetime(start_date)

        print("Load data...")
        data = load_data(lookback_period)
        data, time_series = extract_time_series(data)
        print("Run strategy...")
        result = strategy_wrap(data, None)
        result, state = unpack_result(result)
        log_info("---")

        qndc.set_max_datetime(end_date)

        log_info("Load full data...")
        data = load_data(test_period + lookback_period)
        data, time_series = extract_time_series(data)
        if len(time_series) < 1:
            log_err("Time series is empty")
            return

        # ---

        log_info("---")
        result, state = run_iterations(time_series, data, window, start_date, lookback_period, strategy_wrap, step, collect_all_states)
        if result is None:
            return

        log_info("Load data for cleanup and analysis...")
        min_date = time_series[0] - np.timedelta64(60, 'D')
        data = qndata.load_data_by_type(competition_type, min_date=str(min_date)[:10])
        result = qnout.clean(result, data, competition_type)
        result.name = competition_type
        log_info("Write result...")
        qnout.write(result)
        qnstate.write(state)

        if analyze:
            log_info("---")
            analyze_results(output=result, data=data, kind=competition_type, build_plots=build_plots,
                            start=start_date,
                            check_correlation=check_correlation)

        if args_count > 1:
            return result, state
        else:
            return result
    finally:
        qndc.set_max_datetime(None)


def run_iterations(time_series, data, window, start_date, lookback_period, strategy, step, collect_all_states):
    def copy_window(data, dt, tail):
        return copy.deepcopy(window(data, dt, tail))

    log_info("Run iterations...\n")

    ts = np.sort(time_series)
    outputs = []
    all_states = []

    output_time_coord = ts[ts >= start_date]
    output_time_coord = output_time_coord[::step]

    i = 0

    sys.stdout.flush()

    with progressbar.ProgressBar(max_value=len(output_time_coord), poll_interval=1) as p:
        state = None
        for t in output_time_coord:
            tail = copy_window(data, t, lookback_period)
            result = strategy(tail, copy.deepcopy(state))
            output, state = unpack_result(result)
            if type(output) != xr.DataArray:
                log_err("Output is not xarray!")
                return
            if set(output.dims) != {'asset'} and set(output.dims) != {'asset', 'time'}:
                log_err("Wrong output dimensions. ", output.dims, "Should contain only:", {'asset', 'time'})
                return
            if 'time' in output.dims:
                output = output.sel(time=t)
            output = output.drop(['field', 'time'], errors='ignore')
            outputs.append(output)
            if collect_all_states:
                all_states.append(state)
            i += 1
            p.update(i)

    sys.stderr.flush()

    log_info("Merge outputs...")
    output = xr.concat(outputs, pd.Index(output_time_coord, name=qndata.ds.TIME))

    return output, all_states if collect_all_states else state


def standard_window(data, max_date: np.datetime64, lookback_period:int):
    min_date = max_date - np.timedelta64(lookback_period,'D')
    return data.loc[dict(time=slice(min_date, max_date))]


def extract_time_series(data):
    if type(data) == tuple:
        return data
    else:
        return data, data.time.values


def is_submitted():
    return os.environ.get("SUBMISSION_ID", "") != ""


def is_single_pass():
    return os.environ.get("IS_SINGLE_PASS", "") != ""


def should_run_iterations():
    return os.environ.get("RUN_ITERATIONS", "") != ""


def unpack_result(result):
    state = None
    if type(result) == tuple:
        if len(result) > 1:
            state = result[1]
        if len(result) == 0:
            log_err("ERROR! The result tuple is empty.")
        result = result[0]
    if result is None:
        log_err("ERROR! Strategy output is None!")
    return result, state


def analyze_results(output, data, kind, build_plots, start=None, check_correlation=True):
    log_info("Analyze results...")

    if len(output.time) == 0 or len(output.asset) == 0:
        log_err("ERROR! Output is empty!")
        return

    log_info("Check...")
    qnout.check(output=output, data=data, kind=kind, check_correlation=check_correlation)
    log_info("---")
    log_info("Align...")
    output = qnout.align(output, data, start)
    log_info("Calc global stats...")
    stat_global = qnstat.calc_stat(data, output)
    stat_global = stat_global.loc[output.time[0]:]
    if not build_plots:
        log_info(stat_global.to_pandas().tail())
        return
    log_info("---")
    log_info("Calc stats per asset...")
    stat_per_asset = qnstat.calc_stat(data, output, per_asset=True)
    stat_per_asset = stat_per_asset.loc[output.time.values[0]:]

    if is_notebook():
        build_plots_jupyter(output, stat_global, stat_per_asset)
    else:
        build_plots_dash(output, stat_global, stat_per_asset)


def build_plots_dash(output, stat_global, stat_per_asset):
    log_info("---")
    print("Build plots...")
    try:
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        import pandas as pd
        import plotly.express as px
        import dash_table
        import qnt.graph as qngraph

        from dash.dependencies import Input, Output

        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.Label("Select the asset (or leave blank to display the overall stats):"),
            dcc.Dropdown(
                id='assets_dropdown',
                options=[{'label': '', 'value': ''}] + [{'label': i, 'value': i} for i in output.asset.values.tolist()],
                value=''
            ),
            html.H3("Output:"),
            html.Label('Row offset:'),
            html.Div(dcc.Slider(
                id='output_row_slider',
                min=0,
                max=max(0, len(output.time) - 10),
                value=max(0, len(output.time) - 10),
                step=1
            ), style={'width': '100%'}),
            html.Label('Column offset:'),
            html.Div(dcc.Slider(
                id='output_column_slider',
                min=0,
                max=max(0, len(output.asset) - 10),
                value=0,
                step=1
            ), style={'width': '100%'}),
            dash_table.DataTable(
                id='output_table',
            ),

            html.H3("Stats:"),
            html.Label('Row offset:'),
            html.Div(dcc.Slider(
                id='stats_row_slider',
                min=0,
                max=max(0, len(stat_global.time) - 10),
                value=max(0, len(stat_global.time) - 10),
                step=1
            ), style={'width': '100%'}),
            dash_table.DataTable(
                id='stats_table',
            ),

            html.H3("Equity (PnL)"),
            dcc.Graph(id='equity_chart'),

            html.H3("Underwater"),
            dcc.Graph(id='underwater_chart'),

            html.H3("Sharpe Ratio"),
            dcc.Graph(id='sharpe_ratio_chart'),

            html.H3("Bias"),
            dcc.Graph(id='bias_chart')
        ])

        @app.callback(
            [
                Output('equity_chart', 'figure'),
                Output('underwater_chart', 'figure'),
                Output('sharpe_ratio_chart', 'figure'),
                Output('bias_chart', 'figure'),
                Output('output_column_slider', 'max'),
            ],
            [Input('assets_dropdown', 'value')])
        def asset_changed(asset):
            if asset in stat_per_asset.asset.values.tolist():
                stat = stat_per_asset.sel(asset=asset)
                max_col_offset = 0
            else:
                stat = stat_global
                max_col_offset = max(0, len(output.asset) - 10)

            plots = make_major_plots(stat)

            return plots + (max_col_offset,)

        @app.callback(
            [Output('output_table', 'columns'), Output("output_table", "data")],
            [Input('assets_dropdown', 'value'),Input("output_row_slider", "value"), Input("output_column_slider", "value")]
        )
        def update_output_table(asset,row_offset, column_offset):
            out = output
            out = out[row_offset:row_offset+10,:]
            if asset in out.asset.values.tolist():
                out = out.loc[:,[asset]]
            else:
                out = out[:,column_offset:column_offset+10]

            cols = [{"name": "time", "id": "time"}] + [
                {"name": i, "id": i} for i in sorted(out.asset[column_offset:column_offset+10].values.tolist())
            ]
            data = out.to_pandas().reset_index().to_dict('records')
            for i in data:
                for k in i.keys():
                    i[k] = "%.6f" % i[k] if k != 'time' else i[k].isoformat()[:13]
            return cols, data

        @app.callback(
            [Output('stats_table', 'columns'), Output("stats_table", "data")],
            [Input('assets_dropdown', 'value'),Input("stats_row_slider", "value")]
        )
        def update_stats_table(asset, row_offset):
            if asset in stat_per_asset.asset.values.tolist():
                stat = stat_per_asset.sel(asset=asset)
            else:
                stat = stat_global
            cols = [{"name": "time", "id": "time"}] + [
                {"name": i, "id": i} for i in sorted(stat.field.values.tolist())
            ]
            data = stat[row_offset:row_offset+10].to_pandas().reset_index().to_dict('records')
            for i in data:
                for k in i.keys():
                    i[k] = "%.6f" % i[k] if k != 'time' else i[k].isoformat()[:13]
            return cols, data

        log_info("Run Dash... Open the link below in your browser.")

        #opens browser
        import webbrowser, os
        from threading import Timer
        Timer(1, lambda: webbrowser.open("http://127.0.0.1:"+os.environ.get('PORT', '8050'), new=2)).start()

        app.run_server(dev_tools_hot_reload=False)
    except:
        import logging
        logging.exception("can't start dash")


def build_plots_jupyter(output, stat_global, stat_per_asset):
    print("Build plots...")
    log_info("---")

    def display_scrollable_output_table(output):
        output = output.to_pandas()
        tail_r=10
        tail_c=10

        def show_table(row_offset, column_offset):
            try:
                from IPython.display import display
                display(output.iloc[row_offset:row_offset + tail_r, column_offset:column_offset + tail_c])
            except:
                log_info(output.iloc[row_offset:row_offset + tail_r, column_offset:column_offset + tail_c])

        if is_interact():
            try:
                from ipywidgets import interact, interactive, fixed, interact_manual, Layout, IntSlider
                import ipywidgets as widgets

                interact(show_table,
                         row_offset=IntSlider(
                             max(0, len(output) - tail_r), 0, max(0, len(output) - tail_r), 1,
                             layout=Layout(width='90%')
                         ),
                         column_offset=IntSlider(
                             0, 0, max(0, len(output.columns) - tail_c), 1,
                             layout=Layout(width='90%')
                         )
                         )
            except:
                show_table(len(output) - tail_r, 0)
        else:
            show_table(len(output) - tail_r, 0)

    def display_scrollable_stats_table(stat):
        stat = stat.to_pandas()
        tail=10
        def show_table(offset):
            try:
                from IPython.display import display
                display(stat[offset:offset+tail])
            except:
                log_info(stat[offset:offset + tail])
        if is_interact():
            try:
                from ipywidgets import interact, interactive, fixed, interact_manual, Layout, IntSlider
                import ipywidgets as widgets

                interact(show_table,
                         offset=IntSlider(
                             max(0, len(stat)-tail), 0, max(0, len(stat)-tail), 1,
                             layout=Layout(width='90%')
                         )
                    )
            except:
                show_table(len(stat)-tail)
        else:
            show_table(len(stat)-tail)

    def show_asset_stat(asset):
        if asset in stat_per_asset.asset.values.tolist():
            out = output.sel(asset=[asset])
            stat = stat_per_asset.sel(asset=asset)
        else:
            out = output
            stat = stat_global
        log_info("Output:")
        display_scrollable_output_table(out)
        log_info("Stats:")
        display_scrollable_stats_table(stat)
        make_major_plots(stat)
        log_info("---")

    if is_interact():
        try:
            from ipywidgets import interact, interactive, fixed, interact_manual
            import ipywidgets as widgets

            log_info("Select the asset (or leave blank to display the overall stats):")
            interact(show_asset_stat, asset=widgets.Combobox(options=[''] + output.asset.values.tolist()))
        except:
            show_asset_stat('')
    else:
        show_asset_stat('')


if __name__ == '__main__':
    import qnt.ta as qnta


    def load_data(period):
        data = qndata.futures_load_data(tail=period)
        return data

    def strategy(data):
        close = data.sel(field='close')
        sma200 = qnta.sma(close, 200).isel(time=-1)
        sma20 = qnta.sma(close, 20).isel(time=-1)
        return xr.where(sma200 < sma20, 1, -1)

    backtest(
        competition_type="futures",
        load_data=load_data,
        lookback_period=365,
        test_period=2*365,
        strategy=strategy
    )

    #
    # def load_data(period):
    #     futures = qndata.futures_load_data(tail=period)
    #     crypto = qndata.crypto_load_data(tail=period)
    #     #     display(futures)
    #     #     display(crypto)
    #     return {"futures": futures, "crypto": crypto}, futures.time.values
    #
    # def window(data, max_date: np.datetime64, lookback_period: int):
    #     min_date = max_date - np.timedelta64(lookback_period, 'D')
    #     return {
    #         "futures": data['futures'].sel(time=slice(min_date, max_date)),
    #         "crypto": data['crypto'].sel(time=slice(min_date, max_date)),
    #     }
    #
    # def strategy(data):
    #     close = data['futures'].sel(field='close')
    #
    #     close_last = data['futures'].sel(field='close').shift(time=1)
    #     close_change = close - close_last
    #     #     display(close)
    #
    #     close_crypto = data['crypto'].sel(field='close')
    #     window_biffer = data
    #
    #     close_last_crypto = data['crypto'].sel(field='close').shift(time=1)
    #     close_change_crypto = close_crypto - close_last_crypto
    #     #     display(close_crypto)
    #
    #     # .isel(time=-1) get the last day
    #     sma200 = qnta.sma(close_change, 20).isel(time=-1)
    #     sma200_crypto = qnta.sma(close_change_crypto, 20).isel(time=-1).sel(asset='BTC')
    #     return xr.where(sma200 > sma200_crypto, 1, -1)
    #
    #
    # backtest(
    #     competition_type="futures",
    #     load_data=load_data,
    #     lookback_period=1 * 365,
    #     test_period=1 * 365,
    #     strategy=strategy,
    #     window=window
    # )
