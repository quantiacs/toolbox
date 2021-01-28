import typing as tp
import xarray as xr
import pandas as pd
import numpy as np
import os, sys

import datetime
import progressbar

import qnt.output as qnout
import qnt.data as qndata
import qnt.data.common as qndc
import qnt.stats as qnstat
from qnt.graph import is_notebook, make_major_plots
from qnt.log import log_info, log_err


DataSet = tp.Union[xr.DataArray,dict]


def backtest(*,
             competition_type: str,
             load_data: tp.Callable[[int], tp.Union[DataSet,tp.Tuple[DataSet,np.ndarray]]],
             strategy: tp.Callable[[DataSet], xr.DataArray],
             lookback_period: int = 365,
             test_period: int = 365*15,
             start_date: tp.Union[np.datetime64, str, datetime.datetime, datetime.date, None] = None,
             window: tp.Union[tp.Callable[[DataSet,np.datetime64,int], DataSet], None] = None,
             step: int = 1,
             analyze: bool = True,
             build_plots: bool = True,
             ):
    """

    :param competition_type: "futures" | "stocks" | "cryptofutures" | "stocks_long" | "crypto"
    :param load_data: data load function, accepts tail arg, returns time series and data
    :param lookback_period: calendar days period for one iteration
    :param strategy: accepts data, returns weights distribution for the last day
    :param test_period: test period (calendar days)
    :param start_date: start date for backtesting, overrides test period
    :param step: step size
    :param window: function which isolates data for one iterations
    :param analyze: analyze the output and calc stats
    :param build_plots: build plots (require analyze=True)
    :return:
    """
    qndc.track_event("BACKTEST")
    if window is None:
        window = standard_window

    log_info("Run last pass...")
    print("Load data...")
    data = load_data(lookback_period)
    data, time_series = extract_time_series(data)
    print("Run pass...")
    result = strategy(data)
    if result is None:
        log_err("ERROR! Strategy output is None!")
    else:
        log_info("Ok.")

    if is_submitted():
        log_info("Load data for cleanup...")
        data = qndata.load_data_by_type(competition_type, assets=result.asset.values.tolist(), tail=60)
        result = qnout.clean(result, data)
        result.name = competition_type
        qnout.write(result)
        return result

    log_info("---")

    if start_date is None:
        start_date = pd.Timestamp.today().to_datetime64() - np.timedelta64(test_period-1, 'D')
    else:
        start_date = pd.Timestamp(start_date).to_datetime64()
        test_period = (pd.Timestamp.today().to_datetime64() - start_date) / np.timedelta64(1, 'D')

    log_info("Run first pass...")
    try:
        qndc.MAX_DATETIME_LIMIT = pd.Timestamp(start_date).to_pydatetime()
        qndc.MAX_DATE_LIMIT = qndc.MAX_DATETIME_LIMIT.date()
        print("Load data...")
        data = load_data(lookback_period)
        data, time_series = extract_time_series(data)
        print("Run pass...")
        result = strategy(data)
        if result is None:
            log_err("ERROR! Strategy output is None!")
        else:
            log_info("Ok.")
    finally:
        qndc.MAX_DATE_LIMIT = None
        qndc.MAX_DATETIME_LIMIT = None

    log_info("---")

    log_info("Load full data...")
    data = load_data(test_period + lookback_period)
    data, time_series = extract_time_series(data)
    if len(time_series) < 1:
        log_err("Time series is empty")
        return

    log_info("---")
    result = run_iterations(time_series, data, window, start_date, lookback_period, strategy, step)
    if result is None:
        return

    log_info("Load data for cleanup and analysis...")
    min_date = time_series[0] - np.timedelta64(60, 'D')
    data = qndata.load_data_by_type(competition_type, assets=result.asset.values.tolist(), min_date=str(min_date)[:10])

    result = qnout.clean(result, data, competition_type)
    result.name = competition_type
    qnout.write(result)

    if analyze:
        log_info("---")
        analyze_results(result, data, competition_type, build_plots)

    return result


def run_iterations(time_series, data, window, start_date, lookback_period, strategy, step):
    log_info("Run iterations...\n")

    ts = np.sort(time_series)
    outputs = []

    output_time_coord = ts[ts >= start_date]
    output_time_coord = output_time_coord[::step]

    i = 0

    sys.stdout.flush()

    with progressbar.ProgressBar(max_value=len(output_time_coord), poll_interval=1) as p:
        for t in output_time_coord:
            tail = window(data, t, lookback_period)
            output = strategy(tail)
            if type(output) != xr.DataArray:
                log_err("Output is not xarray!")
                return
            if output.dims != (qndata.ds.ASSET,):
                log_err("Wrong output dimensions. ", output.dims, "is not", (qndata.ds.ASSET,))
                return
            output = output.drop('time', errors='ignore')
            outputs.append(output)
            i += 1
            p.update(i)

    sys.stderr.flush()

    log_info("Merge outputs...")
    output = xr.concat(outputs, pd.Index(output_time_coord, name=qndata.ds.TIME))

    return output


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


def is_interact():
    return 'NONINTERACT' not in os.environ


def analyze_results(output, data, kind, build_plots):
    log_info("Analyze results...")

    if len(output.time) == 0 or len(output.asset) == 0:
        log_err("ERROR! Output is empty!")
        return

    log_info("Check...")
    qnout.check(output, data, kind)
    log_info("---")
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

        app.run_server()
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
