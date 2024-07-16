import random
import itertools
import copy
from functools import reduce
import inspect
import operator
import logging
import math
import progressbar
import multiprocessing
import numpy as np

import qnt.backtester as qnbt
import qnt.log as qnlog  
import qnt.stats as qns
import qnt.graph as qng
import qnt.ta as qnta
import tabulate

def optimize_strategy(data, output_function, argument_generator,
                      stats_function=None, stats_to_weight=None, lookback_period = 365,
                      workers=1):
    """
    :param data: the input data for your strategy
    :param output_function: your strategy, generates outputs by data
    :param argument_generator: generates additional arguments for your strategy
    :param stats_function: calculate statistics for the strategy, see standard_stats_function
    :param stats_to_weight: converts statistics to weight in order to select the best iteration
    :param lookback_period: lookback period, only for the multi-pass backtester
    :param workers: parallel workers count, you can set it equal to os.cpu_count()
    :return:
    """

    if stats_function is None:
        stats_function = standard_stats_function

    if stats_to_weight is None:
        stats_to_weight = standard_stats_to_weight
        
    return optimize(
        TargetFunction(data, output_function, stats_function, lookback_period),
        argument_generator,
        stats_to_weight,
        workers
    )


def optimize(target_function, argument_generator, result_to_weight, workers=1):
    iterations = []
    length = None
    try:
        length = len(argument_generator)
    except:
        pass

    in_queue = multiprocessing.Queue(workers*2)
    out_queue = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(
            name="optimize_worker#" + str(i),
            target=optimize_worker_target,
            args=(target_function, result_to_weight, in_queue, out_queue)
        ) for i in range(workers)
    ]
    for w in workers:
        w.daemon = True
        w.start()

    with progressbar.ProgressBar(max_value=length, poll_interval=1) as bar:
        bar.update(0)

        def pull_out_queue(final=False):
            while not out_queue.empty() or final and bar.value < bar.max_value:
                res = out_queue.get()
                iterations.append(dict(args=res[0],result=res[1],weight=res[2],exception=res[3]))
                bar.update(bar.value + 1)

        for args in argument_generator:
            in_queue.put(args)
            pull_out_queue()

        pull_out_queue(True)

    for w in workers:
        w.terminate()
        w.join()

    return dict(iterations=iterations, best_iteration=max(*iterations, key=lambda i: i.get('weight', float('-inf'))))


def random_range_args_generator(iterations, **ranges):
    """
    Generates argument using random distribution
    :param iterations: number of samples
    :param ranges: range(...) objects or lists, defines arguments' domains
    :return:
    """
    res = (tuple(r[random.randrange(len(r))] for r in ranges.values()) for _ in range(iterations))
    res = (dict(zip(ranges.keys(), r)) for r in res)
    return IterWithLen(res, iterations)


def full_range_args_generator(**ranges):
    """
    Cartesian production, generates all possible combinations of the arguments
    :param ranges: range(...) objects or lists, defines arguments' domains
    :return:
    """
    res = itertools.product(*(r for r in ranges.values()))
    res = (dict(zip(ranges.keys(), r)) for r in res)
    length = reduce(operator.mul, (len(r) for r in ranges.values()))
    return IterWithLen(res, length)


def standard_stats_function(data, output, lookback_period = 365):
    """
    Calculates statistics for the iteration output.
    :param data: market data
    :param output: weights
    :param lookback_period: lookback period for the multi-pass backtester
    :return: dict
    """
    start_date = str(max((data.time[0].values), np.datetime64(qns.get_default_is_start_date_for_type(data.name)))).split('T')[0]
    args = inspect.getfullargspec(output).args
    with qnlog.Settings(info=False,err=False):
        if 'state' in args:
            output = qnbt.backtest(competition_type=data.name, 
                            lookback_period=lookback_period,
                            start_date=start_date,
                            strategy=output,
                            analyze=False,
                            build_plots=False,
                            collect_all_states=False)[0]
            stat = qns.calc_stat(data, output.sel(time=slice(start_date, None)))
        else:
            stat = qns.calc_stat(data, output(data).sel(time=slice(start_date, None)))
    return stat.isel(time=-1).to_pandas().to_dict()


def fast_stats_function(data, output):
    close = data.sel(field='close').dropna('time', how='all').fillna(0)
    prev_output = output.shift(time=1)
    prev_close = close.shift(time=1)
    returns = (close - prev_close) * prev_output / prev_close
    atr = qnta.atr(data.sel(field='high'), data.sel(field='low'), data.sel(field='close'), 14)
    slippage = abs(output / close - prev_output / prev_close) * 0.05 * atr
    relative_returns = returns - slippage
    agg_relative_returns = (relative_returns * abs(output)).sum('asset')
    agg_relative_returns = agg_relative_returns.where(np.isfinite(agg_relative_returns))
    mean_return = (np.exp(np.log(1 + agg_relative_returns).mean()) - 1).values
    volatility = (np.std(agg_relative_returns)).values
    sharpe_ratio = mean_return / volatility
    return dict(
        sharpe_ratio=sharpe_ratio,
        mean_return=mean_return,
        volatility=volatility
    )


def standard_stats_to_weight(stat):
    """
    Converts the statistics to weight.
    :param stat: dict
    :return:
    """
    res = stat.get('sharpe_ratio', float('-inf'))
    if math.isfinite(res):
        return res
    else:
        return float('-inf')


class IterWithLen(object):
    def __init__(self, wrappee, length):
        self.wrappee = wrappee
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.wrappee.__iter__()


class TargetFunction(object):
    def __init__(self, data, output_func, stats_func, lookback_period):
        self.data = data
        self.output_func = output_func
        self.stats_func = stats_func
        self.lookback_period = lookback_period

    def __call__(self, **args):
        try:
            output = copy.deepcopy(self.output_func)
            defaults = output.__defaults__
            arg_names = output.__code__.co_varnames[:output.__code__.co_argcount]
            new_defaults = tuple(args.get(arg, defaults[i]) for i, arg in enumerate(arg_names[-len(defaults):]))
            output.__defaults__ = new_defaults
            stats = self.stats_func(self.data, output, self.lookback_period)
            return stats
        except:
            logging.exception("unexpected")


def optimize_worker_target(target_func, result_to_weight, in_queue, out_queue):
    while True:
        args = in_queue.get()
        res = None
        weight = None
        exc = None
        try:
            res = target_func(**args)
            weight = result_to_weight(res)
        except Exception as e:
            logging.exception("exception in worker")
            exc = e
        out_queue.put((args, res, weight, exc))


def build_plot(results):
    """
    Displays the optimization result as scatter3d
    :param results:
    :return:
    """
    if qng.is_notebook():
        build_plot_jupyter(results)
    else:
        build_plot_dash(results)


def build_plot_jupyter(results):
    import plotly.graph_objs as go
    from ipywidgets import Output, VBox, HBox
    from IPython.display import  display

    data, field_list = prepare_data_for_chart(results)

    def show_plot(coord_x=field_list[0], coord_y=field_list[1], coord_z=field_list[2], coord_color=field_list[3]):
        fig = build_plotly_scatter3d(data, field_list, coord_x, coord_y, coord_z, coord_color, None, None)
        fig.update_layout(height=500)
        widget = go.FigureWidget(fig)

        out = Output()

        @out.capture(clear_output=True)
        def handle_click(trace, points, state):
            try:
                selected_number = points.point_inds[0]
                point = data[selected_number]
                res = tabulate.tabulate([i for i in point.items()], floatfmt=".2f")
                print("Selected:\n" + res)


                size = [15]*len(data)
                symbol = ['circle']*len(data)

                if selected_number is not None:
                    size[selected_number] = 25
                    symbol[selected_number] = 'diamond'

                with widget.batch_update():
                    widget.data[0].marker.size = size
                    widget.data[0].marker.symbol = symbol
            except:
                pass

        widget.data[0].on_click(handle_click)

        display(HBox([widget, out]))

    if qng.is_interact():
        try:
            from ipywidgets import interact, interactive, fixed, interact_manual, Layout, Dropdown
            import ipywidgets as widgets

            widget = interactive(show_plot,
                     coord_x=Dropdown(
                         options=field_list,
                         value=field_list[0]
                     ),
                     coord_y=Dropdown(
                         options=field_list,
                         value=field_list[1]
                     ),
                     coord_z=Dropdown(
                         options=field_list,
                         value=field_list[2]
                     ),
                     coord_color=Dropdown(
                         options=field_list,
                         value=field_list[2]
                     )
                 )

            controls = HBox(widget.children[:-1])
            output = widget.children[-1]
            display(VBox([controls, output]))
            widget.update()
        except:
            show_plot()
    else:
        show_plot()


def build_plot_dash(results):
    data, field_list = prepare_data_for_chart(results)
    option_list = [{'label': k, 'value': k} for k in field_list]

    import dash
    from dash import dcc
    from dash import html
    from dash.dependencies import Input, Output, State

    app = dash.Dash(__name__)
    selector_width = str(max(len(f) for f in field_list) + 6) + "ex"
    app.layout = html.Div([
        html.Div([
            dcc.Store(id='scene_camera'),
            dcc.Store(id='selected_number'),
            html.Div([
                html.Label("x:",style={'line-height': '36px', "margin-right": "1ex"}),
                dcc.Dropdown(id='coord_x', options=option_list, value=option_list[0]['value'],
                             style={"width": selector_width, "margin-bottom": "1ex"}, clearable=False),
            ], style={'display':'flex', 'justify-content': 'flex-end'}),
            html.Div([
                html.Label("y:",style={'line-height': '36px', "margin-right": "1ex"}),
                dcc.Dropdown(id='coord_y', options=option_list, value=option_list[1]['value'],
                             style={"width": selector_width, "margin-bottom": "1ex"}, clearable=False),
            ], style={'display':'flex', 'justify-content': 'flex-end'}),
            html.Div([
                html.Label("z:",style={'line-height': '36px', "margin-right": "1ex"}),
                dcc.Dropdown(id='coord_z', options=option_list, value=option_list[2]['value'],
                             style={"width": selector_width, "margin-bottom": "1ex"}, clearable=False),
            ], style={'display':'flex', 'justify-content': 'flex-end'}),
            html.Div([
                html.Label("color:",style={'line-height': '36px', "margin-right": "1ex"}),
                dcc.Dropdown(id='coord_color', options=option_list, value=option_list[3]['value'],
                             style={"width": selector_width, "margin-bottom": "1ex"}, clearable=False),
            ], style={'display':'flex', 'justify-content': 'flex-end'}),
            html.Div([
                html.Label("Selected:"),
                html.Pre('', id='selected')
            ])
        ],),
        html.Div([
            dcc.Graph(id="chart", style=dict(height='100vh')),
        ], style=dict(width='100%'))
    ],  style=dict(display='flex'))

    @app.callback(
        Output("chart", "figure"),
        [
            Input("coord_x", "value"),
            Input("coord_y", "value"),
            Input("coord_z", "value"),
            Input("coord_color", "value"),
            Input("selected_number", "data"),
            State("scene_camera", "data")
        ]
    )
    def update_bar_chart(coord_x, coord_y, coord_z, coord_color, selected_number, scene_camera=None):
        return build_plotly_scatter3d(data, field_list, coord_x, coord_y, coord_z, coord_color, selected_number, scene_camera)

    @app.callback([Output("selected", "children"), Output("selected_number", "data")], [Input('chart', 'clickData')])
    def select_point(click_data):
        try:
            n = click_data['points'][0]['pointNumber']
            point = data[n]
            res = tabulate.tabulate([i for i in point.items()], floatfmt=".2f")
            return res, n
        except:
            return dash.no_update, dash.no_update

    @app.callback(Output('scene_camera', 'data'), Input("chart", "relayoutData"))
    def store_scene_camera(relayoutData):
        if relayoutData and "scene.camera" in relayoutData:
            return relayoutData["scene.camera"]
        else:
            return dash.no_update

    #opens browser
    import webbrowser, os
    from threading import Timer
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:"+os.environ.get('PORT', '8050'), new=2)).start()

    app.run_server(dev_tools_hot_reload=False)


def prepare_data_for_chart(results):
    arg_fields = [k for k in results['best_iteration']['args'].keys()]
    stat_fields = [k for k in [
        "sharpe_ratio",
        "max_drawdown",
        "mean_return",
        "volatility",
        "equity",
        "avg_turnover",
        "avg_holding_time",
    ] if k in results['best_iteration']['result'].keys()]
    field_list = arg_fields + stat_fields

    data = [dict(list(zip(arg_fields, i['args'].values()))
                 + list(zip(stat_fields, (i['result'][j] for j in stat_fields))))
            for i in results['iterations'] if i['result'] is not None]

    return data, field_list


def build_plotly_scatter3d(data, field_list, coord_x, coord_y, coord_z, coord_color, selected_number, scene_camera=None):
    import plotly.express as px
    fig = px.scatter_3d(data, x=coord_x, y=coord_y, z=coord_z, color=coord_color,
                        hover_data=dict((i, ':.2f') for i in field_list))
    fig.update_layout(clickmode='event')
    if scene_camera is not None:
        fig.update_layout(scene_camera=scene_camera)

    size = [15]*len(data)
    symbol = ['circle']*len(data)

    if selected_number is not None:
        size[selected_number] = 25
        symbol[selected_number] = 'diamond'
    fig.update_traces(marker_size=size, marker_opacity=1, marker_symbol=symbol)

    return fig
