import plotly.offline as ply
import plotly.graph_objs as go
import math
import logging
import os


def is_interact():
    return 'NONINTERACT' not in os.environ


def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True


if is_notebook():
    ply.init_notebook_mode(connected=True)
    try:
        from IPython import get_ipython
        if get_ipython().config['IPKernelApp']['kernel_class'].startswith('google.colab.'):
            import plotly.io as pio
            pio.renderers.default = 'colab'
    except:
        pass


def make_major_plots(stats):
    pd_stat = stats.to_pandas()
    try:
        performance = pd_stat.loc[:, 'equity']
        equity_fig = make_plot_filled(performance.index, performance, name=" PnL (Equity)", type="log")
    except:
        equity_fig = None
        logging.exception("can't build equity chart")

    try:
        UWchart = pd_stat.loc[:, 'underwater']
        underwater_fig = make_plot_filled(UWchart.index, UWchart, color="darkred", name="Underwater Chart", range_max=0)
    except:
        underwater_fig = None
        logging.exception("can't build underwater chart")

    try:
        # show rolling Sharpe ratio on a 3-year basis:
        SRchart = pd_stat.loc[:, 'sharpe_ratio']
        mv = SRchart.iloc[-len(SRchart)//2:].min()
        xv = SRchart.iloc[-len(SRchart)//2:].max()
        mv, xv = mv - 0.03 * (xv - mv), xv + 0.03 * (xv - mv)
        sharpe_ratio_fig = make_plot_filled(SRchart.index, SRchart, color="#F442C5", name="Rolling SR",
                                                    range_min = mv, range_max = xv)
    except:
        sharpe_ratio_fig = None
        logging.exception("can't build sharpe_ratio chart")

    try:
        # show bias chart:
        biaschart = pd_stat.loc[:, 'bias']
        bias_fig = make_plot_filled(biaschart.index, biaschart, color="#5A6351", name="Bias Chart")
    except:
        bias_fig = None
        logging.exception("can't build bias chart")

    if not is_notebook():
        return equity_fig, underwater_fig, sharpe_ratio_fig, bias_fig


def make_plot(index, data, color="#17BECF", width=3, name="chart", range_min = None, range_max = None, type = None):
    """Makes a 2d scatter plot using index and data."""

    table = go.Scatter(x=index, y=data, line=dict(color=color, width=width), name=name)

    data_ = [table]

    min_ = data.min() if range_min is None else range_min
    max_ = data.max() if range_max is None else range_max

    if type == "log":
        min_ = math.log(min_, 10)
        max_ = math.log(max_, 10)

    range = [min_ - (max_ - min_)*0.05, max_ + (max_ - min_)*0.05]

    layout = go.Layout(showlegend=True, yaxis=dict(range = range, type=type))

    fig = go.Figure(data=data_, layout=layout)

    ply.iplot(fig)


def make_plot_double(index, data1, data2, color1="#17BECF", color2="#BA1244", width=3,
                     name1="chart1", name2="chart2"):
    """Makes a 2d scatter plot using index and two data sources."""

    table1 = go.Scatter(x=index, y=data1, line=dict(color=color1, width=width), name=name1)
    table2 = go.Scatter(x=index, y=data2, line=dict(color=color2, width=width), name=name2)

    data = [table1, table2]
    layout = go.Layout(showlegend=True)

    fig = go.Figure(data=data, layout=layout)

    ply.iplot(fig)


def make_plot_filled(index, data, color="#17BECF", width=3, name="chart", range_min = None, range_max = None,
                     type = None, show=True):
    """Makes a filled 2d scatter plot using index and data."""
    
    table = go.Scatter(x=index, y=data, line=dict(color=color, width=width), fill="tonexty", name=name)
    
    data_ = [table]

    min_ = data.min() if range_min is None else range_min
    max_ = data.max() if range_max is None else range_max

    if type == "log":
        min_ = math.log(min_, 10)
        max_ = math.log(max_, 10)

    range = [min_ - (max_ - min_)*0.05, max_ + (max_ - min_)*0.05]

    layout = go.Layout(showlegend=True, yaxis=dict(range = range, type=type))

    fig = go.Figure(data=data_, layout=layout)

    if is_notebook():
        ply.iplot(fig)
    else:
        return fig
