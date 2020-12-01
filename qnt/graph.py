import plotly.offline as ply
import plotly.graph_objs as go
import math

ply.init_notebook_mode(connected=True)


def make_plot(index, data, color="#17BECF", width=3, name="chart", range_min = None, range_max = None, type = None):
    """Makes a 2d scatter plot using index and data."""
    
    table = go.Scatter(x=index, y=data, line=dict(color=color, width=width), name=name)
    
    data_ = [table]
    
    if (range_min == None) and (range_max == None):
        min_ = data.min()
        max_ = data.max()
        range=[min_ - (max_ - min_)*0.05, max_ + (max_ - min_)*0.05]
    
    elif (range_min != None) and (range_max != None):
        range = [range_min, range_max]
    
    elif (range_min != None):
        max_ = data.max()
        range = [range_min, max_ + (max_ - range_min)*0.03]
        
    else:
        min_ = data.min()
        range = [min_ - (range_max - min_)*0.03, range_max]     
        
    if type == "log": 
         range = [math.log(range[0], 10), math.log(range[1], 10)]   
    
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


def make_plot_filled(index, data, color="#17BECF", width=3, name="chart", range_min = None, range_max = None, type = None):
    """Makes a filled 2d scatter plot using index and data."""
    
    table = go.Scatter(x=index, y=data, line=dict(color=color, width=width), fill="tonexty", name=name)
    
    data_ = [table]
    
    if (range_min == None) and (range_max == None):
        min_ = data.min()
        max_ = data.max()
        range=[min_ - (max_ - min_)*0.05, max_ + (max_ - min_)*0.05]
        
    elif (range_min != None) and (range_max != None):
        range = [range_min, range_max]
    
    elif (range_min != None):
        max_ = data.max()
        range = [range_min, max_ + (max_ - range_min)*0.03]
        
    else:
        min_ = data.min()
        range = [min_ - (range_max - min_)*0.03, range_max]         
        
    if type == "log": 
         range = [math.log(range[0], 10), math.log(range[1], 10)]   
        
    layout = go.Layout(showlegend=True, yaxis=dict(range = range, type=type))
                       
    fig = go.Figure(data=data_, layout=layout)
    
    ply.iplot(fig)
