# Basic Libraries
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import seaborn as sb
import matplotlib.pyplot as plt  # we only need pyplot
from bokeh.io import output_file
from bokeh.models import HoverTool, ColumnDataSource
from pkg_resources import get_provider
sb.set()  # set the default Seaborn style for graphics
from bokeh.tile_providers import get_provider, OSM
from bokeh.plotting import figure
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure, output_file, show

dataframe = pd.read_csv("C:/Users/Administrator/Downloads/ExportDataFrame.csv", header=0)

# converting all longitude and latitude data to type float, so bokeh can read coordinates
dataframe['customer_lat'] = dataframe['customer_lat'].astype('float')
dataframe['customer_lng'] = dataframe['customer_lng'].astype('float')
dataframe['seller_lat'] = dataframe['seller_lat'].astype('float')
dataframe['seller_lng'] = dataframe['seller_lng'].astype('float')

# Bokeh maps are in mercator. Convert lat lon fields to mercator units for plotting
def wgs84_to_web_mercator(df, lon, lat):
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi / 360.0)) * k
    return df
df = wgs84_to_web_mercator(dataframe, 'customer_lng', 'customer_lat')

# Establishing a zoom scale for the map. The scale variable will also determine proportions for hexbins and bubble maps so that everything looks visually appealing.
scale = 2000
x = df['x']
y = df['y']

# The range for the map extents is derived from the lat/lon fields. This way the map is automatically centered on the plot elements.
x_min = int(x.mean() - (scale * 350))
x_max = int(x.mean() + (scale * 350))
y_min = int(y.mean() - (scale * 350))
y_max = int(y.mean() + (scale * 350))

# Defining the map tiles to use. I use OSM, but you can also use ESRI images or google street maps.
tile_provider = get_provider(OSM)

# Establish the bokeh plot object and add the map tile as an underlay. Hide x and y axis.
plot = figure(
    title='Brazil O-list Sales by location',
    match_aspect=True,
    tools='wheel_zoom,pan,reset,save',
    x_range=(x_min, x_max),
    y_range=(y_min, y_max),
    x_axis_type='mercator',
    y_axis_type='mercator'
)

plot.grid.visible = True

map = plot.add_tile(tile_provider)
map.level = 'underlay'

plot.xaxis.visible = False
plot.yaxis.visible = False

# If in Jupyter, use the output_notebook() method to display the plot in-line. If not, you can use output_file() or another method to save your map.
output_file('hexmap.html')

# function takes scale (defined above), the initialized plot object, and the converted dataframe with mercator coordinates to create a hexbin map
def hex_map(plot, df, scale, leg_label='Hexbin Heatmap'):
    r, bins = plot.hexbin(x, y, size=scale * 0.5, hover_color='pink', hover_alpha=0.2, legend_label=leg_label)
    hex_hover = HoverTool(tooltips=[('count', '@c')], mode='mouse', point_policy='follow_mouse', renderers=[r])
    hex_hover.renderers.append(r)
    plot.tools.append(hex_hover)
    plot.legend.location = "top_right"
    plot.legend.click_policy = "hide"

# Create the hexbin map
hex_map(plot=plot,
        df=dataframe,
        scale=scale,
        leg_label='Sale')

show(plot)