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

# creating individual dataframes for all numerical variables

dataframe = pd.read_csv("ExportDataFrame.csv", header=0)

pricedf = pd.DataFrame(dataframe['price'])
freightdf = pd.DataFrame(dataframe['freight_value'])
volumedf = pd.DataFrame(dataframe['volume'])
weightdf = pd.DataFrame(dataframe['product_weight_g'])
reviewdf = pd.DataFrame(dataframe['review_score'])
daysdf = pd.DataFrame(dataframe['delivery_days'])

def geo():

    st.write('## Geospatial Data Analysis')

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
    st.bokeh_chart(plot)
    st.write(plot)

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

    # let's see our handiwork!
    st.bokeh_chart(plot)


    st.write("### First and Second Map : Location of buyers/sellers ")
    st.write(
        "From our analysis, we can see the rough distribution of our variables and using geospatial data, we get an overview into where our sellers and buyers are located for our dataset. From our first and second map, we can see the exact locations of O-list deliveries by sellers and buyers. When zoomed out, it is apparent that most deliveries come from main cities like Sao Paulo and Rio de Janeiro, nevertheless it is also important to realise that there are clusters of deliveries made to various other cities as well (Curitiba, Joinville, Florianopolis, Londrina, Juiz de Fora, Belo Horizonte). This gives us an idea of where we could have warehouses situated, and where we can invest into more delivery services. According to this inital plot, it would seem that it is economical to invest into delivery trucks and personnel near the main cities, and perhaps have relatively less resources invested for others. The nature and size of the resources to be invested can only be decided upon further detailed data analysis. When zoomed into specific cities, we can understand the specific neighborhoods that attract most customers for the e-commerce business. For example, within Sao Paulo, when zoomed in, we can see the city center hexmap to be a yellowish colour. This lgihter colour indicates a higher number of sales, showing how this specific area within the city attracts most customers. This shows us that most potential and target customers live in this area.")

    st.write("### Third Map : Bubble Map ")
    st.write(
        "We can use inbuilt functionality within the 'bokeh' module, to derive insights. From our third map, each orange bubble is actually representative of a sale, wherein the radius of the bubble is defined by the delivery days. For example if a certain order took 11 days to deliver then the bubble would be alot more bigger and noticable as compared to a bubble represenating 2 day delivery. This helps us gain isight into the locations that tend to have longer delivery durations, because we can analyse the map to recognise locations that have larger bubbles as opposed to locations with smaller bubbles. From the map above it is apparent that locations: 1) West of Floresta da Tijura near Rio de Janeiro 2) Saito 3) Ribeirao Preto 4) Salvador")

    st.write("### Weight ")
    st.write(
        "For orders with longer delivery days, it is also common for weight of the product to factor in. For example, it will take longer for distributors to deliver a wooden wardrobe as opposed to delivering a book. If you hover along the large bubbles, from the interactive legend it is also apparent that some of these deliveries are actually of heavy products. Hence for locations like North of Salvador, a longer delivery duration may be justified. ")

    st.write("### Review Rating ")
    st.write(
        "The above locations have long delivery durations, and perhaps distributors can work on increasing efficiency in these locations. Furthermore, if you hover across the bubble plots at these locations, it is also apparent that the review score for these high delivery duration orders are very low (around 1-3). This proves that longer delivery days in these locations is a factor that reduces cutomer satisfaction. ")
