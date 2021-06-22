# Basic Libraries
import seaborn as sb
import home
from univariate import *
import bivariate
import linearreg
import geoplot
import conclusion
import streamlit as st
import pandas as pd
sb.set()  # set the default Seaborn style for graphics

streamlit.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='expanded')

# creating individual dataframes for all numerical variables
dataframe = pd.read_csv("ExportDataFrame.csv", header=0)

###introducing side bar to the web page 
# Sidebar Navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:',
    ['Home',
     'Univariate Plots',
     'Bivariate & Multivariate Plots',
     'Geospatial Plots & Analysis',
     'Machine Learning',
     'Conclusion and Recommendation'])

if options == 'Home':
    home.home(dataframe)
elif options == 'Univariate Plots':
    univariate_tab(dataframe)
elif options == 'Bivariate & Multivariate Plots':
    bivariate.biplot()
elif options == 'Geospatial Plots & Analysis':
    geoplot.geo()
elif options == 'Machine Learning':
    linearreg.reg()
elif options == 'Conclusion and Recommendation':
    conclusion.rec()









# =============================================================================
# pricedf = pd.DataFrame(dataframe['price'])
# freightdf = pd.DataFrame(dataframe['freight_value'])
# volumedf = pd.DataFrame(dataframe['volume'])
# weightdf = pd.DataFrame(dataframe['product_weight_g'])
# reviewdf = pd.DataFrame(dataframe['review_score'])
# daysdf = pd.DataFrame(dataframe['delivery_days'])
# =============================================================================





