# Basic Libraries
import seaborn as sb
import home
import univariate
import bivariate
import MLtab
import geoplot
import conclusion
import streamlit as st
import pandas as pd
sb.set()  # set the default Seaborn style for graphics

st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='expanded')

# creating individual dataframes for all numerical variables
dataframe = pd.read_csv("ExportDataFrame.csv", header=0)
#pricedf = pd.DataFrame(dataframe['price'])
#freightdf = pd.DataFrame(dataframe['freight_value'])
#volumedf = pd.DataFrame(dataframe['volume'])
#weightdf = pd.DataFrame(dataframe['product_weight_g'])
#reviewdf = pd.DataFrame(dataframe['review_score'])
#daysdf = pd.DataFrame(dataframe['delivery_days'])

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
    univariate.univariate_tab(dataframe)

elif options == 'Bivariate & Multivariate Plots':
    bivariate.bivariate_tab(dataframe)

elif options == 'Geospatial Plots & Analysis':
    geoplot.geo(dataframe)

elif options == 'Machine Learning':
    MLtab.ML_tab(dataframe)

elif options == 'Conclusion and Recommendation':
    conclusion.rec()


