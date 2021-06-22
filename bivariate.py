# Basic Libraries
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt  # we only need pyplot
import plotly.express as px
from univariate import *


# function to plot a plotly scatter plot
@st.cache
def plot(df, variable):
    fig = px.scatter(df, x="delivery_days", y=variable)
    return fig


# function to create multivariate heatmap
def heatmap(numDF):
    fig = plt.figure(figsize=(12, 12))
    sb.heatmap(numDF.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f")
    return fig


# function to compute analysis based on summary statistics
@st.cache
def bivariate_cor(dataframe, variable_type, old_dataframe):
    if variable_type == "review_score":
        cor = dataframe["delivery_days"].corr(dataframe["review_score"])
        return cor
    else:
        cor = dataframe["delivery_days"].corr(dataframe[variable_type])
        cor = round(cor, 2)
        return cor


# function to display plot
def bivariate_plot(dataframe, variable_type, old_dataframe):
    if variable_type == 'review_score':
        jointDF = pd.concat((old_dataframe['delivery_days'], dataframe['review_score']), axis=1)
        fig = heatmap(jointDF)
        return fig
    else:
        return plot(dataframe, variable_type)


def bivariate_tab(dataframe):
    old_dataframe = dataframe

    # EDA: BIVARIATE PLOTS
    st.write('## Exploratory Data Analysis: Bivariate Plots')
    st.write(
        "Since this investigation aims to analyse O-list's delivery, we chose to focus on Delivery Duration. This "
        "refers to the time taken from the day the order was placed, up until the product was actually delivered to "
        "the customers doorstep. Hence, here we graphically present the above various factors that could affect "
        "deliver against delivery duration. Select the variable from the drop down menu below to view it's Bivariate "
        "plot..")

    outlier = st.checkbox('Remove Outliers')
    if outlier:
        dataframe = remove_outlier_IQR(dataframe)
        dataframe["delivery_days"] = old_dataframe["delivery_days"]
    else:
        dataframe = old_dataframe

    # select box to choose variable to plot against delivery duration
    variable_types = ["price",
                      "volume",
                      "product_weight_g",
                      "freight_value",
                      "review_score"]
    # maybe add 'Boxplot' after fixes
    variable_type = st.selectbox("Choose your variable", variable_types)

    # command to show plot + analysis
    st.write(bivariate_plot(dataframe, variable_type, old_dataframe))
    st.write("The correlation of", variable_type, "against delivery_days is: ", bivariate_cor(dataframe, variable_type, old_dataframe))

    # MULTIVARIABLE STATISTICS
    st.write('## Exploratory Data Analysis: Multivariate Plots')
    st.write(
        "It is also worth to explore whether the chosen variables have any correlation between themselves. Having a "
        "strong positive or a strong negative corelation suggests a relationship between the two variables. In data "
        "science we can use this relationship to derive a mathematical model, which can then be useful in predicting "
        "future values.")
    st.write(
        "Here is a corelation heatmap, where the colour of each cell represents a corelation value. This allows us to "
        "visually spot out strong and weak correlation within our data.")

    # correlation heatmap
    st.write('### Correlation: Heatmap')
    numDF = pd.DataFrame(dataframe[['delivery_days', 'price', 'volume', 'product_weight_g', 'freight_value',
                                    'review_score']])
    st.write(heatmap(numDF))

    # explanation + analysis on correlation heatmap
    st.write("Here we focus on the first row, since we want to explore the correlation of various variables with "
             "delivery days")
    st.write("This is an example of how important color can be within data visualisation. Here freight value has the "
             "highest correlation to delivery days, which suggests this is a variable has a relatively strong "
             "positive correlation. Review score being a darker coloured cell, holds a negative coefficient, "
             "suggesting a strong negative correlation.")
