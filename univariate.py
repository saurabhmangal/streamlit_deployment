# Basic Libraries
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt  # we only need pyplot
import plotly.express as px
import streamlit

sb.set()  # set the default Seaborn style for graphics


# dataframe = pd.read_csv("ExportDataFrame.csv", header=0)

@st.cache
def uni_plot(series, variable_type, plot_type):
    if plot_type == "Boxplot":
        fig = px.box(series, y=variable_type)
    if plot_type == "Histogram":
        fig = px.histogram(series, x=variable_type)
    if plot_type == "Violin Plot":
        fig = px.violin(series, y=variable_type)

    return (fig)

@st.cache
def count_plot(series, variable_type):
    fig, axes = plt.subplots(figsize=(5, 5))
    sb.countplot(x=series[variable_type])
    return fig

@st.cache(allow_output_mutation=True)
def remove_outlier_IQR(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    series_wo_outlier = series[~((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))]
    print(series_wo_outlier)
    return series_wo_outlier


def streamlit_uni_analysis(series, variable_type, plot_type):
    if variable_type == 'review_score':
        st.write("Since Review Scores can only hold values from 1 to 5, this is a categorical variable. Hence it would "
                 "be more accurate to conduct categorical univariate statistics, as in with a count plot, as opposed "
                 "to conventional numerical plots like box plot,histogram and violin plot.")
        fig, axes = plt.subplots(figsize=(5, 5))
        sb.countplot(x = 'review_score', data = series)
        st.write(fig)
    else:
        st.write(uni_plot(series, variable_type, plot_type))  # .show()

    st.subheader("Basic Summary Statistics")
    st.write("Following are the statistical observations for:", variable_type)
    series = series[variable_type]
    st.write(series.describe())

    std = int(series.std())
    max_val = int(series.max())
    if std >= (max_val / 2):
        st.write(
            "This plot has a high standard deviation relative to it's maximum point. This suggests that data in "
            "this plot is not clustered to a mean and is more spread out.")
    else:
        st.write(
            "This plot has a low standard deviation relative to it's maximum. This suggests that most points are "
            "clustered to a mean.")


def univariate_tab(dataframe):
    # EDA: UNIVARIATE PLOTS
    st.write('## Exploratory Data Analysis: Univariate Plots')
    st.write("Here we have chosen 5 numerical variables that we believe have potential to affect delivery service. We "
             "have presented, Boxplots, Histograms and Violin plots for the same. (Price, Freight Value, Volume, "
             "Weight, Review Score).")

    # select boxes so user can choose variable and type of plot
    variable_types = ["price",
                      "volume",
                      "product_weight_g",
                      "freight_value",
                      "review_score",
                      "delivery_days"]

    plots = ["Boxplot", "Histogram", "Violin Plot"]

    variable_type = st.selectbox("Choose your variable", variable_types)
    plot_type = st.selectbox("Choose which plot you want", plots)

    # calling above method and branching according to user input
    outlier = st.checkbox('Remove Outliers')
    # variable_type = variable_types[0]
    # plot_type = plots[0]
    # st.write(plot_type)
    # st.write(variable_type)
    print(plot_type)
    print(variable_type)

    # variable_type = "review_score"

    if outlier:
        series_wo_outlier = remove_outlier_IQR(dataframe[variable_type])
        streamlit_uni_analysis(series_wo_outlier, variable_type, plot_type)
    else:
        streamlit_uni_analysis(dataframe, variable_type, plot_type)
