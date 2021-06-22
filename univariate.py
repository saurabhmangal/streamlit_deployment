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

<<<<<<< HEAD

@st.cache(allow_output_mutation=True)
def remove_outlier_IQR(series, var_type):
=======
@st.cache(allow_output_mutation=True)
def remove_outlier_IQR(series):
>>>>>>> 685266dc7977f23f7bd7434072bacc8a95e96a5d
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

#     if (variable_type != "review_score"):
#         st.write(uni_plot(dataframe[variable_type], variable_type,plot_type))#.show()

# else: st.write("Since Review Scores can only hold values from 1 to 5, this is a categorical variable. Hence it
# would " "be more accurate to conduct categorical univariate statistics, as in with a count plot, as opposed to "
# "conventional numerical plots like box plot,histogram and violin plot.") st.write(count_plot(dataframe[
# variable_type],variable_type))

#     st.subheader("Basic Summary Statistics")
#     st.write("FOllowing are the statistical observations for:", variable_type)
#     st.write(dataframe[variable_type].describe())

#     std = int(dataframe[variable_type].std())
#     max_val = int(dataframe[variable_type].max())
#     if std >= (max_val / 2):
#         st.write(
#             "This plot has a high standard deviation relative to it's maximum point. This suggests that data in "
#             "this plot is not clustered to a mean and is more spread out.")
#     else:
#         st.write(
#             "This plot has a low standard deviation relative to it's maximum. This suggests that most points are "
#             "clustered to a mean.")


# fig = px.box(dataframe["price"],y="price")
# st.write(fig)

# =============================================================================
#
#     fig, axes = plt.subplots(figsize=(10, 10))
#     st.write(sb.countplot(x='review_score', data=dataframe["review_score"]))
#     st.pyplot(fig)
#
#
# else:
#     st.write("Since Review Scores can only hold values from 1 to 5, this is a categorical variable. Hence it would "
#              "be more accurate to conduct categorical univariate statistics, as in with a count plot, as opposed to "
#              "conventional numerical plots like box plot,histogram and violin plot.")
#     fig, axes = plt.subplots(figsize=(10, 10))
#     st.write(sb.countplot(x='review_score', data=dataframe["review_score"]))
#     st.pyplot(fig)
#
#
#
#
#     stats(reviewdf)
#
#     def uni_plot(df, variable_type,plot_type):
#         if (variable_type != "Review Rating"):
#             if (plot_type == "Boxplot"):
#                 fig = px.box(df, y=variable_type)
#             if (plot_type == "Histogram"):
#                 fig = px.histogram(df, x=variable_type)
#             if (plot_type == "Violin Plot"):
#                 fig = px.violin(df, y=variable_type)
#         else
#             st.write(
#                 "Since Review Scores can only hold values from 1 to 5, this is a categorical variable. Hence it would "
#                 "be more accurate to conduct categorical univariate statistics, as in with a count plot, as opposed to "
#                 "conventional numerical plots like box plot,histogram and violin plot.")
#             fig, axes = plt.subplots(figsize=(10, 10))
#             st.write(sb.countplot(x='review_score', data=dataframe["review_score"]))
#             st.pyplot(fig)
#             stats(reviewdf)
#         return (fig)
#
#     st.write(uni_plot)
#     stats(variable_type,dataframe)
#
#
#
#
# # =============================================================================
# #     # method to display boxplot
# #     def boxplot(df, variable_type):
# #         fig = px.box(df, y=variable_type)
# #         st.write(fig)
# #
# #     # method to display histplot
# #     def histplot(df, variable_type):
# #         fig = px.histogram(df, x=variable_type)
# #         st.write(fig)
# #
# #     # method to display violinplot
# #     def violinplot(df, variable_type):
# #         fig = px.violin(df, y=variable_type)
# #         st.write(fig)
# # =============================================================================
#
#     def show_plot(kind: str, plot: str):
#         st.write(kind)
#         if kind == "Price":
#             variable = "price"
#             if plot == "Boxplot":
#                 boxplot(pricedf, variable)
#             if plot == "Histogram":
#                 histplot(pricedf, variable)
#             if plot == "Violin Plot":
#                 violinplot(pricedf, variable)
#             stats(pricedf)
#         elif kind == "Volume":
#             variable = "volume"
#             if plot == "Boxplot":
#                 boxplot(volumedf, variable)
#             if plot == "Histogram":
#                 histplot(volumedf, variable)
#             if plot == "Violin Plot":
#                 violinplot(volumedf, variable)
#             stats(volumedf)
#         elif kind == "Weight":
#             variable = "product_weight_g"
#             if plot == "Boxplot":
#                 boxplot(weightdf, variable)
#             if plot == "Histogram":
#                 histplot(weightdf, variable)
#             if plot == "Violin Plot":
#                 violinplot(weightdf, variable)
#             stats(weightdf)
#         elif kind == "Freight value":
#             variable = "freight_value"
#             if plot == "Boxplot":
#                 boxplot(freightdf, variable)
#             if plot == "Histogram":
#                 histplot(freightdf, variable)
#             if plot == "Violin Plot":
#                 violinplot(freightdf, variable)
#             stats(freightdf)
#         elif kind == "Review Rating":
#             st.write(
#                 "Since Review Scores can only hold values from 1 to 5, this is a categorical variable. Hence it would "
#                 "be more accurate to conduct categorical univariate statistics, as in with a count plot, as opposed to "
#                 "conventional numerical plots like box plot,histogram and violin plot.")
#             p, axes = plt.subplots(figsize=(10, 10))
#             st.write(sb.countplot(x='review_score', data=dataframe["review_score"]))
#             st.pyplot(p)
#             stats(reviewdf)
#         elif kind == "Delivery Duration":
#             variable = "delivery_days"
#             if plot == "Boxplot":
#                 boxplot(daysdf, variable)
#             if plot == "Histogram":
#                 histplot(daysdf, variable)
#             if plot == "Violin Plot":
#                 violinplot(daysdf, variable)
#             stats(daysdf)
#
#
#      # method to display summary statistics and compute dynamic analysis on statistcs
#     def stats(variable_type,dataframe):
#         df = pd.Dataframe(dataframe[variable_type])
#         st.subheader("Basic Summary Statistics")
#         st.write(df.describe())
#         # count = df.count()
#         # min = int(df.min())
#         # mean = int(df.mean())
#         std = int(df.std())
#         max = int(df.max())
#         if std >= (max / 2):
#             st.write(
#                 "This plot has a high standard deviation relative to it's maximum point. This suggests that data in "
#                 "this plot is not clustered to a mean and is more spread out.")
#         else:
#             st.write(
#                 "This plot has a low standard deviation relative to it's maximum. This suggests that most points are "
#                 "clustered to a mean.")
#
#             # function with nested selection statements to navigate functionality based on user's input
#
#
#     # method to remove outliers and return new dataset
#     def remove_outlier_IQR(df):
#         Q1 = df.quantile(0.25)
#         Q3 = df.quantile(0.75)
#         IQR = Q3 - Q1
#         df_final = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]
#         return df_final
#
#     # calling above method and branching according to user input
#     outlier = st.checkbox('Remove Outliers')
#     if outlier:
#         pricedf = remove_outlier_IQR(pricedf)
#         volumedf = remove_outlier_IQR(volumedf)
#         weightdf = remove_outlier_IQR(weightdf)
#         freightdf = remove_outlier_IQR(freightdf)
#         daysdf = remove_outlier_IQR(daysdf)
#     else:
#         pricedf = pd.DataFrame(dataframe['price'])
#         freightdf = pd.DataFrame(dataframe['freight_value'])
#         volumedf = pd.DataFrame(dataframe['volume'])
#         weightdf = pd.DataFrame(dataframe['product_weight_g'])
#         daysdf = pd.DataFrame(dataframe['delivery_days'])
#
#     # command to display plot based on select box input
#     show_plot(kind=variable_type, plot=plot_type)
# '''
# =============================================================================
