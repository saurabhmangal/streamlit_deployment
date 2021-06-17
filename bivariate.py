# Basic Libraries
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt  # we only need pyplot
import plotly.express as px

# function to plot a plotly scatter plot
def plot(df, variable):
    fig = px.scatter(df, x="delivery_days", y=variable)
    st.write(fig)
    cor = df["delivery_days"].corr(df[variable])
    cor = round(cor, 2)
    st.write("The correlation of delivery days and", variable, "is", cor)

def biplot():
    # creating individual dataframes for all numerical variables
    dataframe = pd.read_csv("ExportDataFrame.csv", header=0)
    old_dataframe = dataframe

    # EDA: BIVARIATE PLOTS
    st.write('## Exploratory Data Analysis: Bivariate Plots')
    st.write(
        "Since this investigation aims to analyse O-list's delivery, we chose to focus on Delivery Duration. This "
        "refers to the time taken from the day the order was placed, up until the product was actually delivered to "
        "the customers doorstep. Hence, here we graphically present the above various factors that could affect "
        "deliver against delivery duration. Select the variable from the drop down menu below to view it's Bivariate "
        "plot..")

    # select box to choose variable to plot against delivery duration
    types2 = (
        "Price",
        "Volume",
        "Weight",
        "Freight value",
        "Review Rating",
    )  # maybe add 'Boxplot' after fixes
    variable_type2 = st.selectbox("Choose your variable", types2)

    # function to display plot and compute analysis based on summary statistics
    def show_biplot(kind: str):
        st.write(kind)
        if kind == "Price":
            variable = "price"
            plot(dataframe, variable)
        elif kind == "Volume":
            variable = "volume"
            plot(dataframe, variable)
        elif kind == "Weight":
            variable = "product_weight_g"
            plot(dataframe, variable)
        elif kind == "Freight value":
            variable = "freight_value"
            plot(dataframe, variable)
        elif kind == "Review Rating":
            jointDF = pd.concat((old_dataframe['delivery_days'], dataframe['review_score']), axis=1)
            fig5 = plt.figure()
            sb.heatmap(jointDF.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f")
            st.pyplot(fig5)
            st.write(
                "This data is more categorical than it is numerical. Hence, a correlation heatmap was made instead of "
                "a scatter plot.")
            cor = dataframe["delivery_days"].corr(dataframe["review_score"])
            st.write("The correlation of delivery days and review score is", cor)

    # function to remove outliers
    def remove_outlier_IQR(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_final = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]
        return df_final

    # calling above function and branching according to user input
    outlier = st.checkbox('Remove Outliers')
    if outlier:
        dataframe = remove_outlier_IQR(dataframe)
        dataframe["delivery_days"] = old_dataframe["delivery_days"]
    else:
        dataframe = old_dataframe

    # command to show plot + analysis
    show_biplot(kind=variable_type2)

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
    fig7 = plt.figure(figsize=(12, 12))
    numDF = pd.DataFrame(dataframe[['delivery_days', 'price', 'volume', 'product_weight_g', 'freight_value',
                                    'review_score']])
    sb.heatmap(numDF.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f")
    st.pyplot(fig7)

    # explanation + analysis on correlation heatmap
    st.write("Here we focus on the first row, since we want to explore the correlation of various variables with "
             "delivery days")
    st.write("This is an example of how important color can be within data visualisation. Here freight value has the "
             "highest correlation to delivery days, which suggests this is a variable has a relatively strong "
             "positive correlation. Review score being a darker coloured cell, holds a negative coefficient, "
             "suggesting a strong negative correlation.")
