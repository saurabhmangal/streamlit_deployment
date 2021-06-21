# Basic Libraries
import streamlit as st
import pandas as pd
import seaborn as sb

sb.set()  # set the default Seaborn style for graphics


def home():
    dataframe = pd.read_csv("C:/Users/Administrator/Downloads/ExportDataFrame.csv", header=0)
    st.title("Analysing O-list's Delivery Service")
    st.write("O-list is a Brazilian based e-commerce company that provides a platform for both sellers to display and "
             "buyers to buy various types of products. This investigation takes a data driven approach to analyse and "
             "understand O-list's current delivery services, and whether there is any scope for improvement. This "
             "analysis aims to answer the problem statement i.e **What factors affect Delivery Service?**")
    st.write("O-list provided real time data regarding delivery on Kaggle.com, and below is a cleaned version of the "
             "same data. After careful consideration, we have chosen relevant variables, primary keys/composite keys "
             "and have merged various datasets. We have also cleaned the data to ensure data integrity (no data "
             "redundancy/no incomplete data). ")
    st.dataframe(dataframe)
