# Basic Libraries
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt  # we only need pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sin, cos, sqrt, atan2, radians
from sklearn import metrics
from univariate import *

sb.set()  # set the default Seaborn style for graphics


# calculating distance between customer and seller based on coordinates provided
@st.cache
def calculate_distance(dataframe):
    # approximate radius of earth in km
    R = 6373.0
    # calculating and storing distance as a record in a new column in main dataframe
    for i in range(0, len(dataframe)):
        lat1 = radians(dataframe.loc[i, 'customer_lat'])
        lon1 = radians(dataframe.loc[i, 'customer_lng'])
        lat2 = radians(dataframe.loc[i, 'seller_lat'])
        lon2 = radians(dataframe.loc[i, 'seller_lng'])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        dataframe.loc[i, 'distance'] = distance
    return dataframe


def subplot(y_train, y_train_pred, y_test, y_test_pred):
    # Plot the Predictions vs the True values for TRAIN data
    f, axes = plt.subplots(1, 2, figsize=(24, 12))
    axes[0].scatter(y_train, y_train_pred, color="blue")
    axes[0].plot(y_train, y_train, 'w-', linewidth=1)
    axes[0].set_xlabel("True values of the Response Variable (Train)")
    axes[0].set_ylabel("Predicted values of the Response Variable (Train)")

    # Plot the Predictions vs the True values for TEST data
    axes[1].scatter(y_test, y_test_pred, color="green")
    axes[1].plot(y_test, y_test, 'w-', linewidth=1)
    axes[1].set_xlabel("True values of the Response Variable (Test)")
    axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
    return f


@st.cache
# function to generate table with statistics on accuracy + goodness of fit of model
def table(y_train, y_train_pred, y_test, y_test_pred):
    table_data = [["Explained Variance (R^2):", str(metrics.r2_score(y_train, y_train_pred)),
                   str(metrics.r2_score(y_test, y_test_pred))],
                  ["Mean Squared Error (MSE):", str(mean_squared_error(y_train, y_train_pred)),
                   str(mean_squared_error(y_test, y_test_pred))],
                  ["Mean Absolute Error (MAE):", str(mean_absolute_error(y_train, y_train_pred)),
                   str(mean_absolute_error(y_test, y_test_pred))]]
    table_display = pd.DataFrame(table_data, columns=["Statistics", " Train Dataset", "Test Dataset"])
    return table_display


@st.cache
# function that gives dynamic analysis based on numerical values in statistic table
def table_analysis(y_train, y_train_pred, y_test, y_test_pred):
    if (metrics.r2_score(y_train, y_train_pred)) < 0.3 and mean_squared_error(y_train, y_train_pred) > 50 \
            and mean_squared_error(y_test, y_test_pred) > 50:
        text = "Model is under fit, as the Variance is low but bias is high (high MSE values for train and test data)"
    elif mean_squared_error(y_train, y_train_pred) < mean_squared_error(y_test, y_test_pred):
        text = "Model is over fit, as train data MSE is lower than test data MSE"
    else:
        text = "Model is fit, with relatively low variance, low train and test MSE, and is not bias since train data " \
               "MSE is not smaller than test data MSE "

    return text


# making a residual plot + hard coded analysis
def residual(y_train, y_train_pred, y_test, y_test_pred):
    f, axes = plt.subplots(1, 2, figsize=(24, 12))
    sb.residplot(y_train, y_train_pred, ax=axes[0])
    sb.residplot(y_test, y_test_pred, ax=axes[1])
    return f

def remove_outlier_dataframe(dataframe):
    for i in list(dataframe):
            dataframe[i] = remove_outlier_IQR(dataframe[i])
            dataframe.dropna(inplace=True)
    return (dataframe)

def drop_column(dataframe, column_name):
    return(dataframe.drop([column_name],axis=1))

def Linear_model(dataframe):
    
    vals_consider = ["delivery_days","freight_value", "price", "volume", "product_weight_g", "distance"]
    
    st.write("## Multivariate Linear Regression")
    # x_val = ["freight_value", "price", "volume", "product_weight_g", "distance"]
    # y_val = ["delivery_days"]
    
    dataframe = dataframe[vals_consider].copy()
    # check box to remove outliers
    
    # check box asking user whether they want to consider distance as a parameter
    distancecb = st.checkbox("Include 'distance' as a parameter to improve R^2 value")
    if not distancecb:
        st.write("distance check box done")
        dataframe = drop_column(dataframe, "distance")
        
    st.write(list(dataframe))
    
    st.write(list(dataframe))
    st.write(len(dataframe))
    st.write(dataframe.isna().sum())
    
    outlierCB = st.checkbox("Remove Outliers")
    if outlierCB:
        st.write("Outlier chek box working")
        dataframe = remove_outlier_dataframe(dataframe)
    
    st.write(list(dataframe))
    st.write(len(dataframe))
    st.write(dataframe.isna().sum())
    
       
    
    
    y = pd.DataFrame(dataframe["delivery_days"])
    X = dataframe[dataframe.columns.drop('delivery_days')]

    
    st.write(list(X))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    st.write("### Visualising predicted and actual data")

    # Linear Regression using Train Data
    linreg = LinearRegression()  # create the linear regression object
    linreg.fit(X_train, y_train)  # train the linear regression model
    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    # Coefficients of the Linear Regression line
    # st.write('Intercept of Regression: b = ', int(linreg.intercept_))
    # st.write('Coefficients of Regression: a = ', linreg.coef_)
    # st.text("")

    # plotting train and test data in graphs side by side
    st.write(subplot(y_train, y_train_pred, y_test, y_test_pred))

    # making error table
    st.write("### Error metrics")
    st.write("Using the above model we tried to predict delivery duration values. Below are the statistics "
             "describing how accurate our model is in predicting further data.")
    st.write(table(y_train, y_train_pred, y_test, y_test_pred))

    # dynamic analysis based on statistic table
    st.write(table_analysis(y_train, y_train_pred, y_test, y_test_pred))

    # make residual plot + analysis
    st.write("## Residual Plots")
    st.write(residual(y_train, y_train_pred, y_test, y_test_pred))
    st.write(
        "Considering that points in the residual plot above seemed to cluster near the origin, suggests a pattern. "
        "Hence, Heteroskedasticity is present")


# RANDOM FOREST REGRESSION
def RandomForest_model(dataframe):
    
    y = pd.DataFrame(dataframe["delivery_days"])
    X = pd.DataFrame(dataframe[["freight_value", "price", "volume", "product_weight_g", "distance"]])
    
    st.write("## Random Forest Regression")

    old_dataframe = dataframe
    # check box to remove outliers
    outlierCB = st.checkbox("Remove Outliers")
    if outlierCB:
        dataframe = remove_outlier_IQR(dataframe)
    else:
        dataframe = old_dataframe

    y = pd.DataFrame(dataframe["delivery_days"])
    X = pd.DataFrame(dataframe[["freight_value", "price", "volume", "product_weight_g", "distance"]])

    # check box asking user whether they want to consider distance as a parameter
    distancecb = st.checkbox("Include 'distance' as a parameter to improve R^2 value")
    if distancecb:
        y = pd.DataFrame(dataframe["delivery_days"])
        X = pd.DataFrame(dataframe[["freight_value", "price", "volume", "product_weight_g", "distance"]])
    else:
        y = pd.DataFrame(dataframe["delivery_days"])
        X = pd.DataFrame(dataframe[["freight_value", "price", "volume", "product_weight_g"]])

    # splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # Feature Scaling

    st.write("### Visualising predicted and actual data")

    # Create a Gaussian Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)  # Train the model using the training sets y_pred=clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # plotting train and test data in graphs side by side
    st.write(subplot(y_train, y_train_pred, y_test, y_test_pred))

    # creating a statistic table
    st.write("### Error metrics")
    st.write("Using the above model we tried to predict delivery duration values. Below are the statistics "
             "describing how accurate our model is in predicting further data.")
    st.write(table(y_train, y_train_pred, y_test, y_test_pred))

    # dynamic analysis based on statistic table
    st.write(table_analysis(y_train, y_train_pred, y_test, y_test_pred))

    # make residual plot + analysis
    st.write("## Residual Plots")
    st.write(residual(y_train, y_train_pred, y_test, y_test_pred))
    st.write(
        "Considering that points in the residual plot above seemed to cluster near the origin, suggests a pattern. "
        "Hence, Heteroskedasticity is present")


def ML_tab(dataframe):
    models = ("Linear Regression", "Random Forest Regression")
    model_type = st.selectbox("Choose your model", models)

    dataframe = calculate_distance(dataframe)
    
    st.write(list(dataframe))

    if model_type == "Linear Regression":
        Linear_model(dataframe)
    elif model_type == "Random Forest Regression":
        RandomForest_model(dataframe)


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()

# Import the model we are using
# from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators=400, random_state=0)
# Train the model on training data
# rf.fit(X_train, y_train)

# Make predictions on the test set
# Use the forest's predict method on the test data
# y_pred = rf.predict(X_test)

# Visualising a single decision tree
# Import tools needed for visualization
# Pull out one tree from the forest
# tree = rf.estimators_[5]
#  st.write(tree)

from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
#  from sklearn.metrics import accuracy_score, f1_score

#  model = RandomForestRegressor(max_depth=5, max_features=None, n_jobs=-1)
#  model.fit(X_train, y_train)

# tree = model.estimators_[5]
# st.write(model)

#  y_train_pred = model.predict(X_train)
#  y_test_pred = model.predict(X_test)
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# a = [0.03623213, -1.8993383, 0.00532808]

#  for i in range(0, 28948):
# value = a[0] * df3.loc[i, 'freight_value'] + a[1] * df3.loc[i, 'review_score'] + a[2] * df3.loc[
#     i, 'distance'] + 16.143208754
# df3.loc[i, 'estimated_ML'] = value

# actual_delivery = df3['delivery_days']
# given_estimate = df3['estimated_days']
# ML_estimate = df3['estimated_ML']
