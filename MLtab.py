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


@st.cache
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


@st.cache
# making a residual plot + hard coded analysis
def residual(y_train, y_train_pred, y_test, y_test_pred):
    f, axes = plt.subplots(1, 2, figsize=(24, 12))
    sb.residplot(y_train, y_train_pred, ax=axes[0])
    sb.residplot(y_test, y_test_pred, ax=axes[1])
    return f


def Linear_model(dataframe, X, y):
    st.write("## Multivariate Linear Regression")

    # check box asking user whether they want to consider distance as a parameter
    distancecb = st.checkbox("Include 'distance' as a parameter to improve R^2 value")
    if distancecb:
        pass
    else:
        X.drop('distance')

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
    subplot(y_train, y_train_pred, y_test, y_test_pred)

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
def RandomForest_model(dataframe, X, y):
    st.write("## Random Forest Regression")

    st.write("### Cleaning and Splitting Data")

    # splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # Feature Scaling

    # Create a Gaussian Classifier
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)  # Train the model using the training sets y_pred=clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # plotting train and test data in graphs side by side
    subplot(y_train, y_train_pred, y_test, y_test_pred)

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

    calculate_distance(dataframe)
    y = pd.DataFrame(dataframe["delivery_days"])
    X = pd.DataFrame(dataframe[["freight_value", "price", "volume", "product_weight_g", 'distance']])

    if model_type == "Linear Regression":
        Linear_model(dataframe, X, y)
    elif model_type == "Random Forest Regression":
        RandomForest_model(dataframe, X, y)