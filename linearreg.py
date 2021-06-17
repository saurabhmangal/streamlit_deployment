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

# creating individual dataframes for all numerical variables
dataframe = pd.read_csv("ExportDataFrame.csv", header=0)
pricedf = pd.DataFrame(dataframe['price'])
freightdf = pd.DataFrame(dataframe['freight_value'])
volumedf = pd.DataFrame(dataframe['volume'])
weightdf = pd.DataFrame(dataframe['product_weight_g'])
reviewdf = pd.DataFrame(dataframe['review_score'])
daysdf = pd.DataFrame(dataframe['delivery_days'])

@st.cache
def reg():
    # calculating distance between customer and seller based on coordinates provided
    def calculate_distance():
        # approximate radius of earth in km
        R = 6373.0

        # calculating and storing distance as a record in a new column in main dataframe
        for i in range(0, 29120):
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

    def subplot(y_train, y_train_pred, y_test, y_test_pred):
        st.write("### Visualising predicted and actual data")

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
        st.write(f)

    # function to generate table with statistics on accuracy + goodness of fit of model
    def table(y_train, y_train_pred, y_test, y_test_pred):
        st.write("### Goodness of fit of model")
        st.write("Using the above model we tried to predict delivery duration values. Below are the statistics "
                 "describing how accurate our model is in predicting further data.")

        table_data = [["Explained Variance (R^2):", str(metrics.r2_score(y_train, y_train_pred)),
                       str(metrics.r2_score(y_test, y_test_pred))],
                      ["Mean Squared Error (MSE):", str(mean_squared_error(y_train, y_train_pred)),
                       str(mean_squared_error(y_test, y_test_pred))],
                      ["Mean Absolute Error (MAE):", str(mean_absolute_error(y_train, y_train_pred)),
                       str(mean_absolute_error(y_test, y_test_pred))]]

        table_display = pd.DataFrame(table_data, columns=["Statistics", " Train Dataset", "Test Dataset"])
        st.write(table_display)

    # function that gives dynamic analysis based on numerical values in statistic table
    def table_analysis(y_train, y_train_pred, y_test, y_test_pred):
        if (metrics.r2_score(y_train, y_train_pred)) < 0.3 and mean_squared_error(y_train, y_train_pred) > 50 \
                and mean_squared_error(y_test, y_test_pred) > 50:
            st.write(
                "Model is under fit, as the Variance is low but bias is high (high MSE values for train and test data)")
        elif mean_squared_error(y_train, y_train_pred) < mean_squared_error(y_test, y_test_pred):
            st.write("Model is over fit, as train data MSE is lower than test data MSE")
        else:
            st.write("Model is fit, with relatively low variance, low train and test MSE, and is not bias since "
                     "train data MSE is not smaller than test data MSE")

    # making a residual plot + hard coded analysis
    def residual(y_train, y_train_pred, y_test, y_test_pred):
        st.write("## Residual Plots")
        f, axes = plt.subplots(1, 2, figsize=(24, 12))
        sb.residplot(y_train, y_train_pred, ax=axes[0])
        sb.residplot(y_test, y_test_pred, ax=axes[1])
        st.pyplot(f)
        st.write(
            "Considering that points in the residual plot above seemed to cluster near the origin, suggests a pattern. "
            "Hence, Heteroskedasticity is present")

    # MULTIVARIATE LINEAR REGRESSION
    def linearreg():

        # linear regression plot + calculations
        def plot():
            # Split the Dataset into Train and Test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

            # Linear Regression using Train Data
            linreg = LinearRegression()  # create the linear regression object
            linreg.fit(X_train, y_train)  # train the linear regression model

            # Predict Total values corresponding to HP
            y_train_pred = linreg.predict(X_train)
            y_test_pred = linreg.predict(X_test)
            st.write("Using the original data set, we tried to use the variables, freight value, price, volume, "
                     "review score and weight to derive a model to predict the delivery duration of a product.")

            # Coefficients of the Linear Regression line
            # st.write('Intercept of Regression: b = ', int(linreg.intercept_))
            # st.write('Coefficients of Regression: a = ', linreg.coef_)
            # st.text("")

            # plotting train and test data in graphs side by side
            subplot(y_train, y_train_pred, y_test, y_test_pred)

            # creating a statistic table
            table(y_train, y_train_pred, y_test, y_test_pred)

            # dynamic analysis based on statistic table
            table_analysis(y_train, y_train_pred, y_test, y_test_pred)

            # make residual plot + analysis
            residual(y_train, y_train_pred, y_test, y_test_pred)

        # main linear regression code begins here
        st.write("## Multivariate Linear Regression")

        # add column in distance into dataset
        calculate_distance()

        # check box asking user whether they want to consider distance as a parameter
        distancecb = st.checkbox("Include 'distance' as a parameter to improve R^2 value")

        # branching statements depending on user input
        if distancecb:
            y = pd.DataFrame(dataframe["delivery_days"])
            X = pd.DataFrame(
                dataframe[["freight_value", "price", "volume", "review_score", "product_weight_g", "distance"]])
            plot()
        else:
            y = pd.DataFrame(dataframe["delivery_days"])
            X = pd.DataFrame(dataframe[["freight_value", "price", "volume", "review_score", "product_weight_g"]])
            plot()

    # RANDOM FOREST REGRESSION
    def deep(df3):

        st.write("## Random Forest Regression")

        st.write("### Cleaning and Splitting Data")
        # creating new dataframe to store numerical variables
        numDF3 = pd.DataFrame(
            df3[['delivery_days', 'price', 'volume', 'product_weight_g', 'freight_value', 'review_score', 'distance']])

        # saving parameters/features/independent variables in X
        X = numDF3.iloc[:, 1:7].values

        # saving delivery day (dependant variable/what we want to estimate) in y
        y = numDF3.iloc[:, 0].values

        # displaying dataframe to allow user to visualise what we have done
        st.write("First we have created a new dataframe to store the numerical variables from our main dataframe.")
        st.write(numDF3)

        # splitting data into train and test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)  # Feature Scaling

        # Create a Gaussian Classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # plotting train and test data in graphs side by side
        subplot(y_train, y_train_pred, y_test, y_test_pred)

        # Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics

        # Model Accuracy, how often is the classifier correct?
        st.write("Accuracy of train data:", metrics.accuracy_score(y_train, y_train_pred))
        st.write("Accuracy of test data:", metrics.accuracy_score(y_test, y_test_pred))

        from sklearn.ensemble import RandomForestClassifier

        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)

        # creating a statistic table
        table(y_train, y_train_pred, y_test, y_test_pred)

        # dynamic analysis based on statistic table
        table_analysis(y_train, y_train_pred, y_test, y_test_pred)

        # make residual plot + analysis
        residual(y_train, y_train_pred, y_test, y_test_pred)

        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                               oob_score=False, random_state=None, verbose=0,
                               warm_start=False)

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

    models = ("Linear Regression", "Random Forest Regression")

    model_type = st.selectbox("Choose your model", models)

    if model_type == "Linear Regression":
        linearreg()
    elif model_type == "Random Forest Regression":
        df3 = dataframe[dataframe['delivery_days'] <= 50]
        df3 = df3.dropna()
        df3.reset_index(drop=True, inplace=True)
        deep(df3)
