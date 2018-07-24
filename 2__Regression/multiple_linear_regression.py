import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset and setting explanatory variable and dependent variable
dataset = pd.read_csv('50_startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# encoding categorical data (here state name)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# avoiding the dummy variable trap
x = x[:,1:]

# splitting dataset
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(
        x, y, test_size = 0.2, random_state = 0)


# fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set results
y_pred = regressor.predict(x_test)

# building the optimal model using backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_optimal = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_ols.summary()

x_optimal = x[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_ols.summary()

x_optimal = x[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_ols.summary()

x_optimal = x[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_ols.summary()

x_optimal = x[:, [0,3]]
regressor_ols = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_ols.summary()
