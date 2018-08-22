# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the results on test set
y_pred = regressor.predict(X_test)

# visualising the results of training set in matplot(graph)
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.plot(X_train, y_train, color="yellow")
xlabel = "experience"
ylabel = "Salary"
plt.title="Salary vs experience"
plt.show()
