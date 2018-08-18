# Polynomial Linear Regression
""" 
AIM :: Predict Salary of new employee based on model trained on salary of 
current employees as per their positions(LEVEL).
"""

import pandas as pd
import numpy as np
import matplotlib as plt

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]
