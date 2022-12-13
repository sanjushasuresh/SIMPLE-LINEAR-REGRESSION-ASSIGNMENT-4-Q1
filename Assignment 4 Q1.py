# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:06:46 2022

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("delivery_time.csv")
df.head()
df.shape
df.isnull().sum() 
# There are no null values
df.describe()

# EDA (boxplot, scatterplot, histogram)
df.boxplot("Delivery Time", vert=False)
df.boxplot("Sorting Time", vert=False)
# There are no outliers
# In Delivery Time, since the median is towards the upper whisker the plot is negatively skewed and IQR is approx. 7
# In Sorting Time, the median is correct in middle at 6, so there is no skewness and IQR value = 8-4=4

df.plot.scatter(x="Delivery Time", y="Sorting Time")
# Here, if Sorting Time increases then Delivery Time also increases

df["Delivery Time"].hist()
df["Sorting Time"].hist()
# Both the graphs are not bell shaped and have gap

df.corr()
# Both variables are strong positively correlated and the correlation b/w the variables is 0.825997


# Splitting the variables
Y = df[["Delivery Time"]]
X = df[["Sorting Time"]]


# Model fitting
# MODEL 1
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
y1 = LR.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 7.79
 
rmse = np.sqrt(mse).round(4)
# RMSE = 2.79

r2_score(Y,y1)
# r2score = 0.68227 (68%)



# Transformations
# MODEL 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1=LR.predict(np.log(X))

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 7.47

rmse = np.sqrt(mse).round(3)
# RMSE = 2.73

r2_score(Y,y1)
# r2score = 0.69544 (69%)

#create log-transformed data
df_log = np.log(df)
#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)
#create histograms
axs[0].hist(df, edgecolor='black')
axs[1].hist(df_log, edgecolor='black')
#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')



# MODEL 3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1=LR.predict(np.sqrt(X))

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 7.46

rmse = np.sqrt(mse).round(3)
# RMSE = 2.73

from sklearn.metrics import r2_score
r2_score(Y,y1)
# r2score = 0.69580 (69%)

#create sqrt-transformed data
df_sqrt = np.sqrt(df)
#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)
#create histograms
axs[0].hist(df, edgecolor='black')
axs[1].hist(df_sqrt, edgecolor='black')
#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Square Root Transformed Data')



# MODEL 4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1=LR.predict(X**2)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y,y1)
# MSE = 9.06

rmse = np.sqrt(mse).round(3)
# RMSE = 3.011

r2_score(Y,y1)
# r2score = 0.63028 (63%)

#create cbrt-transformed data
df_cbrt = np.cbrt(df)
#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)
#create histograms
axs[0].hist(df, edgecolor='black')
axs[1].hist(df_cbrt, edgecolor='black')
#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Cube Root Transformed Data')

# Inference : Delivery time is predicted using sorting time and the best model selected is model 3
# which is transformed using squared tranformation because the graph is bell shaped and its rscore is 0.69580.


