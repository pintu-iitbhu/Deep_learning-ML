#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:50:57 2020

@author: edith
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datset= pd.read_csv('50_Startups.csv')
X=datset.iloc[:,:-1].values
y=datset.iloc[:,4]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label=LabelEncoder()
X[:,3]=label.fit_transform(X[:,3])
onehotencoder= OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
"""from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
X=sc_x.fit_transform(X)
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,3,5]]

result=regresor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
print(result.summary())







