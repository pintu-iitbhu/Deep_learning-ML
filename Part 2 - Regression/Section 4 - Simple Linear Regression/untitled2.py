#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:05:04 2020

@author: edith
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset= pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


from sklearn.preprocessing import StandardScaler
SC_x= StandardScaler()
X_SC=SC_x.fit_transform(X)
SC_y= StandardScaler()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X_SC,y, test_size=.2, random_state=0)

from sklearn.linear_model import LinearRegression
linear= LinearRegression()
linear.fit(X_train,y_train)

y_pred= linear.predict(X_train)

plt.scatter(X_train, y_train,color='red')
plt.plot(X_train,y_pred,color='blue')