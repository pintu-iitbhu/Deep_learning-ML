#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:16:12 2020

@author: edith
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,[1]].values
y=dataset.iloc[:,[2]].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)
regressor.predict(X)

plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X),color='blue')
"""
def sigmoid(x,bet1,bet2):
    y=1/(1+ np.exp(-bet1(x-bet2)))
    return y

from scipy.optimize import curve_fit
popt, pcov =curve_fit(sigmoid,X,y)

x = np.linspace(1,10, 10).astype(int)
X=X/max(X)
plt.figure(figsize=(8,5))
y = sigmoid(X, *popt)
plt.plot(X, y, 'ro', label='data')
plt.plot(X,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
"""

