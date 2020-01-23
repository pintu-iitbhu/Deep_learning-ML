#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:57:18 2020

@author: edith
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,[1]].values
y=dataset.iloc[:,[2]].values

t=np.array([6.5])
t=t.reshape(1,1)
from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor(criterion='mse',n_estimators=10,random_state=0)
RFR.fit(X,y)
y_pred=RFR.predict(t)

X_x=np.arange(min(X),max(X), 0.01)
X_x=X_x.reshape(len(X_x),1)
plt.scatter(X,y,color='red')
plt.scatter(t,y_pred,color='black')
plt.plot(X_x,RFR.predict(X_x),color='blue')

