#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 22:22:05 2020

@author: edith
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset= pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label= LabelEncoder()
X[:,0]=label.fit_transform(X[:,0])

onehotencoder= OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray()
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test= train_test_split(X,y,test_size=0.2,random_state= 0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_train)


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly)
lin_reg= LinearRegression()
lin_reg.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X_poly),color='blue')