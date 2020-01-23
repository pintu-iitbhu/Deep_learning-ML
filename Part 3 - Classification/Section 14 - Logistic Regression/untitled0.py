#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:15:23 2020

@author: edith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
dataset = pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[1,2,3]].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
X[:,0]=label.fit_transform(X[:,0])
onehot=OneHotEncoder(categorical_features=[0])
X=onehot.fit_transform(X).toarray()
X=X[:,2:4]

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_sc=sc_x.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train ,X_test,y_train,y_test= train_test_split(X_sc,y,test_size=.2,random_state=0)

from sklearn.linear_model import LogisticRegression
log_regressor=LogisticRegression()
log_regressor.fit(X_train,y_train)

y_pred=log_regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, log_regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()