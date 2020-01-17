#Data preprocessing 

#importing Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Dataset
dataset = pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#Taking Care of Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy="mean",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding Categoriacl values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

# Dummy Encoding of Dataset
onehotencoder= OneHotEncoder(categorical_features= [0])
X= onehotencoder.fit_transform(X).toarray()

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#splitting the Data into training and test Data set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X, Y, test_size=0.2,random_state=0)

# Feature Scalling of dataset Standardistion and Normalizing
from sklearn.preprocessing import StandardScaler
cs_X= StandardScaler()
x_train=cs_X.fit_transform(x_train)
x_test=cs_X.transform(x_test)

print(Y)




