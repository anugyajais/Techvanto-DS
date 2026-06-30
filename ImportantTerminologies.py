"""The machine learning process-
1-Data Preprocessing
2-Modelling
3-Evaluation

Training set and Test Set Data SPLIT

Feature Scaling

Simple linear reg



"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv("Datasets/sample.csv")

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

x
y
print(x)
print(y)

# feature scaling is for equal distribution of data and reduce noise and remove outliers
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x=np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train , y_test= train_test_split(x ,y , test_size=0.2, random_state=42)

# normalisation vs standardisation

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train
x_test
