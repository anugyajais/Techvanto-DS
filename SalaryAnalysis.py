import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv(r'Datasets/Salary_Data.csv.xls')
dataset
X = dataset.iloc[:, :-1].values  # Features (Years of Experience)
X
y = dataset.iloc[:, -1].values   # Target (Salary)
y

# since just one feature hence 'simple' linear regression
# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show