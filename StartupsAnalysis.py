import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the startup CSV file from the datasets folder
df = pd.read_csv('Datasets/50_Startups.csv')
df
# multiple factors are affectinig the profit value(target)
# hence multiple linear reg.

x= df.iloc[:,:-1]
x
y=df.iloc[:,-1]
y

# Identify categorical columns (assuming 'State' is the only categorical column)
categorical_cols = ['State']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')

x = np.array(ct.fit_transform(x))
x

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42) 

# Create and train the multiple linear regression model
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predict the test set results
ypred = regressor.predict(xtest)

# Optionally, print or compare predictions
print("Predicted profits:", ypred)
print("Actual profits:", ytest.values)

ypred= ypred.reshape(len(ypred),1)
ypred

# Visualize actual vs predicted profits for the test set
plt.figure(figsize=(10,6))
plt.scatter(range(len(ytest)), ytest, color='blue', label='Actual Profits')
plt.scatter(range(len(ypred)), ypred, color='red', label='Predicted Profits', marker='x')
plt.title('Actual vs Predicted Profits (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Profit')
plt.legend()
plt.plot(range(len(ytest)), ytest.values, color='blue', linestyle='-', marker='o', label='Actual Profits')
plt.plot(range(len(ypred)), ypred, color='red', linestyle='--', marker='x', label='Predicted Profits')
plt.show()