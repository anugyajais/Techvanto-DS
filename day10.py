import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Iris Dataset Preprocessing Script
# This script downloads the Iris dataset, handles missing values, and encodes categorical variables.
# All steps are performed in a linear fashion without using custom functions.

# Step 1: Download the Iris dataset

df=pd.read_csv(r"Datasets/iris.data.csv")
# Step 2: Load the dataset into a pandas DataFrame
# The dataset does not have headers, so we add them manually.
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(r"Datasets/iris.csv", header=None, names=column_names)

# Step 3: Handle missing values
# Check for missing values and fill them with the column mean (for numeric columns)
# For demonstration, let's assume missing values are represented as NaN
df.replace('?', np.nan, inplace=True)
for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
	df[col] = pd.to_numeric(df[col], errors='coerce')
	if df[col].isnull().sum() > 0:
		df[col].fillna(df[col].mean(), inplace=True)

# Step 4: Encoding categorical variables
# Encode the 'species' column using Label Encoding
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Step 5: Save the preprocessed dataset (optional)
df.to_csv("iris_preprocessed.csv", index=False)

# Display the first few rows of the preprocessed DataFrame
print(df.head())