import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load data from the salary dataset in the dataset folder
df = pd.read_csv('Datasets/Salary_Data.csv')

df
# Select only numeric columns for scaling and clustering
numeric_df = df.select_dtypes(include=[np.number])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numeric data
scaled_data = scaler.fit_transform(df[['YearsExperience', 'Salary']])

# Convert back to DataFrame for readability
scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
print(scaled_df)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(scaled_data)
df['Cluster'] = kmeans.fit_predict(scaled_data)
plt.scatter(df['YearsExperience'], df['Salary'], c=df['Cluster'], cmap='viridis', s=100, alpha=0.6)
plt.title('KMeans Clustering on Salary Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()
# Add cluster labels to the original DataFrame
# df['Cluster'] = kmeans.labels_

# Print the DataFrame with cluster assignments
print(df)



# ---------------------------------------------------------------

# Load data from a different dataset in the Datasets folder
df2 = pd.read_csv('Datasets/iris.data.csv')

# Select only numeric columns for scaling and clustering
numeric_df2 = df2.select_dtypes(include=[np.number])

# Initialize MinMaxScaler
scaler2 = MinMaxScaler()

# Fit and transform the numeric data
scaled_data2 = scaler2.fit_transform(numeric_df2)

# Convert back to DataFrame for readability
scaled_df2 = pd.DataFrame(scaled_data2, columns=numeric_df2.columns)
print(scaled_df2)

# Apply KMeans clustering (let's use 3 clusters for Iris)
kmeans2 = KMeans(n_clusters=3, random_state=0)
kmeans2.fit(scaled_data2)
df2['Cluster'] = kmeans2.fit_predict(scaled_data2)

# Visualize clusters using the first two numeric features
feature_cols = numeric_df2.columns[:2]
plt.scatter(df2[feature_cols[0]], df2[feature_cols[1]], c=df2['Cluster'], cmap='viridis', s=100, alpha=0.6)
plt.title('KMeans Clustering on Iris Data')
plt.xlabel(feature_cols[0])
plt.ylabel(feature_cols[1])
plt.grid(True)
plt.show()

# Print the DataFrame with cluster assignments
print(df2)