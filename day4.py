import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# newArr =np.arange(640)
# mulltidimArray = newArr.reshape(40,4,4)
# arr2=newArr.reshape(4,4,40)
# newArr.shape
# mulltidimArray.shape
# mulltidimArray[0]
# arr2[0]
# newArr[np.newaxis]
# newArr.shape

newArr = np.arange(640)
mulltidimArray = newArr.reshape(40, 4, 4)
arr2 = newArr.reshape(4, 4, 40)
print(newArr.shape)
print(mulltidimArray.shape)
print(mulltidimArray[0])
print(arr2[0])
print(newArr[np.newaxis])
print(newArr[np.newaxis].shape)

data = pd.read_csv(r"D:\unishitz\Techvanto DS\Datasets\Covid Dataset.csv")

print(data.info())
print(data.describe().T)

print(data['COVID-19'].value_counts())

print(data.isnull().sum())
data = data.dropna()

data['COVID-19'] = data['COVID-19'].map({'Yes': 1, 'No': 0})
le = LabelEncoder()
data['Breathing Problem'] = le.fit_transform(data['Breathing Problem'])

# Plot actual data
plt.figure(figsize=(10, 5))
plt.scatter(data['Breathing Problem'], data['COVID-19'], alpha=0.5)
plt.xlabel('Breathing Problem')
plt.ylabel('COVID-19')
plt.title('Scatter: Breathing Problem vs COVID-19')
plt.show()

# Bar plot: Breathing Problem vs COVID-19 rate
breathing_covid = data.groupby('Breathing Problem')['COVID-19'].mean()
plt.figure(figsize=(6,4))
breathing_covid.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xticks([0, 1], ['No Breathing Problem', 'Breathing Problem'], rotation=0)
plt.ylabel('COVID-19 Rate')
plt.title('COVID-19 Rate by Breathing Problem')
plt.show()

# Correlation matrix
print(data.corr(numeric_only=True)[['Breathing Problem', 'COVID-19']])

# Compute correlation matrix
corr_matrix = data.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True)
plt.title("Correlation Heatmap of COVID Dataset", fontsize=16)
plt.show()

if 'Wearing Masks' in data.columns:
    data = data.drop(columns=['Wearing Masks'], axis=1)

x = data.drop('COVID-19', axis=1)
y = data['COVID-19']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Step 5: Predict cases and evaluate
y_pred = rf_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

start_date = "2022-01-01"
data['date'] = pd.date_range(start=start_date, periods=len(data), freq='D')
future_dates = pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
print(future_dates)


