import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error	
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


cars_data = pd.read_csv("Datasets/cars.csv")
cars_data.head()
cars_data.shape
cars_data.columns
cars_data.info()	
cars_data.describe()
# cars_data["Electric_Range"].unique()
# cars_data["Electric_Range"].isna().sum()
cars_data = cars_data.replace(np.nan, 0)  # Replace NaN with 0
cars_data.shape
df_cleaned = cars_data.dropna(inplace=True)
df_cleaned = cars_data.dropna()

df= cars_data[["Cylinders"],["Valves_Per_Cylinder"],["Doors"],["Seating_Capacity"],["Number_of_Airbags"],["USB_Ports"],["Year"]]
