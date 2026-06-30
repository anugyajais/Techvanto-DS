import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

ipldata=pd.read_csv(r"Datasets\ipl.csv")
ipldata
ipldata.shape
ipldata.columns	
ipldata.head(15)

ipldata.drop(['mid','venue','bowler','striker','non-striker'], axis=1, inplace=True)
ipldata

ipldata['bat_team'].unique()
teams= ipldata['bat_team'].unique()
teams=teams.tolist()

csk= ipldata[ipldata['bat_team']=='Chennai Super Kings']
csk	

ipldata= ipldata[ipldata['overs'] >=5]
ipldata
ipldata['date'] = pd.to_datetime(ipldata['date'])
ipldata.info()
corr_matrix = ipldata.corr(numeric_only=True)


plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of IPL Data")
plt.show()



X = ipldata.drop(['total', 'date', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'striker', 'non-striker'], axis=1, errors='ignore')
y = ipldata['total']

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Linear Regression): {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")


# df_encoded = pd.get_dummies(df, columns=['bat_team', 'bowl_team'])


# X = df_encoded.drop(['total'], axis=1)
# y = df_encoded['total']

#dropping date as it not works with scikit-learn
X.drop(columns=['date'], inplace=True)
#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
rmse = (mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Total Runs")
plt.ylabel("Predicted Total Runs")
plt.title("Actual vs Predicted Total Runs")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()


# df_8 = df_date[(df_date['date'].dt.year >= 2008) & (df_date['date'].dt.year <= 2016)]
ipldata.describe().T
ipldata.isnull().any()
df_copy = ipldata.copy()
x=df_copy.drop(['total'], axis='columns')
y= df_copy['overs']
# Using GridSearchCV to find the best algorithm for this problem
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
# # Creating a function to calculate best model for this problem
# def find_best_model(X, y):
#     models = {
#         'linear_regression': {
#             'model': LinearRegression(),
#             'parameters': {
#                 'normalize': [True, False]
#             }
#         },
#         'lasso': {
#             'model': Lasso(),
#             'parameters': {
#                 'alpha': [1, 2],
#                 'selection': ['random', 'cyclic']
#             }
#         }
#     }
    
# models = {
#     'decision_tree': {
#         'model': DecisionTreeRegressor(),
#         'parameters': {
#             'criterion': ['mse', 'friedman_mse'],
#             'splitter': ['best', 'random']
#         }
#     },
#     'random_forest': {
#         'model': RandomForestRegressor(criterion='mse'),
#         'parameters': {
#             'n_estimators': [5, 10, 15, 20]
#         }
#     },
#     'knn': {
#         'model': KNeighborsRegressor(algorithm='auto'),
#         'parameters': {
#             'n_neighbors': [2, 5, 10, 20]
#         }
#     }
# }

# scores = []

# for model_name, model_params in models.items():
# 	clf = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, return_train_score=False)
# 	clf.fit(X, y)
# 	scores.append({
# 		'model': model_name,
# 		'best_score': clf.best_score_,
# 		'best_params': clf.best_params_
# 	})


