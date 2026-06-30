# employee attriition prediction using logistic regression
# aka- will employee leave the company or not
# support vector machine


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r"Datasets\WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.shape
data.columns
data.info()
data.describe()
print("Preprocessing dataset...")
# Convert target variable to binary
data['Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
	data[col] = le.fit_transform(data[col])
	X = data.drop('Attrition', axis=1)
	y = data['Attrition']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Neural Network": MLPClassifier(max_iter=500)
}