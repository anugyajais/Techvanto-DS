# Converted from CreditCardFraud.ipynb
# Notebook -> Python conversion. Markdown reduced to a few comment lines.

# --- markdown (reduced) ---
# # <h1 style="font-family: Trebuchet MS; padding: 20px; font-size: 40px; color: #FFD700; text-align: center; line-height: 0.55;background-color: #3B3B3C"><b>Credit Card Fraud Detection</b><br></h1>

# --- markdown (reduced) ---
# <center>
#     <img src="https://tse4.mm.bing.net/th/id/OIP.b8bt2jer5KdrVDCXq2sfNgAAAA?rs=1&pid=ImgDetMain&o=7&rm=3" alt="Credit Card Fraud Detection" width="50%">
# </center>

# --- markdown (reduced) ---
# ### <center>Dataset Attributes</center>
# - **V1 - V28** : Numerical features that are a result of PCA transformation.
# - **Time** : Seconds elapsed between each transaction and the 1st transaction.

# --- markdown (reduced) ---
# ### Notebook Contents :
# - Dataset Information
# - Data Visualization

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Dataset Information</div></center>

# --- markdown (reduced) ---
# ### Import the Necessary Libraries :

# --- code cell ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
pd.options.display.float_format = '{:.2f}'.format

# --- code cell ---
data = pd.read_csv('../data/creditcard.csv')
data.head()

# --- markdown (reduced) ---
# ### Data Info :

# --- code cell ---
data.shape

# --- code cell ---
data.columns

# --- code cell ---
data.info()

# --- code cell ---
sns.heatmap(data.isnull(),cmap = 'magma',cbar = False)

# --- markdown (reduced) ---
# - **No null values** present in the data!

# --- code cell ---
data.describe()

# --- code cell ---
fraud = data[data['Class'] == 1].describe().T
nofraud = data[data['Class'] == 0].describe().T

colors = ['#FFD700','#3B3B3C']

fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (5,15))
plt.subplot(2,2,1)
sns.heatmap(fraud[['mean']][:15],annot = True,cmap = colors,linewidths = 0.5,linecolor = 'black',cbar = False,fmt = '.2f')
plt.title('Fraud Samples : Part 1');

plt.subplot(2,2,2)
sns.heatmap(fraud[['mean']][15:30],annot = True,cmap = colors,linewidths = 0.5,linecolor = 'black',cbar = False,fmt = '.2f')
plt.title('Fraud Samples : Part 2');

plt.subplot(2,2,3)
sns.heatmap(nofraud[['mean']][:15],annot = True,cmap = colors,linewidths = 0.5,linecolor = 'black',cbar = False,fmt = '.2f')
plt.title('No Fraud Samples : Part 1');

plt.subplot(2,2,4)
sns.heatmap(nofraud[['mean']][15:30],annot = True,cmap = colors,linewidths = 0.5,linecolor = 'black',cbar = False,fmt = '.2f')
plt.title('No Fraud Samples : Part 2');

fig.tight_layout(w_pad = 2)

# --- markdown (reduced) ---
# - **Mean** values of features for **Fraud** & **No Fraud** cases!
# - For **No Fraud** cases, **V1 - V28** mean values are almost **0** for all the cases. Mean **Amount**, 88.29, is less than the mean transaction amount, 122.21, of the **Fraud** cases.
# - **Time** taken for **No Fraud** transactions is more than those for **Fraud** transactions.

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Data Visualization</div></center>

# --- markdown (reduced) ---
# ### Target Variable Visualization (Class) :

# --- code cell ---
fraud = len(data[data['Class'] == 1]) / len(data) * 100
nofraud = len(data[data['Class'] == 0]) / len(data) * 100
fraud_percentage = [nofraud, fraud]

fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (20,5))
plt.subplot(1,2,1)
plt.pie(fraud_percentage, labels = ['No Fraud','Fraud'], autopct='%1.1f%%', startangle = 90, colors = colors,
       wedgeprops = {'edgecolor' : 'black', 'linewidth': 1, 'antialiased' : True})

plt.subplot(1,2,2)
ax = sns.countplot(data=data, x='Class', hue='Class', legend=False, palette=colors, edgecolor='black')
# Set custom x-axis labels
plt.xticks([0, 1], ['No Fraud', 'Fraud'])
# Add value labels on top of bars
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, 
            int(rect.get_height()), horizontalalignment='center', fontsize=11)
plt.title('Number of Fraud Cases')

# --- markdown (reduced) ---
# - The data is clearly **highly unbalanced** with majority of the transactions being **No Fraud**.
# - Due to highly unbalanced data, the classification model will bias its prediction towards the majority class, **No Fraud**.
# - Hence, data balancing becomes a crucial part in building a robust model.

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Feature Selection</div></center>

# --- markdown (reduced) ---
# ### Correlation Matrix :

# --- code cell ---
sns.heatmap(data.corr(),cmap = colors,cbar = True)

# --- markdown (reduced) ---
# - There are too many features in the dataset and it is difficult to understand anything.
# - Hence, we will plot the correlation map only with the target variable.

# --- code cell ---
corr = data.corrwith(data['Class']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlation']
fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (5,10))

plt.subplot(1,2,1)
sns.heatmap(corr.iloc[:15,:],annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',cbar = False)
plt.title('Part 1')

plt.subplot(1,2,2)
sns.heatmap(corr.iloc[15:30],annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',cbar = False)
plt.title('Part 2')

fig.tight_layout(w_pad = 2)

# --- markdown (reduced) ---
# - For feature selection, we will **exclude** the features having correlation values between **[-0.1,0.1]**.
# - V4, V11 are positively correlated and V7, V3, V16, V10, V12, V14, V17 are negatively correlated with the **Class** feature.

# --- markdown (reduced) ---
# ### ANOVA Test :

# --- code cell ---
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# --- code cell ---
features = data.loc[:,:'Amount']
target = data.loc[:,'Class']

best_features = SelectKBest(score_func = f_classif,k = 'all')
fit = best_features.fit(features,target)

featureScores = pd.DataFrame(data = fit.scores_,index = list(features.columns),columns = ['ANOVA Score']) 
featureScores = featureScores.sort_values(ascending = False,by = 'ANOVA Score')

fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (5,10))

plt.subplot(1,2,1)
sns.heatmap(featureScores.iloc[:15,:],annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',cbar = False, fmt = '.2f')
plt.title('ANOVA Score : Part 1')

plt.subplot(1,2,2)
sns.heatmap(featureScores.iloc[15:30],annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black',cbar = False, fmt = '.2f')
plt.title('ANOVA Score : Part 2')

fig.tight_layout(w_pad = 2)

# --- markdown (reduced) ---
# - Higher the value of the ANOVA score, higher the importance of that feature with the target variable.
# - From the above plot, we will reject features with values less than 50.
# - In this case, we will create 2 models based on features selected from the **Correlation Plot** & **ANOVA Score**.

# --- markdown (reduced) ---
# #### Dataset for Model based on Correlation Plot :

# --- code cell ---
df1 = data[['V3','V4','V7','V10','V11','V12','V14','V16','V17','Class']].copy(deep = True)
df1.head()

# --- markdown (reduced) ---
# #### Dataset for Model based on ANOVA Score :

# --- code cell ---
df2 = data.copy(deep = True)
df2.drop(columns = list(featureScores.index[20:]),inplace = True)
df2.head()

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Data Balancing</div></center>
# - In order to cope with unbalanced data, there are 2 options :
#     - **Undersampling** : Trim down the majority samples of the target variable.

# --- code cell ---
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# --- markdown (reduced) ---
# #### Data Balancing for Model based on Correlation Plot :

# --- code cell ---
over = SMOTE(sampling_strategy = 0.5)
under = RandomUnderSampler(sampling_strategy = 0.1)
f1 = df1.iloc[:,:9].values
t1 = df1.iloc[:,9].values

steps = [('under', under),('over', over)]
pipeline = Pipeline(steps=steps)
f1, t1 = pipeline.fit_resample(f1, t1)
Counter(t1)

# --- markdown (reduced) ---
# #### Data Balancing for Model based on ANOVA Score :

# --- code cell ---
over = SMOTE(sampling_strategy = 0.5)
under = RandomUnderSampler(sampling_strategy = 0.1)
f2 = df2.iloc[:,:20].values
t2 = df2.iloc[:,20].values

steps = [('under', under),('over', over)]
pipeline = Pipeline(steps=steps)
f2, t2 = pipeline.fit_resample(f2, t2)
Counter(t2)

# --- markdown (reduced) ---
# ### Calculation for Data Balancing :
# - **Sampling Strategy** : It is a ratio which is the common paramter for oversampling and undersampling.
# - **Sampling Strategy** : **( Samples of Minority Class ) / ( Samples of Majority Class )**

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Modeling</div></center>

# --- code cell ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
# from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve

# --- code cell ---
x_train1, x_test1, y_train1, y_test1 = train_test_split(f1, t1, test_size = 0.20, random_state = 2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(f2, t2, test_size = 0.20, random_state = 2)

# --- markdown (reduced) ---
# - Splitting the data into **80 - 20 train - test** groups.

# --- code cell ---
def model(classifier,x_train,y_train,x_test,y_test):
    
    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
    RocCurveDisplay.from_estimator(classifier, x_test, y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()
    
def model_evaluation(classifier,x_test,y_test):
    
    # Confusion Matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
    
    # Classification Report
    print(classification_report(y_test,classifier.predict(x_test)))

# --- markdown (reduced) ---
# ### 1] Logistic Regression :

# --- code cell ---
from sklearn.linear_model import LogisticRegression

# --- code cell ---
classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2')

# --- markdown (reduced) ---
# #### Model based on Correlation Plot :

# --- code cell ---
model(classifier_lr,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_lr,x_test1,y_test1)

# --- markdown (reduced) ---
# #### Model based on ANOVA Score :

# --- code cell ---
model(classifier_lr,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_lr,x_test2,y_test2)

# --- markdown (reduced) ---
# ### 2] Support Vector Classifier :

# --- code cell ---
from sklearn.svm import SVC

# --- code cell ---
classifier_svc = SVC(kernel = 'linear',C = 0.1)

# --- markdown (reduced) ---
# #### Model based on Correlation Plot :

# --- code cell ---
model(classifier_svc,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_svc,x_test1,y_test1)

# --- markdown (reduced) ---
# #### Model based on ANOVA Score :

# --- code cell ---
model(classifier_svc,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_svc,x_test2,y_test2)

# --- markdown (reduced) ---
# ### 3] Decision Tree Classifier :

# --- code cell ---
from sklearn.tree import DecisionTreeClassifier

# --- code cell ---
classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)

# --- markdown (reduced) ---
# #### Model based on Correlation Plot :

# --- code cell ---
model(classifier_dt,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_dt,x_test1,y_test1)

# --- markdown (reduced) ---
# #### Model based on ANOVA Score :

# --- code cell ---
model(classifier_dt,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_dt,x_test2,y_test2)

# --- markdown (reduced) ---
# ### 4] Random Forest Classifier :

# --- code cell ---
from sklearn.ensemble import RandomForestClassifier

# --- code cell ---
classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)

# --- markdown (reduced) ---
# #### Model based on Correlation Plot :

# --- code cell ---
model(classifier_rf,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_rf,x_test1,y_test1)

# --- markdown (reduced) ---
# #### Model based on ANOVA Score :

# --- code cell ---
model(classifier_rf,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_rf,x_test2,y_test2)

# --- markdown (reduced) ---
# ### 5] K-Nearest Neighbors :

# --- code cell ---
from sklearn.neighbors import KNeighborsClassifier

# --- code cell ---
classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)

# --- markdown (reduced) ---
# #### Model based on Correlation Plot :

# --- code cell ---
model(classifier_knn,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_knn,x_test1,y_test1)

# --- markdown (reduced) ---
# #### Model based on ANOVA Score :

# --- code cell ---
model(classifier_knn,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_knn,x_test2,y_test2)

# --- markdown (reduced) ---
# ### ML Alogrithm Results Table :
# #### Results Table for models based on Correlation Plot :
# |Sr. No.|ML Algorithm|Cross Validation Score|ROC AUC Score|F1 Score (Fraud)|

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Conclusion</div></center>
# - This is a great dataset to learn about binary classification problem with unbalanced data.
# - As the features are disguised, feature selection cannot be assisted based on the domain knowledge of the topic. Statistical tests hold the complete importance to select features for modeling.

# --- markdown (reduced) ---
# ### References :
# - https://www.chargebackgurus.com/blog/credit-card-fraud-detection
# - https://www.cnbc.com/select/what-is-a-credit-card/

# --- markdown (reduced) ---
# # <center><div style="font-family: Trebuchet MS; background-color: #3B3B3C; color: #FFD700; padding: 12px; line-height: 1;">Thank You!</div></center>

