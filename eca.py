# -*- coding: utf-8 -*-
"""eca.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13KfkjyHWyJ2v8Dp5V_O2UQegNCxwWjzd
"""

import pandas as pd
import re
import numpy as np
import itertools

#masukkan dataset yang dibutuhkan dengan alamat penyimpanan yang tepat dan simpan kedalam sebuah variabel
dir = '/content/processed.hungarian.data'
#buatlah iterasi untuk membaca dataset
with open(dir, encoding='Latin1') as file:
  lines = [line.strip() for line in file]


lines[0:10]

# Splitting the strings and creating a list of lists
data_list = [line.split(",") for line in lines]

# Replace '?' with NaN
data_list = [[np.nan if x == '?' else x for x in row] for row in data_list]

# Create a Pandas DataFrame
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.DataFrame(data_list, columns=columns)

# Convert columns to numeric (excluding 'sex' and 'target' columns)
numeric_columns = df.columns.difference(['sex', 'target'])
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Display the DataFrame
df

df.isnull().sum() * 100 / len(df)

# banyak missing value
# slope       64.625850
# ca          98.979592
# thal        90.476190

df=df.drop(columns=['ca', 'thal','slope'])
df

# trestbps     0.340136
# chol         7.823129
# fbs          2.721088
# restecg      0.340136
# thalach      0.340136
# exang        0.340136
# do imputation e.g mean median etc

df.dropna(axis=0, subset=['trestbps','chol','fbs','restecg','thalach','exang'], inplace=False)
df.info()

# fill nan dengan mean tiap kolom

df['trestbps']=df['trestbps'].fillna(df['trestbps'].mean())
df['chol']=df['chol'].fillna(df['chol'].mean())
df['fbs']=df['fbs'].fillna(df['fbs'].mean())
df['restecg']=df['restecg'].fillna(df['restecg'].mean())
df['thalach']=df['thalach'].fillna(df['thalach'].mean())
df['exang']=df['exang'].fillna(df['exang'].mean())
df.info()

# ubah tipe data ke float
df=df.astype(float)
df.info()

df[df.duplicated(keep=False)]

df = df.drop_duplicates()

#spliting train dan test
X = df.drop("target",axis=1).values
y = df.iloc[:,-1]

# cek imbalance 'target'

df['target'].value_counts()

# visualize
import seaborn as sns

ax=sns.countplot(df, x=df['target'])
ax.bar_label(ax.containers[0], fontsize=10);

# do over_sampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

import matplotlib.pyplot as plt

target_sebelum_SMOTE = y
target_sesudah_SMOTE = y_smote_resampled

fig, axes = plt.subplots(1, 2, figsize=(10,5))

ax=sns.countplot(df, x=target_sebelum_SMOTE,ax=axes[0])
ax.set_title('Sebelum Oversample')

ax=sns.countplot(df, x=target_sesudah_SMOTE,ax=axes[1])
ax.set_title('Sesudah Oversample')

target_sesudah_SMOTE.value_counts()

# cek std(standar deviasi)
# jika standar deviasi tinggi, maka data cenderung tersebar lebih luas dari nilai rata-rata, yang menunjukkan variasi yang besar dalam data tersebut
# Semakin dekat standar deviasi ke nol, semakin rendah variabilitas data dan rata-rata semakin dapat diandalkan

df.describe()

# beberapa std kolom jauh dari nol maka perlu normalisasi
# Normalisasi pada dasarnya adalah teknik perubahan skala yang mana kita merubah nilai dari data kedalam skala diantara 0–1.
# Teknik ini biasa juga disebut sebagai Min-Max scaling.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_smote_resampled_normal = scaler.fit_transform(X_smote_resampled)
len(X_smote_resampled_normal)

dfcek1 = pd.DataFrame(X_smote_resampled_normal)
dfcek1.describe()

from sklearn.model_selection import train_test_split
# membagi fitur dan target menjadi data train dan test (untuk yang oversample saja)
X_train, X_test, y_train, y_test = train_test_split(X_smote_resampled, y_smote_resampled, test_size=0.2, random_state=42,stratify=y_smote_resampled)
# membagi fitur dan target menjadi data train dan test (untuk yang oversample + normalization)
X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(X_smote_resampled_normal, y_smote_resampled, test_size=0.2, random_state=42,stratify = y_smote_resampled)

from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,confusion_matrix,precision_score
def evaluation(Y_test,Y_pred):
  acc = accuracy_score(Y_test,Y_pred)
  rcl = recall_score(Y_test,Y_pred,average = 'weighted')
  f1 = f1_score(Y_test,Y_pred,average = 'weighted')
  ps = precision_score(Y_test,Y_pred,average = 'weighted')
  metric_dict={'accuracy': round(acc,3),
  'recall': round(rcl,3),
  'F1 score': round(f1,3),
  'Precision score': round(ps,3)
  }
  return print(metric_dict)

"""#KNN"""

# masuk ke model development
# Pada tahap ini kita akan akan memulai membangun model dengan algoritma KNN dengan nilai neighbors yaitu 3.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

knn_model = KNeighborsClassifier(n_neighbors = 3)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
# Evaluate the KNN model
print("K-Nearest Neighbors (KNN) Model:")
accuracy_knn_smote = round(accuracy_score(y_test,y_pred_knn),3)
print("Accuracy:", accuracy_knn_smote)
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))

evaluation(y_test,y_pred_knn)

cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#Random Forest"""

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf_smote = round(accuracy_score(y_test, y_pred_rf),3)
print("Accuracy:",accuracy_rf_smote)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

evaluation(y_test,y_pred_rf)

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')

"""#XGB"""

xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
# Evaluate the XGBoost model
print("\nXGBoost Model:")
accuracy_xgb_smote = round(accuracy_score(y_test, y_pred_xgb),3)
print("Accuracy:",accuracy_xgb_smote)
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))

evaluation(y_test,y_pred_xgb)

cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#KNN+normalisasi"""

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_normal, y_train_normal)

y_pred_knn = knn_model.predict(X_test_normal)
# Evaluate the KNN model
print("K-Nearest Neighbors (KNN) Model:")
accuracy_knn_smote_normal = round(accuracy_score(y_test_normal,y_pred_knn),3)
print("Accuracy:", accuracy_knn_smote_normal)
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_knn))

evaluation(y_test_normal,y_pred_knn)

cm = confusion_matrix(y_test_normal, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#Random Forest+normalisasi"""

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_normal, y_train_normal)

y_pred_rf = rf_model.predict(X_test_normal)
# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf_smote_normal = round(accuracy_score(y_test_normal, y_pred_rf),3)
print("Accuracy:",accuracy_rf_smote_normal )
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_rf))

evaluation(y_test_normal,y_pred_rf)

cm = confusion_matrix(y_test_normal, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#XGB+Normalisasi"""

xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
xgb_model.fit(X_train_normal, y_train_normal)

y_pred_xgb = xgb_model.predict(X_test_normal)
# Evaluate the XGBoost model
print("\nXGBoost Model:")
accuracy_xgb_smote_normal = round(accuracy_score(y_test_normal, y_pred_xgb),3)
print("Accuracy:",accuracy_xgb_smote_normal)
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_xgb))

evaluation(y_test_normal,y_pred_xgb)

cm = confusion_matrix(y_test_normal, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#KNN+nomalisasi+oversample"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

knn_model = KNeighborsClassifier()
param_grid = {
  "n_neighbors": range(3, 21),
  "metric": ["euclidean", "manhattan", "chebyshev"],
  "weights": ["uniform", "distance"],
  "algorithm": ["auto", "ball_tree", "kd_tree"],
  "leaf_size": range(10, 61),
}
knn_model = RandomizedSearchCV(estimator=knn_model, param_distributions=param_grid, n_iter=100, scoring="accuracy", cv=5)
knn_model.fit(X_train_normal, y_train_normal)
best_params = knn_model.best_params_
print(f"Best parameters: {best_params}")

y_pred_knn = knn_model.predict(X_test_normal)
# Evaluate the KNN model
print("K-Nearest Neighbors (KNN) Model:")
accuracy_knn_smote_normal_Tun = round(accuracy_score(y_test_normal,y_pred_knn),3)
print("Accuracy:", accuracy_knn_smote_normal_Tun)
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_knn))

evaluation(y_test_normal,y_pred_knn)

cm = confusion_matrix(y_test_normal, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#Random Forest+normalisasi+oversample"""

rf_model = RandomForestClassifier()
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [ 10, 15],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"]
# "random_state": [42, 100, 200]
}
rf_model = RandomizedSearchCV(rf_model, param_grid, n_iter=100, cv=5, n_jobs=-1)
rf_model.fit(X_train_normal, y_train_normal)
best_params = rf_model.best_params_
print(f"Best parameters: {best_params}")

y_pred_rf = rf_model.predict(X_test_normal)
# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf_smote_normal_Tun = round(accuracy_score(y_test_normal, y_pred_rf),3)
print("Accuracy:",accuracy_rf_smote_normal_Tun)
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_rf))

evaluation(y_test_normal,y_pred_rf)

cm = confusion_matrix(y_test_normal, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

"""#XGB+Normalisasi+oversample"""

xgb_model = XGBClassifier()
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 200],
    "gamma": [0, 0.1],
    "colsample_bytree": [0.7, 0.8],
}
xgb_model = RandomizedSearchCV(xgb_model, param_grid, n_iter=10, cv=5, n_jobs=-1)
xgb_model.fit(X_train_normal, y_train_normal)
best_params = xgb_model.best_params_
print(f"Best parameters: {best_params}")

y_pred_xgb = xgb_model.predict(X_test_normal)
# Evaluate the XGBoost model
print("\nXGBoost Model:")
accuracy_xgb_smote_normal_Tun = round(accuracy_score(y_test_normal, y_pred_xgb),3)
print("Accuracy:",accuracy_xgb_smote_normal_Tun)
print("Classification Report:")
print(classification_report(y_test_normal, y_pred_xgb))

evaluation(y_test_normal,y_pred_xgb)

cm = confusion_matrix(y_test_normal, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

import pickle

with open('xgb_normalisasi_oversample_tuning.pkl', 'wb') as files:
    pickle.dump(xgb_model, files)

with open('xgb_normalisasi_oversample_tuning.pkl', 'rb') as files:
    pickle.load(files)