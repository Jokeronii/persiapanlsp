# -*- coding: utf-8 -*-
"""Soal_LSP_ADC_coba.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18Asfn1H8qVP-uKhHeyFAKAwAJdej3iJ_

### Daftar Isi
* [1) Mengumpulkan Data](#h1)
* [2) Menelaah Data](#h2)
* [3) Memvalidasi Data](#h3)
* [4) Menetukan Object Data](#h4)
* [5) Membersihkan Data](#h5)
* [6) Mengkonstruksi Data](#h6)
* [7) Menentukan Label Data](#h7)
* [8) Membangun Model](#h8)
* [9) Mengevaluasi Hasil Pemodelan](#h9)
* [10) Optimasi Model Klasifikasi](#h10)

## 1) Mengumpulkan Data <a class="anchor" id="h1"></a>
"""

# Load library yang diperlukan
import pandas as pd
import re
import numpy as np
import itertools

# Load data menjadi data frame
dataset = pd.read_csv('breast-cancer-wisconsin.data', header=None)

# menampilkan data
dataset

# Mengganti nama kolom dalam dataset
dataset.columns = ['Sample_code_number','Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape', 'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']

# Lakukan pengecekan apakah dataset sudah benar dengan menampilkan 5 data teratas
dataset.info()

"""## 2) Menelaah Data <a class="anchor" id="h2"></a>"""

# Menampilkan informasi dari file dataset
dataset.info()

# Menampilkan deskripsi dari file dataset
dataset.describe()

import seaborn as sns
import matplotlib.pyplot as plt

# dibawah adalah contoh kode program untuk fitur 1
sns.set(font_scale=1.0)
dataset['Clump_thickness'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Clump_thickness', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Clump_thickness", y=1.02);

"""### Isi Koding 1"""

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Clump_thickness'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Clump_thickness", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Clump_thickness", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Uniformity_of_cell_size'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Uniformity_of_cell_size', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Uniformity_of_cell_size", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Uniformity_of_cell_shape'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Uniformity_of_cell_shape', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Uniformity_of_cell_shape", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Marginal_adhesion'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Marginal_adhesion', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Marginal_adhesion", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Single_epithelial_cell_size'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Single_epithelial_cell_size', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Single_epithelial_cell_size", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Bare_nuclei'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Bare_nuclei', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi Bare_nuclei", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Bland_chromatin'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Bland_chromatin', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi 'Bland_chromatin'", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Normal_nucleoli'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Normal_nucleoli', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi 'Normal_nucleoli'", y=1.02);

# tuliskan kode program untuk Menampilkan distribusi kelas dari semua fitur
sns.set(font_scale=1.0)
dataset['Mitoses'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel('Mitoses', labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Distribusi 'Mitoses'", y=1.02);

"""## 3) Memvalidasi Data <a class="anchor" id="h3"></a>

Dalam aktivitas ini, Anda harus dapat melakukan pengecekan atas data yang Anda gunakan apakah terdapat missing value, noisy data, atau data yang tidak sesuai lainnya
"""

dataset.info()

"""###Essay 1
Deskripsikan Temuan Anda disini:

sejauh ini tidak ada missing data/null. hanya saja bare_nuclei itu object

## 4) Menentukan Objek Data <a class="anchor" id="h4"></a>
"""

import numpy as np

dataset

dataset.info()

"""###Essay 2
Deskripsikan Temuan Anda disini:

bare_nuclei itu object

## 5) Membersihkan Data <a class="anchor" id="h5"></a>
"""

# menghitung nilai Null pada dataset
dataset.isnull().sum()

# mendeteksi keberadaan nilai Null
dataset.loc[:, dataset.isnull().any()].columns

# Mengubah tipe data kolom 'Bare_nuclei' menjadi integer
dataset['Bare_nuclei'] = pd.to_numeric(dataset['Bare_nuclei'], errors='coerce').fillna(0).astype(int)

# Menampilkan Informasi dari data
dataset.info()

dataset.head()

dataset.corr()

import missingno as msno

"""### Isi Koding 2"""

dataset.isnull().sum()

# Menampilkan data duplikat
duplicate_rows = dataset.duplicated()
print("All Duplicate Rows:")
dataset[dataset.duplicated(keep=False)]

# Menghapus data duplikat, menyimpan data dalam variabel baru "dataClean"
dataClean = dataset.drop_duplicates()
print("All Duplicate Rows:")
dataClean[dataClean.duplicated(keep=False)]

dataset.info()

dataClean.info()

"""Temuan:

## 6) Menkonstruksi Data <a class="anchor" id="h2"></a>
"""

# Menampilkan distribusi kelas dari target
print(dataClean['Class'].value_counts())
sns.set(font_scale=1.0)
dataClean['Class'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Status Pasien kanker payudara", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Status Pasien kanker payudara", y=1.02);

# Menampilkan Korelasi antar Fitur
correlation = dataClean.corr()
plt.subplots(figsize = (12,12))
sns.heatmap(correlation.round(2),
            annot = True,
            vmax = 1,
            square = True,
            cmap = 'RdYlGn_r')
plt.show()

# Menampilkan Boxplot untuk melihat adanya Outlayer
dataClean.plot(kind='box',subplots=True,layout=(5,6), sharex=False,figsize = (20,20),
                           title='figure 1: Data distributions of all features')
plt.show()

#menampilkan deskripsi data yang sudah dibersihkan
dataClean.describe()

dataClean.info()

dataClean

"""###Essay 3
Deskripsikan Temuan Anda disini

## 7) Menentukan Label Data <a class="anchor" id="h7"></a>
"""

from sklearn.model_selection import train_test_split

X_norm= dataClean.drop("Sample_code_number",axis=1).values
y = dataClean['Class']

# perbandingan data training dan data testing adalah 70 : 30
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

"""## 8) Membangun Model <a class="anchor" id="h8"></a>"""

# import library pemodelan yang digunakan
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

clean_classifier_nb = GaussianNB()
clean_classifier_nb .fit(X_train, y_train)

clean_classifier_dt = DecisionTreeClassifier(random_state=42)
clean_classifier_dt .fit(X_train, y_train)

clean_classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clean_classifier_rf .fit(X_train, y_train)

"""## 9) Mengevaluasi Hasil Pemodelan <a class="anchor" id="h9"></a>"""

from sklearn.metrics import classification_report
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

y_pred_nb = clean_classifier_nb.predict(X_test)

# Evaluate the Naive Bayes model
print("\nNaive Bayes Model:")
accuracy_nb = round(accuracy_score(y_test, y_pred_nb),3)
print("Accuracy:",accuracy_nb)
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))

evaluation(y_test,y_pred_nb)

y_pred_dt = clean_classifier_dt.predict(X_test)

# Evaluate the Decission Tree model
print("\nDecission Tree Model:")
accuracy_dt = round(accuracy_score(y_test, y_pred_dt),3)
print("Accuracy:",accuracy_dt)
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

evaluation(y_test,y_pred_dt)

y_pred_rf = clean_classifier_rf.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model:")
accuracy_rf = round(accuracy_score(y_test, y_pred_rf),3)
print("Accuracy:",accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

evaluation(y_test,y_pred_rf)

model_comp = pd.DataFrame({'Model': ['Naive Bayes','Decision Tree','Random Forest'], 'Accuracy': [accuracy_nb * 100, accuracy_dt * 100, accuracy_rf * 100]})

# Membuat bar plot dengan keterangan jumlah
fig, ax = plt.subplots()
bars = plt.bar(model_comp['Model'], model_comp['Accuracy'], color=['red', 'green', 'blue'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Clean Data')
plt.xticks(rotation=45, ha='right')  # Untuk memutar label sumbu x agar lebih mudah dibaca

# Menambahkan keterangan jumlah di atas setiap bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()

"""###Essay 4
Deskripsikan Temuan Anda disini

## 10) Optimasi Model Klasifikasi <a class="anchor" id="h10"></a>
"""

#nilai confusion matrix untuk model dengan akurasi tertinggi

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

columns_to_drop = ['Class', 'Bare_nuclei']
X_selected= dataClean.drop(columns_to_drop, axis=1).values
y = dataClean['Class']

X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y, test_size=0.3, random_state=42)

selected_classifier_nb = GaussianNB()
selected_classifier_nb.fit(X_train_selected, y_train_selected)

y_pred_nb_selected = selected_classifier_nb.predict(X_test_selected)

# Evaluate the optimize model
print("\nNaive Bayes Model:")
accuracy_nb_selected = round(accuracy_score(y_test_selected, y_pred_nb_selected),3)
print("Accuracy:",accuracy_nb_selected)
print("Classification Report:")
print(classification_report(y_test_selected, y_pred_nb_selected))

evaluation(y_test_selected,y_pred_nb_selected)

cm = confusion_matrix(y_test, y_pred_nb_selected)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('True')
plt.ylabel('Predict')
plt.show()

model_comp = pd.DataFrame({'Model': ['Sebelum optimasi','setelah Optimasi'], 'Accuracy': [accuracy_nb*100, accuracy_nb_selected*100]})

# Membuat bar plot dengan keterangan jumlah
fig, ax = plt.subplots()
bars = plt.bar(model_comp['Model'], model_comp['Accuracy'], color=['red', 'blue'])
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Optimal Data')
plt.xticks(rotation=45, ha='right')  # Untuk memutar label sumbu x agar lebih mudah dibaca

# Menambahkan keterangan jumlah di atas setiap bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.show()

"""###Essay 5
Deskripsikan Temuan Anda disini
"""