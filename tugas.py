import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv('https://github.com/NaseemMasaid/tugasDatmin/blob/main/diabetes.csv')
print(diabetes.columns)

diabetes.head()

print("Dimensi kumpulan data diabetes : {}".format(diabetes.shape))

diabetes.groupby('Outcome').size()

diabetes.hist(figsize=(9, 9))

diabetes.groupby('Outcome').hist(figsize=(9, 9))

#Phase 2— Data Cleaning
diabetes.isnull().sum()

diabetes.isna().sum()

print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])
print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())

print("Total : ", diabetes[diabetes.Glucose == 0].shape[0])
print(diabetes[diabetes.Glucose == 0].groupby('Outcome')['Age'].count())

print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])
print(diabetes[diabetes.SkinThickness == 0].groupby('Outcome')['Age'].count())

# Cetak "Total : " dan jumlah baris dari dataframe diabetes yang memiliki nilai BMI sama dengan 0 menggunakan atribut shape
print("Total : ", diabetes[diabetes.BMI == 0].shape[0])
# Cetak jumlah baris dari dataframe diabetes yang memiliki nilai BMI sama dengan 0, dibagi berdasarkan kelompok Outcome (0 atau 1), dan hitung berdasarkan kolom Age menggunakan metode groupby dan count
print(diabetes[diabetes.BMI == 0].groupby('Outcome')['Age'].count())

# Cetak "Total : " dan jumlah baris dari dataframe diabetes yang memiliki nilai Insulin sama dengan 0 menggunakan atribut shape
print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])
# Cetak jumlah baris dari dataframe diabetes yang memiliki nilai Insulin sama dengan 0, dibagi berdasarkan kelompok Outcome (0 atau 1), dan hitung berdasarkan kolom Age menggunakan metode groupby dan count
print(diabetes[diabetes.Insulin == 0].groupby('Outcome')['Age'].count())

#Phase 3— Feature Engineering
# Buat dataframe baru bernama diabetes_mod yang berisi data dari dataframe diabetes yang tidak memiliki nilai BloodPressure, BMI, atau Glucose sama dengan 0 menggunakan operator logika &
diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
# Cetak dimensi (jumlah baris dan kolom) dari dataframe diabetes_mod menggunakan atribut shape
print(diabetes_mod.shape)

# Buat daftar bernama fitur (variabel independen) yang akan digunakan untuk memprediksi respons (variabel dependen)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# Buat matriks fitur X yang berisi nilai-nilai dari kolom-kolom fitur yang dipilih dari dataframe diabetes_mod
X = diabetes_mod[feature_names]
# Buat vektor respons y yang berisi nilai-nilai dari kolom Outcome (0 atau 1) dari dataframe diabetes_mod
y = diabetes_mod.Outcome

#Phase 4— Model Selection
# Impor kelas KNeighborsClassifier dari modul sklearn.neighbors untuk menggunakan metode klasifikasi berdasarkan tetangga terdekat
from sklearn.neighbors import KNeighborsClassifier
# Impor kelas SVC dari modul sklearn.svm untuk menggunakan metode klasifikasi berdasarkan support vector machine
from sklearn.svm import SVC
# Impor kelas LogisticRegression dari modul sklearn.linear_model untuk menggunakan metode klasifikasi berdasarkan regresi logistik
from sklearn.linear_model import LogisticRegression
# Impor kelas DecisionTreeClassifier dari modul sklearn.tree untuk menggunakan metode klasifikasi berdasarkan pohon keputusan
from sklearn.tree import DecisionTreeClassifier
# Impor kelas GaussianNB dari modul sklearn.naive_bayes untuk menggunakan metode klasifikasi berdasarkan naive bayes dengan asumsi distribusi normal
from sklearn.naive_bayes import GaussianNB
# Impor kelas RandomForestClassifier dari modul sklearn.ensemble untuk menggunakan metode klasifikasi berdasarkan hutan acak, yaitu ensemble dari pohon keputusan yang dibangun secara acak
from sklearn.ensemble import RandomForestClassifier
# Impor kelas GradientBoostingClassifier dari modul sklearn.ensemble untuk menggunakan metode klasifikasi berdasarkan gradient boosting, yaitu ensemble dari pohon keputusan yang dibangun secara bertahap dengan mengurangi kesalahan prediksi sebelumnya
from sklearn.ensemble import GradientBoostingClassifier

# Impor fungsi train_test_split dari modul sklearn.model_selection untuk membagi data menjadi dua subset: data latih dan data uji, dengan proporsi yang dapat ditentukan
from sklearn.model_selection import train_test_split
# Impor fungsi cross_val_score dari modul sklearn.model_selection untuk melakukan validasi silang, yaitu proses evaluasi model dengan menggunakan beberapa subset data latih dan data uji yang berbeda-beda
from sklearn.model_selection import cross_val_score
# Impor kelas StratifiedKFold dari modul sklearn.model_selection untuk melakukan validasi silang dengan metode stratified k-fold, yaitu membagi data menjadi k lipatan yang seimbang dalam hal proporsi kelas respons di setiap lipatan
from sklearn.model_selection import StratifiedKFold
# Impor fungsi accuracy_score dari modul sklearn.metrics untuk menghitung akurasi prediksi model, yaitu persentase kasus yang diprediksi dengan benar oleh model
from sklearn.metrics import accuracy_score

# Initial model selection process
models = []

# Membuat list kosong untuk menampung model-model yang akan dicoba

models.append(('KNN', KNeighborsClassifier()))
# Menambahkan model KNN (K-Nearest Neighbor) ke dalam list models

models.append(('SVC', SVC(gamma='scale')))
# Menambahkan model SVC (Support Vector Classifier) dengan parameter gamma='scale' ke dalam list models

models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=4000)))
# Menambahkan model Logistic Regression dengan parameter solver='lbfgs' dan max_iter=4000 ke dalam list models

models.append(('DT', DecisionTreeClassifier()))
# Menambahkan model Decision Tree ke dalam list models

models.append(('GNB', GaussianNB()))
# Menambahkan model Naive Bayes ke dalam list models

models.append(('RF', RandomForestClassifier(n_estimators=100)))
# Menambahkan model Random Forest dengan 100 estimators ke dalam list models

models.append(('GB', GradientBoostingClassifier()))
# Menambahkan model Gradient Boosting ke dalam list models

# Bagi data menjadi data latih dan data uji dengan proporsi 75:25 secara default menggunakan fungsi train_test_split
# Gunakan parameter stratify untuk memastikan proporsi kelas respons sama di data latih dan data uji
# Gunakan parameter random_state untuk menentukan seed untuk pembangkit bilangan acak yang digunakan untuk membagi data
# Simpan hasil pembagian data dalam empat variabel: X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)

# Buat dua daftar kosong bernama names dan scores untuk menyimpan nama dan skor akurasi dari setiap model
names = []
scores = []

# Lakukan iterasi untuk setiap pasangan nama dan model dalam daftar models
for name, model in models:
    # Latih model dengan data latih X_train dan y_train menggunakan metode fit
    model.fit(X_train, y_train)
    # Buat prediksi dengan model untuk data uji X_test menggunakan metode predict
    y_pred = model.predict(X_test)
    # Hitung skor akurasi dari prediksi dengan membandingkan dengan nilai sebenarnya y_test menggunakan fungsi accuracy_score
    scores.append(accuracy_score(y_test, y_pred))
    # Tambahkan nama model ke daftar names
    names.append(name)

# Buat dataframe pandas bernama tr_split yang berisi dua kolom: Name dan Score, dengan nilai dari daftar names dan scores
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
# Cetak dataframe tr_split untuk melihat hasil evaluasi model
print(tr_split)

# Buat objek strat_k_fold yang merupakan instance dari kelas StratifiedKFold dengan parameter n_splits=10, shuffle=True, dan random_state=10
# Ini berarti data akan dibagi menjadi 10 lipatan yang seimbang dalam hal proporsi kelas respons di setiap lipatan, dan lipatan akan diacak dengan seed 10
strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)


# Buat dua daftar kosong bernama names dan scores untuk menyimpan nama dan skor akurasi rata-rata dari setiap model
names = []
scores = []

# Lakukan iterasi untuk setiap pasangan nama dan model dalam daftar models
for name, model in models:

    # Hitung skor akurasi rata-rata dari model dengan menggunakan fungsi cross_val_score yang menerima parameter model, X, y, cv=strat_k_fold, dan scoring='accuracy'
    # Ini berarti model akan dilatih dan dievaluasi dengan menggunakan validasi silang stratified k-fold yang telah ditentukan sebelumnya
    score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    # Tambahkan nama model ke daftar names
    names.append(name)
    # Tambahkan skor akurasi rata-rata ke daftar scores
    scores.append(score)

# Buat dataframe pandas bernama kf_cross_val yang berisi dua kolom: Name dan Score, dengan nilai dari daftar names dan scores
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
# Cetak dataframe kf_cross_val untuk melihat hasil evaluasi model
print(kf_cross_val)

# Buat objek axis yang merupakan hasil dari fungsi barplot dari pustaka seaborn yang menerima parameter x, y, dan data
# Ini berarti membuat plot batang horizontal yang menampilkan skor akurasi dari setiap model klasifikasi yang tersimpan dalam dataframe kf_cross_val
axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
# Atur label sumbu x dan y dari plot dengan menggunakan metode set dari objek axis
axis.set(xlabel='Classifier', ylabel='Accuracy')

# Lakukan iterasi untuk setiap objek patch dalam daftar axis.patches
# Objek patch adalah representasi grafis dari batang plot
for p in axis.patches:
    # Dapatkan tinggi dari objek patch menggunakan metode get_height
    height = p.get_height()

# Tampilkan plot dengan menggunakan fungsi show dari pustaka matplotlib.pyplot
plt.show()

from sklearn.feature_selection import RFECV

logreg_model = LogisticRegression(solver='lbfgs', max_iter=4000)

rfecv = RFECV(estimator=logreg_model, step=1, cv=strat_k_fold, scoring='accuracy')

rfecv.fit(X, y)
plt.figure()

plt.title('Skor CV Regresi Logistik vs Tidak Ada Fitur')

plt.xlabel("Jumlah fitur yang dipilih")

plt.ylabel("Skor validasi silang (nb klasifikasi yang benar)")

plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
rfecv.cv_results_["mean_test_score"])

plt.show()

feature_importance = list(zip(feature_names, rfecv.support_))
new_features = []

for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
print(new_features)

X_new = diabetes_mod[new_features]

initial_score = cross_val_score(logreg_model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Akurasi awal : {} ".format(initial_score))

fe_score = cross_val_score(logreg_model, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Akurasi setelah Pemilihan Fitur : {} ".format(fe_score))
gb_model = GradientBoostingClassifier()

gb_rfecv = RFECV(estimator=gb_model, step=1, cv=strat_k_fold, scoring='accuracy')
gb_rfecv.fit(X, y)

plt.figure()
plt.title('Skor Gradient Boost CV vs Jumlah Fitur')
plt.xlabel("Jumlah fitur yang dipilih")
plt.ylabel("Skor validasi silang (nb klasifikasi yang benar)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
rfecv.cv_results_["mean_test_score"])
plt.show()

feature_importance = list(zip(feature_names, gb_rfecv.support_))

new_features = []

for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])

print(new_features)

X_new_gb = diabetes_mod[new_features]

initial_score = cross_val_score(gb_model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Akurasi awal : {} ".format(initial_score))

fe_score = cross_val_score(gb_model, X_new_gb, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Akurasi setelah Pemilihan Fitur : {} ".format(fe_score))

from sklearn.model_selection import GridSearchCV

c_values = list(np.arange(1, 10))

param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
]

grid = GridSearchCV(LogisticRegression(), param_grid, cv=strat_k_fold, scoring='accuracy')
grid.fit(X_new, y)

print(grid.best_params_)
print(grid.best_estimator_)

logreg_new = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')

initial_score = cross_val_score(logreg_new, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Akurasi akhir : {} ".format(initial_score))
