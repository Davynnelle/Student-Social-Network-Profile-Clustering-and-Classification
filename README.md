# Deskripsi

Proyek ini terdiri dari dua bagian utama:

1. **Clustering**: Mengelompokkan profil siswa berdasarkan atribut demografi, minat, dan aktivitas sosial tanpa menggunakan label (unsupervised learning).
2. **Klasifikasi**: Membangun model klasifikasi untuk memprediksi kelas tertentu (misalnya gender atau kategori lain) dari data yang sudah dilabeli.

Kode ini dirancang untuk membantu memahami alur kerja end-to-end pada analisis data, mulai dari Exploratory Data Analysis (EDA), preprocessing, hingga penerapan algoritma machine learning.

---

## Struktur Repository

```plaintext
├── README.md                                                   # Dokumentasi proyek
├── [Clustering]_Submission_Akhir_BMLP_Dhea_Yuza_Fadiya.ipynb   # Notebook untuk bagian clustering
├── [Klasifikasi]_Submission_Akhir_BMLP_Dhea_Yuza_Fadiya.ipynb  # Notebook untuk bagian klasifikasi
├── Clustering_Marketing.csv                                    # Dataset mentah (tanpa label)
├── dataset_clustering.csv                                      # Data hasil clustering
```

---

## Prasyarat

Pastikan untuk menginstal dependensi berikut (contoh menggunakan `pip`):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 1. Clustering

**Notebook**: `[Clustering]_Submission_Akhir_BMLP_Dhea_Yuza_Fadiya.ipynb`

### 1.1. Import Library
- `pandas`, `numpy` untuk manipulasi data.
- `matplotlib`, `seaborn` untuk visualisasi.
- `IsolationForest` untuk deteksi outlier.
- `StandardScaler`, `RobustScaler` untuk scaling.
- `LabelEncoder` untuk encoding fitur kategorikal.
- `PCA`, `KMeans`, `silhouette_score` untuk clustering dan evaluasi.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

### 1.2. Memuat Dataset

- Mount Google Drive.
- Baca file CSV ke DataFrame pandas.

```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Clustering_Marketing.csv')
df.head()
```

### 1.3. Exploratory Data Analysis (EDA)

1. Cek bentuk data (`df.shape`) dan tipe kolom (`df.info()`).
2. Identifikasi missing values (`df.isnull().sum()`).
3. Statistik deskriptif untuk kolom numerik (`df.describe()`).
4. Distribusi data: histogram, boxplot.
5. Korelasi antar variabel: heatmap.
6. Pairplot terpilih.

### 1.4. Data Preprocessing

1. **Missing Value**:
   - `gender`: isi dengan "Tidak Diketahui".
   - `age`: ubah ke numerik, isi na dengan rata-rata per `gradyear`.
2. **Outlier**:
   - Gunakan IQR untuk memfilter `age` pada rentang 13–21.
3. **Feature Engineering**:
   - Gabungkan frekuensi kata menjadi kategori baru: `social`, `lifestyle`, `personal`, `sports`, `religion`, `entertaiment`.
4. **Encoding**:
   - `LabelEncoder` untuk kolom `gender`.
5. **Scaling**:
   - `RobustScaler` untuk kolom hasil agregasi.

### 1.5. Pembangunan Model Clustering

1. Cari nilai _optimal_ K (2–20) dengan _Silhouette Score_.
2. Jalankan KMeans pada K optimal.
3. Evaluasi _silhouette_score_.

```python
# Menentukan optimal K
silhouette_scores = []
for k in range(2, 21):
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(data_encoded)
    silhouette_scores.append(silhouette_score(data_encoded, labels))
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2

# Training KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data_encoded)
score = silhouette_score(data_encoded, clusters)
```

### 1.6. Visualisasi dan Interpretasi

- **Elbow Method** (Inertia) dan _Silhouette Score_ untuk berbagai K.
- **PCA** ke 2 dimensi untuk scatter plot cluster.
- Hitung rata-rata fitur tiap cluster dan jumlah anggota.
- Tampilkan barplot jumlah anggota setiap cluster.
- Interpretasi karakteristik dan rekomendasi strategi.

### 1.7. Export Hasil

```python
df['cluster'] = clusters
df.to_csv('output/dataset_clustering.csv', index=False)
```

---

## 2. Klasifikasi

**Notebook**: `[Klasifikasi]_Submission_Akhir_BMLP_Dhea_Yuza_Fadiya.ipynb`

### 2.1. Import Library

Pada tahap ini, impor pustaka Python yang dibutuhkan untuk pemrosesan data, pelatihan model, dan evaluasi:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
```

### 2.2. Memuat Dataset Hasil Clustering

Mount Google Drive (Colab) dan muat dataset hasil clustering:

```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/dataset_clustering.csv')
df.head()
print("Jumlah baris dan kolom:", df.shape)
df.info()
df.describe()
```

### 2.3. Exploratory Data Analysis & Preprocessing

1. **Visualisasi Distribusi Fitur**: Histogram untuk semua kolom.
2. **Penanganan Nilai Ekstrem**: Clip `NumberOffriends` dan `age` pada kuantil 1%–99%.
3. **Korelasi Fitur**: Heatmap korelasi untuk mengidentifikasi hubungan linear.

```python
# Contoh clipping
df['NumberOffriends'] = df['NumberOffriends'].clip(
    lower=df['NumberOffriends'].quantile(0.01),
    upper=df['NumberOffriends'].quantile(0.99)
)
df['age'] = df['age'].clip(
    lower=df['age'].quantile(0.01),
    upper=df['age'].quantile(0.99)
)

# Heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur')
plt.show()
```

### 2.4. Data Splitting

Ubah `pca_clusters` menjadi target biner (`High` vs `Low`) berdasarkan median, lalu bagi data:

```python
threshold = df['pca_clusters'].median()
y = (df['pca_clusters'] > threshold).astype(int)
X = df.drop(['pca_clusters'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling fitur numerik
to_scale = X.select_dtypes(include=['float64','int64']).columns
scaler = MinMaxScaler()
X_train[to_scale] = scaler.fit_transform(X_train[to_scale])
X_test[to_scale]  = scaler.transform(X_test[to_scale])

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set:     {X_test.shape}, {y_test.shape}")
```

### 2.5. Pelatihan Model Klasifikasi

Inisialisasi dan latih beberapa algoritma:

```python
models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    print(f"{name} trained.")
```

### 2.6. Evaluasi Model

Gunakan metrik `accuracy`, `precision`, `recall`, `f1-score`, dan `confusion matrix`:

```python
def evaluate(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

for name, y_pred in predictions.items():
    metrics = evaluate(y_test, y_pred)
    print(f"
{name} Metrics:")
    print(metrics)
```

Visualisasikan `confusion matrix`:

```python
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','High'], yticklabels=['Low','High'])
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
```

### 2.7. Interpretasi Hasil

- Bandingkan performa tiap model berdasarkan metrik.  
- Identifikasi indikasi **overfitting** (hasil training vs testing).  
- Pilih model terbaik (mis. SVM untuk generalisasi, Random Forest untuk akurasi tinggi, dll.).

### 2.8. Hyperparameter Tuning (Opsional)

Gunakan `GridSearchCV` atau `RandomizedSearchCV` untuk mencari kombinasi parameter terbaik:

```python
# Contoh GridSearchCV untuk Random Forest
from sklearn.model_selection import GridSearchCV
param_grid = { 'n_estimators': [50,100,200], 'max_depth': [None,10,20] }
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

### 2.9. Tindakan Lanjutan

1. **Feature Selection**: Pilih subset fitur paling relevan.  
2. **Balancing Data**: Atasi ketidakseimbangan kelas, jika ada.  
3. **Ensemble Learning**: Kombinasikan model untuk performa lebih baik.  
4. **Analisis Kesalahan**: Teliti sampel yang sering salah prediksi untuk wawasan lebih jauh.


📄 Note: This project is based on an final submission from the course "Belajar Machine Learning untuk Pemula" on Dicoding Indonesia. The goal was to implement and expand on the core concepts introduced in the course.
