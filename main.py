import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --- 1. Pemuatan Data ---
# Ganti 'imdb_clean.csv' dengan path file Anda yang sebenarnya
df = pd.read_csv("D:\COLLEGE!!/FIFTH SEMESTER/DATA MINING/DataMiningTeori/tugas_rapid_miner/imdb_clean.csv")

# SOLUSI: Hapus SEMUA baris di mana 'rating' (target) adalah NaN.
# Ini penting karena cross_validate TIDAK bisa bekerja jika target mengandung NaN.
df_clean = df.dropna(subset=['rating']).copy()

# Target Anda (seperti di Orange)
TARGET_COLUMN = 'rating'
y = df_clean[TARGET_COLUMN] # Menggunakan data yang sudah bersih

# Fitur yang akan digunakan
FEATURES = ['runtime', 'metascore', 'gross(M)', 'release_year', 'genre']
X = df_clean[FEATURES] # Menggunakan data yang sudah bersih

# --- 3. Pra-pemrosesan Data (Meniru Select Columns, Impute, dan Preprocess) ---

# Tentukan kolom numerik dan kategorikal
numerical_features = ['runtime', 'metascore', 'gross(M)', 'release_year']
categorical_features = ['genre'] # 'director' dan 'title' diabaikan karena kardinalitas tinggi

# Pipeline untuk kolom Numerik: Impute dengan Mean
numerical_pipeline = Pipeline([
    # Mengatasi nilai hilang dengan rata-rata
    ('imputer', SimpleImputer(strategy='mean'))
])

# Pipeline untuk kolom Kategorikal: Impute dengan modus, lalu One-Hot Encode
categorical_pipeline = Pipeline([
    # Mengatasi nilai hilang dengan modus (nilai paling sering muncul)
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # One-Hot Encoding (mengubah kategori menjadi kolom biner)
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Menggabungkan semua preprocessing (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    # Menghapus baris yang mengandung nilai hilang di TARGET (Rating)
    # Ini penting agar CV tidak error
    remainder='drop'
)

# --- 4. Definisi Model ---
# Mendefinisikan model regresi seperti yang Anda gunakan di Orange
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# --- 5. Pipeline Penuh (Preprocess + Model) ---
pipelines = {
    name: Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ]) for name, model in models.items()
}

# --- 6. Evaluasi Model (Meniru Test and Score dengan Cross-Validation) ---

# Menggunakan K-Fold Cross-Validation (K=10 secara default di Orange)
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

# Metrik yang digunakan untuk evaluasi (sesuai Test and Score Orange)
scoring_metrics = {
    'R2': 'r2',
    'MAE': 'neg_mean_absolute_error', # Sklearn menggunakan negatif MAE
    'RMSE': 'neg_root_mean_squared_error' # Sklearn menggunakan negatif RMSE
}

results = {}

print("Memulai Evaluasi Model dengan 10-Fold Cross-Validation...")

for name, pipeline in pipelines.items():
    # Menjalankan Cross-Validation
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring_metrics,
        n_jobs=-1 # Menggunakan semua inti CPU untuk komputasi cepat
    )

    # Menghitung Rata-rata Skor CV
    results[name] = {
        'R2': scores['test_R2'].mean(),
        # Mengubah kembali nilai MAE dan RMSE menjadi positif
        'MAE': -scores['test_MAE'].mean(),
        'RMSE': -scores['test_RMSE'].mean()
    }

# --- 7. Tampilan Hasil Akhir ---
results_df = pd.DataFrame(results).T # Transpose untuk tampilan yang lebih baik

print("\n--- Hasil Evaluasi Model (10-Fold Cross-Validation) ---")
print(results_df.round(4))

# --- 8. Pelatihan Model Terbaik untuk Prediksi (Random Forest) ---
# Melatih model Random Forest terakhir kali pada seluruh data bersih

best_model_name = 'Random Forest'
best_pipeline = pipelines[best_model_name]

# Menghapus baris dengan nilai hilang di target sebelum pelatihan akhir
df_final = df.dropna(subset=[TARGET_COLUMN]).copy()
X_final = df_final[FEATURES]
y_final = df_final[TARGET_COLUMN]

print(f"\nMelatih Model Terbaik ({best_model_name}) pada Seluruh Data...")
best_pipeline.fit(X_final, y_final)

# Contoh Prediksi (opsional)
# new_data = pd.DataFrame({
#     'runtime': [150], 'metascore': [85], 'gross(M)': [100.0],
#     'release_year': [2020], 'genre': ['Action']
# })
# prediction = best_pipeline.predict(new_data)
# print(f"Prediksi Rating untuk film baru: {prediction[0]:.2f}")

# Simpan model untuk digunakan di Streamlit
# import joblib
# joblib.dump(best_pipeline, 'random_forest_model.pkl')
# print("\nModel Random Forest disimpan sebagai 'random_forest_model.pkl'")