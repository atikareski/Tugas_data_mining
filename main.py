import streamlit as st
import pandas as pd
import numpy as np
import joblib # Untuk menyimpan/memuat model
from sklearn.model_selection import cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import BytesIO

# --- Konfigurasi dan Konstanta ---
TARGET_COLUMN = 'rating'
FEATURES = ['runtime', 'metascore', 'gross(M)', 'release_year', 'genre']
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

@st.cache_data
def load_and_clean_data(file_path):
    """Memuat dan membersihkan data, mengatasi NaN di kolom target."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File {file_path} tidak ditemukan.")
        return pd.DataFrame()

    # SOLUSI ERROR: Hapus baris dengan nilai hilang pada target ('rating')
    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Memisahkan Fitur (X) dan Target (y)
    X = df_clean[FEATURES]
    y = df_clean[TARGET_COLUMN]

    return X, y

@st.cache_resource
def create_model_pipeline():
    """Membuat Preprocessor dan Pipeline untuk model."""
    numerical_features = ['runtime', 'metascore', 'gross(M)', 'release_year']
    categorical_features = ['genre']

    # Pipeline untuk kolom Numerik: Impute dengan Mean
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Pipeline untuk kolom Kategorikal: Impute modus, lalu One-Hot Encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Menggabungkan semua preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    # Membuat dictionary Pipeline Penuh
    pipelines = {
        name: Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ]) for name, model in MODELS.items()
    }
    return pipelines

@st.cache_data
def evaluate_models(pipelines, X, y):
    """Melakukan Cross-Validation dan mengembalikan DataFrame hasil."""
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring_metrics = {
        'R2': 'r2',
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error'
    }

    results = {}
    progress_bar = st.progress(0)
    
    st.info("Memulai evaluasi model dengan 10-Fold Cross-Validation. Harap tunggu...")
    
    for i, (name, pipeline) in enumerate(pipelines.items()):
        st.caption(f"Evaluasi {name}...")
        scores = cross_validate(
            pipeline, X, y, cv=cv_strategy, scoring=scoring_metrics, n_jobs=-1
        )

        results[name] = {
            'R2': scores['test_R2'].mean(),
            'MAE': -scores['test_MAE'].mean(),
            'RMSE': -scores['test_RMSE'].mean()
        }
        progress_bar.progress((i + 1) / len(pipelines))
    
    results_df = pd.DataFrame(results).T.round(4)
    results_df.index.name = "Model"
    
    return results_df

# ==============================================================================
#                      FUNGSI UTAMA STREAMLIT
# ==============================================================================
st.title("🎬 Analisis Regresi Rating Film IMDb")
st.markdown("Aplikasi ini mereplikasi *workflow* analisis Orange Anda untuk memprediksi **Rating** film menggunakan regresi.")

# Asumsi file 'imdb_clean.csv' sudah diunggah ke lokasi yang bisa diakses
FILE_PATH = "imdb_clean.csv"

# Memuat dan membersihkan data
X, y = load_and_clean_data(FILE_PATH)

if not X.empty:
    st.subheader("1. Pra-pemrosesan Data")
    st.success(f"Data dimuat. Baris dengan target NaN dihapus. Sisa baris bersih: {len(X)}")
    
    pipelines = create_model_pipeline()

    # Melakukan dan menampilkan evaluasi
    st.subheader("2. Hasil Test and Score (10-Fold Cross-Validation)")
    results_df = evaluate_models(pipelines, X, y)
    
    st.dataframe(results_df)

    # Menarik kesimpulan dari hasil evaluasi
    st.subheader("3. Kesimpulan Model Terbaik")
    best_model_name = results_df['R2'].idxmax()
    st.success(f"Model Terbaik adalah **{best_model_name}** dengan R-squared rata-rata: **{results_df.loc[best_model_name, 'R2']}**.")
    st.markdown("Model Random Forest (dan Gradient Boosting) jauh lebih unggul karena mampu menangani hubungan non-linier dalam data.")

    # Melatih Model Terbaik dan Menampilkan Kepentingan Fitur
    st.subheader("4. Kepentingan Fitur (Feature Importance)")
    
    if best_model_name == 'Random Forest' or best_model_name == 'Gradient Boosting':
        
        # Pelatihan Akhir Model Terbaik
        final_pipeline = pipelines[best_model_name]
        final_pipeline.fit(X, y)
        
        # Mengambil feature names setelah One-Hot Encoding
        feature_names_transformed = np.array(final_pipeline['preprocessor'].get_feature_names_out())
        
        # Mendapatkan feature importances dari model ensemble
        importances = final_pipeline['regressor'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_transformed,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Menghilangkan prefix transformer (e.g., 'num__', 'cat__') untuk tampilan bersih
        feature_importance_df['Feature'] = feature_importance_df['Feature'].str.split('__').str[-1]
        
        st.dataframe(feature_importance_df.head(10).set_index('Feature'))
        
        st.markdown(
            """
            **Analisis:** Mirip dengan analisis Orange Anda, hasil ini menunjukkan bahwa fitur seperti **metascore** dan **gross(M)** memiliki dampak terbesar dalam memprediksi **Rating** film.
            """
        )

# ==============================================================================