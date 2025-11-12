import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

TARGET_COLUMN = 'rating'
FEATURES = ['runtime', 'metascore', 'gross(M)', 'release_year', 'genre']
BEST_MODEL = RandomForestRegressor(random_state=42, n_estimators=100)
FILE_PATH = "imdb_clean.csv"

@st.cache_data
def load_data(file_path):
    """Memuat dan membersihkan data target."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File {file_path} tidak ditemukan. Pastikan 'imdb_clean.csv' ada di direktori yang sama.")
        return pd.DataFrame()

    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()

    unique_genres = df_clean['genre'].dropna().unique().tolist()
    
    return df_clean, unique_genres

@st.cache_resource
def train_model(df_clean):
    """Melatih model Random Forest Regressor dengan Preprocessing Pipeline."""
    
    # 1. Pisahkan X dan y
    X = df_clean[FEATURES]
    y = df_clean[TARGET_COLUMN]

    numerical_features = ['runtime', 'metascore', 'gross(M)', 'release_year']
    categorical_features = ['genre']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', BEST_MODEL)
    ])

    pipeline.fit(X, y)
    
    return pipeline, preprocessor

# --- FUNGSI UTAMA STREAMLIT ---
def main():
    st.set_page_config(layout="wide", page_title="Prediktor Rating Film IMDb")
    
    # Pemuatan data
    df_clean, unique_genres = load_data(FILE_PATH)
    
    if df_clean.empty:
        return

    st.title("🎬 Prediktor Rating Film IMDb")

    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.header("Masukkan Kriteria Film")
        
        # 1. Input Metascore
        metascore = st.slider(
            "Metascore (Skor Kritikus):",
            min_value=0, max_value=100, value=75, step=1,
            help="Skor yang diberikan oleh kritikus (rotten tomatoes/metacritic)."
        )

        # 2. Input Gross/Pendapatan
        gross = st.number_input(
            "Gross (Pendapatan Kotor Global, dalam Juta USD):",
            min_value=0.0, max_value=2000.0, value=50.0, step=1.0, format="%.2f",
            help="Total pendapatan global film. Menggunakan skala dalam Juta USD (50.0 = 50 Juta USD)."
        )

        # 3. Input Runtime/Durasi
        runtime = st.slider(
            "Runtime (Durasi Film, dalam Menit):",
            min_value=60, max_value=240, value=120, step=5,
            help="Berapa lama durasi film ini?"
        )
        
        # 4. Input Genre
        genre = st.selectbox(
            "Genre Utama:",
            options=unique_genres,
            help="Pilih salah satu genre utama."
        )
        
        # 5. Input Tahun Rilis
        release_year = st.slider(
            "Tahun Rilis:",
            min_value=1920, max_value=pd.Timestamp('now').year, value=2024, step=1
        )

        if st.button("Prediksi Rating IMDb", type="primary"):
            
            # Siapkan data input
            input_data = pd.DataFrame({
                'runtime': [runtime], 
                'metascore': [metascore], 
                'gross(M)': [gross],
                'release_year': [release_year], 
                'genre': [genre]
            })

            pipeline, _ = train_model(df_clean)
            predicted_rating = pipeline.predict(input_data)[0]
            
            with col2:
                st.header("Hasil Prediksi")
                st.markdown("---")
                
                # Tampilkan hasil
                st.metric(
                    label="Rating IMDb Diprediksi", 
                    value=f"{predicted_rating:.2f} / 10"
                )

# Jalankan fungsi utama
if __name__ == "__main__":

    main()


