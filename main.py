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

# --- Konfigurasi dan Konstanta ---
TARGET_COLUMN = 'rating'
FEATURES = ['runtime', 'metascore', 'gross(M)', 'release_year', 'genre']
BEST_MODEL = RandomForestRegressor(random_state=42, n_estimators=100)
FILE_PATH = "imdb_clean.csv"

# --- Fungsi Pemuatan dan Pelatihan Data (Cache) ---

# Menggunakan st.cache_data untuk caching data
@st.cache_data
def load_data(file_path):
    """Memuat dan membersihkan data target."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File {file_path} tidak ditemukan. Pastikan 'imdb_clean.csv' ada di direktori yang sama.")
        return pd.DataFrame()

    # SOLUSI ERROR: Hapus SEMUA baris di mana 'rating' (target) adalah NaN.
    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()
    
    # Ambil daftar unik genre untuk input Streamlit
    unique_genres = df_clean['genre'].dropna().unique().tolist()
    
    return df_clean, unique_genres

# Menggunakan st.cache_resource untuk caching model dan pipeline
@st.cache_resource
def train_model(df_clean):
    """Melatih model Random Forest Regressor dengan Preprocessing Pipeline."""
    
    # 1. Pisahkan X dan y
    X = df_clean[FEATURES]
    y = df_clean[TARGET_COLUMN]

    # 2. Definisikan Kolom dan Pipeline Preprocessing
    numerical_features = ['runtime', 'metascore', 'gross(M)', 'release_year']
    categorical_features = ['genre']

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Perhatian: handle_unknown='ignore' agar OHE tidak error saat prediksi data baru
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

    # 3. Pipeline Penuh
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', BEST_MODEL)
    ])

    # 4. Pelatihan Model
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
    st.markdown("Gunakan model **Random Forest** (berdasarkan analisis Orange Anda) untuk memprediksi Rating IMDb film berdasarkan kriteria utama.")
    st.markdown("---")
    
    # --- Side Bar: Model Training & Summary ---
    with st.sidebar:
        st.header("Ringkasan Model")
        st.info("Model **Random Forest Regressor** sudah dilatih pada seluruh data yang bersih. Kinerja model ini terbukti paling akurat dalam evaluasi Anda.")
        st.write(f"Jumlah baris data bersih yang digunakan: **{len(df_clean)}**")
        
        # Tampilkan Kepentingan Fitur
        st.subheader("Faktor Utama (Feature Importance)")
        
        # Latih model dan dapatkan imporstansi fitur
        pipeline, preprocessor = train_model(df_clean)
        
        # Hanya tampilkan Feature Importance jika model adalah Random Forest (atau GB)
        if isinstance(pipeline['regressor'], RandomForestRegressor):
            importances = pipeline['regressor'].feature_importances_
            
            # Mendapatkan nama fitur setelah OHE
            feature_names_transformed = pipeline['preprocessor'].get_feature_names_out()
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names_transformed,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Bersihkan nama fitur
            feature_importance_df['Feature'] = feature_importance_df['Feature'].str.split('__').str[-1]
            
            st.dataframe(feature_importance_df.head(5).set_index('Feature'), use_container_width=True)
            st.caption("Metascore dan Gross adalah faktor dominan dalam model Anda.")


    # --- Main Content: Input Prediksi ---
    col1, col2 = st.columns([1.5, 2])
    
    with col1:
        st.header("Masukkan Kriteria Film")
        
        # 1. Input Metascore (Paling Penting)
        metascore = st.slider(
            "Metascore (Skor Kritikus):",
            min_value=0, max_value=100, value=75, step=1,
            help="Skor yang diberikan oleh kritikus (rotten tomatoes/metacritic). Paling berpengaruh!"
        )

        # 2. Input Gross/Pendapatan (Penting Kedua)
        gross = st.number_input(
            "Gross (Pendapatan Kotor Global, dalam Juta USD):",
            min_value=0.0, max_value=2000.0, value=50.0, step=1.0, format="%.2f",
            help="Total pendapatan global film. Gunakan skala dalam Juta USD (50.0 = 50 Juta USD)."
        )

        # 3. Input Runtime/Durasi (Penting Ketiga)
        runtime = st.slider(
            "Runtime (Durasi Film, dalam Menit):",
            min_value=60, max_value=240, value=120, step=5,
            help="Berapa lama durasi film ini?"
        )
        
        # 4. Input Genre
        genre = st.selectbox(
            "Genre Utama:",
            options=unique_genres,
            help="Pilih salah satu genre utama. Pengaruhnya relatif kecil."
        )
        
        # 5. Input Tahun Rilis (Kurang Penting)
        release_year = st.slider(
            "Tahun Rilis:",
            min_value=1920, max_value=pd.Timestamp('now').year, value=2024, step=1
        )

        # Tombol Prediksi
        if st.button("Prediksi Rating IMDb", type="primary"):
            
            # Siapkan data input
            input_data = pd.DataFrame({
                'runtime': [runtime], 
                'metascore': [metascore], 
                'gross(M)': [gross],
                'release_year': [release_year], 
                'genre': [genre]
            })
            
            # Lakukan Prediksi
            pipeline, _ = train_model(df_clean)
            predicted_rating = pipeline.predict(input_data)[0]
            
            with col2:
                st.header("Hasil Prediksi")
                st.markdown("---")
                
                # Tampilkan hasil
                st.metric(
                    label="Rating IMDb Diprediksi", 
                    value=f"{predicted_rating:.2f} / 10", 
                    delta=f"Model: {pipeline['regressor'].__class__.__name__}"
                )
                
                # Tampilkan pesan berdasarkan rating
                if predicted_rating >= 8.5:
                    st.balloons()
                    st.success("Rating ini menunjukkan film Kualitas Masterpiece dan sangat mungkin masuk daftar 'Top Rated'!")
                elif predicted_rating >= 7.5:
                    st.info("Rating ini menunjukkan film Kualitas Sangat Baik dan sukses!")
                else:
                    st.warning("Rating di bawah 7.5; Film ini mungkin kurang mendapatkan sambutan hangat dari penonton.")

# Jalankan fungsi utama
if __name__ == "__main__":
    main()