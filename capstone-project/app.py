import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set page config
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeaa7;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model dan preprocessing objects
@st.cache_resource
def load_model():
    # Debug: Cek file apa aja yang ada
    import os
    st.write("🔍 **Debug - Files in current directory:**")
    files_in_dir = os.listdir('.')
    for file in sorted(files_in_dir):
        file_size = os.path.getsize(file) if os.path.isfile(file) else "folder"
        st.write(f"• {file} ({file_size} bytes)")
    
    # Cek isi folder capstone-project
    if os.path.exists('capstone-project'):
        st.write("\n📂 **Files in capstone-project folder:**")
        capstone_files = os.listdir('capstone-project')
        for file in sorted(capstone_files):
            file_path = os.path.join('capstone-project', file)
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "folder"
            st.write(f"• {file} ({file_size} bytes)")
    
    try:
        # Path yang benar - ada di subfolder capstone-project
        model_path = 'capstone-project/model_obesitas_optimal.pkl'
        scaler_path = 'capstone-project/scaler.pkl'
        
        # Cek apakah file ada
        if not os.path.exists(model_path):
            st.error(f"❌ {model_path} TIDAK DITEMUKAN!")
            return None, None, None
            
        if not os.path.exists(scaler_path):
            st.error(f"❌ {scaler_path} TIDAK DITEMUKAN!")
            return None, None, None
        
        st.success("✅ Kedua file ditemukan di capstone-project folder!")
        
        # Load model optimal
        st.write("🔄 Loading model...")
        model = pickle.load(open(model_path, 'rb'))
        model_name = "Model Obesitas Optimal"
        st.success(f"✅ Model loaded: {type(model).__name__}")
        
        # Load scaler
        st.write("🔄 Loading scaler...")
        scaler = pickle.load(open(scaler_path, 'rb'))
        st.success(f"✅ Scaler loaded: {type(scaler).__name__}")
        
        return model, scaler, model_name
        
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {str(e)}")
        return None, None, None
        
    except Exception as e:
        st.error(f"❌ Error loading: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None, None, None

# Function untuk preprocessing input sesuai dengan Colab
def preprocess_input(data):
    """
    Preprocessing sesuai dengan yang dilakukan di Google Colab
    """
    
    # Urutan kolom sesuai dengan dataset asli (berdasarkan Colab)
    columns_order = [
        'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ]
    
    # Create DataFrame dengan urutan kolom yang benar
    df = pd.DataFrame([data], columns=columns_order)
    
    # Encoding sesuai dengan yang dilakukan di Colab
    
    # 1. Gender encoding (Male=1, Female=0)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # 2. family_history_with_overweight (yes=1, no=0)
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
    
    # 3. FAVC - Frequent consumption of high caloric food (yes=1, no=0)
    df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})
    
    # 4. CAEC - Consumption of food between meals
    caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['CAEC'] = df['CAEC'].map(caec_mapping)
    
    # 5. SMOKE (yes=1, no=0)
    df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})
    
    # 6. SCC - Calories consumption monitoring (yes=1, no=0)
    df['SCC'] = df['SCC'].map({'yes': 1, 'no': 0})
    
    # 7. CALC - Alcohol consumption
    calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['CALC'] = df['CALC'].map(calc_mapping)
    
    # 8. MTRANS - Transportation used
    mtrans_mapping = {
        'Walking': 0, 'Bike': 1, 'Motorbike': 2, 
        'Public_Transportation': 3, 'Automobile': 4
    }
    df['MTRANS'] = df['MTRANS'].map(mtrans_mapping)
    
    return df

# Main app
def main():
    st.markdown('<h1 class="main-header">🏥 Prediksi Tingkat Obesitas</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">📊 Aplikasi Machine Learning untuk memprediksi tingkat obesitas berdasarkan gaya hidup dan kondisi fisik</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, model_name = load_model()
    
    if model is None:
        st.stop()
    
    # Success message if model loaded
    st.success(f"✅ {model_name} berhasil dimuat!")
    
    # Sidebar untuk input
    st.sidebar.header("📋 Input Data Responden")
    st.sidebar.markdown("Silakan isi semua informasi berikut:")
    
    # Input fields dengan layout yang rapi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Informasi Dasar")
        gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
        age = st.slider("Usia (tahun)", 14, 61, 25)
        height = st.number_input("Tinggi Badan (m)", 1.45, 1.98, 1.70, step=0.01, format="%.2f")
        weight = st.number_input("Berat Badan (kg)", 39.0, 173.0, 70.0, step=0.1, format="%.1f")
        
        st.subheader("👨‍👩‍👧‍👦 Riwayat Keluarga")
        family_history = st.selectbox(
            "Apakah ada anggota keluarga yang pernah/sedang mengalami kelebihan berat badan?", 
            ["no", "yes"]
        )
        
    with col2:
        st.subheader("🍽️ Kebiasaan Makan")
        favc = st.selectbox("Apakah Anda sering mengonsumsi makanan tinggi kalori?", ["no", "yes"])
        fcvc = st.number_input("Seberapa sering Anda makan sayuran? (1=jarang, 3=sering)", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.number_input("Berapa kali Anda makan besar dalam sehari?", 1.0, 4.0, 3.0, step=1.0)
        caec = st.selectbox("Apakah Anda makan camilan di antara waktu makan?", 
                           ["no", "Sometimes", "Frequently", "Always"])
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🚭 Gaya Hidup")
        smoke = st.selectbox("Apakah Anda merokok?", ["no", "yes"])
        ch2o = st.number_input("Berapa liter air yang Anda minum setiap hari?", 1.0, 3.0, 2.0, step=0.1)
        scc = st.selectbox("Apakah Anda memantau asupan kalori harian?", ["no", "yes"])
        
    with col4:
        st.subheader("🏃‍♂️ Aktivitas")
        faf = st.number_input("Seberapa sering Anda melakukan aktivitas fisik? (0=tidak pernah, 3=sangat sering)", 
                             0.0, 3.0, 1.0, step=0.1)
        tue = st.number_input("Berapa jam Anda menggunakan perangkat teknologi per hari?", 
                             0.0, 2.0, 1.0, step=0.1)
        calc = st.selectbox("Seberapa sering Anda mengonsumsi alkohol?", 
                           ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Jenis transportasi apa yang biasa Anda gunakan?", 
                             ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])
    
    # Predict button
    st.markdown("---")
    if st.button("🔍 PREDIKSI TINGKAT OBESITAS", type="primary"):
        with st.spinner("🔄 Sedang menganalisis data..."):
            # Prepare input data dengan urutan yang benar
            input_data = {
                'Gender': gender,
                'Age': age,
                'Height': height,
                'Weight': weight,
                'family_history_with_overweight': family_history,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }
            
            try:
                # Preprocess input
                processed_data = preprocess_input(input_data)
                
                # Scale the data
                input_scaled = scaler.transform(processed_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Labels sesuai dengan dataset (berdasarkan Colab analysis)
                obesity_levels = [
                    'Insufficient_Weight',
                    'Normal_Weight', 
                    'Obesity_Type_I',
                    'Obesity_Type_II',
                    'Obesity_Type_III',
                    'Overweight_Level_I',
                    'Overweight_Level_II'
                ]
                
                predicted_level = obesity_levels[prediction]
                confidence = max(prediction_proba) * 100
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                # Color coding based on result
                if predicted_level in ['Insufficient_Weight', 'Normal_Weight']:
                    result_class = "healthy"
                    emoji = "✅"
                    status = "SEHAT"
                elif predicted_level in ['Overweight_Level_I', 'Overweight_Level_II']:
                    result_class = "warning"
                    emoji = "⚠️"
                    status = "PERLU PERHATIAN"
                else:  # Obesity types
                    result_class = "danger"
                    emoji = "🚨"
                    status = "BERISIKO TINGGI"
                
                # Convert prediction to Indonesian
                level_translation = {
                    'Insufficient_Weight': 'Berat Badan Kurang',
                    'Normal_Weight': 'Berat Badan Normal',
                    'Overweight_Level_I': 'Kelebihan Berat Badan Tingkat I',
                    'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
                    'Obesity_Type_I': 'Obesitas Tipe I',
                    'Obesity_Type_II': 'Obesitas Tipe II',
                    'Obesity_Type_III': 'Obesitas Tipe III'
                }
                
                indonesian_level = level_translation.get(predicted_level, predicted_level)
                
                st.markdown(f"""
                <div class="result-box {result_class}">
                    {emoji} <strong>{status}</strong><br>
                    Tingkat Obesitas: {indonesian_level}<br>
                    Confidence: {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed analysis
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    # BMI calculation
                    bmi = weight / (height ** 2)
                    st.subheader("📏 Analisis BMI")
                    st.metric("BMI Anda", f"{bmi:.2f}")
                    
                    if bmi < 18.5:
                        bmi_category = "Underweight"
                        bmi_color = "🔵"
                    elif 18.5 <= bmi < 25:
                        bmi_category = "Normal"
                        bmi_color = "🟢"
                    elif 25 <= bmi < 30:
                        bmi_category = "Overweight"
                        bmi_color = "🟡"
                    else:
                        bmi_category = "Obese"
                        bmi_color = "🔴"
                        
                    st.write(f"{bmi_color} **Kategori BMI:** {bmi_category}")
                
                with col_analysis2:
                    # Top 3 probabilities
                    st.subheader("📈 Top 3 Prediksi")
                    prob_df = pd.DataFrame({
                        'Level': obesity_levels,
                        'Probability': prediction_proba
                    })
                    prob_df = prob_df.sort_values('Probability', ascending=False).head(3)
                    
                    for idx, row in prob_df.iterrows():
                        translated = level_translation.get(row['Level'], row['Level'])
                        st.write(f"**{translated}:** {row['Probability']*100:.1f}%")
                
                # Recommendations
                st.subheader("💡 Rekomendasi Kesehatan")
                
                if predicted_level == 'Normal_Weight':
                    st.success("""
                    🎉 **Selamat!** Berat badan Anda dalam kategori normal.
                    - Pertahankan pola makan seimbang
                    - Lanjutkan aktivitas fisik rutin
                    - Jaga hidrasi yang cukup
                    """)
                elif predicted_level == 'Insufficient_Weight':
                    st.info("""
                    📈 **Perlu Peningkatan Berat Badan**
                    - Konsultasi dengan ahli gizi untuk program penambahan berat badan
                    - Tingkatkan asupan kalori sehat
                    - Fokus pada latihan kekuatan otot
                    """)
                elif 'Overweight' in predicted_level:
                    st.warning("""
                    ⚖️ **Perlu Penurunan Berat Badan**
                    - Kurangi konsumsi makanan tinggi kalori
                    - Tingkatkan frekuensi dan intensitas olahraga
                    - Perbanyak konsumsi sayuran dan buah
                    - Pantau asupan kalori harian
                    """)
                else:  # Obesity types
                    st.error("""
                    🏥 **Konsultasi Medis Diperlukan**
                    - Segera konsultasi dengan dokter atau ahli gizi
                    - Pertimbangkan program penurunan berat badan terstruktur
                    - Evaluasi kondisi kesehatan secara menyeluruh
                    - Ubah gaya hidup secara bertahap dan konsisten
                    """)
                
                # Feature importance visualization (if you want to add this)
                st.subheader("🔍 Faktor yang Mempengaruhi")
                
                # Create a simple analysis of input factors
                risk_factors = []
                if favc == 'yes':
                    risk_factors.append("Sering konsumsi makanan tinggi kalori")
                if faf < 1:
                    risk_factors.append("Aktivitas fisik kurang")
                if family_history == 'yes':
                    risk_factors.append("Riwayat keluarga obesitas")
                if ch2o < 2:
                    risk_factors.append("Konsumsi air kurang")
                if tue > 1.5:
                    risk_factors.append("Penggunaan teknologi berlebihan")
                
                if risk_factors:
                    st.write("⚠️ **Faktor risiko yang teridentifikasi:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.write("✅ **Gaya hidup Anda sudah cukup baik!**")
                    
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {str(e)}")
                st.write("Silakan periksa input data Anda dan coba lagi.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Capstone Project - Bengkel Koding Data Science</strong><br>
        🎓 Universitas Dian Nuswantoro<br>
        📅 Semester Genap 2024/2025</p>
        <p><em>⚠️ Disclaimer: Hasil prediksi ini hanya untuk tujuan edukasi dan tidak menggantikan konsultasi medis profesional.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
