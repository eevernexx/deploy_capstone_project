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
    try:
        # Pakai file yang ADA dan WORKING
        st.write("🔄 Loading files...")
        
        # Load scaler yang matching dengan model
        try:
            scaler = pickle.load(open('scaler_simple.pkl', 'rb'))
            st.success("✅ scaler_simple.pkl loaded (matching model)")
        except FileNotFoundError:
            scaler = pickle.load(open('scaler_OLD.pkl', 'rb'))
            st.warning("⚠️ Using scaler_OLD.pkl - may have compatibility issues")
            st.info("💡 For best results, upload scaler_simple.pkl")
        
        # Load label encoders yang working
        label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
        st.success("✅ label_encoders.pkl loaded")
        
        # Load feature names yang working
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        st.success("✅ feature_names.pkl loaded")
        
        # Coba load model simple dulu, kalau ga ada pakai yang lama
        model = None
        model_name = ""
        
        try:
            model = pickle.load(open('model_simple.pkl', 'rb'))
            model_name = "Model Simple (New)"
            st.success("✅ model_simple.pkl loaded")
        except FileNotFoundError:
            st.warning("⚠️ model_simple.pkl not found, need to upload this file")
            st.info("💡 Please upload model_simple.pkl to root directory")
            return None, None, None, None, None
        
        st.success("🎉 ALL WORKING FILES LOADED!")
        
        return model, scaler, label_encoders, feature_names, model_name
        
    except Exception as e:
        st.error(f"❌ Error loading: {str(e)}")
        return None, None, None, None, None

# Function untuk preprocessing input
def preprocess_input_smart(data, label_encoders, feature_names):
    """
    Preprocessing yang smart - adaptif dengan file yang ada
    """
    
    st.write(f"🔍 **Expected features:** {feature_names}")
    
    # Buat DataFrame
    df = pd.DataFrame([data])
    
    # Encoding berdasarkan label_encoders yang ada
    # (Ini lebih akurat karena pakai encoder yang sama dengan training)
    
    # Gender encoding
    if 'gender' in label_encoders:
        gender_encoder = label_encoders['gender']
        # Mapping manual berdasarkan classes_
        if hasattr(gender_encoder, 'classes_'):
            classes = gender_encoder.classes_
            # Female biasanya 0, Male biasanya 1 (alphabetical)
            gender_map = {cls: idx for idx, cls in enumerate(classes)}
            df['Gender'] = df['Gender'].map(gender_map)
        else:
            # Fallback manual
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    else:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    # Family history
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1})
    
    # FAVC
    df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
    
    # CAEC - Always, Frequently, Sometimes, no (alphabetical order)
    caec_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
    df['CAEC'] = df['CAEC'].map(caec_map)
    
    # SMOKE
    df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1})
    
    # SCC
    df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})
    
    # CALC - Always, Frequently, Sometimes, no
    calc_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
    df['CALC'] = df['CALC'].map(calc_map)
    
    # MTRANS - Automobile, Bike, Motorbike, Public_Transportation, Walking
    mtrans_map = {
        'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 
        'Public_Transportation': 3, 'Walking': 4
    }
    df['MTRANS'] = df['MTRANS'].map(mtrans_map)
    
    # Handle missing columns yang dibutuhkan feature_names
    for col in feature_names:
        if col not in df.columns:
            if col == 'BMI':
                df['BMI'] = df['Weight'] / (df['Height'] ** 2)
                st.write("✅ BMI calculated and added")
            else:
                df[col] = 0
                st.write(f"⚠️ {col} set to default value 0")
    
    # Reorder sesuai feature_names EXACT ORDER
    df_final = df[feature_names]  # Ini akan otomatis reorder sesuai urutan feature_names
    
    st.write(f"🔍 **Expected order:** {feature_names}")
    st.write(f"🔍 **Final columns:** {list(df_final.columns)}")
    st.write(f"🔍 **Final shape:** {df_final.shape}")
    st.write(f"🔍 **Order match:** {list(df_final.columns) == feature_names}")
    
    return df_final

# Main app
def main():
    st.markdown('<h1 class="main-header">🏥 Prediksi Tingkat Obesitas</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">📊 Aplikasi Machine Learning untuk memprediksi tingkat obesitas berdasarkan gaya hidup dan kondisi fisik</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_encoders, feature_names, model_name = load_model()
    
    if model is None:
        st.error("❌ Please upload model_simple.pkl to continue")
        st.info("💡 Steps: Go to GitHub → Upload model_simple.pkl to root directory → Refresh this app")
        st.stop()
    
    # Success message
    st.success(f"✅ {model_name} berhasil dimuat!")
    st.info(f"🔍 Model menggunakan {len(feature_names)} features")
    
    # Input fields
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
            # Prepare input data
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
                # Preprocessing
                processed_data = preprocess_input_smart(input_data, label_encoders, feature_names)
                
                # Scale data
                st.write(f"🔍 **Before scaling - column order:** {list(processed_data.columns)}")
                st.write(f"🔍 **Expected by model:** {feature_names}")
                st.write(f"🔍 **Match:** {list(processed_data.columns) == feature_names}")
                
                # Ensure exact order match
                if list(processed_data.columns) != feature_names:
                    st.warning("⚠️ Column order mismatch! Reordering...")
                    processed_data = processed_data[feature_names]
                    st.write(f"🔍 **After reorder:** {list(processed_data.columns)}")
                
                input_scaled = scaler.transform(processed_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Obesity levels (gunakan yang dari label_encoders kalau ada)
                if 'target' in label_encoders and hasattr(label_encoders['target'], 'classes_'):
                    obesity_levels = list(label_encoders['target'].classes_)
                else:
                    # Fallback default
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
                
                # Color coding
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
                
                # Translation
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
                
                # BMI Analysis
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
                
                # Recommendations
                st.subheader("💡 Rekomendasi Kesehatan")
                
                if predicted_level == 'Normal_Weight':
                    st.success("🎉 **Selamat!** Berat badan Anda dalam kategori normal. Pertahankan pola hidup sehat!")
                elif predicted_level == 'Insufficient_Weight':
                    st.info("📈 **Perlu Peningkatan Berat Badan** - Konsultasi dengan ahli gizi untuk program penambahan berat badan yang sehat.")
                elif 'Overweight' in predicted_level:
                    st.warning("⚖️ **Perlu Penurunan Berat Badan** - Tingkatkan aktivitas fisik dan atur pola makan yang lebih seimbang.")
                else:  # Obesity types
                    st.error("🏥 **Konsultasi Medis Diperlukan** - Segera konsultasi dengan dokter atau ahli gizi untuk program penurunan berat badan yang aman.")
                    
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
