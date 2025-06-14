import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Set page config with custom theme
st.set_page_config(
    page_title="AI Obesity Predictor 🤖",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #333;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-left: 4px solid #f093fb;
        padding-left: 1rem;
    }
    
    .input-container {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(5px);
    }
    
    .result-container {
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .result-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .healthy {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d5016;
        border-color: #a8edea;
    }
    
    .warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #8b4513;
        border-color: #ffecd2;
    }
    
    .danger {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #8b0000;
        border-color: #ff9a9e;
    }
    
    .predict-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        margin: 2rem 0;
    }
    
    .predict-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .recommendation-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        backdrop-filter: blur(5px);
    }
    
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    
    .sidebar .stSlider > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .success-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .loading-spinner {
        border: 4px solid rgba(102, 126, 234, 0.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model dan preprocessing objects
@st.cache_resource
def load_model():
    try:
        st.write("🔄 Loading AI models...")
        
        try:
            scaler = pickle.load(open('scaler_simple.pkl', 'rb'))
        except FileNotFoundError:
            scaler = pickle.load(open('scaler_OLD.pkl', 'rb'))
        
        label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        
        model = None
        model_name = ""
        
        try:
            model = pickle.load(open('model_simple.pkl', 'rb'))
            model_name = "Advanced AI Model v2.0"
        except FileNotFoundError:
            return None, None, None, None, None
        
        return model, scaler, label_encoders, feature_names, model_name
        
    except Exception as e:
        st.error(f"❌ Error loading: {str(e)}")
        return None, None, None, None, None

# Function untuk preprocessing input
def preprocess_input_smart(data, label_encoders, feature_names):
    df = pd.DataFrame([data])
    
    # Gender encoding
    if 'gender' in label_encoders:
        gender_encoder = label_encoders['gender']
        if hasattr(gender_encoder, 'classes_'):
            classes = gender_encoder.classes_
            gender_map = {cls: idx for idx, cls in enumerate(classes)}
            df['Gender'] = df['Gender'].map(gender_map)
        else:
            df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    else:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'no': 0, 'yes': 1})
    df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
    
    caec_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
    df['CAEC'] = df['CAEC'].map(caec_map)
    
    df['SMOKE'] = df['SMOKE'].map({'no': 0, 'yes': 1})
    df['SCC'] = df['SCC'].map({'no': 0, 'yes': 1})
    
    calc_map = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
    df['CALC'] = df['CALC'].map(calc_map)
    
    mtrans_map = {
        'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 
        'Public_Transportation': 3, 'Walking': 4
    }
    df['MTRANS'] = df['MTRANS'].map(mtrans_map)
    
    for col in feature_names:
        if col not in df.columns:
            if col == 'BMI':
                df['BMI'] = df['Weight'] / (df['Height'] ** 2)
            else:
                df[col] = 0
    
    df_final = df[feature_names]
    return df_final

# Create BMI gauge chart
def create_bmi_gauge(bmi_value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = bmi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "BMI Score", 'font': {'size': 24, 'color': '#333'}},
        delta = {'reference': 22.5, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#51cf66"}},
        gauge = {
            'axis': {'range': [None, 40], 'tickcolor': '#333'},
            'bar': {'color': "#667eea", 'thickness': 0.8},
            'steps': [
                {'range': [0, 18.5], 'color': "#74c0fc"},
                {'range': [18.5, 25], 'color': "#51cf66"},
                {'range': [25, 30], 'color': "#ffd43b"},
                {'range': [30, 40], 'color': "#ff6b6b"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': bmi_value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "#333", 'family': "Poppins"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Create prediction confidence chart
def create_confidence_chart(prediction_proba, obesity_levels):
    fig = go.Figure(data=[
        go.Bar(
            x=obesity_levels,
            y=prediction_proba * 100,
            marker_color='rgba(102, 126, 234, 0.8)',
            marker_line_color='rgba(102, 126, 234, 1.0)',
            marker_line_width=2,
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'AI Prediction Confidence Distribution',
            'x': 0.5,
            'font': {'size': 18, 'color': '#333', 'family': 'Poppins'}
        },
        xaxis_title="Obesity Categories",
        yaxis_title="Confidence (%)",
        height=400,
        font={'color': "#333", 'family': "Poppins"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.8)',
        xaxis={'tickangle': -45}
    )
    return fig

# Main app
def main():
    # Header with animation
    st.markdown("""
    <div class="main-container">
        <h1 class="main-header">🤖 AI Obesity Predictor</h1>
        <p class="sub-header">
            ✨ Prediksi tingkat obesitas dengan teknologi AI terdepan<br>
            🎯 Analisis komprehensif berdasarkan gaya hidup dan kondisi fisik Anda
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, label_encoders, feature_names, model_name = load_model()
    
    if model is None:
        st.error("❌ AI Model tidak dapat dimuat")
        st.stop()
    
    # Success message with animation
    st.markdown(f"""
    <div class="main-container success-animation">
        <h3 style="color: #51cf66; text-align: center; font-weight: 600;">
            ✅ {model_name} Ready! 🚀
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar untuk input dengan styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-bottom: 2rem;">
            <h2>📋 Data Input</h2>
            <p>Lengkapi semua informasi di bawah ini</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input sections with modern cards
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">👤 Informasi Personal</h3>', unsafe_allow_html=True)
        
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            gender = st.selectbox("🚻 Jenis Kelamin", ["Female", "Male"], key="gender")
            age = st.slider("🎂 Usia (tahun)", 14, 61, 25, key="age")
        with input_col2:
            height = st.number_input("📏 Tinggi Badan (m)", 1.45, 1.98, 1.70, step=0.01, format="%.2f", key="height")
            weight = st.number_input("⚖️ Berat Badan (kg)", 39.0, 173.0, 70.0, step=0.1, format="%.1f", key="weight")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">👨‍👩‍👧‍👦 Riwayat Keluarga</h3>', unsafe_allow_html=True)
        family_history = st.selectbox(
            "🧬 Apakah ada anggota keluarga yang pernah/sedang mengalami kelebihan berat badan?", 
            ["no", "yes"], key="family"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🍽️ Pola Makan</h3>', unsafe_allow_html=True)
        
        food_col1, food_col2 = st.columns(2)
        with food_col1:
            favc = st.selectbox("🍔 Sering konsumsi makanan tinggi kalori?", ["no", "yes"], key="favc")
            fcvc = st.number_input("🥗 Frekuensi makan sayuran (1-3)", 1.0, 3.0, 2.0, step=0.1, key="fcvc")
        with food_col2:
            ncp = st.number_input("🍽️ Jumlah makan besar per hari", 1.0, 4.0, 3.0, step=1.0, key="ncp")
            caec = st.selectbox("🍿 Frekuensi ngemil", ["no", "Sometimes", "Frequently", "Always"], key="caec")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🏃‍♂️ Aktivitas & Lifestyle</h3>', unsafe_allow_html=True)
        
        activity_col1, activity_col2 = st.columns(2)
        with activity_col1:
            smoke = st.selectbox("🚭 Apakah Anda merokok?", ["no", "yes"], key="smoke")
            ch2o = st.number_input("💧 Konsumsi air per hari (liter)", 1.0, 3.0, 2.0, step=0.1, key="water")
            scc = st.selectbox("📊 Memantau asupan kalori?", ["no", "yes"], key="scc")
        with activity_col2:
            faf = st.number_input("🏋️ Frekuensi aktivitas fisik (0-3)", 0.0, 3.0, 1.0, step=0.1, key="faf")
            tue = st.number_input("📱 Penggunaan teknologi per hari (jam)", 0.0, 2.0, 1.0, step=0.1, key="tue")
            calc = st.selectbox("🍷 Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"], key="calc")
        
        mtrans = st.selectbox("🚗 Jenis transportasi utama", 
                             ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"], key="transport")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick BMI preview
        if height > 0 and weight > 0:
            bmi_preview = weight / (height ** 2)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">BMI Preview</div>
                <div class="metric-value">{bmi_preview:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI gauge
            st.plotly_chart(create_bmi_gauge(bmi_preview), use_container_width=True, config={'displayModeBar': False})
    
    # Predict button with enhanced styling
    if st.button("🔍 ANALISIS DENGAN AI", key="predict_btn"):
        # Create loading animation
        with st.container():
            st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
            
        with st.spinner("🤖 AI sedang menganalisis data Anda..."):
            # Prepare input data
            input_data = {
                'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
                'family_history_with_overweight': family_history, 'FAVC': favc,
                'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke,
                'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue,
                'CALC': calc, 'MTRANS': mtrans
            }
            
            try:
                # Preprocessing
                processed_data = preprocess_input_smart(input_data, label_encoders, feature_names)
                input_scaled = scaler.transform(processed_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Obesity levels
                if 'target' in label_encoders and hasattr(label_encoders['target'], 'classes_'):
                    obesity_levels = list(label_encoders['target'].classes_)
                else:
                    obesity_levels = [
                        'Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
                        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II'
                    ]
                
                predicted_level = obesity_levels[prediction]
                confidence = max(prediction_proba) * 100
                
                # Results section
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0;">
                    <h2 style="color: #333; font-weight: 600;">🎯 Hasil Analisis AI</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Color coding and status
                if predicted_level in ['Insufficient_Weight', 'Normal_Weight']:
                    result_class = "healthy"
                    emoji = "✅"
                    status = "KONDISI SEHAT"
                elif predicted_level in ['Overweight_Level_I', 'Overweight_Level_II']:
                    result_class = "warning"
                    emoji = "⚠️"
                    status = "PERLU PERHATIAN"
                else:
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
                
                # Main result display
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown(f"""
                    <div class="result-container {result_class}">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{emoji}</div>
                        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">{status}</div>
                        <div style="font-size: 1.2rem; margin-bottom: 1rem;">{indonesian_level}</div>
                        <div style="font-size: 1rem; opacity: 0.8;">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    # BMI detailed analysis
                    bmi = weight / (height ** 2)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">BMI Analysis</div>
                        <div class="metric-value">{bmi:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if bmi < 18.5:
                        bmi_category, bmi_color = "Underweight", "🔵"
                    elif 18.5 <= bmi < 25:
                        bmi_category, bmi_color = "Normal", "🟢"
                    elif 25 <= bmi < 30:
                        bmi_category, bmi_color = "Overweight", "🟡"
                    else:
                        bmi_category, bmi_color = "Obese", "🔴"
                    
                    st.markdown(f"""
                    <div style="text-align: center; color: #333; font-size: 1.1rem; margin-top: 1rem;">
                        {bmi_color} <strong>{bmi_category}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence distribution chart
                st.plotly_chart(create_confidence_chart(prediction_proba, obesity_levels), use_container_width=True, config={'displayModeBar': False})
                
                # Enhanced recommendations
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                st.markdown("### 💡 Rekomendasi Personal AI")
                
                if predicted_level == 'Normal_Weight':
                    st.success("🎉 **Excellent!** Berat badan Anda dalam kategori ideal. Pertahankan pola hidup sehat ini!")
                    st.info("✨ **Tips Maintenance:** Lanjutkan olahraga rutin, pola makan seimbang, dan jaga hidrasi tubuh.")
                elif predicted_level == 'Insufficient_Weight':
                    st.info("📈 **Action Plan:** Program penambahan berat badan sehat diperlukan.")
                    st.warning("🏥 **Rekomendasi:** Konsultasi dengan ahli gizi untuk diet penambahan massa tubuh yang aman.")
                elif 'Overweight' in predicted_level:
                    st.warning("⚖️ **Improvement Zone:** Penurunan berat badan bertahap sangat direkomendasikan.")
                    st.info("🏃‍♂️ **Action Items:** Tingkatkan aktivitas fisik, kurangi kalori tinggi, perbanyak sayuran dan protein.")
                else:  # Obesity types
                    st.error("🏥 **Medical Attention Required:** Konsultasi medis segera diperlukan untuk program penurunan berat badan yang aman dan efektif.")
                    st.warning("⚠️ **Important:** Kondisi ini dapat meningkatkan risiko penyakit kardiovaskular, diabetes, dan komplikasi kesehatan lainnya.")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {str(e)}")
                st.write("Silakan periksa input data Anda dan coba lagi.")
    
    # Enhanced Footer
    st.markdown("""
    <div class="footer">
        <h3>🎓 Capstone Project - AI Health Analytics</h3>
        <p><strong>Bengkel Koding Data Science</strong></p>
        <p>🏛️ Universitas Dian Nuswantoro | 📅 Semester Genap 2024/2025</p>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
            <p><em>⚠️ Disclaimer: Hasil prediksi AI ini adalah untuk tujuan edukasi dan penelitian. 
            Tidak menggantikan konsultasi medis profesional. Selalu konsultasikan kondisi kesehatan Anda dengan dokter.</em></p>
        </div>
        <div style="margin-top: 1rem;">
            <p>🤖 Powered by Advanced Machine Learning • 🔬 Built with Streamlit & Python</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
