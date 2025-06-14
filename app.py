import streamlit as st
import pandas as pd
import numpy as np

st.title("🔍 DEBUG: Step by Step Chart Testing")

# Test if basic charts work at all
st.subheader("Step 1: Basic Chart Test")
test_data = pd.DataFrame({
    'Values': [10, 20, 30, 40]
}, index=['A', 'B', 'C', 'D'])

try:
    st.bar_chart(test_data)
    st.success("✅ Basic chart works!")
except Exception as e:
    st.error(f"❌ Basic chart failed: {e}")

# Test chart with prediction-like data
st.subheader("Step 2: Prediction-like Data")
fake_predictions = np.array([0.1, 0.6, 0.05, 0.15, 0.03, 0.04, 0.03])
obesity_levels = [
    'Insufficient_Weight',
    'Normal_Weight', 
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III',
    'Overweight_Level_I',
    'Overweight_Level_II'
]

level_translation = {
    'Insufficient_Weight': 'Berat Badan Kurang',
    'Normal_Weight': 'Berat Badan Normal',
    'Overweight_Level_I': 'Kelebihan Berat Badan Tingkat I',
    'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
    'Obesity_Type_I': 'Obesitas Tipe I',
    'Obesity_Type_II': 'Obesitas Tipe II',
    'Obesity_Type_III': 'Obesitas Tipe III'
}

try:
    chart_labels = [level_translation.get(level, level) for level in obesity_levels]
    chart_probs = fake_predictions * 100
    
    prob_chart_data = pd.DataFrame({
        'Probabilitas (%)': chart_probs
    }, index=chart_labels)
    
    st.bar_chart(prob_chart_data)
    st.success("✅ Prediction chart works!")
except Exception as e:
    st.error(f"❌ Prediction chart failed: {e}")

# Test lifestyle data
st.subheader("Step 3: Lifestyle Data")
try:
    lifestyle_data = {
        'Pola Makan': 75,
        'Aktivitas': 60,
        'Kebiasaan Sehat': 80
    }
    
    lifestyle_df = pd.DataFrame({
        'Skor': list(lifestyle_data.values())
    }, index=list(lifestyle_data.keys()))
    
    st.bar_chart(lifestyle_df)
    st.success("✅ Lifestyle chart works!")
except Exception as e:
    st.error(f"❌ Lifestyle chart failed: {e}")

# Test BMI data
st.subheader("Step 4: BMI Reference Data")
try:
    bmi_ref_data = pd.DataFrame({
        'BMI Range': [18.5, 6.5, 5, 10]
    }, index=['Underweight (<18.5)', 'Normal (18.5-25)', 'Overweight (25-30)', 'Obese (>30)'])
    
    st.bar_chart(bmi_ref_data)
    st.success("✅ BMI chart works!")
except Exception as e:
    st.error(f"❌ BMI chart failed: {e}")

# Test risk factors
st.subheader("Step 5: Risk Factors")
try:
    risk_data = pd.DataFrame({
        'Jumlah': [2, 3]
    }, index=['Faktor Risiko', 'Faktor Protektif'])
    
    st.bar_chart(risk_data)
    st.success("✅ Risk chart works!")
except Exception as e:
    st.error(f"❌ Risk chart failed: {e}")

st.info("Upload kode ini sebagai app.py dan kasih tau mana yang error!")
