import streamlit as st
import os

st.title("🔍 DEBUG - File Structure Check")

# Cek semua file di directory
st.subheader("📂 All Files in Root Directory:")
try:
    all_files = os.listdir('.')
    for file in sorted(all_files):
        if os.path.isfile(file):
            file_size = os.path.getsize(file)
            st.write(f"• **{file}** ({file_size} bytes)")
        else:
            st.write(f"• **{file}** (folder)")
except Exception as e:
    st.error(f"Error listing files: {e}")

# Cek file .pkl specifically
st.subheader("🎯 PKL Files Only:")
pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
for pkl_file in pkl_files:
    file_size = os.path.getsize(pkl_file)
    st.success(f"✅ {pkl_file} ({file_size} bytes)")

if not pkl_files:
    st.error("❌ No .pkl files found!")

# Test load each pkl file
st.subheader("🧪 Test Loading Each PKL File:")
import pickle

for pkl_file in pkl_files:
    try:
        with open(pkl_file, 'rb') as f:
            obj = pickle.load(f)
        st.success(f"✅ {pkl_file} - Loaded successfully: {type(obj).__name__}")
    except Exception as e:
        st.error(f"❌ {pkl_file} - Error: {str(e)}")

# Manual file existence check
st.subheader("🎯 Manual File Check:")
files_to_check = [
    'model_simple.pkl',
    'scaler_simple.pkl', 
    'label_encoders.pkl',
    'feature_names.pkl',
    'model_obesitas_optimal_OLD.pkl',
    'scaler_OLD.pkl'
]

for file_name in files_to_check:
    if os.path.exists(file_name):
        file_size = os.path.getsize(file_name)
        st.success(f"✅ {file_name} EXISTS ({file_size} bytes)")
    else:
        st.error(f"❌ {file_name} NOT FOUND")

st.info("💡 Upload this as app.py temporarily to see exact file structure")
