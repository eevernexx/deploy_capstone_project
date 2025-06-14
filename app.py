import streamlit as st
import pandas as pd
import numpy as np

st.title("🧪 Test Chart - Pastikan Chart Muncul")

st.write("Jika chart di bawah ini muncul, berarti Streamlit chart berfungsi:")

# Test 1: Simple Line Chart
st.subheader("📈 Test 1: Line Chart")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.line_chart(chart_data)

# Test 2: Simple Bar Chart  
st.subheader("📊 Test 2: Bar Chart")
bar_data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [23, 45, 56, 78]
})
st.bar_chart(bar_data.set_index('Category'))

# Test 3: Area Chart
st.subheader("📊 Test 3: Area Chart")
area_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['x', 'y', 'z'])
st.area_chart(area_data)

# Test 4: Simple Metrics
st.subheader("📊 Test 4: Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

if st.button("Generate Random Chart"):
    random_data = pd.DataFrame({
        'Random A': np.random.randn(10),
        'Random B': np.random.randn(10),
    })
    st.bar_chart(random_data)

st.info("Jika semua chart di atas muncul, berarti Streamlit charts working!")
