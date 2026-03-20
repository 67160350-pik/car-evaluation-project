import streamlit as st
import joblib
import numpy as np

# =========================
# Load model
# =========================
model = joblib.load("model.pkl")

# =========================
# UI
# =========================
st.set_page_config(page_title="Car Evaluation", layout="centered")

st.title("Car Evaluation Prediction")
st.write("แอปนี้ใช้สำหรับทำนายคุณภาพของรถจากข้อมูลที่คุณเลือก")

st.markdown("---")

# =========================
# Input Section
# =========================
st.subheader("Input Features")

col1, col2 = st.columns(2)

with col1:
    buying = st.selectbox("Buying (ราคาซื้อ)", [0,1,2,3])
    doors = st.selectbox("Doors (จำนวนประตู)", [0,1,2,3])
    lug_boot = st.selectbox("Lug Boot (ที่เก็บของ)", [0,1,2,3])

with col2:
    maint = st.selectbox("Maint (ค่าบำรุงรักษา)", [0,1,2,3])
    persons = st.selectbox("Persons (จำนวนที่นั่ง)", [0,1,2,3])
    safety = st.selectbox("Safety (ความปลอดภัย)", [0,1,2,3])

st.markdown("---")

# =========================
# Prediction
# =========================
if st.button("Predict"):
    data = np.array([[buying, maint, doors, persons, lug_boot, safety]])

    pred = model.predict(data)
    prob = model.predict_proba(data)

    st.subheader("Result")

    st.success(f"Prediction: {pred[0]}")
    st.info(f"Confidence: {prob.max():.2f}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("This project is for educational purposes only.")
