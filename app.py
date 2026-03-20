import streamlit as st
import joblib
import numpy as np

# โหลด model + encoder
model = joblib.load("model.pkl")

st.title("Car Evaluation")

buying = st.selectbox("Buying", [0,1,2,3])
maint = st.selectbox("Maint", [0,1,2,3])
doors = st.selectbox("Doors", [0,1,2,3])
persons = st.selectbox("Persons", [0,1,2,3])
lug_boot = st.selectbox("Lug Boot", [0,1,2,3])
safety = st.selectbox("Safety", [0,1,2,3])

if st.button("Predict"):
    try:
        data = np.array([[buying, maint, doors, persons, lug_boot, safety]])
        pred = model.predict(data)
        prob = model.predict_proba(data)

        st.success(f"Prediction: {pred[0]}")
        st.write(f"Confidence: {prob.max():.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
