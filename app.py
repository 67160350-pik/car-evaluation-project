import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load(model.pkl)

st.title(Car Evaluation App)

st.write(แอปนี้ใช้สำหรับทำนายคุณภาพของรถจากข้อมูลที่เลือก)

# อธิบาย feature
st.subheader(Input Features)
st.write(กรุณาเลือกค่าของแต่ละ feature)

buying = st.selectbox(Buying (ราคาซื้อ), [0,1,2,3])
maint = st.selectbox(Maint (ค่าบำรุงรักษา), [0,1,2,3])
doors = st.selectbox(Doors (จำนวนประตู), [0,1,2,3])
persons = st.selectbox(Persons (จำนวนที่นั่ง), [0,1,2,3])
lug_boot = st.selectbox(Lug Boot (ขนาดที่เก็บของ), [0,1,2,3])
safety = st.selectbox(Safety (ความปลอดภัย), [0,1,2,3])

# ปุ่มทำนาย
if st.button(Predict)
    data = np.array([[buying, maint, doors, persons, lug_boot, safety]])
    
    pred = model.predict(data)
    prob = model.predict_proba(data)

    st.subheader(Result)
    st.success(fPrediction {pred[0]})
    st.write(fConfidence {prob.max().2f})

# disclaimer
st.caption(This model is for educational purposes only.)