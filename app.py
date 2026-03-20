import streamlit as st
import joblib
import numpy as np

# =========================
# Style (พื้นหลังม่วงอ่อน)
# =========================
st.markdown("""
    <style>
    .stApp {
        background-color: #f3e8ff;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Load model
# =========================
try:
    model = joblib.load("model.pkl")
except:
    st.error("❌ โหลดโมเดลไม่ได้")
    st.stop()

# =========================
# Title
# =========================
st.title("🚗 Car Evaluation App")
st.write("เลือกคุณสมบัติของรถ แล้วระบบจะทำนายคุณภาพให้")

st.markdown("---")

# =========================
# Input (แบบสมจริง)
# =========================
col1, col2 = st.columns(2)

with col1:
    buying = st.selectbox("💰 Buying Price", ["low", "med", "high", "vhigh"])
    doors = st.selectbox("🚪 Number of Doors", ["2", "3", "4", "5more"])
    lug_boot = st.selectbox("🧳 Luggage Boot Size", ["small", "med", "big"])

with col2:
    maint = st.selectbox("🔧 Maintenance Cost", ["low", "med", "high", "vhigh"])
    persons = st.selectbox("👨‍👩‍👧 Capacity", ["2", "4", "more"])
    safety = st.selectbox("🛡 Safety Level", ["low", "med", "high"])

st.markdown("---")

# =========================
# Mapping (แปลงเป็นตัวเลข)
# =========================
buying_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
maint_map = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
doors_map = {"2": 0, "3": 1, "4": 2, "5more": 3}
persons_map = {"2": 0, "4": 1, "more": 2}
lug_map = {"small": 0, "med": 1, "big": 2}
safety_map = {"low": 0, "med": 1, "high": 2}

# class mapping (แสดงผลให้เข้าใจง่าย)
class_map = {
    0: "unacceptable ❌",
    1: "acceptable ⚠️",
    2: "good 👍",
    3: "very good ⭐"
}

# =========================
# Predict
# =========================
if st.button("🔮 Predict"):
    try:
        data = np.array([[
            buying_map[buying],
            maint_map[maint],
            doors_map[doors],
            persons_map[persons],
            lug_map[lug_boot],
            safety_map[safety]
        ]])

        pred = model.predict(data)
        prob = model.predict_proba(data)

        result = class_map.get(pred[0], pred[0])

        st.success(f"🚀 Prediction: {result}")
        st.info(f"📊 Confidence: {prob.max():.2f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Made for educational purposes")
