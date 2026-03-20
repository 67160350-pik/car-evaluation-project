import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# โหลดโมเดลและ encoders
model = joblib.load("model.pkl")
le_dict = joblib.load("encoders.pkl")

# ---------- Page Config ----------
st.set_page_config(
    page_title="Car Evaluation App",
    page_icon="🚗",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
.main .block-container {
    background: linear-gradient(180deg, #b39ddb, #7b1fa2);
    color: white;
}

.card {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

.stButton>button {
    background: linear-gradient(45deg, #7b1fa2, #9c27b0);
    color: white;
    font-weight: bold;
    padding: 12px 28px;
    border-radius: 12px;
    border: none;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🚗 Car Evaluation App")
st.markdown("<h4 style='text-align:center;color:white'>🚀 Predict Car Quality & Explore Options</h4>", unsafe_allow_html=True)
st.divider()

# ---------- Input ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    buying = st.selectbox("💰 Buying Price", ["low","med","high","vhigh"])
    maint = st.selectbox("🔧 Maintenance", ["low","med","high","vhigh"])
    doors = st.selectbox("🚪 Doors", ["2","3","4","5more"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    persons = st.selectbox("👥 Persons", ["2","4","more"])
    lug_boot = st.selectbox("🧳 Luggage Boot", ["small","med","big"])
    safety = st.selectbox("🛡️ Safety", ["low","med","high"])
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------- Encode ----------
def encode_input(val, col):
    return le_dict[col].transform([val])[0]

# ---------- Predict ----------
if st.button("🚀 Predict", use_container_width=True):

    data_input = np.array([[
        encode_input(buying,"buying"),
        encode_input(maint,"maint"),
        encode_input(doors,"doors"),
        encode_input(persons,"persons"),
        encode_input(lug_boot,"lug_boot"),
        encode_input(safety,"safety")
    ]])
    
    pred = model.predict(data_input)[0]
    pred_class = le_dict["class"].inverse_transform([pred])[0]
    
    color_map = {"unacc":"#ff4b4b","acc":"#ffc107","good":"#4caf50","vgood":"#2196f3"}
    display_map = {"unacc":"Unacceptable ❌","acc":"Acceptable ⚠️","good":"Good 👍","vgood":"Very Good ⭐"}

    # ---------- Prediction Result ----------
    st.markdown(f"""
    <div class="card" style="text-align:center; border:2px solid {color_map[pred_class]}">
        <h2>🚗 Prediction Result</h2>
        <h1 style="color:{color_map[pred_class]}">{display_map[pred_class]}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Score Analysis + Graph ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("## 📈 Score Analysis")

    scores = np.array([
        encode_input(buying,"buying"),
        encode_input(maint,"maint"),
        encode_input(doors,"doors"),
        encode_input(persons,"persons"),
        encode_input(lug_boot,"lug_boot"),
        encode_input(safety,"safety")
    ]) * 25

    total_score = scores.sum()
    avg_score = scores.mean()

    st.markdown(f"📊 **Total Score:** {total_score}")
    st.markdown(f"📊 **Average Score:** {avg_score:.2f}")

    # กราฟ
    fig, ax = plt.subplots()
    ax.plot(scores, marker='o')
    ax.set_title("Score Distribution")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Score")

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Suggestions ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 💡 Suggestions")

    suggestions = {
        "unacc":["❌ ลดราคาซื้อ","❌ เพิ่มความปลอดภัย"],
        "acc":["⚠️ ปรับค่าบางอย่างเล็กน้อย"],
        "good":["👍 รถดีแล้ว ใช้งานได้"],
        "vgood":["⭐ รถเหมาะสมที่สุด"]
    }

    for s in suggestions[pred_class]:
        st.markdown(f"- {s}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- About Project ----------
st.divider()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## 📘 About Project")

st.markdown("""
- 🚗 Features: Buying price, Maintenance, Doors, Persons, Luggage Boot, Safety  
- 🎯 Target: Car quality (Unacceptable, Acceptable, Good, Very Good)  
- 🤖 Model: RandomForest Classifier  
- 📊 ใช้สำหรับทำนายคุณภาพรถจากข้อมูลที่เลือก
""")

st.markdown('</div>', unsafe_allow_html=True)
