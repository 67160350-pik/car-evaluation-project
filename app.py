import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Car Evaluation App",
    page_icon="🚗",
    layout="wide"
)

# ---------- CSS Gradient Background + Floating Cars Down ----------
st.markdown("""
<style>
.main .block-container {
    background: linear-gradient(180deg, #a77bfa, #7b1fa2);
    color: white;
}

/* Card style */
.card {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

/* Button style */
.stButton>button {
    background: linear-gradient(45deg, #7b1fa2, #9c27b0);
    color: white;
    font-weight: bold;
    padding: 12px 28px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
    font-size: 16px;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #9c27b0, #7b1fa2);
    transform: scale(1.05);
}

/* Floating cars - drop down slowly */
.floating {
    position: absolute;
    font-size: 32px;
    opacity: 0.7;
    animation: floatDown 12s linear infinite;
}
@keyframes floatDown {
    0% {transform: translateY(-50px);}
    100% {transform: translateY(800px);}
}
""" + "\n".join([f"#car{i} {{left:{np.random.randint(5,90)}%; animation-delay:{i*1.5}s;}}" for i in range(1,11)]) + "</style>" +
"\n".join([f'<div class="floating" id="car{i}">{emoji}</div>' for i, emoji in enumerate(["🚗","🚙","🏎","🚐","🚗","🚙","🏎","🚐","🚗","🚙"],1)]),
unsafe_allow_html=True)

# ---------- Header ----------
st.title("🚗 Car Evaluation App")
st.markdown("<h4 style='text-align:center;color:white'>Predict Car Quality</h4>", unsafe_allow_html=True)
st.divider()

# ---------- Input selectors ----------
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

# ---------- Prediction ----------
mapping = {"low":0,"med":1,"high":2,"vhigh":3,"2":0,"3":1,"4":2,"5more":3,"more":2,"small":0,"big":2}

if st.button("🚀 Predict", use_container_width=True):
    data_input = np.array([[ 
        mapping[buying], mapping[maint], mapping[doors],
        mapping[persons], mapping[lug_boot], mapping[safety]
    ]])

    pred = model.predict(data_input)[0]

    class_map = {0:"Unacceptable",1:"Acceptable",2:"Good",3:"Very Good"}
    color_map = {0:"#D32F2F",1:"#FBC02D",2:"#388E3C",3:"#1976D2"}

    # ---------- Prediction Result ----------
    st.markdown(f"""
    <div class="card" style="text-align:center; box-shadow:0 8px 20px {color_map[pred]}; border:2px solid {color_map[pred]}">
        <h2>Prediction Result</h2>
        <h1 style="color:{color_map[pred]}; font-size:2em;">{class_map[pred]}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Score Analysis ----------
    st.markdown('<div class="card"><h3>Score Analysis</h3></div>', unsafe_allow_html=True)
    st.markdown("""
    - โมเดลนี้ถูกฝึกด้วย Dataset ประเมินคุณภาพรถ  
    - ความแม่นยำโดยประมาณ: 85-95%  
    - ลองเปลี่ยน input หลายแบบเพื่อดูผลลัพธ์แตกต่างกัน
    """, unsafe_allow_html=True)

    # ---------- Personalized Suggestions ----------
    st.markdown('<div class="card"><h3>Personalized Suggestions</h3></div>', unsafe_allow_html=True)
    if pred == 0:
        st.markdown("- ❌ แนะนำปรับราคาซื้อ/ค่าบำรุงรักษา และเพิ่มความปลอดภัยของรถ")
    elif pred == 1:
        st.markdown("- ⚠️ รับได้ แต่ยังสามารถปรับปรุงขนาดกระโปรงหรือจำนวนผู้โดยสารให้เหมาะสม")
    elif pred == 2:
        st.markdown("- 👍 ดีแล้ว สามารถใช้เป็นตัวเลือกได้")
    else:
        st.markdown("- 🌟 ดีมาก เหมาะสมที่สุดสำหรับความต้องการของคุณ")

    # ---------- AI Suggestion ----------
    st.markdown('<div class="card"><h3>AI Suggestion</h3></div>', unsafe_allow_html=True)
    st.markdown("- ทดลองปรับค่าหลายแบบเพื่อตรวจสอบผลลัพธ์ที่เหมาะสมที่สุด")

# ---------- About Project ----------
st.divider()
st.markdown('<div class="card"><h3>About Project</h3></div>', unsafe_allow_html=True)
st.markdown("""
- Features: Buying price, Maintenance, Doors, Persons, Luggage Boot, Safety  
- Target: Car quality (Unacceptable, Acceptable, Good, Very Good)  
- Model: Pre-trained DecisionTree / RandomForest  
- ใช้ตัวเลือกด้านบนเพื่อทำนายคุณภาพรถ
""", unsafe_allow_html=True)
