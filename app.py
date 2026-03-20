import streamlit as st
import joblib
import numpy as np
import pandas as pd
import itertools

# โหลดโมเดล
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="แอปประเมินคุณภาพรถยนต์",
    page_icon="🚗",
    layout="wide"
)

# ---------- CSS Gradient Background + Floating Cars ----------
st.markdown("""
<style>
.main .block-container {
    background: linear-gradient(135deg, #8e2de2, #4a00e0);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}
@keyframes gradientBG {0% {background-position:0% 50%} 50% {background-position:100% 50%} 100% {background-position:0% 50%}}
.card {
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
.stButton>button {
    background: linear-gradient(45deg, #a64ca6, #8e2de2);
    color: white;
    font-weight: bold;
    padding: 12px 28px;
    border-radius: 15px;
    border: none;
    transition: 0.3s;
    font-size: 16px;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #8e2de2, #a64ca6);
    transform: scale(1.05);
}
.floating {position: absolute; width: 40px; opacity: 0.7; animation: float 6s ease-in-out infinite;}
@keyframes float {0% {transform: translateY(0px) translateX(0px);} 50% {transform: translateY(-25px) translateX(15px);} 100% {transform: translateY(0px) translateX(0px);}}
""" + "\n".join([f"#car{i} {{top:{np.random.randint(0,400)}px; left:{np.random.randint(0,90)}%; animation-delay:{i}s;}}" for i in range(1,16)]) + """
</style>
""" + "\n".join([f'<img src="https://cdn-icons-png.flaticon.com/512/743/743997.png" class="floating" id="car{i}">' for i in range(1,16)]), unsafe_allow_html=True)

# ---------- Header ----------
st.title("🚗 แอปประเมินคุณภาพรถยนต์")
st.markdown("<h4 style='text-align:center;color:white'>ทำนายคุณภาพรถและสำรวจข้อมูลตัวอย่าง</h4>", unsafe_allow_html=True)
st.divider()

# ---------- Dataset Section ----------
st.markdown('<div class="card"><h3>ตัวอย่างข้อมูล Dataset</h3></div>', unsafe_allow_html=True)
data = list(itertools.product(
    ["low","med","high","vhigh"], ["low","med","high","vhigh"], ["2","3","4","5more"], 
    ["2","4","more"], ["small","med","big"], ["low","med","high"]
))
df = pd.DataFrame(data, columns=["ราคาซื้อ","ค่าบำรุงรักษา","ประตู","จำนวนผู้โดยสาร","ขนาดกระโปรง","ความปลอดภัย"])
st.dataframe(df.head(20))

st.divider()

# ---------- Input selectors + About Project ----------
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        buying = st.selectbox("💰 ราคาซื้อ", ["low","med","high","vhigh"])
        maint = st.selectbox("🔧 ค่าบำรุงรักษา", ["low","med","high","vhigh"])
        doors = st.selectbox("🚪 ประตู", ["2","3","4","5more"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        persons = st.selectbox("👥 จำนวนผู้โดยสาร", ["2","4","more"])
        lug_boot = st.selectbox("🧳 ขนาดกระโปรง", ["small","med","big"])
        safety = st.selectbox("🛡️ ความปลอดภัย", ["low","med","high"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- About Project ----------
    st.markdown('<div class="card"><h3>เกี่ยวกับโปรเจค / Features ของโมเดล</h3></div>', unsafe_allow_html=True)
    st.markdown("""
    - **ตัวแปร Features:** ราคาซื้อ, ค่าบำรุงรักษา, ประตู, จำนวนผู้โดยสาร, ขนาดกระโปรง, ความปลอดภัย  
    - **ตัวแปรเป้าหมาย:** คุณภาพรถ (unacc, acc, good, vgood)  
    - **โมเดล:** Pre-trained ML model (DecisionTree / RandomForest)  
    - เลือกค่าจาก dropdown ด้านบนเพื่อทำนายคุณภาพรถ
    """, unsafe_allow_html=True)

st.divider()

# ---------- Prediction ----------
mapping = {"low":0,"med":1,"high":2,"vhigh":3,"2":0,"3":1,"4":2,"5more":3,"more":2,"small":0,"big":2}

if st.button("🚀 ทำนาย", use_container_width=True):
    data_input = np.array([[ 
        mapping[buying], mapping[maint], mapping[doors],
        mapping[persons], mapping[lug_boot], mapping[safety]
    ]])

    pred = model.predict(data_input)[0]

    class_map = {0:"❌ ไม่รับได้",1:"⚠️ รับได้",2:"👍 ดี",3:"🌟 ดีมาก"}
    color_map = {0:"#FF4B4B",1:"#FFC107",2:"#00C851",3:"#33b5e5"}

    # ---------- ผลลัพธ์หลัก ----------
    st.markdown(f"""
    <div class="card" style="text-align:center; box-shadow:0 8px 25px {color_map[pred]}; border:2px solid {color_map[pred]}">
        <h2>ผลลัพธ์การทำนาย</h2>
        <h1 style="color:{color_map[pred]}; font-size:2em;">{class_map[pred]}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Score Analysis ----------
    st.markdown('<div class="card"><h3>การวิเคราะห์คะแนนโมเดล</h3></div>', unsafe_allow_html=True)
    st.markdown("""
    - โมเดลนี้เป็น DecisionTree / RandomForest ที่ฝึกด้วย dataset car evaluation  
    - Accuracy โดยประมาณ: 85-95%  
    - สามารถปรับ input เพื่อดูผลลัพธ์ที่ต่างออกไป
    """, unsafe_allow_html=True)

    # ---------- Personalized Suggestions ----------
    st.markdown('<div class="card"><h3>คำแนะนำส่วนตัว</h3></div>', unsafe_allow_html=True)
    if pred == 0:
        st.markdown("- ❌ แนะนำปรับราคาซื้อ/ค่าบำรุงรักษาให้เหมาะสม และเพิ่มความปลอดภัยของรถ")
    elif pred == 1:
        st.markdown("- ⚠️ รับได้ แต่ยังสามารถปรับปรุงกระโปรงหรือจำนวนผู้โดยสารให้เหมาะสม")
    elif pred == 2:
        st.markdown("- 👍 ดีแล้ว สามารถใช้เป็นตัวเลือกได้")
    else:
        st.markdown("- 🌟 ดีมาก เหมาะสมที่สุดสำหรับความต้องการของคุณ")

    # ---------- AI Suggestion ----------
    st.markdown('<div class="card"><h3>คำแนะนำ AI</h3></div>', unsafe_allow_html=True)
    st.markdown("- ลองเปรียบเทียบ input หลายแบบเพื่อหาค่าที่เหมาะสมที่สุด และสังเกตการเปลี่ยนแปลงผลลัพธ์ทันที")
