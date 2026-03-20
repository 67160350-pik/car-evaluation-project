import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดโมเดลและ encoders
model = joblib.load("model.pkl")
le_dict = joblib.load("encoders.pkl")

# ---------- Page Config ----------
st.set_page_config(
    page_title="Car Evaluation App",
    page_icon="🚗",
    layout="wide"
)

# ---------- CSS Background & Floating Cars ----------
st.markdown("""
<style>
.main .block-container {
    background: linear-gradient(180deg, #b39ddb, #7b1fa2);
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
st.markdown("<h4 style='text-align:center;color:white'>Predict Car Quality & Explore Options</h4>", unsafe_allow_html=True)
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
def encode_input(val, col):
    le = le_dict[col]
    return le.transform([val])[0]

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
    
    # ---------- Prediction Result ----------
    color_map = {"unacc":"#D32F2F","acc":"#FBC02D","good":"#388E3C","vgood":"#1976D2"}
    display_map = {"unacc":"Unacceptable","acc":"Acceptable","good":"Good","vgood":"Very Good"}
    
    st.markdown(f"""
    <div class="card" style="text-align:center; box-shadow:0 8px 20px {color_map[pred_class]}; border:2px solid {color_map[pred_class]}">
        <h2>Prediction Result</h2>
        <h1 style="color:{color_map[pred_class]}; font-size:2em;">{display_map[pred_class]}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------- Score Analysis ----------
    st.markdown('<div class="card"><h3>Score Analysis</h3></div>', unsafe_allow_html=True)
    score_text = {
        "unacc":"โมเดลชี้ว่ารถนี้ไม่เหมาะสม แนะนำปรับราคาซื้อ/ค่าบำรุงรักษาและเพิ่มความปลอดภัย",
        "acc":"รถรับได้ แต่ยังสามารถปรับปรุงบางอย่างให้เหมาะสมมากขึ้น",
        "good":"รถดีแล้ว สามารถใช้เป็นตัวเลือกได้",
        "vgood":"รถดีมาก เหมาะสมที่สุดสำหรับความต้องการของคุณ"
    }
    st.markdown(f"- {score_text[pred_class]}")
    
    # ---------- Personalized Suggestions ----------
    st.markdown('<div class="card"><h3>Personalized Suggestions</h3></div>', unsafe_allow_html=True)
    suggestions = {
        "unacc":["ลดราคาซื้อหรือค่าบำรุงรักษา","เพิ่มความปลอดภัยของรถ","ตรวจสอบจำนวนผู้โดยสารและประตู"],
        "acc":["ปรับขนาดกระโปรงให้เหมาะสม","ตรวจสอบความปลอดภัยเพิ่มเติม","ปรับราคาซื้อ/ค่าบำรุงรักษาเล็กน้อย"],
        "good":["เหมาะสำหรับครอบครัวหรือใช้งานทั่วไป","สามารถปรับปรุงขนาดกระโปรงเล็กน้อย","ตรวจสอบความปลอดภัยเพิ่มเติม"],
        "vgood":["เหมาะสมที่สุดสำหรับทุกการใช้งาน","ประสิทธิภาพสูงสุด","ความปลอดภัยครบถ้วน"]
    }
    for s in suggestions[pred_class]:
        st.markdown(f"- {s}")
    
    # ---------- AI Suggestion ----------
    st.markdown('<div class="card"><h3>AI Suggestion</h3></div>', unsafe_allow_html=True)
    ai_texts = {
        "unacc":["ลองปรับ input หลายแบบเพื่อตรวจสอบรถที่เหมาะสมที่สุด","ใช้ model แนะนำการปรับปรุงรถ"],
        "acc":["คุณสามารถปรับค่าบางอย่างเพื่อเพิ่มคะแนน","AI แนะนำให้ลองเปรียบเทียบหลาย configuration"],
        "good":["รถนี้เหมาะสม คุณสามารถลองเปรียบเทียบกับตัวเลือกอื่นๆ","AI แนะนำให้ตรวจสอบ input อื่นๆเพื่อหาที่ดีที่สุด"],
        "vgood":["รถนี้เหมาะที่สุดแล้ว","สามารถทดลอง input อื่นๆเพื่อยืนยันความเหมาะสม","AI แนะนำการปรับค่าขั้นสูงเพื่อเพิ่มประสิทธิภาพ"]
    }
    for t in ai_texts[pred_class]:
        st.markdown(f"- {t}")
    
    # ---------- Graphs ----------
    st.markdown('<div class="card"><h3>Graphs</h3></div>', unsafe_allow_html=True)
    
    # Distribution of input features
    input_df = pd.DataFrame([{
        "buying": buying, "maint": maint, "doors": doors,
        "persons": persons, "lug_boot": lug_boot, "safety": safety
    }])
    for col in input_df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, palette="Set2", order=df[col].unique(), ax=ax)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)
    
    # Class distribution
    fig2, ax2 = plt.subplots()
    sns.countplot(x="class", data=df, palette="Set1", order=df["class"].unique(), ax=ax2)
    plt.title("Class Distribution")
    st.pyplot(fig2)

# ---------- About Project ----------
st.divider()
st.markdown('<div class="card"><h3>About Project</h3></div>', unsafe_allow_html=True)
st.markdown("""
- Features: Buying price, Maintenance, Doors, Persons, Luggage Boot, Safety  
- Target: Car quality (Unacceptable, Acceptable, Good, Very Good)  
- Model: Pre-trained RandomForest with balanced class  
- ใช้ตัวเลือกด้านบนเพื่อทำนายคุณภาพรถ
""", unsafe_allow_html=True)
