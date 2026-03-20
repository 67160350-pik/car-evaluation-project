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

# ---------- CSS Gradient Background + Floating Cars ----------
st.markdown("""
<style>
.main .block-container {
    background: linear-gradient(135deg, #6a0dad, #8a2be2);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: white;
}
@keyframes gradientBG {0% {background-position:0% 50%} 50% {background-position:100% 50%} 100% {background-position:0% 50%}}
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
    transition: 0.3s;
    font-size: 16px;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #9c27b0, #7b1fa2);
    transform: scale(1.05);
}
.floating {position: absolute; width: 40px; opacity: 0.7; animation: float 6s ease-in-out infinite;}
@keyframes float {0% {transform: translateY(0px) translateX(0px);} 50% {transform: translateY(-20px) translateX(10px);} 100% {transform: translateY(0px) translateX(0px);}}
""" + "\n".join([f"#car{i} {{top:{np.random.randint(0,400)}px; left:{np.random.randint(0,90)}%; animation-delay:{i}s;}}" for i in range(1,16)]) + """
</style>
""" + "\n".join([f'<img src="https://cdn-icons-png.flaticon.com/512/743/743997.png" class="floating" id="car{i}">' for i in range(1,16)]), unsafe_allow_html=True)

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
mapping = {"low":0,"med":1,"high":2,"vhigh":3,"2":0,"3":1,"4":2,"5more":3,"more":2,"small":0,"big":2}

if st.button("🚀 Predict", use_container_width=True):
    data_input = np.array([[ 
        mapping[buying], mapping[maint], mapping[doors],
        mapping[persons], mapping[lug_boot], mapping[safety]
    ]])

    pred = model.predict(data_input)[0]

    # Result mapping - elegant color palette
    class_map = {0:"Unacceptable",1:"Acceptable",2:"Good",3:"Very Good"}
    color_map = {0:"#D32F2F",1:"#FBC02D",2:"#388E3C",3:"#1976D2"}

    # Prediction Result
    st.markdown(f"""
    <div class="card" style="text-align:center; box-shadow:0 8px 20px {color_map[pred]}; border:2px solid {color_map[pred]}">
        <h2>Prediction Result</h2>
        <h1 style="color:{color_map[pred]}; font-size:2em;">{class_map[pred]}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Score Analysis
    st.markdown('<div class="card"><h3>Score Analysis</h3></div>', unsafe_allow_html=True)
    st.markdown("""
    - The model is trained on the Car Evaluation Dataset  
    - Approximate Accuracy: 85-95%  
    - Try different inputs to see how the prediction changes
    """, unsafe_allow_html=True)

    # Personalized Suggestions
    st.markdown('<div class="card"><h3>Personalized Suggestions</h3></div>', unsafe_allow_html=True)
    if pred == 0:
        st.markdown("- ❌ Consider adjusting buying price, maintenance, or safety to improve the car rating.")
    elif pred == 1:
        st.markdown("- ⚠️ Acceptable, but some improvements in luggage or safety may increase rating.")
    elif pred == 2:
        st.markdown("- 👍 Good choice, suitable for most needs.")
    else:
        st.markdown("- 🌟 Excellent! Optimal configuration for your requirements.")

    # AI Suggestion
    st.markdown('<div class="card"><h3>AI Suggestion</h3></div>', unsafe_allow_html=True)
    st.markdown("- Try testing multiple input combinations to find the optimal car configuration and compare results.")

# ---------- About Project ----------
st.divider()
st.markdown('<div class="card"><h3>About Project</h3></div>', unsafe_allow_html=True)
st.markdown("""
- Features: Buying price, Maintenance, Doors, Persons, Luggage Boot, Safety  
- Target: Car quality (Unacceptable, Acceptable, Good, Very Good)  
- Model: Pre-trained DecisionTree / RandomForest  
- Input values from selectors above to predict car quality
""", unsafe_allow_html=True)
