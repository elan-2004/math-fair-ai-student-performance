import streamlit as st
import pickle
import numpy as np

# =========================
# Load Model
# =========================
with open("student_performance_model.pkl", "rb") as file:
    model = pickle.load(file)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="✨",
    layout="wide"
)

# =========================
# Custom CSS (Purple Theme)
# =========================
st.markdown("""
<style>
/* Page background */
body {
    background: linear-gradient(180deg, #faf6ff, #ffffff);
}

/* Main header */
.main-title {
    font-size: 44px;
    font-weight: 900;
    color: #5b2aff;
    text-shadow: 1px 1px 3px #cfc4ff;
    margin-bottom: 0.3rem;
}

/* Section titles under the line */
.section-label {
    font-size: 16px;
    font-weight: 800;
    color: #6f3cff;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

/* Card styling around text/content */
.card {
    background: #ffffff;
    padding: 20px 25px;
    border-radius: 18px;
    box-shadow: 0px 0px 20px rgba(180, 150, 255, 0.18);
    margin-bottom: 30px;
}

/* Purple pill */
.prob-pill {
    background: #f3e8ff;
    color: #5a00ff;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
    margin-top: 14px;
}

/* Predict button */
.stButton>button {
    background: linear-gradient(90deg, #7f3dff, #ff6bdf);
    color: white;
    border-radius: 25px;
    font-size: 18px;
    padding: 10px 26px;
    border: none;
    transition: 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Result boxes */
.result-pass {
    background-color: #e6ffe9;
    padding: 15px;
    border-radius: 10px;
    color: #0f8a00;
    font-size: 20px;
    font-weight: 700;
}
.result-fail {
    background-color: #ffe6e6;
    padding: 15px;
    border-radius: 10px;
    color: #cc0000;
    font-size: 20px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<h1 class='main-title'>Student Performance AI</h1>", unsafe_allow_html=True)
st.write("**Probability & Statistics – Math Fair Project**")
st.write("Instructor: Dr. Najma Saleem • Course: Probability & Statistics_202")

st.write("---")

# =========================
# LAYOUT: 3 COLUMNS
# =========================
col1, col2, col3 = st.columns([1.1, 1, 1.1])

# ---- OVERVIEW ----
with col1:
    st.markdown('<p class="section-label">OVERVIEW</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### What this AI does")
    st.write(
        "- Uses **study hours**, **sleep hours**, and **attendance**.\n"
        "- Predicts if a student will **Pass** or **Fail**.\n"
        "- Outputs a **probability** between 0 and 1 for passing."
    )
    st.write("Shows how simple data + probability can power an AI model.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- INPUTS ----
with col2:
    st.markdown('<p class="section-label">INPUTS</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Features used by the model")
    st.write(
        "- **Study Hours per day** (average)\n"
        "- **Sleep Hours per night** (average)\n"
        "- **Attendance (%)**"
    )

    st.markdown(
        '<span class="prob-pill">Model output: probability of passing</span>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ---- DEMO ----
with col3:
    st.markdown('<p class="section-label">DEMO</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Try the AI")

    study = st.number_input(
        "Study hours per day",
        min_value=0.0,
        max_value=12.0,
        step=1.0,
        value=3.0,
    )
    sleep = st.number_input(
        "Sleep hours per night",
        min_value=0.0,
        max_value=12.0,
        step=1.0,
        value=6.0,
    )
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        step=1,
        value=80,
    )

    if st.button("Predict"):
        X = np.array([[study, sleep, attendance]])
        prob_pass = model.predict_proba(X)[0][1]
        prediction = "Pass" if prob_pass >= 0.5 else "Fail"

        if prediction == "Pass":
            st.markdown("<div class='result-pass'>Prediction: Pass</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-fail'>Prediction: Fail</div>", unsafe_allow_html=True)

        st.write(f"**Probability of passing: {prob_pass:.2f}**")

    st.markdown("</div>", unsafe_allow_html=True)
