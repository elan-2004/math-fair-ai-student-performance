import streamlit as st
import numpy as np
import pickle
import sklearn  # ensure scikit-learn loads correctly

# ---------- LOAD THE TRAINED MODEL ----------
with open("student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pass_probability(study_hours, sleep_hours, attendance):
    X = np.array([[study_hours, sleep_hours, attendance]])
    prob = model.predict_proba(X)[0, 1]
    return float(prob)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Learns from Student Performance",
    page_icon="📊",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
CUSTOM_CSS = """
<style>
.main {
    background-color: #f7f5f5;
}
h1, h2, h3 {
    color: #2b2c6f;
    font-family: 'Helvetica', sans-serif;
}
.card {
    padding: 1.2rem 1.4rem;
    background-color: #ffffff;
    border-radius: 0.8rem;
    border: 1px solid #e0e0e0;
    box-shadow: 0 0 10px rgba(0,0,0,0.02);
    font-size: 0.95rem;
}
.subtitle {
    color: #5f61e6;
    font-weight: 700;
    font-size: 1.0rem;
    margin-bottom: 0.4rem;
}
.small-label {
    font-size: 0.85rem;
    color: #666666;
}
.prob-pill {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    background-color: #f8f0ff;
    border: 1px dashed #7a4bff;
    font-weight: 600;
    color: #4e3299;
}
.footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #888888;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- LAYOUT ----------
st.title("AI Learns from Student Performance")
st.markdown("### Probability & Statistics – Math Fair Project")
st.markdown("**Instructor:** Dr. Najma Saleem  &nbsp;&nbsp; • &nbsp;&nbsp; **Course:** Probability & Statistics_202")

st.write("---")

col1, col2, col3 = st.columns([1.1, 1, 1.1])

# ---- LEFT ----
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Introduction")
    st.write(
        "This project shows how **AI can learn from probability**. "
        "We use student-style data (study hours, sleep, and attendance) "
        "to train a model that predicts whether a student is likely to **pass or fail**. "
        "It connects probability, data analysis, and AI decision-making."
    )

    st.markdown("### Problem Statement")
    st.write(
        "Students differ in their study habits, sleep hours, and class attendance. "
        "These differences affect academic performance. "
        "Our question: **Can an AI model learn these patterns and predict pass/fail outcomes?**"
    )

    st.markdown('<span class="subtitle">Dataset & Features</span>', unsafe_allow_html=True)
    st.write("- Study Hours per day\n"
             "- Sleep Hours per night\n"
             "- Attendance (%)\n"
             "- Final Result (Pass/Fail)")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- MIDDLE ----
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### How the AI Learns")
    st.write(
        "- The AI looks for **patterns** in the data.\n"
        "- It learns which combinations of study hours, sleep, and attendance usually lead to **passing**.\n"
        "- It then uses **probability** to estimate the chance that a new student will pass."
    )

    st.markdown('<p class="prob-pill">Importance of Probability</p>', unsafe_allow_html=True)
    st.write(
        "The model outputs a **probability** between 0 and 1. "
        "This shows how statistics and probability power AI."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- RIGHT ----
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Try the AI Demo")

    study = st.number_input("Study Hours", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80, step=5)

    if st.button("Predict"):
        prob = predict_pass_probability(study, sleep, attendance)
        label = "Pass" if prob >= 0.5 else "Fail"
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{prob:.2f}**")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>AI Learning from Student Performance</div>", unsafe_allow_html=True)
