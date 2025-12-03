import streamlit as st
import numpy as np
import pickle
import sklearn  # needed so unpickling the model works


# --------- LOAD TRAINED MODEL ----------
with open("student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_pass_probability(study_hours, sleep_hours, attendance):
    """Return probability of passing (0–1) and class label."""
    X = np.array([[study_hours, sleep_hours, attendance]])
    prob_pass = model.predict_proba(X)[0, 1]
    label = "Pass" if prob_pass >= 0.5 else "Fail"
    return float(prob_pass), label


# --------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Learns from Student Performance",
    page_icon="📊",
    layout="wide",
)

# --------- SIMPLE STYLING ----------
CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #f8f6ff 0%, #fdfcf6 100%);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

h1 {
    font-weight: 800;
    letter-spacing: 0.03em;
    color: #252554;
}

h2, h3 {
    color: #33345c;
}

.top-subtitle {
    font-size: 0.95rem;
    color: #55557a;
}

/* Card style */
.card {
    padding: 1.2rem 1.4rem;
    background-color: #ffffff;
    border-radius: 1rem;
    border: 1px solid #e2ddff;
    box-shadow: 0 8px 18px rgba(68, 58, 150, 0.06);
    font-size: 0.95rem;
}

/* Section labels */
.section-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b5bf3;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

/* Probability pill */
.prob-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background-color: #f2ebff;
    border: 1px dashed #7a4bff;
    font-weight: 600;
    color: #4e3299;
    font-size: 0.8rem;
    margin-top: 0.3rem;
}

/* Inputs */
.stNumberInput > label {
    font-size: 0.9rem;
    color: #3b3b5f;
    font-weight: 600;
}

/* Predict button */
.stButton > button {
    width: 100%;
    border-radius: 999px;
    background: linear-gradient(135deg, #6f5cf5, #ff9fd8);
    color: white;
    border: none;
    padding: 0.5rem 0.9rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 6px 15px rgba(111, 92, 245, 0.35);
}
.stButton > button:hover {
    filter: brightness(1.05);
}

/* Footer */
.footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #9b99b5;
    text-align: center;
}

/* Mobile spacing */
@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------- PAGE CONTENT (SHORT VERSION) ----------
st.title("AI Learns from Student Performance")
st.markdown(
    '<p class="top-subtitle">'
    '<b>Probability &amp; Statistics – Math Fair Project</b><br>'
    'Instructor: Dr. Najma Saleem &nbsp; • &nbsp; '
    'Course: Probability &amp; Statistics_202'
    '</p>',
    unsafe_allow_html=True,
)

st.write("---")

col1, col2, col3 = st.columns([1.1, 1, 1.1])

# ---- LEFT: WHAT THE AI DOES ----
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Overview</p>', unsafe_allow_html=True)
    st.markdown("### What this AI does")
    st.write(
        "- Uses **study hours**, **sleep hours**, and **attendance**.\n"
        "- Outputs a prediction: **Pass** or **Fail**.\n"
        "- Shows a **probability** between 0 and 1 for passing."
    )
    st.write(
        "The goal is to show how probability and simple data can power an AI model."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- MIDDLE: FEATURES (VERY SHORT) ----
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Inputs</p>', unsafe_allow_html=True)
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

# ---- RIGHT: DEMO ----
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Demo</p>', unsafe_allow_html=True)
    st.markdown("### Try the AI")

    study = st.number_input(
        "Study hours per day", min_value=0.0, max_value=12.0, value=3.0, step=0.5
    )
    sleep = st.number_input(
        "Sleep hours per night", min_value=0.0, max_value=12.0, value=6.0, step=0.5
    )
    attendance = st.number_input(
        "Attendance (%)", min_value=0, max_value=100, value=80, step=5
    )

    if st.button("Predict"):
        prob, label = predict_pass_probability(study, sleep, attendance)
        st.success(f"Prediction: **{label}**")
        st.write(f"Probability of passing: **{prob:.2f}**")
    else:
        st.info("Set the values and click **Predict** to see the result.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown(
    "<div class='footer'>Full explanation of the project is on our poster at the Math Fair.</div>",
    unsafe_allow_html=True,
)
