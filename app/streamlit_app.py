"""Streamlit demo: score an applicant, explain the decision, suggest recourse.

Run:  streamlit run app/streamlit_app.py
Requires a trained artifact (run `python -m src.train` first).
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import LoanScorer
from src import recourse

st.set_page_config(page_title="Loan Decision Engine", page_icon="🏦", layout="centered")
st.title("🏦 Loan Decision Engine")
st.caption("Risk score · explainable reasons · actionable recourse")


@st.cache_resource
def get_scorer():
    return LoanScorer()


try:
    scorer = get_scorer()
except Exception as e:
    st.error(f"No trained model found — run `python -m src.train` first.\n\n{e}")
    st.stop()

st.sidebar.header("Applicant details")
row = {
    "gender": st.sidebar.selectbox("Gender", ["female", "male", "other"]),
    "is_employed": st.sidebar.selectbox("Employed", [True, False]),
    "job": st.sidebar.selectbox("Job", ["Teacher", "Nurse", "Doctor", "Engineer",
                                         "Accountant", "Lawyer", "Data Scientist", "Data Analyst",
                                         "Software Developer"]),
    "location": st.sidebar.text_input("Location", "Harare"),
    "marital_status": st.sidebar.selectbox("Marital status", ["single", "married", "divorced"]),
    "loan_amount": st.sidebar.number_input("Loan amount", 1000.0, 300000.0, 30000.0, 1000.0),
    "outstanding_balance": st.sidebar.number_input("Outstanding balance", 0.0, 200000.0, 35000.0, 1000.0),
    "interest_rate": st.sidebar.slider("Interest rate", 0.0, 0.5, 0.2, 0.01),
    "number_of_defaults": st.sidebar.slider("Prior defaults", 0, 5, 0),
    "remaining term": st.sidebar.number_input("Remaining term (months)", 1, 120, 48),
    "salary": st.sidebar.number_input("Salary", 250.0, 20000.0, 3000.0, 50.0),
    "age": st.sidebar.slider("Age", 18, 70, 40),
    "disbursemet_date": "2023 06 15",
}
applicant = pd.DataFrame([row])

if st.sidebar.button("Score applicant", type="primary"):
    scored = scorer.predict(applicant).iloc[0]
    prob, decision = scored["default_probability"], scored["decision"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{prob:.1%}")
    c2.metric("Decision", decision)
    c3.metric("Risk band", str(scored["risk_band"]))

    st.subheader("Why this decision")
    st.text(scorer.explain_text(applicant))

    if decision == "DECLINE":
        st.subheader("How to get approved (recourse)")
        rec = recourse.find_recourse(scorer.pipeline, applicant, scorer.reference, scorer.threshold)
        st.text(recourse.recourse_text(rec))

st.sidebar.caption(f"Model: {scorer.meta.get('model_name', '?')} · "
                   f"approve below {scorer.threshold:.0%}")
