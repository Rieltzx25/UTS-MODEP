"""
Part 3 - Monolithic Streamlit App
Load model .pkl, render form, prediksi placement + salary.
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_models():
    clf = joblib.load("models/placement_classifier.pkl")
    reg = joblib.load("models/salary_regressor.pkl")
    return clf, reg


def build_input_df(form_values):
    row = dict(form_values)
    row["academic_score"] = (row["cgpa"]*10 + row["tenth_percentage"] + row["twelfth_percentage"]) / 3
    row["skill_avg"]      = (row["coding_skill_rating"]
                             + row["communication_skill_rating"]
                             + row["aptitude_skill_rating"]) / 3
    return pd.DataFrame([row])


# -- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Placement Predictor")
    st.markdown(
        "Aplikasi monolithic untuk memprediksi **status penempatan kerja** "
        "dan **estimasi gaji** siswa berdasarkan profil akademik, skill, dan gaya hidup."
    )
    st.markdown("---")
    st.subheader("Model Info")
    st.markdown(
        "- **Classifier:** Random Forest  \n"
        "- **Regressor:** Ridge Regression  \n"
        "- **Training:** via MLflow Pipeline"
    )
    st.markdown("---")
    st.caption("UTS Model Deployment 2026 — NIM 2802428103")
    st.caption("[GitHub Repo](https://github.com/Rieltzx25/UTS-MODEP)")


# -- Main -----------------------------------------------------------------
st.title("Placement & Salary Predictor")
st.markdown(
    "Isi form di bawah, lalu klik **Predict**. Regresi gaji hanya dijalankan jika "
    "status prediksinya *Placed*."
)

clf, reg = load_models()

with st.form("prediction_form"):
    st.subheader("Demografi & Akademik")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "CE", "ME"])
    with c2:
        cgpa = st.slider("CGPA", 5.0, 10.0, 8.3, 0.1)
        tenth = st.slider("10th %", 50.0, 100.0, 74.7, 0.1)
    with c3:
        twelfth = st.slider("12th %", 50.0, 100.0, 74.8, 0.1)
        backlogs = st.number_input("Backlogs", 0, 10, 0)

    st.subheader("Skills & Pengalaman")
    c1, c2, c3 = st.columns(3)
    with c1:
        coding  = st.slider("Coding skill (1–5)", 1, 5, 4)
        comm    = st.slider("Communication skill (1–5)", 1, 5, 3)
        apt     = st.slider("Aptitude skill (1–5)", 1, 5, 4)
    with c2:
        projects = st.number_input("Projects completed", 0, 20, 6)
        interns  = st.number_input("Internships completed", 0, 10, 2)
        hacks    = st.number_input("Hackathons participated", 0, 10, 4)
    with c3:
        certs    = st.number_input("Certifications", 0, 20, 3)
        attend   = st.slider("Attendance %", 40.0, 100.0, 72.0, 0.1)
        study    = st.slider("Study hours/day", 0.0, 12.0, 4.0, 0.5)

    st.subheader("Gaya Hidup & Latar Belakang")
    c1, c2, c3 = st.columns(3)
    with c1:
        sleep  = st.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
        stress = st.slider("Stress level (1–10)", 1, 10, 6)
    with c2:
        ptj    = st.selectbox("Part-time job", ["No", "Yes"])
        income = st.selectbox("Family income", ["Low", "Medium", "High"])
        tier   = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    with c3:
        net    = st.selectbox("Internet access", ["Yes", "No"])
        extra  = st.selectbox("Extracurricular", ["Low", "Medium", "High", "Unknown"])

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)


if submitted:
    form = {
        "gender": gender, "branch": branch,
        "cgpa": cgpa, "tenth_percentage": tenth, "twelfth_percentage": twelfth,
        "backlogs": backlogs, "study_hours_per_day": study,
        "attendance_percentage": attend,
        "projects_completed": projects, "internships_completed": interns,
        "coding_skill_rating": coding, "communication_skill_rating": comm,
        "aptitude_skill_rating": apt,
        "hackathons_participated": hacks, "certifications_count": certs,
        "sleep_hours": sleep, "stress_level": stress,
        "part_time_job": ptj, "family_income_level": income,
        "city_tier": tier, "internet_access": net,
        "extracurricular_involvement": extra,
    }
    X = build_input_df(form)

    pred  = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns([1, 1])

    with col1:
        if pred == "Placed":
            st.success(f"### ✅ {pred}")
        else:
            st.error(f"### ❌ {pred}")

        st.markdown("**Confidence:**")
        fig, ax = plt.subplots(figsize=(5, 2))
        colors = ["#DD8452" if c == "Not Placed" else "#4C72B0" for c in classes]
        ax.barh(classes, proba, color=colors)
        for i, p in enumerate(proba):
            ax.text(p + 0.02, i, f"{p:.2%}", va="center")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        if pred == "Placed":
            salary = float(reg.predict(X)[0])
            salary = max(0.0, salary)

            st.markdown("**Estimasi Gaji**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.barh([0], [salary], color="#55A868", height=0.4)
            ax.barh([0], [20], color="#E0E0E0", height=0.4, alpha=0.3, zorder=0)
            ax.text(salary + 0.3, 0, f"{salary:.2f} LPA",
                    va="center", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 22)
            ax.set_yticks([])
            ax.set_xlabel("Salary (LPA)")
            ax.set_title(f"Predicted Salary")
            plt.tight_layout()
            st.pyplot(fig)

            st.info(
                "💡 Rentang gaji di dataset training: **5.2 – 20.0 LPA** "
                "(median ~16.4 LPA untuk siswa Placed)."
            )
        else:
            st.warning(
                "🔒 Regresi gaji tidak dijalankan karena status prediksinya *Not Placed*. "
                "Ini by design: regressor hanya dilatih di subset Placed supaya tidak "
                "terkontaminasi structural zero."
            )

    with st.expander("🔍 Detail Input"):
        st.dataframe(X.T.rename(columns={0: "value"}))
