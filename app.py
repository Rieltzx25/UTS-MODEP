"""
Part 3 - Monolithic Streamlit App
Load model .pkl, render form, prediksi placement + salary.
"""
import joblib
import pandas as pd
import streamlit as st


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
        "dan **estimasi gaji** siswa berdasarkan profil akademik dan skill."
    )
    st.markdown("---")
    st.subheader("Model Info")
    st.markdown(
        "- **Classifier:** Random Forest  \n"
        "- **Regressor:** Ridge Regression  \n"
        "- **Training:** via MLflow Pipeline  \n"
        "- **Fitur input:** 15 (hasil seleksi permutation importance)"
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
    st.subheader("Akademik")
    c1, c2, c3 = st.columns(3)
    with c1:
        cgpa   = st.slider("CGPA", 5.0, 10.0, 8.3, 0.1)
        branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "CE", "ME"])
    with c2:
        tenth    = st.slider("10th %", 50.0, 100.0, 74.7, 0.1)
        backlogs = st.number_input("Backlogs", 0, 10, 0)
    with c3:
        twelfth = st.slider("12th %", 50.0, 100.0, 74.8, 0.1)

    st.subheader("Skills & Pengalaman")
    c1, c2, c3 = st.columns(3)
    with c1:
        coding = st.slider("Coding skill (1–5)", 1, 5, 4)
        comm   = st.slider("Communication skill (1–5)", 1, 5, 3)
        apt    = st.slider("Aptitude skill (1–5)", 1, 5, 4)
    with c2:
        projects = st.number_input("Projects completed", 0, 20, 6)
        interns  = st.number_input("Internships completed", 0, 10, 2)
    with c3:
        hacks = st.number_input("Hackathons participated", 0, 10, 4)
        certs = st.number_input("Certifications", 0, 20, 3)

    st.subheader("Lainnya")
    c1, c2, c3 = st.columns(3)
    with c1:
        stress = st.slider("Stress level (1–10)", 1, 10, 6)
    with c2:
        tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    with c3:
        extra = st.selectbox("Extracurricular", ["Low", "Medium", "High", "Unknown"])

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)


if submitted:
    form = {
        "cgpa": cgpa, "tenth_percentage": tenth, "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "projects_completed": projects, "internships_completed": interns,
        "coding_skill_rating": coding, "communication_skill_rating": comm,
        "aptitude_skill_rating": apt,
        "hackathons_participated": hacks, "certifications_count": certs,
        "stress_level": stress,
        "branch": branch, "city_tier": tier,
        "extracurricular_involvement": extra,
    }
    X = build_input_df(form)

    pred  = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    placed_idx = classes.index("Placed")
    prob_placed = float(proba[placed_idx])

    col1, col2, col3 = st.columns(3)

    with col1:
        if pred == "Placed":
            st.success(f"**Status:** {pred}")
        else:
            st.error(f"**Status:** {pred}")

    with col2:
        st.metric("Probability (Placed)", f"{prob_placed:.2%}")

    with col3:
        if pred == "Placed":
            salary = max(0.0, float(reg.predict(X)[0]))
            st.metric("Estimasi Gaji", f"{salary:.2f} LPA")
        else:
            st.metric("Estimasi Gaji", "—")

    if pred == "Placed":
        st.caption(
            "Rentang gaji di dataset training: 5.2 – 20.0 LPA (median ~16.4 LPA)."
        )
    else:
        st.caption(
            "Regresi gaji tidak dijalankan karena status prediksinya Not Placed "
            "(regressor hanya dilatih di subset Placed)."
        )

    with st.expander("🔍 Detail Input"):
        st.dataframe(X.T.rename(columns={0: "value"}))
