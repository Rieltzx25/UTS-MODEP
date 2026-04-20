"""
Part 4 - Streamlit Frontend (client)
Form input, kirim request ke FastAPI backend, tampilkan hasilnya.
Jalankan: streamlit run client.py
"""
import requests
import streamlit as st


st.set_page_config(page_title="Placement Predictor (API client)", page_icon="🎓", layout="wide")


with st.sidebar:
    st.title("🎓 Placement Predictor")
    st.caption("Frontend Streamlit — decoupled via FastAPI")

    st.markdown("---")
    st.subheader("API Endpoint")
    api_url = st.text_input("Base URL", value="http://localhost:8000")

    if st.button("Ping API"):
        try:
            r = requests.get(f"{api_url}/health", timeout=3)
            if r.ok:
                st.success(f"API reachable — {r.json()}")
            else:
                st.error(f"API error {r.status_code}")
        except Exception as e:
            st.error(f"Gagal connect: {e}")

    st.markdown("---")
    st.caption("UTS Model Deployment 2026 — NIM 2802428103")
    st.caption("[GitHub Repo](https://github.com/Rieltzx25/UTS-MODEP)")


st.title("Placement & Salary Predictor — API Client")
st.markdown(
    "Form ini **tidak load model langsung**. Saat kamu klik Predict, request dikirim "
    "ke backend FastAPI, lalu hasilnya ditampilkan di sini."
)

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

    submitted = st.form_submit_button("🔮 Predict via API", use_container_width=True)


if submitted:
    payload = {
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

    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    except Exception as e:
        st.error(f"Gagal connect ke API: {e}")
        st.stop()

    if not r.ok:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()

    result = r.json()

    st.markdown("---")
    st.subheader("Hasil Prediksi")

    c1, c2, c3 = st.columns(3)
    with c1:
        if result["placement_status"] == "Placed":
            st.success(f"**Status:** {result['placement_status']}")
        else:
            st.error(f"**Status:** {result['placement_status']}")
    with c2:
        st.metric("Probability (Placed)", f"{result['probability_placed']:.2%}")
    with c3:
        if result["salary_lpa"] is not None:
            st.metric("Estimasi Gaji", f"{result['salary_lpa']:.2f} LPA")
        else:
            st.metric("Estimasi Gaji", "—")

    st.caption(result["note"])

    with st.expander("📡 Request & Response detail"):
        st.markdown("**Request Payload:**")
        st.json(payload)
        st.markdown("**Response:**")
        st.json(result)
