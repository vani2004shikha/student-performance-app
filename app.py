# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import hashlib
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

# PDF library
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------------- LOAD LOTTIE ----------------------
def load_lottie(url: str):
    """Load a Lottie animation from a URL; return dict or None."""
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# Mixed theme animations (if a URL fails, the value will be None and we won't call st_lottie on it)
LOTTIE_HOME     = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
LOTTIE_STUDY    = load_lottie("https://assets1.lottiefiles.com/packages/lf20_tutvdkg0.json")
LOTTIE_REMedy   = load_lottie("https://assets7.lottiefiles.com/private_files/lf30_tamcv1wp.json")
LOTTIE_SUCCESS  = load_lottie("https://assets4.lottiefiles.com/packages/lf20_u4yrau.json")
LOTTIE_LOGIN    = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
LOTTIE_DASH     = load_lottie("https://assets1.lottiefiles.com/packages/lf20_i8d0whbs.json")

# ---------------------- PAGE DESIGN / STYLES ----------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered", page_icon="🎓")

# UPDATED: All text is BLACK (#000000)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, #fff9c4 0%, #fffde7 100%);
  background-attachment: fixed;
}}

[data-testid="stHeader"] {{
  background: rgba(0,0,0,0);
}}

.block-container {{
  backdrop-filter: blur(3px);
  background: rgba(224,247,250,0.9);
  padding: 28px;
  border-radius: 14px;
  color: #000000;
}}

h1, h2, h3, p, label, .css-1d391kg, .css-1d391kg * {{
  color: #000000 !important;
}}

.stButton>button {{
  background: linear-gradient(90deg,#ffd54f,#ffecb3);
  color: #000000;
  border-radius: 8px;
}}

[data-testid="stSidebar"] {{
  background: rgba(224,247,250,0.95);
  color: #000000;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown('<div class="block-container">', unsafe_allow_html=True)

# ---------------------- UTILITIES ----------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.csv")
TEACHERS_FILE = os.path.join(DATA_DIR, "teachers.csv")
MODEL_FILE = "model.pkl"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def ensure_default_teacher():
    if not os.path.exists(TEACHERS_FILE):
        df = pd.DataFrame([{"username": "teacher", "password_hash": hash_password("password")}])
        df.to_csv(TEACHERS_FILE, index=False)

ensure_default_teacher()

def load_teachers():
    try:
        return pd.read_csv(TEACHERS_FILE)
    except Exception:
        return pd.DataFrame(columns=["username", "password_hash"])

def authenticate(username, password):
    df = load_teachers()
    if username in df['username'].values:
        stored = df.loc[df['username'] == username, 'password_hash'].values[0]
        return stored == hash_password(password)
    return False

def register_teacher(username, password):
    df = load_teachers()
    if username in df['username'].values:
        return False, "Username already exists"
    new = {"username": username, "password_hash": hash_password(password)}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(TEACHERS_FILE, index=False)
    return True, "Registered"

def save_prediction(record: dict):
    df_new = pd.DataFrame([record])
    if os.path.exists(PREDICTIONS_FILE):
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            df = pd.concat([df, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(PREDICTIONS_FILE, index=False)

@st.cache_resource
def load_model(path=MODEL_FILE):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    class DummyModel:
        def predict(self, X):
            weights = np.array([4,0.2,0.5,1,2])
            return np.clip(X.dot(weights), 0, 100)
    return DummyModel()

model = load_model()

# ---------------------- NAVIGATION ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Prediction", "Teacher Login", "Dashboard"])

# ---------------------- HOME PAGE ----------------------
if page == "Home":
    st.title("🎓 Student Performance Predictor")
    st.subheader("Make learning visible — Predict. Improve. Grow.")

    if LOTTIE_HOME:
        st_lottie(LOTTIE_HOME, height=300)

    st.markdown("""
    This app helps teachers and students understand performance trends,  
    improve study habits, and track progress.
    """)

# ---------------------- PREDICTION PAGE ----------------------
elif page == "Prediction":
    st.header("📚 Predict Student Performance")
    if LOTTIE_STUDY:
        st_lottie(LOTTIE_STUDY, height=250)

    with st.form("prediction_form"):
        name = st.text_input("Student Name")
        class_name = st.text_input("Class")

        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 3.0, step=0.5)
            prev_grade = st.number_input("Previous Grade (%)", 0, 100, 70)
            participation = st.selectbox("Class Participation (0-5)", [0,1,2,3,4,5])
        with col2:
            attendance = st.number_input("Attendance (%)", 0, 100, 75)
            sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, step=0.5)
            show_save = st.checkbox("Save prediction to teacher account")

        submit = st.form_submit_button("Predict Performance 🎯")

    if submit:
        if not name or not class_name:
            st.error("Please enter student name and class.")
        else:
            X = np.array([[study_hours, attendance, prev_grade, sleep_hours, participation]])
            try:
                predicted_score = float(model.predict(X)[0])
            except Exception as e:
                st.error("Model prediction failed: " + str(e))
                predicted_score = 0.0
            predicted_score = max(0, min(100, predicted_score))

            st.success(f"{name} (Class {class_name}) - Predicted Score: {predicted_score:.2f}")

            if predicted_score > 70:
                if LOTTIE_SUCCESS:
                    st_lottie(LOTTIE_SUCCESS, height=200)
                st.balloons()

            # -------------- Visualization --------------
            st.markdown("### 📊 Visual Analysis")
            feature_names = ["Study Hours", "Attendance", "Prev Grade", "Sleep Hours", "Participation"]
            weights = np.array([4,0.2,0.5,1,2])
            vals = np.array([study_hours, attendance, prev_grade, sleep_hours, participation])
            importance = weights * vals
            df_imp = pd.DataFrame({"feature": feature_names, "importance": importance})

            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(df_imp['feature'], df_imp['importance'])
            plt.xticks(rotation=20)
            st.pyplot(fig)

            # Low-score remedial Lottie
            if predicted_score < 50:
                st.warning("⚠️ Low predicted score — Remedial measures recommended:")
                if LOTTIE_REMedy:
                    st_lottie(LOTTIE_REMedy, height=200)
                st.markdown("""
                **Recommendations:**
                - Increase study hours gradually  
                - Improve attendance & participation  
                - Focus on weak areas  
                - Follow a routine  
                - Ask teacher for extra support  
                """)

            # ----------------- SAVE TO TEACHER -----------------
            if show_save:
                if st.session_state.get("teacher_auth", False):
                    teacher = st.session_state.get("teacher_username", "unknown")
                    save_prediction({
                        "timestamp": datetime.now().isoformat(),
                        "teacher": teacher,
                        "student_name": name,
                        "class": class_name,
                        "study_hours": study_hours,
                        "attendance": attendance,
                        "prev_grade": prev_grade,
                        "sleep_hours": sleep_hours,
                        "participation": participation,
                        "predicted_score": predicted_score
                    })
                    st.success("Saved to teacher account!")
                else:
                    st.info("Please login as teacher first.")

            # ----------------- PDF REPORT -----------------
            st.markdown("### 📄 Download PDF Report")
            buf = io.BytesIO()
            try:
                c = canvas.Canvas(buf, pagesize=letter)
                width, height = letter

                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "Student Performance Report")

                c.setFont("Helvetica", 12)
                c.drawString(50, height - 100, f"Student Name: {name}")
                c.drawString(50, height - 120, f"Class: {class_name}")
                c.drawString(50, height - 140, f"Predicted Score: {predicted_score:.2f}")

                # Add remedial note if needed
                y = height - 180
                if predicted_score < 50:
                    c.drawString(50, y, "Remedial Measures Recommended:")
                    y -= 20
                    c.drawString(60, y, "- Increase study hours gradually")
                    y -= 15
                    c.drawString(60, y, "- Improve attendance & participation")
                    y -= 15
                    c.drawString(60, y, "- Focus on weak areas / tutoring")
                    y -= 15

                c.showPage()
                c.save()
                buf.seek(0)
                pdf_bytes = buf.getvalue()
                buf.close()

                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"{name.replace(' ', '_')}_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error("Failed to generate PDF: " + str(e))

# ---------------------- TEACHER LOGIN ----------------------
elif page == "Teacher Login":
    st.header("👩‍🏫 Teacher Login")
    if LOTTIE_LOGIN:
        st_lottie(LOTTIE_LOGIN, height=250)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state["teacher_auth"] = True
            st.session_state["teacher_username"] = username
            st.success("Login Successful!")
        else:
            st.error("Invalid username or password.")

    st.markdown("---")
    st.subheader("Register teacher (optional)")
    new_user = st.text_input("New Username", key="reg_user")
    new_pass = st.text_input("New Password", type="password", key="reg_pass")
    if st.button("Register"):
        ok, msg = register_teacher(new_user, new_pass)
        if ok:
            st.success("Registered new teacher. Please login.")
        else:
            st.error(msg)

# ---------------------- DASHBOARD ----------------------
elif page == "Dashboard":
    st.header("📊 Teacher Dashboard")
    if LOTTIE_DASH:
        st_lottie(LOTTIE_DASH, height=250)

    if not st.session_state.get("teacher_auth", False):
        st.warning("Please login first!")
        st.stop()

    if os.path.exists(PREDICTIONS_FILE):
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            # show only this teacher's saved predictions
            teacher = st.session_state.get("teacher_username")
            df_user = df[df['teacher'] == teacher] if 'teacher' in df.columns else df
            if df_user.empty:
                st.info("No saved predictions for your account yet.")
            else:
                st.dataframe(df_user.sort_values("timestamp", ascending=False))
                st.markdown("### Average Predicted Score (your saved)")
                st.write(df_user['predicted_score'].mean())
                # histogram
                try:
                    import altair as alt
                    chart = alt.Chart(df_user).mark_bar().encode(
                        alt.X("predicted_score:Q", bin=alt.Bin(maxbins=20), title="Predicted Score"),
                        y='count()'
                    ).properties(width=700, height=300)
                    st.altair_chart(chart)
                except Exception:
                    pass
                # download entire teacher history
                csv_bytes = df_user.to_csv(index=False).encode('utf-8')
                st.download_button("Download your predictions (CSV)", csv_bytes, file_name=f"{teacher}_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Failed to load predictions: " + str(e))
    else:
        st.info("No predictions saved yet.")

st.markdown('</div>', unsafe_allow_html=True)
