# app.py (UPDATED ONLY DROPDOWN ARROW + PREDICTED SCORE TEXT COLOR)
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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch


# ---------------------- CONFIG & LOTTIE LOADER ----------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered", page_icon="🎓")

def load_lottie(url: str, timeout: int = 6):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


LOTTIE_HOME     = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
LOTTIE_STUDY    = load_lottie("https://assets1.lottiefiles.com/packages/lf20_tutvdkg0.json")
LOTTIE_REMedy   = load_lottie("https://assets7.lottiefiles.com/private_files/lf30_tamcv1wp.json")
LOTTIE_SUCCESS  = load_lottie("https://assets4.lottiefiles.com/packages/lf20_u4yrau.json")
LOTTIE_LOGIN    = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
LOTTIE_DASH     = load_lottie("https://assets1.lottiefiles.com/packages/lf20_i8d0whbs.json")


# ---------------------- STYLES ----------------------
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
  background: rgba(224,247,250,0.95);
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

[data-testid="stSidebar"] svg {{
  fill: black !important;
  color: black !important;
}}

.predicted-score-box {{
  color: white !important;
  font-size: 18px;
  font-weight: bold;
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
        return False, "Username exists"
    df = pd.concat([df, pd.DataFrame([{
        "username": username, 
        "password_hash": hash_password(password)
    }])], ignore_index=True)
    df.to_csv(TEACHERS_FILE, index=False)
    return True, "Registered"

def save_prediction(record: dict):
    df_new = pd.DataFrame([record])
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(PREDICTIONS_FILE, index=False)

@st.cache_resource
def load_model(path=MODEL_FILE):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except:
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
        st_lottie(LOTTIE_HOME, height=320)
    st.write("This app helps students & teachers understand performance trends.")


# ---------------------- PREDICTION PAGE ----------------------
elif page == "Prediction":
    st.header("📚 Predict Student Performance")
    if LOTTIE_STUDY:
        st_lottie(LOTTIE_STUDY, height=240)

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
        X = np.array([[study_hours, attendance, prev_grade, sleep_hours, participation]])
        predicted_score = float(model.predict(X)[0])
        predicted_score = max(0, min(100, predicted_score))

        st.markdown(
            f'<div class="predicted-score-box">Predicted Score: {predicted_score:.2f}%</div>',
            unsafe_allow_html=True
        )

        if predicted_score > 70:
            if LOTTIE_SUCCESS:
                st_lottie(LOTTIE_SUCCESS, height=200)
            st.balloons()

        # --- Charts ---
        st.markdown("### 📊 Criteria Importance")
        feature_names = ["Study Hours","Attendance","Previous Grade","Sleep Hours","Participation"]
        weights = np.array([4,0.2,0.5,1,2])
        vals = np.array([study_hours, attendance, prev_grade, sleep_hours, participation])
        importance = weights * vals
        importance_norm = (importance / (importance.sum()+1e-9)) * 100
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importance_norm})

        fig1, ax1 = plt.subplots(figsize=(7,3.5))
        ax1.bar(df_imp['feature'], df_imp['importance'])
        plt.xticks(rotation=20)
        st.pyplot(fig1)

        st.markdown("### 📊 Predicted Score vs Previous Grade")
        df_cmp = pd.DataFrame({"Metric":["Previous Grade","Predicted Score"],
                               "Value":[prev_grade,predicted_score]})

        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.bar(df_cmp['Metric'], df_cmp['Value'])
        ax2.set_ylim(0, 100)
        st.pyplot(fig2)

        if predicted_score < 50:
            st.warning("⚠️ Remedial Measures Recommended")
            if LOTTIE_REMedy:
                st_lottie(LOTTIE_REMedy, height=200)
            st.write("""
            - Increase study hours  
            - Improve attendance  
            - Revise weak topics  
            - Seek teacher assistance  
            """)

        # --- PDF generation (unchanged) ---
        try:
            chart1_buf = io.BytesIO()
            fig1.savefig(chart1_buf, format='png', bbox_inches='tight')
            chart1_buf.seek(0)

            chart2_buf = io.BytesIO()
            fig2.savefig(chart2_buf, format='png', bbox_inches='tight')
            chart2_buf.seek(0)

            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            W, H = letter

            c.setFillColorRGB(1.0,0.92,0.39)
            c.rect(0, H-80, W, 80, stroke=0, fill=1)

            c.setFillColor(colors.HexColor("#0d47a1"))
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, H-52, "Student Performance Report")

            box_x, box_y = 40, H-240
            box_w, box_h = W-80, 120
            c.setFillColorRGB(0.88,0.96,0.98)
            c.roundRect(box_x, box_y, box_w, box_h, 8, stroke=0, fill=1)

            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(box_x+12, box_y+box_h-24, f"Student Name: {name}")
            c.drawString(box_x+12, box_y+box_h-42, f"Class: {class_name}")
            c.drawString(box_x+12, box_y+box_h-60, f"Study Hours/day: {study_hours}")
            c.drawString(box_x+220, box_y+box_h-60, f"Attendance: {attendance}%")
            c.drawString(box_x+12, box_y+box_h-78, f"Previous Grade: {prev_grade}%")
            c.drawString(box_x+220, box_y+box_h-78, f"Sleep Hours: {sleep_hours}")

            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, box_y-18, f"Predicted Score: {predicted_score:.2f}%")

            img1 = ImageReader(chart1_buf)
            img2 = ImageReader(chart2_buf)
            c.drawImage(img1, 50, box_y-180, width=250, height=150)
            c.drawImage(img2, 330, box_y-180, width=250, height=150)

            c.showPage()
            c.save()

            pdf_buffer.seek(0)
            st.download_button(
                "Download Styled PDF",
                data=pdf_buffer.getvalue(),
                file_name=f"{name}_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error("PDF Error: " + str(e))


# ---------------------- TEACHER LOGIN ----------------------
elif page == "Teacher Login":
    st.header("👩‍🏫 Teacher Login")
    if LOTTIE_LOGIN:
        st_lottie(LOTTIE_LOGIN, height=220)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state["teacher_auth"] = True
            st.session_state["teacher_username"] = username
            st.success("Login Successful!")
        else:
            st.error("Invalid credentials")

    st.markdown("---")
    st.subheader("Register teacher")
    new_user = st.text_input("New Username", key="reg_user")
    new_pass = st.text_input("New Password", type="password", key="reg_pass")

    if st.button("Register"):
        ok, msg = register_teacher(new_user, new_pass)
        st.success(msg) if ok else st.error(msg)


# ---------------------- DASHBOARD ----------------------
elif page == "Dashboard":
    st.header("📊 Teacher Dashboard")
    if LOTTIE_DASH:
        st_lottie(LOTTIE_DASH, height=250)

    if not st.session_state.get("teacher_auth"):
        st.warning("Please login first!")
        st.stop()

    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
        teacher = st.session_state["teacher_username"]
        df_user = df[df["teacher"] == teacher] if "teacher" in df else df

        st.dataframe(df_user)

        st.write("Average Predicted Score:", df_user["predicted_score"].mean())

        import altair as alt
        chart = alt.Chart(df_user).mark_bar().encode(
            alt.X("predicted_score:Q", bin=True),
            y='count()'
        )
        st.altair_chart(chart)

    else:
        st.info("No saved predictions yet.")


st.markdown('</div>', unsafe_allow_html=True)
