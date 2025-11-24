# app.py - Enhanced Student Performance App (final corrected)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from fpdf import FPDF
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# ---------------------------
# User-supplied local image path (from your session uploads)
# Developer note: using the local path from conversation history so it can be transformed into a URL if needed.
LOCAL_IMG_PATH = "/mnt/data/Screenshot (1004).png"  # <-- path included from conversation history

# ---------------------------
# Styling: light-yellow background + side images + black text
# ---------------------------
PAGE_CSS = f"""
<style>
/* light yellow background */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, #fff9d9 0%, #fffde7 100%);
  background-attachment: fixed;
}}

/* left and right side images */
.side-image {{
  position: fixed;
  top: 0;
  width: 150px;
  height: 100vh;
  background-size: cover;
  background-position: center;
  z-index: 0;
}}
.side-left {{ left: 0; }}
.side-right {{ right: 0; }}

/* central white card */
.block-container {{
  max-width: 980px;
  margin-left: auto;
  margin-right: auto;
  background: rgba(255,255,255,0.98);
  padding: 28px 36px;
  border-radius: 12px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.06);
}}

/* Force text to black for readability */
h1, h2, h3, p, label, .stText, .stMarkdown {{
  color: #000000 !important;
}}

/* input tweaks */
.stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {{
  height:44px;
  border-radius:8px;
  padding-left:10px;
}}
</style>

<div class="side-image side-left" style="background-image: url('https://images.unsplash.com/photo-1529070538774-1843cb3265df?auto=format&fit=crop&w=400&q=60');"></div>
<div class="side-image side-right" style="background-image: url('https://images.unsplash.com/photo-1556012018-1b44f5f0d2b2?auto=format&fit=crop&w=400&q=60');"></div>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.markdown("<div class='block-container'>", unsafe_allow_html=True)

# Optionally show user's uploaded screenshot images (if exist in the environment)
if os.path.exists(LOCAL_IMG_PATH):
    try:
        st.image(LOCAL_IMG_PATH, width=120)
    except:
        pass

# ---------------------------
# Model loading (or fallback train)
# ---------------------------
MODEL_PATH = "model.pkl"
DATA_PATH = "student_data.csv"
model = None

def train_from_csv_and_save():
    df = pd.read_csv(DATA_PATH)
    # Required columns for fallback training
    cols = ['study_hours','attendance_percent','internal_marks','assignments_score','previous_grade','family_support']
    if not all(c in df.columns for c in cols):
        raise ValueError(f"CSV must contain columns: {cols} and 'final_score' (found: {list(df.columns)})")
    X = df[cols]
    y = df['final_score']
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X, y)
    joblib.dump(m, MODEL_PATH)
    return m

# Try to load model; otherwise train if CSV present
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning("Could not load model.pkl: " + str(e))
        model = None
else:
    if os.path.exists(DATA_PATH):
        try:
            model = train_from_csv_and_save()
            st.info("Trained model from student_data.csv and saved model.pkl")
        except Exception as e:
            st.warning("Failed to train from CSV: " + str(e))
            model = None
    else:
        st.info("No model.pkl or student_data.csv found. Upload model.pkl to the repo to use the trained model.")

# ---------------------------
# Sidebar navigation (multi-page)
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Teacher Login", "Dashboard"])

# Simple in-memory session for teacher login
if 'teacher_logged_in' not in st.session_state:
    st.session_state['teacher_logged_in'] = False
    st.session_state['teacher_user'] = ""

# Demo credentials (change before production)
DEMO_USER = "teacher"
DEMO_PASS = "pass123"

# ---------------------------
# Home page
# ---------------------------
if page == "Home":
    st.title("🎓 Student Performance Prediction")
    st.write("This app predicts a student's final score using a small ML model, gives advice, shows charts, and can create a PDF report.")
    st.write("Make sure `model.pkl` is present in the repository root for best results; otherwise add `student_data.csv` with the right columns and the app will attempt to train a model.")
    st.write("Use the Predict page to enter student data. Teachers can log in to save predictions to the Dashboard.")
    st.success("Background: light yellow; text color set to black for readability.")

# ---------------------------
# Predict page
# ---------------------------
elif page == "Predict":
    st.title("Predict Student Performance")

    # Illustrations: show a header image (either local or online)
    col_header_img, _ = st.columns([3,1])
    with col_header_img:
        if os.path.exists(LOCAL_IMG_PATH):
            st.image(LOCAL_IMG_PATH, caption="School Illustration", use_column_width=True)
        else:
            # fallback illustration
            st.image("https://images.unsplash.com/photo-1596496059330-7f91a4a3d5a8?auto=format&fit=crop&w=1200&q=60",
                     caption="School illustration", use_column_width=True)

    st.subheader("Step 1 — Student information (required)")
    col1, col2 = st.columns([2,1])
    with col1:
        student_name = st.text_input("Student Name")
    with col2:
        student_class = st.text_input("Class / Grade")

    if not student_name.strip() or not student_class.strip():
        st.warning("Please enter both Student Name and Class to continue.")
        st.stop()

    st.subheader("Step 2 — Academic & other inputs (all required)")
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
        study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        internal_marks = st.number_input("Internal Marks (out of 100)", min_value=0, max_value=100, value=60)
    with c2:
        assignments_score = st.number_input("Assignments Score (out of 100)", min_value=0, max_value=100, value=65)
        previous_grade = st.number_input("Previous Grade (out of 100)", min_value=0, max_value=100, value=70)
        # family_support now 0..6 as requested
        family_support = st.selectbox("Family Support (0=none .. 6=very strong)", [0,1,2,3,4,5,6], index=3)

    if st.button("Predict 🎯"):
        features = [study_hours, attendance, internal_marks, assignments_score, previous_grade, family_support]
        if any(v is None for v in features):
            st.error("Please fill all fields.")
            st.stop()

        if model is None:
            st.error("No model available. Upload model.pkl to the repo or provide a correct student_data.csv.")
            st.stop()

        X = np.array([features])
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.error("Model prediction failed: " + str(e))
            st.stop()

        pred = float(max(0, min(100, pred)))  # clamp to [0,100]
        st.success(f"Predicted final score for **{student_name} (Class {student_class})**: **{pred:.2f} / 100**")

        # Balloon animation for high score (>80)
        if pred > 80:
            st.balloons()
            # also a celebratory gif (simple animation)
            st.image("https://media.giphy.com/media/26BkNrGhy4DKnbD9u/giphy.gif", width=300)

        # Remedial measures if score < 60
        st.subheader("Recommendations")
        if pred < 50:
            st.warning("⚠️ Predicted score is low (<50). Urgent remedial measures:")
            st.markdown(
                "- Create a daily study timetable and increase study hours gradually.  \n"
                "- Improve attendance and actively participate in classes.  \n"
                "- Seek extra help / tutoring for weak topics.  \n"
                "- Focus on completing and improving assignments and internal tests.  \n"
                "- Practice past papers and take regular quizzes."
            )
        elif pred < 60:
            st.info("Predicted score is below 60. Suggested remedial actions:")
            st.markdown(
                "- Increase focused study time and practice weak topics.  \n"
                "- Review internal assessments and complete extra assignments.  \n"
                "- Use study groups and ask teachers for targeted help."
            )
        elif pred < 75:
            st.info("Predicted score is moderate — aim for improvement with focused revision and practice.")
        else:
            st.success("Good! Maintain the habits to keep performing well.")

        # Charts
        st.subheader("Visual Summary")
        input_df = pd.DataFrame({
            "Metric":["Attendance(%)","Study Hours","Internal(%)","Assignments(%)","Previous Grade(%)","Family Support"],
            "Value":[attendance, study_hours, internal_marks, assignments_score, previous_grade, family_support]
        }).set_index("Metric")
        st.bar_chart(input_df)

        comp_df = pd.DataFrame({
            "Stage":["Internal","Assignments","Predicted"],
            "Score":[internal_marks, assignments_score, pred]
        }).set_index("Stage")
        st.line_chart(comp_df)

        # Save record dict
        record = {
            "timestamp": datetime.now().isoformat(),
            "student_name": student_name,
            "student_class": student_class,
            "attendance": attendance,
            "study_hours": study_hours,
            "internal_marks": internal_marks,
            "assignments_score": assignments_score,
            "previous_grade": previous_grade,
            "family_support": family_support,
            "predicted_score": round(pred,2)
        }

        # If teacher logged in, allow saving
        if st.session_state.get('teacher_logged_in', False):
            if st.button("Save prediction to Dashboard"):
                save_path = "predictions.csv"
                df_new = pd.DataFrame([record])
                if os.path.exists(save_path):
                    try:
                        df_existing = pd.read_csv(save_path)
                        df_all = pd.concat([df_existing, df_new], ignore_index=True)
                    except:
                        df_all = df_new
                else:
                    df_all = df_new
                df_all.to_csv(save_path, index=False)
                st.success("Saved to predictions.csv (Dashboard).")

        # PDF report builder and download
        def build_pdf_bytes(rec):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Student Performance Report", ln=True, align="C")
            pdf.ln(6)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, f"Name: {rec['student_name']}", ln=True)
            pdf.cell(0, 8, f"Class: {rec['student_class']}", ln=True)
            pdf.cell(0, 8, f"Date: {rec['timestamp']}", ln=True)
            pdf.ln(4)
            pdf.cell(0, 8, "Inputs:", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 7, f"- Attendance: {rec['attendance']}", ln=True)
            pdf.cell(0, 7, f"- Study Hours: {rec['study_hours']}", ln=True)
            pdf.cell(0, 7, f"- Internal Marks: {rec['internal_marks']}", ln=True)
            pdf.cell(0, 7, f"- Assignments Score: {rec['assignments_score']}", ln=True)
            pdf.cell(0, 7, f"- Previous Grade: {rec['previous_grade']}", ln=True)
            pdf.cell(0, 7, f"- Family Support: {rec['family_support']}", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Predicted Final Score: {rec['predicted_score']} / 100", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, "Recommendations:", ln=True)
            pdf.set_font("Arial", size=11)
            if rec['predicted_score'] < 60:
                pdf.multi_cell(0, 6, "The predicted score is below 60. Recommended actions: increase study hours, improve attendance, get extra help from teachers, practice more, and focus on internal assessments and assignments.")
            elif rec['predicted_score'] < 75:
                pdf.multi_cell(0, 6, "Moderate performance. Focus on targeted revision, strengthen weak topics, and complete more practice assignments.")
            else:
                pdf.multi_cell(0, 6, "Good performance. Maintain consistency in study and attendance.")
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            return pdf_bytes

        pdf_bytes = build_pdf_bytes(record)
        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"{student_name}_report.pdf", mime="application/pdf")

# ---------------------------
# Teacher Login page
# ---------------------------
elif page == "Teacher Login":
    st.title("Teacher Login")

    # Show login status
    if st.session_state.get('teacher_logged_in', False):
        st.success(f"Logged in as {st.session_state.get('teacher_user')}")
        if st.button("Log out"):
            st.session_state['teacher_logged_in'] = False
            st.session_state['teacher_user'] = ""
            st.experimental_rerun()
        st.write("You can save predictions from the Predict page while logged in. Dashboard shows saved predictions.")
    else:
        with st.form("login"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if user == DEMO_USER and pwd == DEMO_PASS:
                    st.session_state['teacher_logged_in'] = True
                    st.session_state['teacher_user'] = user
                    st.success("Login successful.")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials. (Demo: teacher / pass123)")

# ---------------------------
# Dashboard page
# ---------------------------
elif page == "Dashboard":
    st.title("Teacher Dashboard")
    if not st.session_state.get('teacher_logged_in', False):
        st.warning("Please log in from Teacher Login to view saved predictions.")
        st.stop()

    save_path = "predictions.csv"
    if not os.path.exists(save_path):
        st.info("No saved predictions yet. Save predictions from the Predict page after logging in.")
        st.stop()

    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        st.error("Failed to read predictions.csv: " + str(e))
        st.stop()

    st.subheader("Saved Predictions")
    st.dataframe(df)

    st.subheader("Analytics")
    if 'predicted_score' in df.columns:
        st.metric("Average predicted score", f"{df['predicted_score'].mean():.2f}")
        st.write("Prediction distribution:")
        st.bar_chart(df['predicted_score'])
        st.write("Predictions over time:")
        # ensure timestamp parsed
        try:
            df['timestamp_parsed'] = pd.to_datetime(df['timestamp'])
            df_sorted = df.sort_values('timestamp_parsed')
            st.line_chart(df_sorted.set_index('timestamp_parsed')['predicted_score'])
        except:
            pass

    # Allow download
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download all predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

# ---------------------------
# End central card
# ---------------------------
st.markdown("</div>", unsafe_allow_html=True)
