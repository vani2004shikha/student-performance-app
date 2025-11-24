# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import altair as alt
import hashlib
from datetime import datetime

# optional PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------------------- PAGE DESIGN / STYLES ----------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered", page_icon="🎓")

# Custom CSS for colors, background and fonts
page_bg_img = f"""
<style>
/* Overall app background: light yellow */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, #fff9c4 0%, #fffde7 100%);
  background-attachment: fixed;
}}

/* Header transparent */
[data-testid="stHeader"] {{
  background: rgba(0,0,0,0);
}}

/* Main block container: sky blue with blur */
.block-container {{
  backdrop-filter: blur(3px);
  background: rgba(224,247,250,0.9); /* sky-blue-ish */
  padding: 28px;
  border-radius: 14px;
  color: #000000; /* text black */
}}

/* Make titles and text black */
h1, h2, h3, .css-1d391kg, .css-1d391kg * {{
  color: #000000 !important;
}}

/* Buttons style (optional) */
.stButton>button {{
  background: linear-gradient(90deg,#ffd54f,#ffecb3);
  color: #000000;
  border-radius: 8px;
}}

/* Make sidebar also sky-blue */
[data-testid="stSidebar"] {{
  background: rgba(224,247,250,0.95);
  color: #000000;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Wrap main container
st.markdown('<div class="block-container">', unsafe_allow_html=True)

# ---------------------- Utilities ----------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.csv")
TEACHERS_FILE = os.path.join(DATA_DIR, "teachers.csv")
MODEL_FILE = "model.pkl"

# helper: hash password
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# init a default teacher account if teachers file doesn't exist
def ensure_default_teacher():
    if not os.path.exists(TEACHERS_FILE):
        df = pd.DataFrame([{"username": "teacher", "password_hash": hash_password("password")}])
        df.to_csv(TEACHERS_FILE, index=False)

ensure_default_teacher()

# load teachers
def load_teachers():
    try:
        return pd.read_csv(TEACHERS_FILE)
    except Exception:
        return pd.DataFrame(columns=["username", "password_hash"])

# authenticate teacher
def authenticate(username, password):
    df = load_teachers()
    if username in df['username'].values:
        stored = df.loc[df['username'] == username, 'password_hash'].values[0]
        return stored == hash_password(password)
    return False

# register teacher (optional)
def register_teacher(username, password):
    df = load_teachers()
    if username in df['username'].values:
        return False, "Username already exists"
    new = {"username": username, "password_hash": hash_password(password)}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(TEACHERS_FILE, index=False)
    return True, "Registered"

# save prediction
def save_prediction(record: dict):
    cols = ["timestamp", "teacher", "student_name", "class", "study_hours", "attendance", "prev_grade", "sleep_hours", "participation", "predicted_score"]
    df_new = pd.DataFrame([record])
    if os.path.exists(PREDICTIONS_FILE):
        df = pd.read_csv(PREDICTIONS_FILE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(PREDICTIONS_FILE, index=False)

# load model with caching
@st.cache_resource
def load_model(path=MODEL_FILE):
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            st.warning("Model file exists but couldn't be loaded. Using fallback model. Error: " + str(e))
    # fallback: create a simple dummy model that returns a weighted sum clipped to 0-100
    class DummyModel:
        def predict(self, X):
            # X columns: study_hours, attendance, prev_grade, sleep_hours, participation
            weights = np.array([4.0, 0.2, 0.5, 1.0, 2.0])
            raw = X.dot(weights)
            # normalize by an arbitrary factor
            scores = np.clip(raw, 0, 100)
            return scores
    return DummyModel()

model = load_model()

# ---------------------- APP NAVIGATION (SINGLE FILE MULTI-PAGE) ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Prediction", "Teacher Login", "Dashboard"], index=0)

# small helper for showing a header image + title on Home
def show_header(title, subtitle=None, image_url=None, gif_url=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.write(subtitle)
    cols = st.columns([1,2])
    if image_url:
        cols[0].image(image_url, use_column_width=True)
    if gif_url:
        cols[1].image(gif_url, use_column_width=True)

# ---------------------- PAGE: HOME ----------------------
if page == "Home":
    st.title("🎓 Student Performance Predictor")
    st.write("Welcome! Use this app to predict student performance, get visual insights, and (if you're a teacher) save class results.")
    # Illustration images (Unsplash / public URLs). Replace with your own hosted assets if desired.
    show_header(
        title="Make learning visible — Predict. Improve. Grow.",
        subtitle="Navigate with the left menu. Prediction input, reports, teacher login and dashboard are available.",
        image_url="https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=1000&q=80", # students
        gif_url="https://media.giphy.com/media/3o6ZsXKx2X3vX1o7sI/giphy.gif" # playful animation
    )
    st.markdown("---")
    st.write("Features:")
    st.markdown("""
    - Predict student scores from simple inputs.
    - Visualize factors that influence the predicted score (charts & graphs).
    - Download a PDF/CSV report for the student.
    - Teacher login: save & manage class predictions.
    - Remedial suggestions when the predicted score is below 50.
    """)
    st.info("Tip: default teacher account is `teacher` / `password` (change in Teacher Login).")
    # Add more visuals
    st.image("https://images.unsplash.com/photo-1584697964403-8588f7d1f1ea?w=1200&q=80", caption="Learning together")

# ---------------------- PAGE: PREDICTION ----------------------
elif page == "Prediction":
    st.header("📚 Predict Student Performance")
    st.write("Fill the details below to predict the student's score.")
    # Form to avoid reruns
    with st.form("prediction_form"):
        name = st.text_input("Student Name")
        class_name = st.text_input("Class")
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5, format="%.1f")
            prev_grade = st.number_input("Previous Grade (%)", min_value=0, max_value=100, value=70)
            participation = st.selectbox("Class Participation Level (0-5)", [0,1,2,3,4,5], index=2)
        with col2:
            attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
            sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=7.0, step=0.5, format="%.1f")
            show_save_to_teacher = st.checkbox("Save prediction to teacher account (requires login)", value=False)
        st.write("")  # spacing
        submit = st.form_submit_button("Predict Performance 🎯")

    if submit:
        # Validation
        if not name.strip() or not class_name.strip():
            st.error("Please enter student name and class.")
        else:
            # prepare input and predict
            inputs = np.array([[study_hours, attendance, prev_grade, sleep_hours, participation]])
            try:
                prediction = model.predict(inputs)[0]
                # ensure it's in 0-100
                try:
                    predicted_score = float(prediction)
                except Exception:
                    predicted_score = float(np.asscalar(prediction)) if hasattr(prediction, 'tolist') else float(prediction)
                predicted_score = max(0.0, min(100.0, predicted_score))
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                predicted_score = None

            if predicted_score is not None:
                st.success(f"📘 **{name} (Class {class_name})'s Predicted Score: {predicted_score:.2f}**")
                # show balloons only if > 70
                if predicted_score > 70:
                    st.balloons()

                # Charts and visualizations
                st.markdown("### 📊 Visual Analysis")

                # Bar chart: feature contributions (simple proxy)
                feature_names = ["Study Hours", "Attendance", "Prev Grade", "Sleep Hours", "Participation"]
                # crude "importance" based on normalized product of input and weights used by fallback model or coefficients if present
                importance = []
                if hasattr(model, "coef_"):
                    # linear model coefficients
                    coef = np.abs(model.coef_)
                    vals = np.array([study_hours, attendance, prev_grade, sleep_hours, participation])
                    importance = coef * vals
                else:
                    # fallback heuristic
                    weights = np.array([4.0, 0.2, 0.5, 1.0, 2.0])
                    vals = np.array([study_hours, attendance, prev_grade, sleep_hours, participation])
                    importance = weights * vals
                # normalize for plotting
                importance_norm = (importance / (importance.sum() + 1e-9)) * 100
                df_imp = pd.DataFrame({"feature": feature_names, "importance": importance_norm})

                # matplotlib bar chart
                fig, ax = plt.subplots(figsize=(6,3))
                ax.bar(df_imp['feature'], df_imp['importance'])
                ax.set_ylabel("Relative contribution (%)")
                ax.set_xticklabels(df_imp['feature'], rotation=20, ha='right')
                st.pyplot(fig)

                # Altair scatter: previous vs predicted
                df_scatter = pd.DataFrame({
                    "Metric": ["Previous Grade", "Predicted Score"],
                    "Value": [prev_grade, predicted_score]
                })
                chart = alt.Chart(df_scatter).mark_bar().encode(
                    x='Metric',
                    y='Value'
                )
                st.altair_chart(chart, use_container_width=True)

                # Remedial suggestions if score < 50
                if predicted_score < 50:
                    st.warning("⚠️ Predicted score is below 50 — remedial measures recommended:")
                    st.markdown("""
                    **Suggested Remedial Measures**
                    - Increase daily study hours gradually (aim for consistent *focused* study sessions).
                    - Keep a study routine and short breaks (Pomodoro technique — 25/5).
                    - Improve attendance and class participation — these strongly correlate with better outcomes.
                    - Review previous exam mistakes — focus on weak topics.
                    - Seek tutor or after-school help for specific subjects.
                    - Short, consistent revision sessions before exams (20–30 mins/day).
                    """)
                    # show a supportive image
                    st.image("https://images.unsplash.com/photo-1584697964403-8588f7d1f1ea?w=900&q=80", caption="Remedial guidance & focused learning")

                # Save option & PDF/CSV report
                st.markdown("---")
                st.write("Generate report / save prediction:")

                # prepare report content
                report = {
                    "timestamp": datetime.now().isoformat(timespec='seconds'),
                    "student_name": name,
                    "class": class_name,
                    "study_hours": study_hours,
                    "attendance": attendance,
                    "prev_grade": prev_grade,
                    "sleep_hours": sleep_hours,
                    "participation": participation,
                    "predicted_score": round(predicted_score, 2)
                }

                # Save to teacher if requested and logged in
                teacher_username = None
                if show_save_to_teacher:
                    if "teacher_auth" in st.session_state and st.session_state.teacher_auth:
                        teacher_username = st.session_state.teacher_username
                        rec = report.copy()
                        rec["teacher"] = teacher_username
                        save_prediction(rec)
                        st.success(f"Saved prediction under teacher: {teacher_username}")
                    else:
                        st.info("To save predictions you must login as a teacher (go to Teacher Login page).")

                # Allow download as PDF if reportlab available; else CSV fallback
                buffer = io.BytesIO()
                if REPORTLAB_AVAILABLE:
                    # create simple PDF
                    pdf_filename = f"{name.replace(' ', '_')}_report.pdf"
                    c = canvas.Canvas(buffer, pagesize=letter)
                    width, height = letter
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(72, height - 72, f"Student Performance Report — {name}")
                    c.setFont("Helvetica", 12)
                    y = height - 110
                    for k, v in report.items():
                        c.drawString(72, y, f"{k}: {v}")
                        y -= 20
                    # remedial note
                    if predicted_score < 50:
                        y -= 10
                        c.setFont("Helvetica-Bold", 12)
                        c.drawString(72, y, "Remedial Measures Recommended")
                        y -= 20
                        c.setFont("Helvetica", 10)
                        text = ("Increase study hours, review weak topics, attend tutoring, "
                                "improve class participation and attendance.")
                        c.drawString(72, y, text)
                    c.showPage()
                    c.save()
                    buffer.seek(0)
                    st.download_button("Download PDF Report", buffer, file_name=pdf_filename, mime="application/pdf")
                else:
                    # create CSV in memory
                    csv_buf = io.StringIO()
                    pd.DataFrame([report]).to_csv(csv_buf, index=False)
                    csv_buf.seek(0)
                    st.download_button("Download CSV Report (PDF not available)", csv_buf.getvalue(), file_name=f"{name.replace(' ', '_')}_report.csv", mime="text/csv")

# ---------------------- PAGE: TEACHER LOGIN ----------------------
elif page == "Teacher Login":
    st.header("👩‍🏫 Teacher Login & Account")
    st.write("Login to save predictions for your class. You can also register a new teacher account.")

    if "teacher_auth" not in st.session_state:
        st.session_state.teacher_auth = False
        st.session_state.teacher_username = None

    login_col, reg_col = st.columns(2)

    with login_col:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate(login_user, login_pass):
                st.session_state.teacher_auth = True
                st.session_state.teacher_username = login_user
                st.success(f"Logged in as {login_user}")
            else:
                st.error("Invalid credentials")

        if st.session_state.teacher_auth:
            st.success(f"You're logged in as: {st.session_state.teacher_username}")
            if st.button("Logout"):
                st.session_state.teacher_auth = False
                st.session_state.teacher_username = None
                st.success("Logged out")

    with reg_col:
        st.subheader("Register")
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        if st.button("Register"):
            ok, msg = register_teacher(new_user, new_pass)
            if ok:
                st.success("Registered new teacher. Please login.")
            else:
                st.error(msg)

    st.markdown("---")
    st.write("Saved predictions are stored locally in 'data/predictions.csv'. Teachers can view them on the Dashboard page.")

# ---------------------- PAGE: DASHBOARD ----------------------
elif page == "Dashboard":
    st.header("📈 Dashboard — Class Predictions")
    st.write("View saved predictions (teachers only) and basic analytics.")

    if not os.path.exists(PREDICTIONS_FILE):
        st.info("No saved predictions yet. Teachers can save predictions from the Prediction page.")
    else:
        df = pd.read_csv(PREDICTIONS_FILE)
        # If teacher logged in, allow filter by their username
        if "teacher_auth" in st.session_state and st.session_state.teacher_auth:
            user = st.session_state.teacher_username
            st.subheader(f"Predictions saved by {user}")
            df_user = df[df['teacher'] == user]
            if df_user.empty:
                st.info("You have not saved any predictions yet.")
            else:
                st.dataframe(df_user.sort_values("timestamp", ascending=False))
                # aggregate charts
                avg_score = df_user['predicted_score'].mean()
                st.metric("Average predicted score (your saved)", f"{avg_score:.2f}")

                # histogram of scores
                chart = alt.Chart(df_user).mark_bar().encode(
                    alt.X("predicted_score:Q", bin=alt.Bin(maxbins=20), title="Predicted Score"),
                    y='count()'
                ).properties(width=700, height=300)
                st.altair_chart(chart)

                # Allow teacher to download all their saved predictions
                csv = df_user.to_csv(index=False).encode('utf-8')
                st.download_button("Download your predictions CSV", csv, file_name=f"{user}_predictions.csv", mime="text/csv")
        else:
            st.subheader("All saved predictions (admin view)")
            st.dataframe(df.sort_values("timestamp", ascending=False).head(200))
            # Overall analytics
            st.markdown("### Overall stats")
            avg = df['predicted_score'].mean()
            median = df['predicted_score'].median()
            st.metric("Average predicted score", f"{avg:.2f}")
            st.metric("Median predicted score", f"{median:.2f}")
            # show top/bottom students
            st.markdown("Top 5 predicted scores")
            st.dataframe(df.sort_values("predicted_score", ascending=False).head(5)[["timestamp","teacher","student_name","class","predicted_score"]])
            st.markdown("Bottom 5 predicted scores")
            st.dataframe(df.sort_values("predicted_score", ascending=True).head(5)[["timestamp","teacher","student_name","class","predicted_score"]])

            # aggregated chart
            hist = alt.Chart(df).mark_bar().encode(
                alt.X("predicted_score:Q", bin=alt.Bin(maxbins=25), title="Predicted Score"),
                y='count()'
            ).properties(width=700, height=300)
            st.altair_chart(hist)

st.markdown('</div>', unsafe_allow_html=True)
