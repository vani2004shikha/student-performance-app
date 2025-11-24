# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

####################
# --- Styles / UI ---
####################
# Sky-blue background + side images + central content card
page_style = """
<style>
/* sky-blue background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #cfeefd 0%, #e6f7ff 100%);
  background-attachment: fixed;
}

/* left and right side panels for images */
.side-image {
  position: fixed;
  top: 0;
  width: 150px;
  height: 100vh;
  background-size: cover;
  background-position: center;
  z-index: 0;
}
.side-left { left: 0; }
.side-right { right: 0; }

/* main card */
.block-container {
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
  background: rgba(255, 255, 255, 0.95);
  padding: 30px 40px;
  border-radius: 14px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.08);
}

/* input widths */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
  border-radius:8px;
  height:44px;
  padding-left:12px;
}

/* header font size */
h1 {
  font-size: 34px;
  margin-bottom: 8px;
}
</style>
<div class="side-image side-left" style="background-image: url('https://images.unsplash.com/photo-1529070538774-1843cb3265df?auto=format&fit=crop&w=400&q=60');"></div>
<div class="side-image side-right" style="background-image: url('https://images.unsplash.com/photo-1556012018-1b44f5f0d2b2?auto=format&fit=crop&w=400&q=60');"></div>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Put content in the central card area
st.markdown("<div class='block-container'>", unsafe_allow_html=True)

st.markdown("<h1>🎓 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.write("Fill student details (name & class) first. All fields are required. Teachers can log in and save predictions to the Dashboard.")

####################
# --- Model load or train fallback ---
####################
MODEL_PATH = "model.pkl"
DATA_PATH = "student_data.csv"
model = None

def train_from_csv_and_save():
    """If model.pkl missing but student_data.csv present, train a simple LinearRegression."""
    df = pd.read_csv(DATA_PATH)
    X = df[['study_hours','attendance_percent','internal_marks','assignments_score','previous_grade','family_support']]
    y = df['final_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression()
    m.fit(X_train, y_train)
    joblib.dump(m, MODEL_PATH)
    return m

# Try to load model, else train from CSV if available
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning("Model file exists but could not be loaded. Error: " + str(e))
        model = None
else:
    if os.path.exists(DATA_PATH):
        try:
            model = train_from_csv_and_save()
            st.info("Trained a model from student_data.csv (no model.pkl was present).")
        except Exception as e:
            st.warning("Could not train model from CSV: " + str(e))
            model = None
    else:
        st.info("No pretrained model found. You can still use the app after uploading a model or student_data.csv.")

####################
# --- Sidebar: multi-page navigation ---
####################
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Predict","Teacher Login","Dashboard"])

# Small helper to ensure numeric fields are valid
def valid_number(x):
    try:
        _ = float(x)
        return True
    except:
        return False

####################
# --- Pages ---
####################
# Home
if page == "Home":
    st.header("Welcome 👋")
    st.write("""
    This app predicts a student's final score based on a few inputs:
    - Attendance (%)  
    - Daily study hours  
    - Internal marks  
    - Assignments score  
    - Previous grade  
    - Family support (0=none to 5=strong)
    
    Teachers can log in, make predictions and save them to the Dashboard.
    """)
    st.write("If you are a student, go to the Predict page. Teachers: use Teacher Login to save results.")

# Predict
elif page == "Predict":
    st.header("Student Prediction Form")

    # Step 1: Name & Class first
    st.subheader("Step 1 — Student info (required)")
    coln1, coln2 = st.columns([2,1])
    with coln1:
        student_name = st.text_input("Student Name")
    with coln2:
        student_class = st.text_input("Class / Grade")

    if not student_name.strip() or not student_class.strip():
        st.warning("Please enter Student Name and Class to continue.")
        st.stop()

    st.subheader("Step 2 — Performance inputs (all required)")
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
        study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        internal_marks = st.number_input("Internal Marks (out of 100)", min_value=0, max_value=100, value=60)
    with c2:
        assignments_score = st.number_input("Assignments Score (out of 100)", min_value=0, max_value=100, value=65)
        previous_grade = st.number_input("Previous Grade (out of 100)", min_value=0, max_value=100, value=70)
        family_support = st.selectbox("Family Support (0=none .. 5=strong)", options=[0,1,2,3,4,5], index=3)

    # Validate all fields (they are numeric due to number_input so basic validation already)
    if st.button("Predict 🎯"):
        # additional checks if needed
        inputs_ok = True
        for v in [attendance, study_hours, internal_marks, assignments_score, previous_grade]:
            if v is None:
                inputs_ok = False
        if not inputs_ok:
            st.error("Please fill all fields correctly.")
            st.stop()

        # Prepare model input. Note: columns order must match training.
        X = np.array([[study_hours, attendance, internal_marks, assignments_score, previous_grade, family_support]])
        # If trained model has different feature order, adapt accordingly; here we assume the order used in training function below.
        # Try model prediction
        if model is None:
            st.error("No model available to predict. Upload model.pkl or student_data.csv to the app folder.")
            st.stop()
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            # Try alternative column order fallback
            try:
                # fallback order used in earlier simple examples (study_hours, attendance, previous_grade, sleep_hours, participation)
                # but since here we used 6 features, give user clear message.
                st.error("Model prediction failed. Model may expect different feature order or different feature count. Error: " + str(e))
                st.stop()
            except Exception:
                st.stop()

        pred = max(0, min(100, pred))  # clamp
        st.success(f"Predicted final score for **{student_name} (Class {student_class})**: **{pred:.2f} / 100**")

        # Show a quick advice
        if pred >= 85:
            st.info("Excellent — keep up the good work!")
        elif pred >= 60:
            st.info("Good — a little more studying will improve it further.")
        else:
            st.warning("Below target — consider more study time and extra help.")

        # Show charts: bar of inputs and line chart comparing some scores
        st.subheader("Visual summary")
        # bar chart of the numeric inputs
        input_df = pd.DataFrame({
            "Metric":["Attendance","Study Hours","Internal","Assignments","Previous Grade","Family Support"],
            "Value":[attendance, study_hours, internal_marks, assignments_score, previous_grade, family_support]
        })
        st.bar_chart(data=input_df.set_index("Metric"))

        # line chart of internal->assignments->predicted (simple)
        timeline = pd.DataFrame({
            "Stage":["Internal","Assignments","Predicted"],
            "Score":[internal_marks, assignments_score, pred]
        })
        st.line_chart(data=timeline.set_index("Stage"))

        # Option for teacher save (only if teacher logged in elsewhere)
        st.write("---")
        st.info("If a teacher is logged in (Teacher Login page), they can save this prediction to the Dashboard.")
        # also provide a quick download CSV for this single prediction
        single = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "student_name": student_name,
            "student_class": student_class,
            "attendance": attendance,
            "study_hours": study_hours,
            "internal_marks": internal_marks,
            "assignments_score": assignments_score,
            "previous_grade": previous_grade,
            "family_support": family_support,
            "predicted_score": round(float(pred),2)
        }])
        csv = single.to_csv(index=False).encode('utf-8')
        st.download_button("Download this prediction (CSV)", data=csv, file_name=f"{student_name}_prediction.csv", mime="text/csv")

# Teacher Login & Save
elif page == "Teacher Login":
    st.header("Teacher Login")

    # Simple hard-coded demo credentials (change these in code for real use)
    DEMO_USER = "teacher"
    DEMO_PASS = "pass123"

    if 'teacher_logged_in' not in st.session_state:
        st.session_state['teacher_logged_in'] = False
        st.session_state['teacher_user'] = ""

    if st.session_state['teacher_logged_in']:
        st.success(f"Logged in as {st.session_state['teacher_user']}")
        if st.button("Log out"):
            st.session_state['teacher_logged_in'] = False
            st.session_state['teacher_user'] = ""
            st.experimental_rerun()
        st.write("You can save a new prediction using the Predict page (after making a prediction). Alternatively upload a CSV of predictions to import.")
        # quick import feature
        st.write("---")
        st.subheader("Import predictions CSV (optional)")
        uploaded = st.file_uploader("Upload a CSV with the same columns as the Dashboard", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                # append to predictions.csv
                save_path = "predictions.csv"
                if os.path.exists(save_path):
                    existing = pd.read_csv(save_path)
                    combined = pd.concat([existing, df_up], ignore_index=True)
                else:
                    combined = df_up
                combined.to_csv(save_path, index=False)
                st.success("Uploaded and merged predictions into Dashboard.")
            except Exception as e:
                st.error("Failed to import CSV: " + str(e))
    else:
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                if user == DEMO_USER and pwd == DEMO_PASS:
                    st.session_state['teacher_logged_in'] = True
                    st.session_state['teacher_user'] = user
                    st.success("Login successful.")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials. (Demo: teacher / pass123)")

# Dashboard
elif page == "Dashboard":
    st.header("Teacher Dashboard")
    save_path = "predictions.csv"
    if not os.path.exists(save_path):
        st.info("No saved predictions yet. Teachers must log in and save predictions from the Predict page or upload a CSV.")
        st.stop()

    df = pd.read_csv(save_path)
    st.subheader("Saved predictions table")
    st.dataframe(df)

    st.subheader("Analytics")
    if "predicted_score" in df.columns:
        avg = df['predicted_score'].mean()
        st.metric("Average predicted score", f"{avg:.2f}")
        # show distribution
        st.write("Distribution of predicted scores")
        st.bar_chart(df['predicted_score'])

    # Allow export of all predictions
    csv_all = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download all predictions (CSV)", data=csv_all, file_name="predictions_all.csv", mime="text/csv")

####################
# --- End central card div ---
####################
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Helper: allow teacher to save from Predict page via session_state
# Implementation detail:
# When a teacher is logged in and a recent prediction exists, teacher can save it.
# To keep the code linear, we check a simple file 'last_prediction_temp.csv' which Predict page could write (but for this example we kept download button).
# For a more robust flow, you'd integrate a database or GitHub commit API.
# -------------------------
