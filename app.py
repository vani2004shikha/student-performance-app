import streamlit as st
import joblib
import numpy as np

# ---------------------- PAGE DESIGN ----------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Background Image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("https://images.unsplash.com/photo-1523050854058-8df90110c9f1");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

.block-container {{
backdrop-filter: blur(6px);
background: rgba(255,255,255,0.8);
padding: 30px;
border-radius: 15px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --------------------------------------------------------------------

st.title("🎓 Student Performance Prediction App")
st.write("Fill the details step-by-step to predict your performance.")

# Load Model
model = joblib.load("model.pkl")

# ---------------------- STEP 1: BASIC STUDENT INFO ----------------------

st.subheader("🧑‍🎓 Step 1: Student Details")

name = st.text_input("Enter Student Name")
class_name = st.text_input("Enter Class")

if not name or not class_name:
    st.warning("➡ Please fill the student name and class to continue.")
    st.stop()

# ---------------------- STEP 2: PERFORMANCE INPUTS ----------------------

st.subheader("📚 Step 2: Enter Study & Lifestyle Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("Daily Study Hours", 1.0, 12.0, 3.0)

with col2:
    attendance = st.number_input("Attendance (%)", 1, 100, 75)

prev_grade = st.number_input("Previous Grade", 0, 100, 70)
sleep_hours = st.number_input("Sleep Hours per Day", 1.0, 12.0, 7.0)

participation = st.selectbox("Class Participation Level", [0, 1, 2, 3, 4, 5])

# Validation
if st.button("Predict Performance 🎯"):

    # check all fields
    if (not study_hours) or (not attendance) or (not prev_grade) or (not sleep_hours):
        st.error("All fields are compulsory. Please fill all details.")
        st.stop()

    # Convert to array for model
    inputs = np.array([[study_hours, attendance, prev_grade, sleep_hours, participation]])
    result = model.predict(inputs)[0]

    st.success(f"📘 **{name} (Class {class_name})'s Predicted Score: {result:.2f}**")
    st.balloons()

