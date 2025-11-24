import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load model
model = joblib.load("model.pkl")  # ensure correct filename

# Page config
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

# Background styling
page_style = """
<style>
body {
    background-color: #cce7ff;
}
.block-container {
    padding-top: 2rem;
}
label, h1, h2, h3, p, div {
    color: black !important;
}
.sidebar .sidebar-content {
    background-color: #a3d4ff !important;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Sidebar Images
st.sidebar.image("school_left.png", use_column_width=True)
st.sidebar.image("students_right.png", use_column_width=True)

# Login system for teachers
users = {"teacher": "password123"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Teacher Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login Successful!")
        else:
            st.error("Invalid credentials.")
    st.stop()

# Main App Title
st.title("🎓 Student Performance Prediction App")
st.write("Fill the details below.")

# Step 1: Student Details
st.header("Step 1: Student Information")
col1, col2 = st.columns(2)

with col1:
    student_name = st.text_input("Student Name", max_chars=30)
with col2:
    student_class = st.text_input("Class", max_chars=10)

# Step 2: Academic & Lifestyle Inputs
st.header("Step 2: Academic & Lifestyle Inputs")

attendance = st.slider("Attendance (%)", 0, 100, 75)
study_hours = st.number_input("Daily Study Hours", 0.0, 12.0, 3.0)
internal_marks = st.slider("Internal Marks", 0, 30, 15)
assignment_score = st.slider("Assignment Score", 0, 20, 10)
prev_grade = st.number_input("Previous Grade Marks", 0, 100, 70)
family_support = st.selectbox("Family Support Level", ["Low", "Medium", "High"])

support_mapping = {"Low": 0, "Medium": 1, "High": 2}
fs_val = support_mapping[family_support]

# Prediction
if st.button("Predict Performance"):
    data = [[study_hours, attendance, prev_grade, internal_marks, assignment_score, fs_val]]

    result = model.predict(data)[0]
    st.success(f"Predicted Final Score: {result}")

    # Remedial Suggestions
    if result < 50:
        st.warning("⚠️ The predicted score is low. Suggested Remedial Measures:")
        st.write("- Increase study hours gradually.\n- Improve attendance.\n- Seek help from teachers.\n- Follow a structured timetable.\n- Reduce distractions.")

    # Performance Chart
    st.header("📊 Performance Visualization")
    fig, ax = plt.subplots()
    ax.bar(["Predicted Score"], [result])
    st.pyplot(fig)

    # Save Data
    df = pd.DataFrame({
        "Name": [student_name],
        "Class": [student_class],
        "Attendance": [attendance],
        "Study Hours": [study_hours],
        "Internal Marks": [internal_marks],
        "Assignment Score": [assignment_score],
        "Previous Grade": [prev_grade],
        "Family Support": [family_support],
        "Prediction": [result]
    })

    df.to_csv("student_record.csv", mode="a", header=False, index=False)

    # PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Student Performance Report", ln=True, align="C")
    pdf.ln(10)
    for col in df.columns:
        pdf.cell(200, 8, txt=f"{col}: {df[col][0]}", ln=True)
    pdf.output("student_report.pdf")

    st.download_button("📄 Download Student Report", data=open("student_report.pdf", "rb").read(), file_name="student_report.pdf")
