import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import os
from datetime import datetime

# -----------------------------
# PAGE CONFIG AND BACKGROUND
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #FFFACD; /* Light Yellow */
    color: black;
}
input, select {
    color: black !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# FUNCTION: REMEDIAL SUGGESTIONS
# -----------------------------
def get_remedial_suggestions(score):
    if score >= 60:
        return ""
    return """
### 🔧 Recommended Remedial Measures
- Increase study hours gradually by 1–2 hrs/day  
- Improve attendance (target 85% minimum)  
- Complete all assignments on time  
- Request doubt-clearing sessions with teachers  
- Practice chapter-wise mock tests  
"""

# -----------------------------
# FUNCTION: GENERATE PDF REPORT
# -----------------------------
def generate_pdf(name, class_name, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="Student Performance Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Class: {class_name}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Score: {prediction}", ln=True)

    if prediction < 60:
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt="Remedial Suggestions:\n- Increase study hours\n- Improve attendance\n- Complete assignments\n- Take extra help sessions")

    filename = f"{name}_report.pdf"
    pdf.output(filename)

    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📄 Download PDF Report</a>'

# -----------------------------
# TEACHER LOGIN SYSTEM
# -----------------------------
def teacher_login():
    st.subheader("👨‍🏫 Teacher Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "teacher" and password == "admin123":
            st.session_state["teacher_logged_in"] = True
            st.success("Logged in successfully")
        else:
            st.error("Invalid login details")

# -----------------------------
# MAIN STUDENT PREDICTION FORM
# -----------------------------
def student_prediction():
    st.title("🎓 Student Performance Predictor")

    col1, col2 = st.columns([2, 2])

    with col1:
        st.image(
            "https://cdn.pixabay.com/photo/2017/01/31/20/26/school-2026771_1280.png",
            width=350
        )

    with col2:
        st.image(
            "https://cdn.pixabay.com/photo/2021/02/25/14/59/students-6049319_1280.png",
            width=350
        )

    st.subheader("Enter Student Details")

    name = st.text_input("Student Name", placeholder="Enter full name")
    class_name = st.text_input("Class", placeholder="e.g., 10th A")

    st.subheader("Enter Academic Inputs")

    study_hours = st.number_input("Daily Study Hours", min_value=0.0, max_value=10.0)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
    internal_marks = st.number_input("Internal Assessment Marks (out of 20)", min_value=0, max_value=20)
    assignment_score = st.number_input("Assignment Score (out of 10)", min_value=0, max_value=10)
    previous_grade = st.number_input("Previous Grade Marks (out of 100)", min_value=0, max_value=100)

    if st.button("Predict Score"):
        if name == "" or class_name == "":
            st.error("Please fill all fields!")
        else:
            input_data = [[study_hours, attendance, internal_marks, assignment_score, previous_grade]]
            prediction = model.predict(input_data)[0]
            st.success(f"📘 Predicted Final Score: **{prediction:.2f}**")

            # Balloons for good performance
            if prediction > 80:
                st.balloons()

            # Show remedial measures
            if prediction < 60:
                st.warning(get_remedial_suggestions(prediction))

            # -----------------------------
            # CHART
            # -----------------------------
            st.subheader("📊 Performance Visualization")

            labels = ["Study Hours", "Attendance", "Internal Marks", "Assignment Score", "Previous Grade"]
            values = [study_hours, attendance, internal_marks, assignment_score, previous_grade]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            st.pyplot(fig)

            # -----------------------------
            # PDF REPORT
            # -----------------------------
            st.markdown("---")
            st.subheader("📄 Download Report")
            pdf_link = generate_pdf(name, class_name, prediction)
            st.markdown(pdf_link, unsafe_allow_html=True)

            # SAVE TO CSV (Dashboard)
            save_data = {
                "Timestamp": datetime.now(),
                "Name": name,
                "Class": class_name,
                "Predicted Score": prediction
            }

            df_new = pd.DataFrame([save_data])

            if os.path.exists("saved_predictions.csv"):
                df_existing = pd.read_csv("saved_predictions.csv")
                df_all = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_all = df_new

            df_all.to_csv("saved_predictions.csv", index=False)

# -----------------------------
# DASHBOARD FOR TEACHERS
# -----------------------------
def dashboard():
    st.title("📚 Teacher Dashboard")

    if os.path.exists("saved_predictions.csv"):
        df = pd.read_csv("saved_predictions.csv")
        st.dataframe(df)
    else:
        st.info("No predictions saved yet.")

# -----------------------------
# MULTIPAGE NAVIGATION
# -----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Student Prediction", "Teacher Login", "Teacher Dashboard"]
)

if menu == "Student Prediction":
    student_prediction()

elif menu == "Teacher Login":
    teacher_login()

elif menu == "Teacher Dashboard":
    if st.session_state.get("teacher_logged_in", False):
        dashboard()
    else:
        st.error("You must login as teacher first!")
