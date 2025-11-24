import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("Student Performance Prediction App")
st.write("Enter the student's details to predict the final score.")

study_hours = st.number_input("Study Hours per Day", 0.0, 24.0, 3.0)
attendance = st.number_input("Attendance (%)", 0, 100, 75)
previous_grade = st.number_input("Previous Grade", 0, 100, 70)
sleep_hours = st.number_input("Sleep Hours per Day", 0.0, 24.0, 7.0)
participation = st.selectbox("Participation Level", [0,1,2])

if st.button("Predict Score"):
    X = np.array([[study_hours, attendance, previous_grade, sleep_hours, participation]])
    pred = model.predict(X)[0]
    st.success(f"Predicted Score: {pred:.2f}")
