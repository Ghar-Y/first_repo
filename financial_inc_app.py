# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load("/Users/macbook/Documents/pythonDS/Financial_inclusion_model.pkl")

# Define a function to make predictions
def make_prediction(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    # Make prediction
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]  # Probability of Class 1
    return prediction[0], prob

# Streamlit App Layout
st.title("Financial Inclusion Prediction")
st.write("Predict whether an individual is likely to have a bank account based on their demographic and socio-economic features.")

# Create input fields for features
st.header("Enter the following details:")
country = st.selectbox("Country", ["Kenya", "Uganda", "Tanzania", "Rwanda"])
gender = st.selectbox("Gender", ["Male", "Female"])
relationship = st.selectbox("Relationship with Head", ["Head of Household", "Spouse", "Child", "Parent", "Other Relative", "Non-Relative"])
marital_status = st.selectbox("Marital Status", ["Single/Never Married", "Married", "Divorced", "Widowed", "Dont know"])
education_level = st.selectbox("Education Level", ["No formal education", "Primary education", "Secondary education", "Tertiary education", "Vocational/Specialised training","Other/Dont know/RTA"])
job_type = st.selectbox("Job Type", [
    "Self employed",
    "Informally employed",
    "Farming and Fishing",
    "Remittance Dependent",
    "Other Income",
    "Formally employed Private",
    "No Income",
    "Formally employed Government",
    "Government Dependent",
    "Dont Know/Refuse to answer"
])
location_type = st.selectbox("Location Type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Access", ["No", "Yes"])
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=1)
age = st.number_input("Age of Respondent", min_value=18, max_value=100, value=30)

# Map input values to the encoded format used during training
input_data = {
    "household_size": household_size,
    "age_of_respondent": age,
    "location_type": 1 if location_type == "Urban" else 0,
    "cellphone_access": 1 if cellphone_access == "Yes" else 0,
    f"country_{country}": 1,  # One-hot encoded country
    f"gender_of_respondent_{gender}": 1,
    f"relationship_with_head_{relationship}": 1,
    f"marital_status_{marital_status}": 1,
    f"education_level_{education_level}": 1,
}

# Handle One-Hot Encoding for `job_type`
job_type_categories = [
    "Self employed",
    "Informally employed",
    "Farming and Fishing",
    "Remittance Dependent",
    "Other Income",
    "Formally employed Private",
    "No Income",
    "Formally employed Government",
    "Government Dependent",
    "Dont Know/Refuse to answer"
]
input_data.update({
    f"job_type_{job_type_category}": 1 if job_type == job_type_category else 0
    for job_type_category in job_type_categories
})

# Add a validation button
if st.button("Predict"):
    # Fill missing one-hot columns with 0
    all_features = model.get_booster().feature_names
    input_data = {feature: input_data.get(feature, 0) for feature in all_features}
    
    # Make prediction
    prediction, prob = make_prediction(input_data)
    if prediction == 1:
        st.success(f"The individual is likely to have a bank account. Probability: {prob:.2f}")
    else:
        st.error(f"The individual is unlikely to have a bank account. Probability: {prob:.2f}")