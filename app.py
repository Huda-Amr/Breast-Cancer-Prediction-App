import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

st.title("Breast Cancer Prediction App 🎯")
st.write("Enter the feature values to let the model predict whether it's Benign (0) or Malignant (1).")

# Feature names from the Breast Cancer Wisconsin dataset
feature_names = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean'
]

# Collect user input for all features
inputs = []
for name in feature_names:
    value = st.number_input(f"{name}", value=0.0)
    inputs.append(value)

# Prediction button
if st.button("Predict"):
    data = np.array([inputs])
    prediction = model.predict(data)
    if prediction[0] == 0:
        st.success("✅ Prediction: Benign (non-cancerous)")
    else:
        st.error("⚠️ Prediction: Malignant (cancerous)")


