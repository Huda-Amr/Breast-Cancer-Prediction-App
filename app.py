import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

st.title("Breast Cancer Prediction App 🎯")
st.write("Enter the feature values to let the model predict whether it's Benign (0) or Malignant (1).")

# Feature names from the Breast Cancer Wisconsin dataset
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
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


