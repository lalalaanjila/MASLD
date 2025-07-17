import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature names
feature_names = [    
    "Age", "Sex", "Strength", "Sleep", "Light", "Moderate"]

# Streamlit user interface
st.title("Metabolic dysfunction-associated steatotic liver disease risk prediction calculator")

# age: numerical input
Age = st.number_input("Age:", min_value=40, max_value=70, value=60) 

# Sex: categorical selection
Sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

Strength = st.number_input("Hand grip strength (Max):", min_value=0, max_value=90, value=70) 

Sleep = st.number_input("Average sleep duration:", min_value=0.0, max_value=1.0, value=0.450)

Light = st.number_input("Average light duration:", min_value=0.0, max_value=1.0, value=0.250)

Moderate = st.number_input("Average moderate-vigorous duration:", min_value=0.0, max_value=1.0, value=0.055)

# Process inputs and make predictions
feature_values = [Age, Sex, Strength, Sleep, Light, Moderate]
features = np.array([feature_values])

if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:        
         advice = (            
           f"Our model indicates that you have a high risk of MASLD. "
           f"The predicted probability of developing MASLD is {probability:.1f}%. "
           "While this is an estimate, it suggests that you may be at considerable risk. "
           "I recommend consulting a hepatologist or a liver specialist as soon as possible for further evaluation, "
           "to confirm the diagnosis and discuss appropriate management or treatment options."       
           )
    else:        
         advice = (            
           f"Our model indicates that you have a low risk of MASLD. "
           f"The predicted probability of not having MASLD is {probability:.1f}%. "
           "However, it is still important to maintain a healthy lifestyle to support liver health. "
           "Regular monitoring and periodic check-ups are recommended, especially if you have risk factors such as obesity, diabetes, or a history of liver disease."
           )
    st.write(advice)

    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200) 

    st.image("shap_force_plot.png")