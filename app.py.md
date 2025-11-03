import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Trained Model ---
# Load the model file created by 'model_training.py'
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully")
except FileNotFoundError:
    st.error("Model file ('model.pkl') not found. Please run 'model_training.py' first.")
    st.stop()

# --- 2. Set Up Streamlit Page ---
st.set_page_config(page_title="Vital.AI", page_icon="🩺", layout="wide")

st.title("🩺 Vital.AI: Intelligent Clinical Decision Support")
st.markdown("Transforming complex patient data into rapid, actionable diagnostic insights. [cite: 2]")

st.sidebar.header("Navigation")
# You can add more pages here later, like in your demo screenshot 
st.sidebar.markdown("Main User")
st.sidebar.info("This prototype focuses on the core AI diagnostic engine for **Type 2 Diabetes** prediction. ")

# --- 3. Create Input Form for Patient Data ---
st.header("Patient Biometrics Input [cite: 17]")
st.write("Enter the patient's data to get an AI-driven diagnostic insight.")

# Create columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    preg = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    plas = st.number_input('Glucose', min_value=0, max_value=200, value=120) # Key feature 
    pres = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70)

with col2:
    skin = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    test = st.number_input('Insulin', min_value=0, max_value=900, value=80)
    mass = st.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=70.0, value=32.5, format="%.1f") # Key feature 

with col3:
    pedi = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.45, format="%.3f")
    age = st.number_input('Age', min_value=0, max_value=120, value=45) # Key feature 

# --- 4. Prediction and XAI Logic ---
if st.button('Get Diagnostic Insight', type="primary"):
    
    # Put input data into a DataFrame (model expects this format)
    input_data = pd.DataFrame(
        [[preg, plas, pres, skin, test, mass, pedi, age]],
        columns=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
    )
    
    # Generate prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.divider()
    
    # --- 5. Display Results ---
    st.header("Diagnostic Insight [cite: 25]")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.subheader("Predicted Diagnosis")
        if prediction[0] == 1:
            st.error(f"**Predicted: High Risk (Type 2 Diabetes)** ")
            st.write(f"Confidence: **{prediction_proba[0][1] * 100:.1f}%**")
        else:
            st.success("**Predicted: Low Risk (No Diabetes)**")
            st.write(f"Confidence: **{prediction_proba[0][0] * 100:.1f}%**")
        
        st.write("This AI Clinical Co-Pilot provides data-driven decision support, not a replacement for clinical expertise. [cite: 15, 19]")

    # --- 6. Explainable AI (XAI) Feature ---
    # This is the "Feature Importance" part from your PPT [cite: 74]
    with col_res2:
        st.subheader("Explainable AI (XAI): Key Contributing Factors ")
        st.write("This chart shows the factors the AI considered most important *overall* for its decision-making. [cite: 69]")
        
        # Get feature importances from the model
        importances = model.feature_importances_
        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Pedigree Func.', 'Age']
        
        # Create a DataFrame for easy plotting
        xai_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        xai_df = xai_df.sort_values(by='Importance', ascending=False)
        
        # Use Streamlit's built-in bar chart
        st.bar_chart(xai_df.set_index('Feature'))

        st.caption(f"As shown, **{xai_df.iloc[0]['Feature']}**, **{xai_df.iloc[1]['Feature']}**, and **{xai_df.iloc[2]['Feature']}** are the most critical factors for this model.")