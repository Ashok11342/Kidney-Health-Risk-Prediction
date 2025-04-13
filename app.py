import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import model loader
try:
    from model.model_loader import load_model, get_model_summary
except Exception as e:
    st.error(f"Error importing model loader: {e}")
    logging.error(f"Error importing model loader: {e}")

# Page config
st.set_page_config(
    page_title="Kidney Health Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and introduction
st.title("Kidney Health Risk Prediction")
st.markdown("""
This application uses machine learning to predict kidney health risks based on your medical data.
Enter your information below to get a personalized risk assessment.
""")

# Load model (in a way that it's only loaded once)
@st.cache_resource
def get_model():
    """Load the model or return None if it fails"""
    try:
        return load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")
        return None

# Try to load the model
model = get_model()

# Demo mode flag when model is not available
demo_mode = model is None
if demo_mode:
    st.warning("⚠️ Running in DEMO MODE - The model could not be loaded. Using simulated predictions.")

# Create sidebar for inputs
st.sidebar.header("Patient Information")

# Input form
with st.sidebar.form("patient_info_form"):
    # Basic information
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Health parameters
    st.subheader("Health Parameters")
    blood_pressure = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
    blood_glucose = st.slider("Blood Glucose (mg/dL)", 70, 300, 100)
    blood_urea = st.slider("Blood Urea (mg/dL)", 10, 100, 30)
    serum_creatinine = st.slider("Serum Creatinine (mg/dL)", 0.5, 5.0, 1.0, step=0.1)
    sodium = st.slider("Sodium (mEq/L)", 125, 150, 135)
    potassium = st.slider("Potassium (mEq/L)", 2.5, 7.0, 4.0, step=0.1)
    hemoglobin = st.slider("Hemoglobin (g/dL)", 8.0, 18.0, 14.0, step=0.1)
    albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    
    # Medical history
    st.subheader("Medical History")
    diabetes = st.checkbox("Diabetes")
    hypertension = st.checkbox("Hypertension")
    coronary_artery_disease = st.checkbox("Coronary Artery Disease")
    
    # Submit button
    submitted = st.form_submit_button("Predict Risk")

# Main content area
if submitted:
    st.header("Risk Assessment Results")
    
    # Prepare data for prediction
    # Note: This is a placeholder. Adjust according to your actual model input requirements
    features = np.array([[
        age, 
        1 if gender == "Male" else 0, 
        blood_pressure, 
        blood_glucose, 
        blood_urea, 
        serum_creatinine, 
        sodium, 
        potassium, 
        hemoglobin,
        albumin,
        1 if diabetes else 0,
        1 if hypertension else 0,
        1 if coronary_artery_disease else 0
    ]])
    
    # Track patient data for visualization
    patient_data = {
        'age': age,
        'gender': gender,
        'blood_pressure': blood_pressure,
        'blood_glucose': blood_glucose,
        'blood_urea': blood_urea,
        'serum_creatinine': serum_creatinine,
        'sodium': sodium,
        'potassium': potassium,
        'hemoglobin': hemoglobin,
        'albumin': albumin,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'coronary_artery_disease': coronary_artery_disease
    }
    
    # Use model for prediction if available, otherwise use demo calculations
    with st.spinner("Analyzing your data..."):
        try:
            # Make prediction
            if not demo_mode and model is not None:
                prediction = model.predict(features)
                risk_score = prediction[0][0]  # Adjust based on your model output
            else:
                # Demo mode - calculate a simulated risk score based on input values
                base_risk = 0.2
                
                # Age factor (higher age = higher risk)
                age_factor = min(1.0, age / 100) * 0.3
                
                # Medical conditions
                conditions_factor = 0
                if diabetes:
                    conditions_factor += 0.15
                if hypertension:
                    conditions_factor += 0.15
                if coronary_artery_disease:
                    conditions_factor += 0.1
                
                # Lab values
                lab_factor = 0
                if blood_pressure > 140:
                    lab_factor += 0.1
                if blood_glucose > 120:
                    lab_factor += 0.1
                if serum_creatinine > 1.2:
                    lab_factor += 0.15
                if albumin > 0:
                    lab_factor += 0.05 * albumin
                
                # Calculate final simulated risk
                risk_score = min(0.95, base_risk + age_factor + conditions_factor + lab_factor)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score * 100,
                    title = {'text': "Kidney Disease Risk"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score * 100
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk interpretation
                if risk_score < 0.3:
                    risk_level = "Low"
                    st.success("Your kidney disease risk is relatively low.")
                elif risk_score < 0.7:
                    risk_level = "Moderate"
                    st.warning("Your kidney disease risk is moderate. Consider consulting with a healthcare provider.")
                else:
                    risk_level = "High"
                    st.error("Your kidney disease risk is high. Please consult with a healthcare provider as soon as possible.")
            
            with col2:
                # Contributing factors chart
                st.subheader("Contributing Risk Factors")
                
                # This is a simplified example. In a real app, you would have model-specific logic
                # to determine the contribution of each factor
                risk_factors = {
                    "Age": 0.2 if age > 60 else 0.1,
                    "Blood Pressure": 0.3 if blood_pressure > 140 else 0.1,
                    "Blood Glucose": 0.25 if blood_glucose > 120 else 0.05,
                    "Serum Creatinine": 0.4 if serum_creatinine > 1.2 else 0.1,
                    "Hemoglobin": 0.2 if hemoglobin < 12 else 0.05,
                    "Albumin": 0.1 * albumin
                }
                
                # Add medical history factors
                if diabetes:
                    risk_factors["Diabetes"] = 0.3
                if hypertension:
                    risk_factors["Hypertension"] = 0.25
                if coronary_artery_disease:
                    risk_factors["CAD"] = 0.2
                
                # Create bar chart
                factors_df = pd.DataFrame({
                    'Factor': list(risk_factors.keys()),
                    'Contribution': list(risk_factors.values())
                })
                
                factors_df = factors_df.sort_values('Contribution', ascending=False)
                
                fig = px.bar(
                    factors_df, 
                    x='Contribution', 
                    y='Factor', 
                    orientation='h',
                    color='Contribution',
                    color_continuous_scale=['green', 'yellow', 'red']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Recommendations
            st.subheader("Health Recommendations")
            st.write("Based on your risk assessment, here are some recommendations:")
            
            recommendations = []
            
            if blood_pressure > 130:
                recommendations.append("Work on lowering your blood pressure through diet, exercise, and possibly medication.")
            if blood_glucose > 110:
                recommendations.append("Monitor your blood glucose levels and consider consulting with a healthcare provider.")
            if serum_creatinine > 1.2:
                recommendations.append("Your creatinine level is elevated, which may indicate reduced kidney function.")
            if albumin > 0:
                recommendations.append("The presence of albumin in urine could indicate kidney damage.")
            
            # Add general recommendations
            recommendations.append("Stay hydrated by drinking plenty of water throughout the day.")
            recommendations.append("Maintain a balanced, kidney-friendly diet low in sodium.")
            recommendations.append("Exercise regularly to maintain a healthy weight and improve overall health.")
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Disclaimer
            st.info("Disclaimer: This tool provides an estimate based on the information provided and should not be considered a medical diagnosis. Always consult with a healthcare professional for proper medical advice.")
            
            # Show demo mode notice again if active
            if demo_mode:
                st.warning("⚠️ DEMO MODE: This is a simulated result as the model could not be loaded.")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logging.error(f"Error during prediction: {e}")
else:
    # Initial state or after resetting
    st.info("Enter your information in the sidebar and click 'Predict Risk' to get your assessment.")
    
    # Display information about the project
    st.header("About This Project")
    st.write("""
    This application uses machine learning to predict kidney health risks based on various health parameters.
    The model was trained on medical data and can provide an estimate of kidney disease risk.
    
    ### How It Works
    
    1. Enter your health information in the sidebar
    2. Our machine learning model analyzes your data
    3. View your risk assessment and personalized recommendations
    
    ### Key Health Indicators
    
    * **Blood Pressure**: High blood pressure can damage kidneys over time
    * **Blood Glucose**: Diabetes is a leading cause of kidney disease
    * **Serum Creatinine**: An important indicator of kidney function
    * **Albumin**: Protein in urine that can indicate kidney damage
    """)
    
    # Sample visualization
    st.subheader("Sample Risk Distribution")
    
    # Generate sample data
    np.random.seed(42)
    sample_data = {
        'Age Group': ['25-34', '35-44', '45-54', '55-64', '65-74', '75+'] * 20,
        'Risk Score': np.random.beta(2, 5, 120) * 100
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create visualization
    fig = px.box(df, x='Age Group', y='Risk Score', color='Age Group',
                title='Kidney Disease Risk by Age Group (Sample Data)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Note: The above visualization uses synthetic data for illustration purposes only.")

# Footer
st.markdown("---")
st.markdown("Kidney Health Risk Prediction | Created by Ashok | [GitHub Repository](https://github.com/Ashok11342/Kidney-Health-Risk-Prediction)") 