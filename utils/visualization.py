import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_risk_gauge(risk_score):
    """
    Create a gauge chart to visualize risk score.
    
    Parameters:
    - risk_score: Float between 0 and 1
    
    Returns:
    - Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        title={'text': "Kidney Disease Risk"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
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
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def create_risk_factors_chart(risk_factors):
    """
    Create a horizontal bar chart for risk factors.
    
    Parameters:
    - risk_factors: Dictionary with factors as keys and contribution values as values
    
    Returns:
    - Plotly figure object
    """
    # Convert to DataFrame and sort
    factors_df = pd.DataFrame({
        'Factor': list(risk_factors.keys()),
        'Contribution': list(risk_factors.values())
    })
    
    factors_df = factors_df.sort_values('Contribution', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        factors_df, 
        x='Contribution', 
        y='Factor', 
        orientation='h',
        color='Contribution',
        color_continuous_scale=['green', 'yellow', 'red']
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Contribution to Risk Score",
        yaxis_title=None,
        coloraxis_showscale=False
    )
    
    return fig

def create_age_distribution_chart(sample_size=120):
    """
    Create a sample age distribution chart for demonstration purposes.
    
    Parameters:
    - sample_size: Number of samples to generate
    
    Returns:
    - Plotly figure object
    """
    # Generate sample data
    np.random.seed(42)
    sample_data = {
        'Age Group': ['25-34', '35-44', '45-54', '55-64', '65-74', '75+'] * (sample_size // 6),
        'Risk Score': np.random.beta(2, 5, sample_size) * 100
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create visualization
    fig = px.box(
        df, 
        x='Age Group', 
        y='Risk Score', 
        color='Age Group',
        title='Kidney Disease Risk by Age Group (Sample Data)'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=None,
        yaxis_title="Risk Score (%)",
    )
    
    return fig

def calculate_risk_factors(patient_data):
    """
    Calculate risk factor contributions based on patient data.
    
    Parameters:
    - patient_data: Dictionary of patient information
    
    Returns:
    - Dictionary of risk factors and their contributions
    """
    risk_factors = {}
    
    # Age contribution
    age = patient_data.get('age', 0)
    if age < 30:
        risk_factors["Age"] = 0.05
    elif age < 45:
        risk_factors["Age"] = 0.1
    elif age < 60:
        risk_factors["Age"] = 0.15
    else:
        risk_factors["Age"] = 0.2
    
    # Blood pressure contribution
    bp = patient_data.get('blood_pressure', 0)
    if bp < 120:
        risk_factors["Blood Pressure"] = 0.05
    elif bp < 130:
        risk_factors["Blood Pressure"] = 0.1
    elif bp < 140:
        risk_factors["Blood Pressure"] = 0.2
    else:
        risk_factors["Blood Pressure"] = 0.3
    
    # Blood glucose contribution
    glucose = patient_data.get('blood_glucose', 0)
    if glucose < 100:
        risk_factors["Blood Glucose"] = 0.05
    elif glucose < 120:
        risk_factors["Blood Glucose"] = 0.1
    elif glucose < 150:
        risk_factors["Blood Glucose"] = 0.2
    else:
        risk_factors["Blood Glucose"] = 0.25
    
    # Serum creatinine contribution
    creatinine = patient_data.get('serum_creatinine', 0)
    if creatinine < 0.8:
        risk_factors["Serum Creatinine"] = 0.05
    elif creatinine < 1.2:
        risk_factors["Serum Creatinine"] = 0.1
    elif creatinine < 1.5:
        risk_factors["Serum Creatinine"] = 0.2
    else:
        risk_factors["Serum Creatinine"] = 0.4
    
    # Hemoglobin contribution
    hemoglobin = patient_data.get('hemoglobin', 0)
    if hemoglobin > 14:
        risk_factors["Hemoglobin"] = 0.05
    elif hemoglobin > 12:
        risk_factors["Hemoglobin"] = 0.1
    else:
        risk_factors["Hemoglobin"] = 0.2
    
    # Albumin contribution
    albumin = patient_data.get('albumin', 0)
    risk_factors["Albumin"] = 0.1 * albumin
    
    # Medical history contributions
    if patient_data.get('diabetes', False):
        risk_factors["Diabetes"] = 0.3
    if patient_data.get('hypertension', False):
        risk_factors["Hypertension"] = 0.25
    if patient_data.get('coronary_artery_disease', False):
        risk_factors["CAD"] = 0.2
    
    return risk_factors 