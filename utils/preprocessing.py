import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input_data(input_data):
    """
    Preprocess input data for the model.
    
    Parameters:
    - input_data: Dictionary or DataFrame containing raw input data
    
    Returns:
    - Preprocessed numpy array ready for model prediction
    """
    # Convert dictionary to DataFrame if needed
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Feature engineering (example)
    
    # 1. Age categorization
    if 'age' in input_data.columns:
        input_data['age_group'] = pd.cut(
            input_data['age'], 
            bins=[0, 18, 35, 50, 65, 80, 120],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(int)
    
    # 2. BMI calculation (if height and weight are present)
    if 'height' in input_data.columns and 'weight' in input_data.columns:
        # Height in meters, weight in kg
        input_data['bmi'] = input_data['weight'] / ((input_data['height']/100) ** 2)
    
    # 3. Blood pressure categorization
    if 'blood_pressure' in input_data.columns:
        conditions = [
            (input_data['blood_pressure'] < 120),
            (input_data['blood_pressure'] >= 120) & (input_data['blood_pressure'] < 130),
            (input_data['blood_pressure'] >= 130) & (input_data['blood_pressure'] < 140),
            (input_data['blood_pressure'] >= 140) & (input_data['blood_pressure'] < 180),
            (input_data['blood_pressure'] >= 180)
        ]
        values = [0, 1, 2, 3, 4]  # Normal, Elevated, Stage 1, Stage 2, Crisis
        input_data['bp_category'] = np.select(conditions, values, default=0)
    
    # 4. One-hot encoding for categorical variables
    categorical_cols = ['gender']
    for col in categorical_cols:
        if col in input_data.columns:
            # Convert string values to binary
            if col == 'gender':
                input_data[col] = (input_data[col] == 'Male').astype(int)
    
    # 5. Select features used by the model (adjust based on your model)
    features = [
        'age', 'gender', 'blood_pressure', 'blood_glucose', 'blood_urea',
        'serum_creatinine', 'sodium', 'potassium', 'hemoglobin', 'albumin'
    ]
    
    # Add medical history columns
    binary_cols = ['diabetes', 'hypertension', 'coronary_artery_disease']
    for col in binary_cols:
        if col in input_data.columns:
            features.append(col)
    
    # Ensure all required features are present
    for feature in features:
        if feature not in input_data.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Extract features
    X = input_data[features].values
    
    # Normalization (if required by model)
    # This should be consistent with your training preprocessing
    scaler = get_scaler()
    X_scaled = scaler.transform(X)
    
    return X_scaled

def get_scaler():
    """
    Returns a pre-trained scaler for feature normalization.
    
    In a real application, this would load a saved scaler that was
    trained on the training data.
    """
    # For demonstration, return a dummy scaler
    # In a real app, you would load a saved scaler:
    # scaler = joblib.load('model/scaler.joblib')
    
    # Dummy means and std values (replace with actual values from training)
    means = [50, 0.5, 120, 100, 30, 1.0, 135, 4.0, 14.0, 0, 0, 0, 0]
    stds = [20, 0.5, 20, 30, 10, 0.5, 5, 0.5, 2.0, 1, 1, 1, 1]
    
    scaler = StandardScaler()
    # Set scaler parameters manually
    scaler.mean_ = np.array(means)
    scaler.scale_ = np.array(stds)
    scaler.var_ = np.array(stds) ** 2
    scaler.n_features_in_ = len(means)
    
    return scaler

def validate_input_data(input_data):
    """
    Validates input data for completeness and range checking.
    
    Parameters:
    - input_data: Dictionary containing input data
    
    Returns:
    - tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = [
        'age', 'gender', 'blood_pressure', 'blood_glucose', 
        'blood_urea', 'serum_creatinine'
    ]
    
    # Check for missing fields
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # If missing required fields, return early
    if errors:
        return False, errors
    
    # Range validations
    if input_data['age'] < 18 or input_data['age'] > 100:
        errors.append("Age must be between 18 and 100")
        
    if input_data['blood_pressure'] < 80 or input_data['blood_pressure'] > 200:
        errors.append("Blood pressure must be between 80 and 200 mmHg")
        
    if input_data['blood_glucose'] < 70 or input_data['blood_glucose'] > 300:
        errors.append("Blood glucose must be between 70 and 300 mg/dL")
        
    if input_data['blood_urea'] < 10 or input_data['blood_urea'] > 100:
        errors.append("Blood urea must be between 10 and 100 mg/dL")
        
    if input_data['serum_creatinine'] < 0.5 or input_data['serum_creatinine'] > 5.0:
        errors.append("Serum creatinine must be between 0.5 and 5.0 mg/dL")
    
    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors 