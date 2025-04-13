import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set page title and config
st.set_page_config(
    page_title="Kidney Disease Predictor",
    layout="centered"
)

# App title and description
st.title("Kidney Disease Prediction")
st.write("Upload a CT scan image to predict the kidney condition")

# Define class names
CLASS_NAMES = ['cyst', 'normal', 'stone', 'tumor']

# Simple function to check if model exists
def check_model_exists():
    model_path = 'dqn_resnet_model.h5'
    return os.path.exists(model_path)

# Create model architecture (matching the original training architecture)
@st.cache_resource
def create_model():
    # First create the ResNet50 feature extractor
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Create Q-network architecture
    feature_shape = base_model.output_shape[1:]  # (7, 7, 2048)
    input_layer = Input(shape=feature_shape)
    x = GlobalAveragePooling2D()(input_layer)  # Reduces to (2048,)
    x = Dense(512, activation='relu')(x)
    q_values = Dense(len(CLASS_NAMES), activation='linear')(x)  # Q-values for each class
    q_network = Model(inputs=input_layer, outputs=q_values)
    
    # Complete model pipeline
    combined_model = Model(
        inputs=base_model.input,
        outputs=q_network(base_model.output)
    )
    
    return combined_model, base_model

# Preprocess the image
def preprocess_image(img):
    # Resize to 224x224 (the size expected by ResNet50)
    img = img.resize((224, 224))
    # Convert to array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess for ResNet50
    return preprocess_input(img_array)

# Function to make prediction
def predict_disease(img):
    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    
    # Get models
    combined_model, _ = create_model()
    
    # Check if weights file exists
    model_path = 'dqn_resnet_model.h5'
    if check_model_exists():
        try:
            # Try to load weights
            combined_model.load_weights(model_path, by_name=True, skip_mismatch=True)
        except Exception as e:
            st.warning("Using untrained model for demonstration purposes.")
    else:
        st.warning("Model file not found. Using untrained model for demonstration purposes.")
    
    # Get prediction (Q-values)
    q_values = combined_model.predict(preprocessed_img, verbose=0)[0]
    
    # Apply softmax to convert Q-values to probabilities
    probabilities = tf.nn.softmax(q_values).numpy()
    pred_class = np.argmax(probabilities)
    
    # Map index to class name
    predicted_class = CLASS_NAMES[pred_class]
    confidence = probabilities[pred_class] * 100
    
    return predicted_class, confidence, probabilities

# File uploader
uploaded_file = st.file_uploader("Choose a kidney CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_bytes = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Add a prediction button
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            # Get prediction
            prediction, confidence, probabilities = predict_disease(img)
            
            # Display result
            with col2:
                st.markdown("### Prediction Result")
                st.markdown(f"**Detected condition:** {prediction}")
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Style the prediction based on the result
                if prediction == "normal":
                    st.success(f"The kidney appears normal with {confidence:.1f}% confidence.")
                elif prediction == "cyst":
                    st.warning(f"Kidney cyst detected with {confidence:.1f}% confidence.")
                elif prediction == "stone":
                    st.warning(f"Kidney stone detected with {confidence:.1f}% confidence.")
                elif prediction == "tumor":
                    st.error(f"Kidney tumor detected with {confidence:.1f}% confidence.")
            
            # Create probability bar chart
            st.markdown("### Probability Distribution")
            
            # Create a bar chart using Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=CLASS_NAMES,
                    y=[prob * 100 for prob in probabilities],
                    marker_color=['#FF9999', '#66B2FF', '#FFCC99', '#99CC99'],
                    text=[f"{prob * 100:.1f}%" for prob in probabilities],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Prediction Confidence by Category",
                xaxis_title="Category",
                yaxis_title="Confidence (%)",
                yaxis=dict(range=[0, 100]),
                height=400,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation for medical professionals
            with st.expander("What does this mean?"):
                st.write("""
                - **Confidence score** represents how certain the model is about its prediction.
                - **Bar chart** shows the relative confidence across all possible conditions.
                - Higher bars indicate stronger prediction signals for that condition.
                - If multiple conditions show high confidence, additional testing may be advised.
                """) 