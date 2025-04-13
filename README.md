# Kidney Health Risk Prediction

## Project Overview

This project implements a machine learning model to predict kidney health risks based on medical CT scan images. Using a deep learning approach with a ResNet50 architecture and reinforcement learning techniques, the system can classify kidney scans into four categories: normal, cyst, stone, and tumor. The application provides an interactive web interface built with Streamlit that allows medical professionals to upload kidney CT scan images and receive instant predictions with confidence scores.

## Features

- **CT Scan Analysis**: Analyzes kidney CT scan images to detect abnormalities
- **Multi-class Classification**: Identifies four conditions (normal, cyst, stone, tumor)
- **Confidence Scores**: Provides prediction confidence for each category
- **Interactive UI**: User-friendly Streamlit interface for easy use
- **Visualization**: Displays probability distribution across all categories

## Repository Structure

```
Kidney-Health-Risk-Prediction/
│
├── app.py                   # Main Streamlit application
├── disease_classifer.py     # Model training script
├── dqn_resnet_model.h5      # Trained deep learning model
├── requirements.txt         # Project dependencies
├── output.txt               # Training logs
├── output while training.png # Training visualization
├── heatmap.png              # Confusion matrix visualization
└── README.md                # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/Ashok11342/Kidney-Health-Risk-Prediction.git
cd Kidney-Health-Risk-Prediction
```

### Setting Up a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts.

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Installing Dependencies

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies, including:
- Streamlit (for the web interface)
- TensorFlow (for the machine learning model)
- NumPy (for numerical operations)
- scikit-learn (for data preprocessing)
- Plotly (for interactive visualizations)
- Pillow (for image processing)


## Running the Application

After installing the dependencies, you can run the application using:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Usage Instructions

1. Once the application is running, you'll see a file uploader in the web interface
2. Upload a kidney CT scan image in JPG, JPEG, or PNG format
3. Click the "Predict" button to analyze the image
4. The system will display:
   - The predicted condition (normal, cyst, stone, or tumor)
   - A confidence percentage for the prediction
   - A bar chart showing the confidence distribution across all categories
5. An explanation section is available for interpreting the results

## Model Training

If you want to retrain the model with your own dataset:

1. Organize your dataset into categories (normal, cyst, stone, tumor)
2. Update the input folder path in disease_classifer.py
3. Run the training script:
   ```bash
   python disease_classifer.py
   ```