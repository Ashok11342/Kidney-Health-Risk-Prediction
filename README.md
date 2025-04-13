# Kidney Health Risk Prediction

## Project Overview
This project implements a machine learning model to predict kidney health risks based on medical data. The application provides an interactive web interface built with Streamlit where users can input relevant health parameters and receive risk assessments for kidney-related conditions.

## Features
- Machine learning-based prediction of kidney health risks
- Interactive web interface for data input and visualization
- Comprehensive analysis of contributing risk factors
- Visual reports and insights using Plotly visualizations

## Live Demo
You can access the live application here: [Kidney Health Risk Prediction App](https://kidney-health-risk-prediction.streamlit.app)

## Repository Structure
```
Kidney-Health-Risk-Prediction/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .streamlit/             # Streamlit configuration
│   └── config.toml         # Streamlit theme and settings
├── model/                  # Model directory
│   ├── kidney_model.h5     # Trained deep learning model
│   └── model_loader.py     # Utility to load the model
├── utils/                  # Utility functions
│   ├── preprocessing.py    # Data preprocessing functions
│   └── visualization.py    # Data visualization functions
├── data/                   # Sample data
│   └── sample_input.csv    # Example input data
├── .gitattributes          # Git LFS configuration
└── README.md               # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- Git LFS (for handling large model files)

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
- scikit-learn (for data preprocessing and additional ML utilities)
- Plotly (for interactive visualizations)
- Pillow (for image processing)
-gdown(for downloading .h5 model file)

## Running the Application Locally

After installing the dependencies, you can run the application using:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Deployment Instructions

### Deploying to Streamlit Community Cloud

To deploy this application to Streamlit Community Cloud:

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push
   ```

2. Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud) using your GitHub account

3. Once logged in, click on "New app" button

4. Select your repository, branch, and specify the main file path (`app.py`)

5. Click "Deploy"

6. Your app will be deployed at a URL like: `https://username-repo-name.streamlit.app`

### Handling the .h5 Model File

For large model files (like .h5 files), there are two recommended approaches:

#### Option 1: Git LFS (preferred for files under 100MB)

1. Install [Git LFS](https://git-lfs.github.com/)
2. Initialize Git LFS in your repository:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   ```
3. Add and commit your model file as usual:
   ```bash
   git add model/kidney_model.h5
   git commit -m "Add model file"
   git push
   ```

#### Option 2: Cloud Storage (for files over 100MB)

1. Upload your model to a cloud storage service (Google Drive, Dropbox, AWS S3, etc.)
2. In your code, add logic to download the model at runtime:

```python
import gdown  # Add to requirements.txt

def download_model_if_needed():
    if not os.path.exists('model/kidney_model.h5'):
        # Create directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        # Download file from Google Drive
        url = 'YOUR_GOOGLE_DRIVE_SHARING_URL'
        output = 'model/kidney_model.h5'
        gdown.download(url, output, quiet=False)

# Call this function before loading the model
download_model_if_needed()
```

## Usage Instructions

1. Fill in the required health parameters in the input form
2. Submit the form to get a prediction
3. Review the risk assessment and visualizations
4. Explore the insights and recommendations provided

## Contributing

Contributions to improve the model accuracy, add new features, or enhance the UI are welcome. Please feel free to submit a pull request.

## License

MIT License

Copyright (c) 2023 Ashok

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
