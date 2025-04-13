import os
import tensorflow as tf
import logging

# Uncomment this line to enable gdown
import gdown

def download_model_if_needed():
    """Downloads the model file if it's not already present."""
    if not os.path.exists('model/kidney_model.h5'):
        logging.info("Model file not found. Attempting to download...")
        os.makedirs('model', exist_ok=True)
        
        # Option 1: Use gdown to download from Google Drive
        try:
            # For Google Drive, convert the sharing URL to the direct download format
            # Example: From https://drive.google.com/file/d/FILEID/view?usp=sharing
            # To: https://drive.google.com/uc?id=FILEID
            
            # Replace this with your actual Google Drive file ID
            # Extract this from your Google Drive sharing link
            url = 'https://drive.google.com/file/d/1u2lFcIIZcKEuqsEg0B_ah7SWAM1t2PGm/view?usp=sharing'
            output = 'model/kidney_model.h5'
            gdown.download(url, output, quiet=False)
            logging.info("Model downloaded successfully!")
            return True
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            logging.warning("Please manually place the kidney_model.h5 file in the model directory.")
            return False
    return True

def load_model():
    """Loads and returns the trained model."""
    # Ensure model file exists
    model_exists = download_model_if_needed()
    if not model_exists:
        raise FileNotFoundError("Model file not found. Please add the model file to the model directory.")
    
    # Load the model
    try:
        model = tf.keras.models.load_model('model/kidney_model.h5')
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def get_model_summary(model):
    """Returns a string representation of the model summary."""
    # Redirect the summary to a string
    from io import StringIO
    import sys
    
    # Capture the summary output
    old_stdout = sys.stdout
    string_io = StringIO()
    sys.stdout = string_io
    
    # Print the model summary
    model.summary()
    
    # Restore stdout and get the summary string
    sys.stdout = old_stdout
    summary_string = string_io.getvalue()
    
    return summary_string

if __name__ == "__main__":
    # Test loading the model
    try:
        model = load_model()
        print("Model loaded successfully!")
        print("\nModel Summary:")
        print(get_model_summary(model))
    except Exception as e:
        print(f"Error: {e}") 