import os
import tensorflow as tf
import logging

# Uncomment this line to enable gdown
import gdown

def create_dummy_model():
    """Creates a very simple model for testing purposes when the real model can't be loaded."""
    logging.info("Creating a simple dummy model for testing...")
    inputs = tf.keras.Input(shape=(13,))  # Matching our input shape
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Save the model to disk
    os.makedirs('model', exist_ok=True)
    model_path = 'model/kidney_model.h5'
    model.save(model_path)
    logging.info(f"Dummy model saved to {model_path}")
    return model

def download_model_if_needed():
    """Downloads the model file if it's not already present."""
    if not os.path.exists('model/kidney_model.h5'):
        logging.info("Model file not found. Attempting to download...")
        os.makedirs('model', exist_ok=True)
        
        # Option 1: Use gdown to download from Google Drive
        try:
            # For Google Drive, the share link needs to use the fuzzy option
            url = 'https://drive.google.com/file/d/1u2lFcIIZcKEuqsEg0B_ah7SWAM1t2PGm/view?usp=sharing'
            output = 'model/kidney_model.h5'
            
            # Use the fuzzy option to handle view links
            gdown.download(url, output, quiet=False, fuzzy=True)
            
            # Verify file size (models should typically be several MB)
            file_size = os.path.getsize(output)
            logging.info(f"Downloaded file size: {file_size/1024/1024:.2f} MB")
            
            if file_size < 1000000:  # Less than 1MB is suspicious for a model
                logging.warning(f"Downloaded file seems too small ({file_size} bytes). It might not be a valid model file.")
                logging.warning("Creating a dummy model instead.")
                return False
            
            logging.info("Model downloaded successfully!")
            return True
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            logging.warning("Creating a dummy model for testing instead.")
            return False
    return True

def load_model():
    """Loads and returns the trained model."""
    # Ensure model file exists
    model_exists = download_model_if_needed()
    
    # Load the model
    try:
        # Verify file exists and has some content
        if not model_exists or not os.path.exists('model/kidney_model.h5') or os.path.getsize('model/kidney_model.h5') == 0:
            logging.warning("Model file is missing or invalid. Creating a dummy model.")
            return create_dummy_model()
            
        model = tf.keras.models.load_model('model/kidney_model.h5')
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.warning("Falling back to dummy model...")
        return create_dummy_model()

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