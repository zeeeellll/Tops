import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array 
import numpy as np
import os

# Set page configuration for better layout
st.set_page_config(page_title="Image Classification App", layout="centered")

# --- Load the Trained Model ---
# st.title("📸 Image Classification App")
# ...
@st.cache_resource
def load_model():
    model_path = r'/workspaces/Tops/Deep_Learning/Project/resnet50_model_new.h5' 

    try:
        # st.write(f"Attempting to load model from: {os.path.abspath(model_path)}")
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.info("Please ensure 'resnet50_model.h5' is in the same directory or provide the correct path.")
        return None

model = load_model()

def preprocess_image(img_path, target_size=(150, 150)):
    """Loads and preprocesses an image for model prediction."""
    img = load_img(img_path, target_size=target_size) 

    img_array = img_to_array(img)                     
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 
    
    return img_array

def predict_image_class(model, img_array, class_names):
    """Makes a prediction using the loaded model."""
    if model is None:
        return "Model not loaded", None

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence


def get_class_names(train_path):
    """Extracts class names from the training data directory."""
    if not os.path.exists(train_path):
        st.error(f"Training path not found: {train_path}")
        return []
    class_names = sorted(os.listdir(train_path))

    # Filter out any non-directory files if present
    class_names = [name for name in class_names if os.path.isdir(os.path.join(train_path, name))]
    return class_names


train_data_path = r'C:\Users\MAHADEV\OneDrive\Desktop\Zeel\Data Science\Tops\Fair work\Work\Deep Learning\Project\images\train'
class_names = get_class_names(train_data_path)


if not class_names:
    st.warning("Could not determine class names. Please check your `train_data_path` and directory structure.")
    # Provide a placeholder or default if class names can't be loaded
    class_names = [f"Class {i}" for i in range(7)] # Fallback: adjust '7' to your no_of_classes

# --- Streamlit UI ---
st.title("📸 Image Classification App")
st.write("Upload an image and the model will predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save the uploaded file temporarily to process it
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess and predict
    img_array = preprocess_image("temp_image.jpg")
    predicted_class, confidence = predict_image_class(model, img_array, class_names)

    if predicted_class == "Model not loaded. Cannot predict.":
        st.error(predicted_class)
    else:
        st.success(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

    # Clean up the temporary file
    os.remove("temp_image.jpg")

st.markdown("---")
st.write("This application uses a deep learning model to classify images.")