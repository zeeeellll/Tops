import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import tempfile

# Page config
st.set_page_config(page_title='Cat Dog Classification Model', layout="centered")

# --- Load model ---
@st.cache_resource
def load_model():
    model_path = r'C:\Users\MAHADEV\OneDrive\Desktop\Zeel\Data Science\Tops\Fair work\Work\Deep Learning\Assessment\model2.h5'
    try:
        st.write(f"Attempting to load model from: {os.path.abspath(model_path)}")
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

# --- Get class names from folder ---
def get_class_names(train_path):
    if not os.path.exists(train_path):
        st.error(f"Training path not found: {train_path}")
        return []
    class_names = sorted(os.listdir(train_path))
    class_names = [name for name in class_names if os.path.isdir(os.path.join(train_path, name))]
    return class_names

train_data_path = r'C:\Users\MAHADEV\OneDrive\Desktop\Zeel\Data Science\Tops\Fair work\Work\Deep Learning\Assessment\train'
class_names = get_class_names(train_data_path)

if not class_names:
    st.warning("Could not determine class names from training folder.")
    class_names = [f"Class {i}" for i in range(2)]  # default fallback

# --- Preprocess image ---
def preprocess_image(img_path, target_size=(256, 256)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 256, 256, 3)
    img_array = img_array / 255.0
    return img_array

# --- Predict image class ---
def predict_image_class(model, img_array, class_names):
    if model is None:
        return "Model not loaded", None

    prediction = model.predict(img_array)

    # Binary classification: sigmoid output
    if prediction.shape[1] == 1:
        predicted_class_index = int(prediction[0][0] > 0.5)
        confidence = prediction[0][0] * 100 if predicted_class_index == 1 else (1 - prediction[0][0]) * 100
    else:
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, confidence

# --- Streamlit UI ---
st.title("Cat vs Dog Image Classifier")
st.write("Upload an image, and the model will predict whether it's a **cat** or a **dog**!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_image_path = tmp_file.name

    try:
        img_array = preprocess_image(temp_image_path)
        predicted_class, confidence = predict_image_class(model, img_array, class_names)
        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    finally:
        os.remove(temp_image_path)  # Clean up temp image

st.markdown("---")
st.write("Built with using TensorFlow and Streamlit.")
