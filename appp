import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

model_path = "brain_tumor_cnn_model.h5"
if not os.path.exists(model_path):
    import gdown
    url = "https://drive.google.com/uc?id=1aHyWAONr662uKP30DGz3NE__1Oszz8Xh"
    gdown.download(url, model_path, quiet=False)

# Load trained model
model = tf.keras.models.load_model("brain_tumor_cnn_model.h5")

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Title
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI image and the model will predict the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Output
    st.success(f"✅ Predicted Tumor Type: **{predicted_class.upper()}**")
