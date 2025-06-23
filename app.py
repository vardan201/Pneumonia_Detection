import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io

# Load the trained model
model = load_model("pneumonia_detection_model.keras")

# Define class labels
class_names = ['NORMAL', 'PNEUMONIA']

# Streamlit UI
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict whether it shows signs of Pneumonia.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction)

    # Display result
    if confidence >= 0.5:
        st.error(f"Prediction: PNEUMONIA ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: NORMAL ({(1-confidence)*100:.2f}% confidence)")
