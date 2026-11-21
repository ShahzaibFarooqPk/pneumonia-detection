
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# ---------------------------------
# 1. Page configuration
# ---------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pneumonia Detection System")
st.write("Upload a chest X-ray image for an AI model prediction.")

# ---------------------------------
# 2. Load trained CNN model
# ---------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "custom_pre_trained_model_10.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.success("CNN model loaded successfully!")

# ---------------------------------
# 3. Upload X-ray image
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# 4. Process uploaded image
# ---------------------------------

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    # Convert image to RGB
    img = img.convert("RGB")

    # Resize image to model input size
    img = img.resize((100, 100))

    # Display uploaded image
    st.image(img, caption="Uploaded X-ray", width=300)

    # Convert image into NumPy array
    image_array = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Normalize image
    image_array = image_array / 255.0

    # ---------------------------------
    # 5. Make prediction
    # ---------------------------------

    if st.button("Predict"):

        prediction = model.predict(image_array, verbose=0)[0][0]

        st.write("Model output:", float(prediction))

        # NORMAL = 0
        # PNEUMONIA = 1

        if prediction >= 0.5:
            st.error("Prediction: PNEUMONIA")
        else:
            st.success("Prediction: NORMAL")

        st.caption(
            "Educational demonstration only. "
            "This result is not a medical diagnosis."
        )
