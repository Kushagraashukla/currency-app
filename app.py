import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = "currency_model2.h5"
CLASS_PATH = "class_names.npy"
IMG_SIZE = 224

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Currency Detector", layout="centered")
st.title("💰 Indian Currency Detection")

# ==============================
# LOAD MODEL
# ==============================

import gdown
import os

MODEL_PATH = "currency_model2.h5"
DRIVE_FILE_ID = "1jipkYSGgtrx7AgGfGcHAXB-kU5aCKBj6"


@st.cache_resource
def load_model():

    # download if not exists
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        with st.spinner("Downloading model... (first time only)"):
            gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_PATH, allow_pickle=True)

    return model, class_names


model, class_names = load_model()

# ==============================
# BETTER PREPROCESSING (IMPORTANT)
# ==============================

def preprocess_image(image):

    img = np.array(image)

    # RGBA → RGB
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # ---- CENTER CROP (IMPORTANT) ----
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = w//2 - min_dim//2
    start_y = h//2 - min_dim//2
    img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # ---- reduce noise ----
    img = cv2.GaussianBlur(img, (3,3), 0)

    # resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # convert float (model me rescaling already hai)
    img = img.astype("float32")

    # batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_currency(image):

    img = preprocess_image(image)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return class_names[predicted_index], confidence

# ==============================
# USER CHOICE
# ==============================

option = st.radio(
    "Choose Detection Mode:",
    ["Upload Image", "Live Webcam"]
)

# ==============================
# IMAGE UPLOAD MODE
# ==============================

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload currency image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Detect Currency"):
            with st.spinner("Detecting..."):
                label, confidence = predict_currency(image)

            st.success(f"Detected Currency: ₹{label}")
            st.info(f"Confidence: {confidence:.2f}%")

# ==============================
# LIVE WEBCAM MODE (IMPROVED)
# ==============================

elif option == "Live Webcam":

    st.info("Show currency in center, good lighting, full note visible")

    camera_image = st.camera_input("Take a photo")

    if camera_image:
        image = Image.open(camera_image)

        st.image(image, caption="Captured Image", use_container_width=True)

        with st.spinner("Detecting..."):
            label, confidence = predict_currency(image)

        st.success(f"Detected Currency: ₹{label}")
        st.info(f"Confidence: {confidence:.2f}%")
