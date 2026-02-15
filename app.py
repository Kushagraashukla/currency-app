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

st.set_page_config(
    page_title="Currency Detector",
    layout="centered"
)

# ==============================
# CUSTOM UI STYLE
# ==============================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}

h1 {
    text-align:center;
    color:white;
}

.stButton>button {
    width:100%;
    background-color:#00c6ff;
    color:white;
    font-size:18px;
    border-radius:10px;
}

.result-box {
    padding:20px;
    border-radius:15px;
    background: rgba(255,255,255,0.1);
    text-align:center;
    font-size:22px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

st.title("Indian Currency Detection")
st.markdown("### AI Powered Currency Recognition System")

st.divider()

# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_PATH, allow_pickle=True)
    return model, class_names

model, class_names = load_model()

# ==============================
# PREPROCESS
# ==============================

def preprocess_image(image):

    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = w//2 - min_dim//2
    start_y = h//2 - min_dim//2
    img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    return img

# ==============================
# PREDICT
# ==============================

def predict_currency(image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_names[predicted_index], confidence

# ==============================
# MODE SELECTOR
# ==============================

option = st.radio(
    "Choose Detection Mode",
    ["Upload Image", "Live Webcam"],
    horizontal=True
)

st.divider()

# ==============================
# UPLOAD MODE
# ==============================

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Currency Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:

        col1, col2 = st.columns([1,1])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.write("")
            st.write("")
            if st.button("Detect Currency"):

                with st.spinner("Analyzing image..."):
                    label, confidence = predict_currency(image)

                st.markdown(
                    f"<div class='result-box'>Detected: Rs {label}</div>",
                    unsafe_allow_html=True
                )

                st.progress(int(confidence))
                st.write(f"Confidence: {confidence:.2f}%")

# ==============================
# WEBCAM MODE
# ==============================

elif option == "Live Webcam":

    st.info("Show currency in center, good lighting, full note visible")

    camera_image = st.camera_input("Capture Currency")

    if camera_image:

        col1, col2 = st.columns([1,1])

        with col1:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_container_width=True)

        with col2:
            with st.spinner("Analyzing image..."):
                label, confidence = predict_currency(image)

            st.markdown(
                f"<div class='result-box'>Detected: Rs {label}</div>",
                unsafe_allow_html=True
            )

            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")
