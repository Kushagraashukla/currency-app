import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# SETTINGS

MODEL_PATH = "currency_model2.h5"
CLASS_PATH = "class_names.npy"
IMG_SIZE = 224


# PAGE CONFIG

st.set_page_config(page_title="Currency Detector", layout="wide")

# WEBSITE STYLE UI (HTML + CSS)

st.markdown("""
<style>

body {
    background: #0f172a;
}

.main {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}

/* Navbar */
.navbar {
    padding:15px;
    background:#020617;
    border-radius:10px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    color:white;
}

/* Hero Section */
.hero {
    text-align:center;
    padding:60px 20px;
    color:white;
}

.hero h1 {
    font-size:48px;
}

.hero p {
    font-size:20px;
    color:#94a3b8;
}

/* Cards */
.card {
    padding:30px;
    border-radius:15px;
    background:#1e293b;
    color:white;
    box-shadow:0 10px 30px rgba(0,0,0,0.4);
}

/* Button */
.stButton>button {
    width:100%;
    height:50px;
    font-size:18px;
    background:linear-gradient(to right,#3b82f6,#06b6d4);
    color:white;
    border-radius:10px;
}

/* Result box */
.result {
    padding:20px;
    border-radius:12px;
    background:#16a34a;
    text-align:center;
    font-size:28px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# NAVBAR

st.markdown('<div class="navbar">AI Currency Detection System</div>', unsafe_allow_html=True)

# HERO SECTION

st.markdown("""
<div class="hero">
<h1>Indian Currency Recognition</h1>
<p>Deep Learning powered currency detection system</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# LOAD MODEL (UNCHANGED BACKEND)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_PATH, allow_pickle=True)
    return model, class_names

model, class_names = load_model()

# PREPROCESS (UNCHANGED)

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

# PREDICT (UNCHANGED)

def predict_currency(image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_names[predicted_index], confidence

# MODE SELECTOR

option = st.radio(
    "Choose Mode",
    ["Upload Image", "Live Webcam"],
    horizontal=True
)

st.divider()


# UPLOAD MODE


if option == "Upload Image":

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Currency Image")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if uploaded_file:
            if st.button("Detect Currency"):

                with st.spinner("Processing..."):
                    label, confidence = predict_currency(image)

                st.markdown(
                    f"<div class='result'>Detected: Rs {label}</div>",
                    unsafe_allow_html=True
                )

                st.progress(int(confidence))
                st.write(f"Confidence: {confidence:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# WEBCAM MODE
# ==============================

elif option == "Live Webcam":

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        camera_image = st.camera_input("Capture Currency")

        if camera_image:
            image = Image.open(camera_image)
            st.image(image, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if camera_image:
            with st.spinner("Processing..."):
                label, confidence = predict_currency(image)

            st.markdown(
                f"<div class='result'>Detected: Rs {label}</div>",
                unsafe_allow_html=True
            )

            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)


