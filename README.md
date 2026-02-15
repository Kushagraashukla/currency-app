# Indian Currency Detection using Deep Learning

This project is a deep learning based system that detects Indian currency notes using images or webcam.  
The main goal of this project is to help visually impaired people identify currency notes easily.

The system takes an image of a currency note, processes it using a trained CNN model, and predicts the denomination.

---

## About the Project

Identifying currency notes can be difficult for blind or visually impaired people.  
This project provides a simple solution using artificial intelligence.

The model is trained on images of Indian currency notes and can detect different denominations from an image or live camera.

---

## Features

- Detects Indian currency notes
- Supports image upload
- Real-time webcam detection
- Web interface using Streamlit
- Simple and easy to use system

---

## Technologies Used

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- OpenCV
- NumPy
- Streamlit

---

## Supported Currency Notes

- ₹10
- ₹20
- ₹50
- ₹100
- ₹200
- ₹500
- ₹2000

---

## How It Works

1. User uploads image or captures currency using webcam.
2. Image is preprocessed (resize, crop, etc.).
3. Model predicts the currency class.
4. Result is shown with confidence score.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/currency-detection.git
cd currency-detection
