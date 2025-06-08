
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

st.set_page_config(page_title="Face Detector - Webcam", layout="centered")
st.title("📸 Real-Time Face Detector with Webcam")
st.markdown("⚠️ This app requires webcam access. Click 'Start Camera' to begin.")

# Capture image from webcam using Streamlit's HTML/JS trick
capture_btn = st.button("📷 Start Camera")

if capture_btn:
    picture = st.camera_input("Take a picture")

    if picture:
        image = Image.open(picture)
        img_array = np.array(image)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.image(img_array, caption="Detected Faces")
