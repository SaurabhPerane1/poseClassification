import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("yoga_pose_classifier1.keras")

model = load_model()

pose_classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

def preprocess_frame(frame):
    img = cv2.resize(frame, (150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Pose Classification - Live Feed")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

start = st.button('Start Webcam')
stop = st.button('Stop Webcam')

FRAME_WINDOW = st.image([])

if start:
    st.session_state.camera_active = True

if stop:
    st.session_state.camera_active = False

if st.session_state.camera_active:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Unable to access camera")
    else:
        ret, frame = camera.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = preprocess_frame(frame_rgb)
            prediction = model.predict(input_img)
            predicted_class = pose_classes[np.argmax(prediction)]
            cv2.putText(frame_rgb, f'Pose: {predicted_class}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(frame_rgb)
        camera.release()
