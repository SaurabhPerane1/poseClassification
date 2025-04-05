import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import tensorflow as tf

pose_classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("yoga_pose_classifier1.keras")

model = load_model()

def preprocess_frame(frame):
    img = cv2.resize(frame, (150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = preprocess_frame(img_rgb)
        prediction = model.predict(input_img)
        predicted_class = pose_classes[np.argmax(prediction)]
        cv2.putText(img, f'Pose: {predicted_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

st.title("Yoga Pose Classifier - Live")

webrtc_streamer(key="pose", video_processor_factory=VideoProcessor)
