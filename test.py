import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import time

last_frame_time = 0
min_interval = 1/10

def video_frame_callback(frame):
    global last_frame_time
    now = time.time()
    if now-last_frame_time<min_interval:
        return None
    last_frame_time = now
    try:    
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (320, 240))
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print("Error in frame processing: ", e)
        return None

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
