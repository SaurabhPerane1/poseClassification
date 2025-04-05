import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

def video_frame_callback(frame):
    try:    
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print("Error in frame processing: ", e)
        return None

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
