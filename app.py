import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
#import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Define available connections for drawing.
connections = {
    'HAND_CONNECTIONS': mp_hands.HAND_CONNECTIONS,
}

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Default to drawing background and using HAND_CONNECTIONS.
        self.draw_background = True
        self.selected_connection = connections['HAND_CONNECTIONS']
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = hands.process(img)
        output_img = img if self.draw_background else np.zeros_like(img)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(output_img, hand_landmarks, self.selected_connection)
        
        return output_img

# Streamlit App
st.title('Hand & Finger Tracking')
st.markdown("This is a demo of hand and finger tracking using [Google's MediaPipe](https://google.github.io/mediapipe/solutions/hands.html).")

st.write("Webcam Feed:")
webrtc_ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True},
)

st.image('https://visitor-badge.glitch.me/badge?page_id=kristyc.mediapipe-hands')
