import streamlit as st
import cv2
import mediapipe as mp

st.title('Test OpenCV and MediaPipe')

st.write('OpenCV version:', cv2.__version__)
st.write('MediaPipe version:', mp.__version__)
