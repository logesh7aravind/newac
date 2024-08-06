import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Draw a blue vertical line in the center of the frame
        height, width, _ = img.shape
        line_color = (255, 0, 0)  # Blue color in BGR
        line_thickness = 2
        cv2.line(img, (width // 2, 0), (width // 2, height), line_color, line_thickness)

        self.frame = img  # Store the frame to capture it later
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Webcam Live Stream with Capture Button")

    # Initialize the video transformer with STUN server configuration
    ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    if ctx.video_transformer:
        # Capture button
        if st.button("Capture"):
            # Check if a frame is available
            if ctx.video_transformer.frame is not None:
                # Get the frame
                captured_frame = ctx.video_transformer.frame

                # Display the captured frame
                st.image(captured_frame, channels="BGR")

                # Save the captured frame as an image file
                cv2.imwrite("captured_image.jpg", captured_frame)
                st.success("Image captured and saved as 'captured_image.jpg'")

if __name__ == "__main__":
    main()
