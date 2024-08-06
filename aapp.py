import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import av

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Load the necklace image with an alpha channel
necklace_image_path = 'necklace_1.png'
necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            center_section = img  # Assuming the whole frame is the center section

            frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    # Extract the bounding box coordinates
                    bboxC = detection.location_data.relative_bounding_box
                    hC, wC, _ = center_section.shape
                    xminC = int(bboxC.xmin * wC)
                    yminC = int(bboxC.ymin * hC)
                    widthC = int(bboxC.width * wC)
                    heightC = int(bboxC.height * hC)
                    xmaxC = xminC + widthC
                    ymaxC = yminC + heightC

                    # Calculate the bottom bounding box coordinates
                    bottom_ymin = ymaxC + 10
                    bottom_ymax = min(ymaxC + 150, hC)

                    # Increase the width of the bounding box
                    xminC -= 20  # Decrease the left side
                    xmaxC += 20  # Increase the right side

                    # Check if the bounding box dimensions are valid
                    if widthC > 0 and heightC > 0 and xmaxC > xminC and bottom_ymax > bottom_ymin:
                        # Resize necklace image to fit the bounding box size
                        resized_image = cv2.resize(necklace_image, (xmaxC - xminC, bottom_ymax - bottom_ymin))

                        # Create a mask from the alpha channel
                        alpha_channel = resized_image[:, :, 3]
                        mask = alpha_channel[:, :, np.newaxis] / 255.0

                        # Apply the mask to the necklace image
                        overlay = resized_image[:, :, :3] * mask

                        # Create a mask for the input image region
                        mask_inv = 1 - mask

                        # Apply the inverse mask to the input image
                        region = center_section[bottom_ymin:bottom_ymax, xminC:xmaxC]
                        resized_mask_inv = None
                        if region.shape[1] > 0 and region.shape[0] > 0:
                            resized_mask_inv = cv2.resize(mask_inv, (region.shape[1], region.shape[0]))
                            resized_mask_inv = resized_mask_inv[:, :, np.newaxis]  # Add an extra dimension

                        if resized_mask_inv is not None:
                            region_inv = region * resized_mask_inv

                            # Combine the resized image and the input image region
                            resized_overlay = None
                            if region_inv.shape[1] > 0 and region_inv.shape[0] > 0:
                                resized_overlay = cv2.resize(overlay, (region_inv.shape[1], region_inv.shape[0]))

                            # Combine the resized overlay and region_inv
                            region_combined = cv2.add(resized_overlay, region_inv)

                            # Replace the neck region in the input image with the combined region
                            center_section[bottom_ymin:bottom_ymax, xminC:xmaxC] = region_combined

            return av.VideoFrame.from_ndarray(center_section, format="bgr24")
        except Exception as e:
            st.error(f"Error occurred: {e}")
            return frame

# Streamlit App
st.title('Face Detection with Necklace Overlay')
st.markdown("This app detects faces and overlays a necklace image on the detected face area.")

st.write("Webcam Feed:")
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True},
)

st.image('https://visitor-badge.glitch.me/badge?page_id=kristyc.mediapipe-face-detection')
