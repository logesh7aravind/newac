import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

# Load the necklace image with an alpha channel
necklace_image_path = 'necklace_1.png'
necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
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

                # Increase the width of the red bounding box
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

def main():
    st.title("Webcam Live Stream with Necklace Overlay")

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
