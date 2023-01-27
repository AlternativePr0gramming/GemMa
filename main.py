import sys
import cv2
import mediapipe
import pykinect_azure as pyk

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager
from leap_listener import LeapListener

# Structure and base logic taken from: https://github.com/gabguerin/Sign-Language-Recognition--MediaPipe-DTW.
# Edited to enable Gemini input and it also includes its own recording program for the Azure Kinect and LM.
# Program runs in Python 3.7.3, other versions could possibly not work.

is_recording = False
LMB_pressed = False
MMB_pressed = False

def on_click(event, x, y, flags, *userdata):
    global LMB_pressed
    global MMB_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        LMB_pressed = True
    elif event == cv2.EVENT_MBUTTONDOWN:
        MMB_pressed = True


if __name__ == "__main__":
    category = sys.argv[1]
    # Create dataset of the videos where landmarks have not been extracted yet
    videos = load_dataset(category)

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    video_reference_signs, ul_reference_signs = load_reference_signs(category, videos)

    # Object that stores MediaPipe results and computes sign similarities
    sign_recorder_cam = SignRecorder(video_reference_signs, ul_reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager(on_click)
    listener = LeapListener()

    # Initialize Azure Kinect.
    pyk.initialize_libraries()
    ak_config = pyk.default_configuration
    ak_config.color_resolution = pyk.K4A_COLOR_RESOLUTION_720P
    ak_config.depth_mode = pyk.K4A_DEPTH_MODE_OFF
    ak_config.camera_fps = pyk.K4A_FRAMES_PER_SECOND_30

    azure_kinect = pyk.start_device(config=ak_config)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.8, min_tracking_confidence=0.8
    ) as holistic:
        while True:
            # Read feed
            capture = azure_kinect.update()
            ret, frame = capture.get_color_image()

            # Make detections
            image, result = mediapipe_detection(frame, holistic)

            if listener.frame < listener.get_frame():
                listener.get_landmarks()

            sign_detected, is_recording = sign_recorder_cam.process_mp_results(result, 
                                                                               listener.lh_landmarks[-1],
                                                                               listener.rh_landmarks[-1])

            # If the Azure Kinect isn't recording, then neither will the 3Di.
            if not is_recording:
                listener.reset_landmarks()

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, result, sign_detected, is_recording)

            pressedKey = cv2.waitKey(1) & 0xFF
            if MMB_pressed and not is_recording:  # Record pressing middle mouse button
                sign_recorder_cam.record()
                MMB_pressed = False
            elif pressedKey == ord("q"):  # Break pressing q
                break
        cv2.destroyAllWindows()
