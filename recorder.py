import pykinect_azure as pyk
import time
import cv2
import sys
from os import path, makedirs, listdir
from leap_listener import LeapListener


LMB_pressed = False
MMB_pressed = False
recording_time = 3 # seconds

def on_click(event, x, y, flags, *userdata):
    global LMB_pressed
    global MMB_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        LMB_pressed = True
    elif event == cv2.EVENT_MBUTTONDOWN:
        MMB_pressed = True

if __name__ == "__main__":
    category = sys.argv[1]
    sign_type = sys.argv[2]

    video_folder_path = path.join("data/videos", category, sign_type)
    UL_folder_path = path.join("data/ultraleapdataset", category, sign_type)
    
    if not path.isdir(video_folder_path) or not path.isdir(UL_folder_path):
        makedirs(video_folder_path, exist_ok=True)
        makedirs(UL_folder_path, exist_ok=True)

    listener = LeapListener()
    # Initialize and start the Azure Kinect.
    pyk.initialize_libraries()
    ak_config = pyk.default_configuration
    ak_config.color_resolution = pyk.K4A_COLOR_RESOLUTION_720P
    ak_config.depth_mode = pyk.K4A_DEPTH_MODE_OFF
    ak_config.camera_fps = pyk.K4A_FRAMES_PER_SECOND_30
    azure_kinect = pyk.start_device(config=ak_config)

    cv2.namedWindow("Recorder")
    cv2.setMouseCallback("Recorder", on_click)
    vid_cod = cv2.VideoWriter_fourcc(*'mp4v')
    font = cv2.FONT_HERSHEY_SIMPLEX
    output = None
    start_time = time.time()
    video_i = len(listdir(video_folder_path))
    is_recording = False

    while True:
        # Read feed
        capture = azure_kinect.update()
        ret, frame = capture.get_color_image()

        if not ret:
            continue

        pressedKey = cv2.waitKey(1) & 0xFF
        if MMB_pressed and not is_recording:  # Record pressing MMB
            video_i += 1
            output = cv2.VideoWriter((video_folder_path + "/%s-%d.mp4" % (sign_type, video_i)), 
                                     vid_cod, 30, (1280, 720))
            start_time = time.time()
            is_recording = True
            MMB_pressed = False
        elif pressedKey == 32:  # Break pressing Space
            break
        if is_recording:
            if time.time() - start_time > recording_time:
                is_recording = False
                MMB_pressed = False
                listener.pickle_landmarks(category, sign_type, video_i)
                listener.reset_landmarks()
            output.write(frame)
            listener.get_landmarks()
            image = cv2.putText(frame, 'Recording...', (50, 50), font,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Recorder", frame)
    if output:
        output.release()

    cv2.destroyAllWindows()
