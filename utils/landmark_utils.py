import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp
from utils.mediapipe_utils import mediapipe_detection


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """Extract the results of both hands and convert them to a np array of size
    if a hand doesn't appear, return an array of zeros

    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 21 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return left_hand, right_hand


def save_landmarks_from_leapdata(leap_dicts, sign_name):
    for i, leap_dict in enumerate(leap_dicts):
        save_array(
            leap_dict["left_hand"], os.path.join(sign_name, f"lh_{i}.pickle")
        )
        save_array(
            leap_dict["right_hand"], os.path.join(sign_name, f"rh_{i}.pickle")
        )


def save_landmarks_from_video(video_name, category):
    landmark_list = {"left_hand": [], "right_hand": []}
    sign_name = video_name.split("-")[0]
    # Set the Video stream
    cap = cv2.VideoCapture(
        os.path.join("data", "videos", category, sign_name, video_name + ".mp4")
    )
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.8, min_tracking_confidence=0.8
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Store results
                left_hand, right_hand = extract_landmarks(results)
                landmark_list["left_hand"].append(left_hand)
                landmark_list["right_hand"].append(right_hand)
            else:
                break
        cap.release()
    # Create the folder of the sign if it doesn't exists
    path = os.path.join("data", "dataset", category, sign_name, video_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Saving the landmark_list in the correct folder
    save_array(
        landmark_list["left_hand"], os.path.join(path, f"lh_{video_name}.pickle")
    )
    save_array(
        landmark_list["right_hand"], os.path.join(path, f"rh_{video_name}.pickle")
    )


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
