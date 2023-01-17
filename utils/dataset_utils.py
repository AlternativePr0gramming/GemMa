import os

import pandas as pd
from tqdm import tqdm

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array


def load_dataset(category):
    """
    Loads and then makes a dataset from the non-present videos.
    """
    videos = [
        file_name.replace(".mp4", "")
        for _, _, files in os.walk(os.path.join("data", "videos", category))
        for file_name in files
        if file_name.endswith(".mp4")
    ]

    dataset = [
        file_name.replace(".pickle", "").replace("lh_", "")
        for _, _, files in os.walk(os.path.join("data", "dataset", category))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("lh_")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")

        for idx in tqdm(range(n)):
            save_landmarks_from_video(videos_not_in_dataset[idx], category)

    return videos


def load_reference_signs(category, videos):
    video_reference_signs = {"name": [], "sign_model": [], "distance": []}
    ul_reference_signs = {"name": [], "sign_model": [], "distance": []}

    for i, video_name in enumerate(videos):
        sign_name = video_name.split("-")[0]
        path_ds_videos = os.path.join("data", "dataset", category, sign_name, video_name)
        path_ds_ul = os.path.join("data", "ultraleapdataset", category, sign_name, video_name)

        left_hand_list = load_array(os.path.join(path_ds_videos, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path_ds_videos, f"rh_{video_name}.pickle"))
        left_hand_list_ul = load_array(os.path.join(path_ds_ul, f"lh_{video_name}.pickle"))
        right_hand_list_ul = load_array(os.path.join(path_ds_ul, f"rh_{video_name}.pickle"))

        video_reference_signs["name"].append(sign_name)
        video_reference_signs["sign_model"].append(SignModel(left_hand_list, right_hand_list))
        video_reference_signs["distance"].append(0)

        ul_reference_signs["name"].append(sign_name)
        ul_reference_signs["sign_model"].append(SignModel(left_hand_list_ul, right_hand_list_ul))
        ul_reference_signs["distance"].append(0)
    
    video_reference_signs = pd.DataFrame(video_reference_signs, dtype=object)
    ul_reference_signs = pd.DataFrame(ul_reference_signs, dtype=object)
    print(f'Dictionary count: {video_reference_signs[["name", "sign_model"]].groupby(["name"]).count()}')
    return video_reference_signs, ul_reference_signs
