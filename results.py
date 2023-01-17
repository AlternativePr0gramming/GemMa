import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from utils.landmark_utils import load_array

categories = {
    "1A": 1,
    "1B": 2,
    "2A": 1,
    "2B": 2,
    "3A": 2,
    "3B": 2
} 

MPUL_lh = []
MPUL_rh = []
MP_rh = []
MP_lh = []


def count_outliers(frames_MP, frames_UL):
    MP_outlier_amount = 0
    MPUL_outlier_amount = 0
    no_hand = [0 * 63]
    
    for i, frame in enumerate(frames_MP):
        if (frame == no_hand).all():
            MP_outlier_amount += 1
            try:
                if (frames_UL[i] == no_hand).all():
                    MPUL_outlier_amount += 1
            except IndexError:
                break
    return MP_outlier_amount, MPUL_outlier_amount

def read_datasets(path_UL, path_MP, hand_amount):
    if hand_amount == 1:
        frames = 0
        empty_frames = 0
        empty_solo_frames = 0

        file_paths_MP = [os.path.join(path, name) for path, _, files in os.walk(path_MP) for name in files if "lh" in name]
        file_paths_UL = [os.path.join(path, name) for path, _, files in os.walk(path_UL) for name in files if "lh" in name]

        for file_MP, file_UL in zip(file_paths_MP, file_paths_UL):
            frames_MP = load_array(file_MP)
            frames_UL = load_array(file_UL)
            frames += frames_MP.shape[0]
            MP_amount, MPUL_amount = count_outliers(frames_MP, frames_UL)
            empty_solo_frames += MP_amount
            empty_frames += MPUL_amount

        MP_lh.append(empty_solo_frames * 100 / frames)
        MPUL_lh.append(empty_frames * 100 / frames)
        MP_rh.append(0)
        MPUL_rh.append(0)

    else:
        frames = 0
        empty_frames_lh = 0
        empty_solo_frames_lh = 0
        empty_frames_rh = 0
        empty_solo_frames_rh = 0

        file_paths_MP = [os.path.join(path, name) for path, _, files in os.walk(path_MP) for name in files]
        file_paths_UL = [os.path.join(path, name) for path, _, files in os.walk(path_UL) for name in files]
        
        for i, (file_MP, file_UL) in enumerate(zip(file_paths_MP, file_paths_UL)):
            frames_MP = load_array(file_MP)
            frames_UL = load_array(file_UL)
            if i % 2 == 0:
                frames += frames_MP.shape[0]
                MP_amount, MPUL_amount = count_outliers(frames_MP, frames_UL)
                empty_solo_frames_lh += MP_amount
                empty_frames_lh += MPUL_amount
            else:
                MP_amount, MPUL_amount = count_outliers(frames_MP, frames_UL)
                empty_solo_frames_rh += MP_amount
                empty_frames_rh += MPUL_amount

        MP_lh.append(empty_solo_frames_lh * 100 / frames)
        MPUL_lh.append(empty_frames_lh * 100 / frames)
        MP_rh.append(empty_solo_frames_rh * 100 / frames)
        MPUL_rh.append(empty_frames_rh * 100 / frames)
    

if __name__ == "__main__":
    for category in categories:
        path_UL = os.path.join("data", "ultraleapdataset", category)
        path_MP = os.path.join("data", "dataset", category)
        read_datasets(path_UL, path_MP, categories[category])

    df = pd.DataFrame({"left hand": MPUL_lh,
                       "right hand": MPUL_rh}, list(categories.keys()))
    #df1 = pd.DataFrame(np.c_[MP_rh, MPUL_rh], list(categories.keys()))
    #ax = df1.plot(kind="bar", edgecolor='black', linewidth=1, stacked=False, width=0.5, 
    #              colormap="Pastel2_r", rot=0)
    #ax.set_xlabel("Category")
    #ax.set_ylabel(r"% of outliers")
    #ax.set_ylim([0, 35])
    ax1 = df.plot(kind="bar", edgecolor='white', linewidth=3, stacked=False, width=0.7, 
                 color=["gray", "black"], rot=0, legend=True)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)
    ax1.set_xlabel("Category")
    ax1.set_ylabel(r"% of outliers")
    ax1.set_ylim([0, 35])

    # df = pd.DataFrame({"left hand": MP_lh,
    #                    "right hand": MP_rh}, list(categories.keys()))
    # df.plot(kind="bar", edgecolor='black', linewidth=1, stacked=False, width=0.7, 
    #                ax=ax, colormap="Pastel2_r", rot=0, legend=False)
    plt.show()
    