import pandas as pd
import numpy as np
from typing import Tuple

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, ul_reference_signs: pd.DataFrame, seq_len=40):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len
        self.predicted_sign = ""

        # List of results stored each frame
        self.recorded_mp_lh = []
        self.recorded_mp_rh = []
        self.recorded_ul_lh = []
        self.recorded_ul_rh = []

        # DataFrames storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs
        self.ul_reference_signs = ul_reference_signs


    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.ul_reference_signs["distance"].values[:] = 0
        self.predicted_sign = ""
        self.is_recording = True


    def process_mp_results(self, results, lh_landmarks, rh_landmarks) -> Tuple[str, bool]:
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
               lh_landmarks: leap motion left output
               rh_landmarks: leap motion right output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_mp_lh) < self.seq_len:
                left_mp_hand, right_mp_hand = extract_landmarks(results)
                self.recorded_mp_lh.append(left_mp_hand)
                self.recorded_mp_rh.append(right_mp_hand)
                self.recorded_ul_lh.append(lh_landmarks)
                self.recorded_ul_rh.append(rh_landmarks)
            else:
                self.compute_distances()
                print(self.reference_signs)
                print(self.ul_reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        if self.predicted_sign == "":
            self._get_sign_predicted()
        return self.predicted_sign, self.is_recording


    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(self.recorded_mp_lh, self.recorded_mp_rh)
        ul_recorded_sign = SignModel(self.recorded_ul_lh, self.recorded_ul_rh)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)
        self.ul_reference_signs = dtw_distances(ul_recorded_sign, self.ul_reference_signs)


    def _get_sign_predicted(self, batch_size=30):
        self.predicted_sign = self._choose_sign(batch_size)

        # Reset variables
        self.recorded_mp_lh = []
        self.recorded_mp_rh = []
        self.recorded_ul_lh = []
        self.recorded_ul_rh = []
        self.is_recording = False
        

    def _choose_sign(self, batch_size):
        n_frames = self.seq_len
        outlier = [0] * 63

        # Checks if the left hand has been fully absent through the entire recording.
        if n_frames in {self.recorded_mp_lh.count(outlier), self.recorded_ul_lh.count(outlier)}:
            mp_outlier_score = (1 - self.recorded_mp_rh.count(outlier) / n_frames)
            ul_outlier_score = (1 - self.recorded_ul_rh.count(outlier) / n_frames)
        else:
            mp_outlier_score = (0.5 - self.recorded_mp_rh.count(outlier) / (2 * n_frames)) + \
                                (0.5 - self.recorded_mp_lh.count(outlier) / (2 * n_frames))
            ul_outlier_score = (0.5 - self.recorded_ul_rh.count(outlier) / (2 * n_frames)) + \
                                (0.5 - self.recorded_ul_lh.count(outlier) / (2 * n_frames))

        # Takes the top $batch_size and then calculates the mean of the likely gestures.
        # The four lowest average distances then get taken for further comparison.
        highest_two = self.reference_signs.iloc[:batch_size]['name'].value_counts()[:2]
        if highest_two[0] > highest_two[1] * 1.5 and ul_outlier_score < 0.2:
            return highest_two.idxmax()

        MP_means = self.reference_signs.iloc[:batch_size].groupby(['name'])['distance'].mean().sort_values().head(4)
        UL_means = self.ul_reference_signs.iloc[:batch_size].groupby(['name'])['distance'].mean().sort_values().head(4)

        # (Reversed) normalization is done as the Ultraleap and MP distances work over different measurements.
        # The normalization is reversed as it seems more intuitive that a higher score indicates a likelier gesture.
        MP_means = (MP_means.max() - MP_means) / (MP_means.max() - MP_means.min())
        UL_means = (UL_means.max() - UL_means) / (UL_means.max() - UL_means.min())

        # Multiply the Panda series by the ratio of outliers in the recording.
        MP_means = MP_means.multiply(mp_outlier_score)
        UL_means = UL_means.multiply(ul_outlier_score)

        print(MP_means)
        print(UL_means)
        print(ul_outlier_score)

        # Add the two series together and take the highest scoring gesture.
        predicted_sign = MP_means.add(UL_means, fill_value=0).idxmax()
        return predicted_sign
