from os import path, makedirs, listdir
from utils.landmark_utils import save_array
import ctypes

lib = ctypes.CDLL("PollingSample.dll")

def get_new_landmarks():
    get_landmarks = lib.getLandmarks
    get_landmarks.restype = ctypes.POINTER(ctypes.c_float)
    return get_landmarks()

class LeapListener():
    def __init__(self):
        self.frame = -1
        self.lh_landmarks = []
        self.rh_landmarks = []
        lib.loadLandmarks()
    
    def retrieve_landmarks(self):
        return self.lh_landmarks, self.rh_landmarks

    def pickle_landmarks(self, category, sign_type, video_number):
        """
        Iterates through all the fingers in a hand and returns their landmarks in a list.
        Also adds a keymark from the wrist.

        :param hand: hand object from Leap's Hand Tracking.
        :return: a list containing 63 3D points as tuples.
        """
        new_dataset = "%s-%d" % (sign_type, video_number)
        makedirs(path.join("data/ultraleapdataset", category, sign_type, new_dataset), exist_ok=True)
        save_array(self.lh_landmarks, path.join("data/ultraleapdataset", category, sign_type, new_dataset,
                                        "lh_%s-%d.pickle" % (sign_type, video_number)))
        save_array(self.rh_landmarks, path.join("data/ultraleapdataset", category, sign_type, new_dataset,
                                        "rh_%s-%d.pickle" % (sign_type, video_number)))

    def reset_landmarks(self):
        self.lh_landmarks = []
        self.rh_landmarks = []

    def get_frame(self):
        getFrame = lib.getFrame
        getFrame.restype = ctypes.c_int64
        self.frame = getFrame()
        return getFrame()

    def get_landmarks(self):
        landmarks = get_new_landmarks()[:126]
        hand_len = len(landmarks) // 2
        self.lh_landmarks.append(landmarks[:hand_len])
        self.rh_landmarks.append(landmarks[hand_len:])

    def _get_landmarks(self):
        get_landmarks = lib.getLandmarks
        get_landmarks.restype = ctypes.POINTER(ctypes.c_float)
        return get_landmarks()
    