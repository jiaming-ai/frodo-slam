import lietorch
import torch
import numpy as np
from mast3r_slam.geometry import get_pixel_coords

class Mapping:
    def __init__(self, keyframes, config):
        self.keyframes = keyframes
        self.config = config
        self.conf_threshold = 1
        self.points = []

        self.subsample_factor = 5


    def get_all_X(self, kf_idxs):
        X = self.keyframes.get_X_by_idxs(kf_idxs) # [T, N, 3]
        C = self.keyframes.get_avg_C_by_idxs(kf_idxs) # [T, N, 1]


        # subsample points
        X = X[::self.subsample_factor]
        C = C[::self.subsample_factor]

        # transform points to world frame
        T_WCs = self.keyframes.get_T_WC_by_idxs(kf_idxs) # [T,1, 8]
        T_Wcs = lietorch.Sim3(T_WCs)  
        X = T_Wcs.act(X) # [T, N, 3]

        X = X.reshape(-1, 3)
        C = C.reshape(-1, 1)

        # remove points with low confidence
        X = X[C > self.conf_threshold]

        return X

    def map_new_frame(self, frame):

        # get dirty keyframes
        kf_idxs = self.keyframes.get_dirty_map_idx()
        if len(kf_idxs) == 0:
            return
        
        Xs = self.get_all_X(kf_idxs) # [N, 3]

        # voxelize