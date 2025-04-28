import time
import lietorch
import torch
# from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
from mast3r_slam.odometry import StraightOrSpinOdometry
import mast3r_slam_backends
from mast3r_slam.height_prior import RectanglePlaneEstimator


class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, config, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.K = K
        self.window_size = self.cfg["window_size"]

        # Store edges as separate lists instead of tensors
        self._ii_list = []
        self._jj_list = []
        self._idx_ii2jj_list = []
        self._idx_jj2ii_list = []
        self._valid_match_j_list = []
        self._valid_match_i_list = []
        self._Q_ii2jj_list = []
        self._Q_jj2ii_list = []

        # Store odometry edges as separate lists
        self._odometry_ii_list = []
        self._odometry_jj_list = []
        self._odometry_delta_T_list = []

    def reset(self):
        # Clear all edge lists
        self._ii_list = []
        self._jj_list = []
        self._idx_ii2jj_list = []
        self._idx_jj2ii_list = []
        self._valid_match_j_list = []
        self._valid_match_i_list = []
        self._Q_ii2jj_list = []
        self._Q_jj2ii_list = []

        # Clear odometry lists
        self._odometry_ii_list = []
        self._odometry_jj_list = []
        self._odometry_delta_T_list = []

    def add_odometry_factors(self, ii, jj, delta_T):
        """Add odometry factors to the factor graph
        Make sure delta_T is a SE3 transformation, from jj to ii, 
        i.e. delta_T = T_ii_inv * T_jj
        Args:
            ii (list): index of the first keyframe
            jj (list): index of the second keyframe
            delta_T (torch.Tensor): SE3 transformation from jj to ii
        """
        delta_T = delta_T.unsqueeze(0).to(self.device)
        self._odometry_ii_list.append(ii)
        self._odometry_jj_list.append(jj)
        self._odometry_delta_T_list.append(delta_T)
    
    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii]).to(self.device)
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj]).to(self.device)
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii]).to(self.device)
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj]).to(self.device)
        shape_i = self.frames[0].img_true_shape 
        shape_j = self.frames[0].img_true_shape

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        # TODO: we can store only subset of Xs and Cs, that are valid and above threshold
        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        # Store valid edges in separate lists instead of a single tuple
        self._ii_list = self._ii_list + ii_tensor.tolist()
        self._jj_list = self._jj_list + jj_tensor.tolist()
        for i in range(len(ii_tensor)):
            self._idx_ii2jj_list.append(idx_i2j[i])
            self._idx_jj2ii_list.append(idx_j2i[i])
            self._valid_match_j_list.append(valid_j[i])
            self._valid_match_i_list.append(valid_i[i])
            self._Q_ii2jj_list.append(Qij[i])
            self._Q_jj2ii_list.append(Qji[i])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def _materialise_edges(self):
        if not self._ii_list:
            return False

        oldest_alive = self.frames.oldest_alive_idx()
        
        # Filter edges based on oldest alive index
        survivors = []
        for i in range(len(self._ii_list)):
            if self._ii_list[i] >= oldest_alive and self._jj_list[i] >= oldest_alive:
                survivors.append(i)
        
        # Keep only surviving edges
        self._ii_list = [self._ii_list[i] for i in survivors]
        self._jj_list = [self._jj_list[i] for i in survivors]
        self._idx_ii2jj_list = [self._idx_ii2jj_list[i] for i in survivors]
        self._idx_jj2ii_list = [self._idx_jj2ii_list[i] for i in survivors]
        self._valid_match_j_list = [self._valid_match_j_list[i] for i in survivors]
        self._valid_match_i_list = [self._valid_match_i_list[i] for i in survivors]
        self._Q_ii2jj_list = [self._Q_ii2jj_list[i] for i in survivors]
        self._Q_jj2ii_list = [self._Q_jj2ii_list[i] for i in survivors]
        
        # Filter odometry edges
        odom_survivors = []
        for i in range(len(self._odometry_ii_list)):
            if self._odometry_ii_list[i] >= oldest_alive and self._odometry_jj_list[i] >= oldest_alive:
                odom_survivors.append(i)
        
        self._odometry_ii_list = [self._odometry_ii_list[i] for i in odom_survivors]
        self._odometry_jj_list = [self._odometry_jj_list[i] for i in odom_survivors]
        self._odometry_delta_T_list = [self._odometry_delta_T_list[i] for i in odom_survivors]

        if not self._ii_list:
            return False

        return True

    def get_unique_kf_idx(self):
        # Convert lists to tensors for processing
        ii = torch.tensor(self._ii_list, dtype=torch.long, device=self.device)
        jj = torch.tensor(self._jj_list, dtype=torch.long, device=self.device)
        return torch.unique(torch.cat([ii, jj]), sorted=True)

    def prep_two_way_edges(self):
        # Convert lists to tensors for processing
        ii = torch.tensor(self._ii_list, dtype=torch.long, device=self.device)
        jj = torch.tensor(self._jj_list, dtype=torch.long, device=self.device)
        idx_ii2jj = torch.stack(self._idx_ii2jj_list)
        valid_match_j = torch.stack(self._valid_match_j_list)
        valid_match_i = torch.stack(self._valid_match_i_list)
        Q_ii2jj = torch.stack(self._Q_ii2jj_list)
        Q_jj2ii = torch.stack(self._Q_jj2ii_list)
        
        # Create two-way edges
        ii_two_way = torch.cat((ii, jj), dim=0)
        jj_two_way = torch.cat((jj, ii), dim=0)
        idx_ii2jj_two_way = torch.cat((idx_ii2jj, torch.stack(self._idx_jj2ii_list)), dim=0)
        valid_match_two_way = torch.cat((valid_match_j, valid_match_i), dim=0)
        Q_ii2jj_two_way = torch.cat((Q_ii2jj, Q_jj2ii), dim=0)
        
        return ii_two_way, jj_two_way, idx_ii2jj_two_way, valid_match_two_way, Q_ii2jj_two_way

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        # compute h bar
        pe = RectanglePlaneEstimator()
        camera_height = StraightOrSpinOdometry._CAMERA_HEIGHT
        s_residuals = []
        h, w = self.frames[0].img.shape[-2:]
        for X in Xs:
            h_bar = pe.run(X.cpu().numpy(),image_size=(h,w))
            if h_bar is not None:
                s_residuals.append(camera_height / h_bar)
            else:
                s_residuals.append(-1)

        return Xs, T_WCs, Cs, s_residuals

    def solve_GN_rays(self):
        has_edges = self._materialise_edges()
        if not has_edges:
            return

        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs, s_bar = self.get_poses_points(unique_kf_idx)
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        # Convert odometry lists to tensors
        odom_ii = torch.tensor(self._odometry_ii_list, dtype=torch.long, device=self.device)
        odom_jj = torch.tensor(self._odometry_jj_list, dtype=torch.long, device=self.device)
        odom_delta_T = torch.cat(self._odometry_delta_T_list, dim=0)[:,0,:]

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        sigma_odom_t = 0.0002
        sigma_odom_r = 0.0005
        sigma_ray = 0.003
        sigma_scale_prior = 0.1
        s_bar = torch.tensor(s_bar)

        dx = mast3r_slam_backends.gauss_newton_rays_odom(
            pose_data.to(self.device),
            Xs.to(self.device),
            Cs.to(self.device),
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            odom_ii,
            odom_jj,
            odom_delta_T,
            s_bar,
            sigma_odom_t,
            sigma_odom_r,
            sigma_ray,
            sigma_dist,
            sigma_scale_prior,
            C_thresh,
            Q_thresh,
            pin,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
