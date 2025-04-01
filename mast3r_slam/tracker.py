import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.local_mapping import FactorGraph
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import (
    mast3r_match_asymmetric, 
    mast3r_inference_mono, 
    mast3r_match_symmetric
)
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r_slam.visualization_utils import visualize_matches
import matplotlib.pyplot as plt
from mast3r_slam.matching import pixel_to_lin, lin_to_pixel
from mast3r_slam.frame import Mode, KeyframesCuda
from mast3r_slam.pgo import PoseGraph
class LocalMapOptimizer:
    def __init__(self, 
                 model,
                 buffer_size=6, 
                 offset_to_current=[-1,-3],
                 device="cuda"):
        assert -min(offset_to_current) < buffer_size,\
            "Window size must be less than the minimum edge"

        self.device = device
        self.cfg = config
        self.frames_buffer = KeyframesCuda(
            config["image_size"][0],
            config["image_size"][1],
            buffer=buffer_size,
            device=device
        )
        self.img_shape = config["image_size"]
        self.buffer_size = buffer_size
        self.offset_to_current = offset_to_current
        self._idx = -1
        self.cache_to_frame_id= {}
        self.graph = FactorGraph(model, self.frames_buffer, device=self.device)

        self.enabled = False

    def add_frame(self, frame, idx):
        if not self.enabled:
            return
        self._idx += 1
        # when new frame is added, some previous factors are invalid
        if self._idx >= self.buffer_size:
            # when idx is wrapped around, remove the factors that contain the old frame
            self.graph.remove_factors_i(self._idx % self.buffer_size)

        # map the idx to the frame id
        cache_idx = self.frames_buffer.append(frame)
        self.cache_to_frame_id[cache_idx] = idx

        self.add_factors()


    def optimize(self, init=False):
        if len(self.graph.factors) < 19:
            return None, None
        return self.graph.solve_GN_rays(init)


    # def prepare_decoder_inputs(self):
    #     """Get the encoder feature for a given frame id
    #     Assumes the buffer is full
    #     """
    #     if self._idx < self.buffer_size:
    #         return None, None, None, None, None, None
    #     idxs_i = [(self._idx + offset) % self.buffer_size for offset in self.offset_to_current]
    #     idxs_j = [self._idx % self.buffer_size] * len(self.offset_to_current)
    #     frames_i = [self.encoder_buffer[idx] for idx in idxs_i]
    #     frames_j = [self.encoder_buffer[idx] for idx in idxs_j]

    #     feats_i = torch.cat([frame[0] for frame in frames_i], dim=0)
    #     feats_j = torch.cat([frame[0] for frame in frames_j], dim=0)
    #     pos_i = torch.cat([frame[1] for frame in frames_i], dim=0)
    #     pos_j = torch.cat([frame[1] for frame in frames_j], dim=0)
    #     return feats_i, pos_i, feats_j, pos_j, idxs_i, idxs_j
    
    def reset(self):
        self.frames_buffer.reset() 
        self._idx = -1
        self.graph.reset()
        self.cache_to_frame_id = {}

    def add_factors(self):
        """Add factors to the graph"""
        idxs_i = [(self._idx + offset) for offset in self.offset_to_current]
        idxs_i = [i % self.buffer_size for i in idxs_i if i >= 0]
        idxs_j = [self._idx % self.buffer_size] * len(idxs_i)
        if len(idxs_i) == 0:
            return
        self.graph.add_factors(idxs_i, idxs_j)
        

class FrameTracker:
    def __init__(self, model, frames, device, states):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.states = states
        # cache the last keyframe
        self.last_kf = None
        self.device = device

        self.reset_idx_f2k()
        
        self.local_opt = PoseGraph(device=self.device)


        # self._offset_to_current = [-1, -3, -5]
        # self._local_window_size = 20

        # # n key frame
        # self._mode = Mode.INIT
        # # the relative id from the current keyframe
        # self.local_opt = LocalMapOptimizer(
        #     model=self.model,
        #     buffer_size=self._local_window_size, 
        #     offset_to_current=self._offset_to_current,
        #     device=self.device
        # )
    
    def reset(self, keyframes):
        """Reset the tracker
        Reset should be called after a new map is created
        """
        # assumes the keyframes contains the first keyframe
        self.keyframes = keyframes
        self.reset_idx_f2k()
        self.last_kf = None

        self.local_opt.reset()
        # self._mode = Mode.INIT
        # self.local_opt.reset()

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def init_tracking(self, frame: Frame):
        """Initialize the tracker
        Args:
            frame (Frame): frame to initialize
        """
        if frame.feat is None:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X_init, C_init)

            # only add encoder result if it's not been added yet
            # self.local_opt.add_frame(frame, 0)
            self.local_opt.add_frame(frame)
            self.local_opt.last_frame_is_keyframe(0)

        self.keyframes.append(frame)
        self.states.set_mode(Mode.TRACKING)
        self.states.set_frame(frame)
        self.last_kf = frame
        self.img_shape = frame.img_true_shape



    def track(self, frame: Frame) -> tuple[list, bool]:
        """Track the frame
        Args:
            frame (Frame): frame to track
        Returns:
            list: list of results
            bool: True if tracking failed
        """
        # only Dff, Dkf is HxWxC
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf, Dff, Dkf \
            = mast3r_match_asymmetric(
            self.model, frame, self.last_kf, idx_i2j_init=self.idx_f2k
        )
        frame.update_pointmap(Xff, Cff)

        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        valid_match_k = valid_match_k[0]
        idx_f2k = idx_f2k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)
        Cf = Cff[idx_f2k]

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ckf > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()

        if match_frac < self.cfg["min_match_frac"]:
            visualize_matches = False

            if visualize_matches:
                # visualize matches
                matchesff = lin_to_pixel(idx_f2k[valid_opt[:, 0]], frame.img.shape[-1])
                matcheskf = lin_to_pixel(torch.arange(valid_opt.shape[0],\
                    device=self.device)[valid_opt[:, 0]], self.last_kf.img.shape[-1])
                visualize_matches(
                    matchesff.cpu().numpy(),
                    matcheskf.cpu().numpy(),
                    frame.img[0].cpu(),
                    self.last_kf.img.cpu(),
                )

            # optionally, use fnn matching
            if self.cfg["use_fnn"]:
                print(f"Running fnn matching for frame {frame.frame_id}")
                matchesff, matcheskf = fast_reciprocal_NNs(
                    Dff,
                    Dkf,
                    device=self.device,
                    dist='dot'
                )
                # need to make sure the index
                idx_kf = pixel_to_lin(
                    torch.tensor(matcheskf.copy(), device=self.device), 
                    self.last_kf.img.shape[-1])
                valid_match_k = torch.zeros_like(valid_match_k, dtype=torch.bool)
                valid_match_k[idx_kf] = True
                
                idx_f2k_valid = pixel_to_lin(
                    torch.tensor(
                        matchesff.copy(), 
                        device=self.device),
                    frame.img.shape[-1])

                idx_f2k = torch.zeros_like(idx_f2k)
                idx_f2k[idx_kf] = idx_f2k_valid



                Qk = torch.sqrt(Qff[idx_f2k] * Qkf)
                Cf = Cff[idx_f2k]

                # Get valid
                # Use canonical confidence average
                valid_Cf = Cf > self.cfg["C_conf"]

                valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
                valid_kf = valid_match_k & valid_Q

                match_frac = valid_opt.sum() / valid_opt.numel()

                if visualize_matches:
                    visualize_matches(
                        matchesff,
                        matcheskf,
                        frame.img[0].cpu(),
                        self.last_kf.img.cpu(),
                    )
                if match_frac < self.cfg["min_match_frac_fnn"]:
                    print(f"Skipped frame {frame.frame_id} after fnn matching")
                    return [], True
                
                # reset idx_f2k
                self.reset_idx_f2k()

            else:
                print(f"Skipped frame {frame.frame_id} without fnn matching")
                return [], True



        use_calib = config["use_calib"]
        if use_calib:
            K = self.last_kf.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, self.last_kf, idx_f2k, self.img_shape, use_calib, K
        )

        try:
            # Track
            if not use_calib:
                # get only valid points
                Xf = Xf[valid_opt[:, 0]]
                Xk = Xk[valid_opt[:, 0]]
                Qk = Qk[valid_opt[:, 0]]
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    self.img_shape,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return [], True

        frame.T_WC = T_WCf

        # add frame to the local optimizer
        # must be called after the pose is updated
        self.local_opt.add_frame(frame)

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        self.last_kf.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap

        unique_valid_match = torch.unique(idx_f2k[valid_kf[:, 0]]).shape[0] / valid_kf.numel()
        new_kf = unique_valid_match < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:
            # update the keyframe in shared memory when new keyframe is created
            self.keyframes[len(self.keyframes) - 1] = self.last_kf
            self.reset_idx_f2k()

            # set the current frame as the last keyframe
            self.last_kf = frame
            idx = self.keyframes.append(frame)

            self.local_opt.last_frame_is_keyframe(idx)

            # print(f"before optimize: {frame.T_WC.data[0]}")
            success = self.local_opt.optimize()
            if success:
                kf_poses, kf_idx = self.local_opt.get_kf_poses()
                self.keyframes.update_T_WCs(kf_poses, kf_idx)

                # update the last keyframe pose
                self.last_kf.T_WC.data = kf_poses[kf_idx == idx][0].to(self.device)
                # print(f"after optimize: {frame.T_WC.data[0]}")




            # perform local optimization if a new keyframe is created
            # self.local_opt.add_frame(frame, idx)

            # T_WCs, unique_kf_idx_cache = self.local_opt.optimize(self._mode == Mode.INIT)
            # if T_WCs is not None:
            #     unique_kf_idx_cache = unique_kf_idx_cache.tolist()
            #     unique_kf_idx = [self.local_opt.cache_to_frame_id[idx] for idx in unique_kf_idx_cache]

            #     self.keyframes.update_T_WCs(T_WCs, torch.tensor(unique_kf_idx))
            #     print(f"Local optimization done for frame {frame.frame_id}")
            #     if self._mode == Mode.INIT:
            #         self._mode = Mode.TRACKING

        return (
            [
                self.last_kf.X_canon,
                self.last_kf.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        """Optimize relative pose using ray distance and distance
        Args:
            Xf (torch.Tensor): only valid points
            Xk (torch.Tensor): only valid points
            T_WCf (torch.Tensor): Frame pose
            T_WCk (torch.Tensor): Keyframe pose
            Qk (torch.Tensor): valid feature point confidence
            valid (torch.Tensor): valid feature points
        """
   
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
