import datetime
import os
import torch
import torch.multiprocessing as mp
import time
import numpy as np

from mast3r_slam.frame import (
    Mode, 
    KeyframesCuda, 
    StatesCuda, 
    create_frame, 
    SharedKeyframes, 
    SharedStates
)
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
)
from mast3r_slam.multiprocess_utils import new_queue
from mast3r_slam.pgo import pos_yaw_to_se3
from mast3r_slam.tracker import FrameTracker
import pypose as pp
from mast3r_slam.visualization import run_visualization
from mast3r_slam.config import config, set_global_config
import logging

def relocalization(frame, keyframes, factor_graph, retrieval_database, config):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            logging.info(f"RELOCALIZING against kf {n_kf - 1} and {kf_idx}")
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                logging.info("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                logging.info("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(config, model, states, keyframes, main2backend, backend2main, K=None):
    set_global_config(config)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, config,K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        try:
            msg = main2backend.get_nowait()
            if isinstance(msg, dict) and "reset" in msg:
                logging.info("Resetting backend")
                factor_graph.reset()
                retrieval_database.reset()

        except Exception as e:
            pass  # No message available

        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database, config)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            logging.info(f"Database retrieval {idx}: {lc_inds}")

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)
                
class VIO:
    def __init__(self, config, img_size, calib=None, device="cuda:0", use_backend=True, visualize=False):
        """
        Initialize the Visual-Inertial Odometry system.
        
        Args:
            config: Configuration dictionary
            img_size: Tuple of (height, width) for the input images
            calib: Optional calibration data
            device: Device to run the model on
        """
        self.config = config
        self.device = device
        self.img_size = img_size
        self.h, self.w = img_size
        self.visualize = visualize
        self.use_backend = use_backend
        self.multi_process = visualize or use_backend
        set_global_config(config)

        # Set up multiprocessing
        mp.set_start_method('spawn', force=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # self.altas = [self.keyframes]
        # Load model
        self.model = load_mast3r(device=self.device)
        
        if self.multi_process:
            manager = mp.Manager()
            self.keyframes = SharedKeyframes(manager, self.h, self.w, device=self.device)
            self.states = SharedStates(manager, self.h, self.w, device=self.device)

        else:
            # more efficient for single process
            self.keyframes = KeyframesCuda(self.h, self.w, device=self.device)

            self.states = StatesCuda(self.h, self.w, device=self.device)
        
        if self.visualize:
            self.main2viz = new_queue(manager, not visualize)
            self.viz2main = new_queue(manager, not visualize)
            self.viz = mp.Process(
                target=run_visualization,
                args=(self.config, self.states, self.keyframes, self.main2viz, self.viz2main),
            )
            self.viz.start()
        
        if use_backend:
            self.main2backend = new_queue(manager, not visualize)
            self.backend2main = new_queue(manager, not visualize)
            self.backend = mp.Process(target=run_backend, args=(config, self.model, self.states, self.keyframes, self.main2backend, self.backend2main))
            self.backend.start()

        
        # Set up calibration
        self.K = None
        if calib is not None:
            config["use_calib"] = True
            self.K = torch.from_numpy(calib).to(self.device, dtype=torch.float32)
            self.keyframes.set_intrinsics(self.K)
        
        # Initialize tracker
        self.tracker = FrameTracker(self.model, self.keyframes, self.device, local_opt_mode=False)
        
        # Initialize state variables
        self.frame_count = 0
        self.loss_track_counter = 0
        self.states.set_mode(Mode.INIT)

    def reset(self):
        """Reset the VIO system"""
        logging.info("Resetting VIO system")
        self.frame_count = 0
        self.loss_track_counter = 0

        # TODO: do we need to keep the old states?
        self.keyframes.reset()
        self.states.reset()
    
        self.tracker.reset(self.keyframes)
        self.states.set_mode(Mode.INIT)

        # if self.visualize:
        #     self.main2viz.put({"new_keyframes": self.keyframes})

        if self.use_backend:
            self.main2backend.put({"reset": True})

    def grab_rgb(self, img, timestamp=None, odom=None):
        """
        Process a new RGB frame and update the pose.
        
        Args:
            img: RGB image as numpy array (H, W, 3) with values in [0, 1]
            timestamp: Optional timestamp for the frame
            odom: Optional odometry data (position, yaw)
            
        Returns:
            success: Boolean indicating if tracking was successful
            pose: Current camera pose as a 4x4 transformation matrix
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Add odometry data if available
        if odom is not None:
            odom_pose = pos_yaw_to_se3(odom['pos'], odom['yaw'])
        else:
            odom_pose = None
            
        frame = create_frame(
            self.frame_count, 
            img, 
            self.states.get_pose(),
            img_size=512, # TODO
            device=self.device, 
            odom=odom_pose
        )
        
        # Initialize tracking if needed
        if self.states.get_mode() == Mode.INIT:
            self.tracker.init_tracking(frame)
            self.states.set_frame(frame)
            self.states.set_mode(Mode.TRACKING)
            self.frame_count += 1
            return True, self._get_current_pose(), True
            
        # Track frame
        match_info, track_success, new_kf = self.tracker.track(frame)
        
        # Handle tracking failure
        if not track_success:
            # TODO: try relocalize if backend is running
            self.loss_track_counter += 1
            if self.loss_track_counter >= self.config["tracking"]["new_map_after_loss_track_N"]:
                logging.info(f"Creating new map after tracking loss for {self.loss_track_counter} frames...")
                self.reset()
                self.tracker.init_tracking(frame)
                self.states.set_frame(frame)
                self.states.set_mode(Mode.TRACKING)
                self.frame_count += 1
                return False, self._get_current_pose(), True
        else:
            # Update frame if tracking is successful
            self.states.set_frame(frame)
            self.loss_track_counter = 0
            if self.use_backend and new_kf:
                self.states.queue_global_optimization(len(self.keyframes) - 1)

        self.frame_count += 1
        return track_success, self._get_current_pose(), new_kf
    
    def _get_current_pose(self):
        """Get the current camera pose as a 4x4 transformation matrix"""
 
        # Convert from Sim3 to SE3 (4x4 matrix)
        pose = pp.Sim3(self.states.T_WC).squeeze(0).matrix().cpu().numpy()
        return pose
    
    def get_pose(self):
        """Get the current camera pose"""
        return self._get_current_pose()
    
    def get_keyframes(self):
        """Get the current keyframes"""
        return self.keyframes
    
    def terminate(self):
        """Terminate the VIO system"""
        self.states.set_mode(Mode.TERMINATED)

if __name__ == "__main__":
    vio = VIO(config_path="config/base_no_fnn.yaml", img_size=(640, 640))
    vio.grab_rgb(np.zeros((640, 640, 3)))
    logging.info(vio.get_pose())
