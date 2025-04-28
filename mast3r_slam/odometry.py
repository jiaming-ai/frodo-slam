#!/usr/bin/env python3
"""
Hybrid wheel + visual odometry for robots that only drive straight or spin in‑place.

NEW 18 Apr 2025
--------------
* `visualize()` — live 2‑D map drawn in an OpenCV window.
* Internal path buffer (`self._path`) for trajectory plotting.

Run this file directly to see the real‑time visualisation.
"""

import base64
import json
import math
import os
import pickle
import random
import statistics
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
import requests
from loguru import logger
import pypose as pp
import torch
import lietorch
# ----------  helpers for ray‑angle estimation --------------------------------
def load_directions_dict(json_file: str) -> Dict[str, List[float]]:
    """Load { "x,y": [dx,dy,dz], … } → dict[str,list[float, float, float]]."""
    with open(json_file, "r") as fp:
        return json.load(fp)


def detect_and_match_orb(
    img1: np.ndarray, img2: np.ndarray, nfeatures: int
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    """Detect + BF‑match ORB features, sorted by distance."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    return k1, k2, sorted(matches, key=lambda m: m.distance)
    
def _bearing_xz(v: np.ndarray) -> float:
    """Return atan2(x, z) (rad) – bearing in ground plane."""
    return math.atan2(v[0], v[2])

def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return unsigned angle (deg) between two 3‑D vectors."""
    v1, v2 = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    dot = np.clip(v1.dot(v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))


def ransac_mode(
    angles: List[float], iters: int = 100, thresh_rad: float = 0.1
) -> float | None:
    """
    Robustly pick the dominant angle with 1‑D RANSAC.
    Returns best median or None if <3 inliers.
    """
    if len(angles) < 3:
        return None
    best_med, best_inliers = None, 0
    for _ in range(iters):
        a0 = random.choice(angles)
        inliers = [a for a in angles if abs(a - a0) <= thresh_rad]
        if len(inliers) > best_inliers:
            best_inliers = len(inliers)
            best_med = statistics.median(inliers)
    if best_inliers < 3:
        return None
    return best_med

def pos_yaw_to_se3(pos, yaw):
    """
    Convert position and yaw to SE(3)
    """
    # ensure tensors
    pos = torch.as_tensor(pos, dtype=torch.float32)
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    # we negate yaw to match your original cos(-yaw), sin(-yaw)
    half = -0.5 * yaw
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)

    # quaternion for rotation about Y-axis: axis = (0,1,0)
    # q = [axis * sin(θ/2), cos(θ/2)]
    q = torch.tensor([0.0, sin_h, 0.0, cos_h], dtype=torch.float32)

    # translation: T[0,3] = -pos[1], T[1,3]=0, T[2,3]=pos[0]
    t = torch.tensor([-pos[1], 0.0, pos[0]], dtype=torch.float32)
    return lietorch.SE3(torch.cat([t, q]).unsqueeze(0))


def set_default_params(robot_type: str = "mini"):
    if robot_type == "mini":
        StraightOrSpinOdometry._WHEEL_DIAM_M = 0.095
        StraightOrSpinOdometry._TRACK_M = 0.160
        StraightOrSpinOdometry._CAMERA_OFFSET_M = 0.075
        StraightOrSpinOdometry._CAMERA_HEIGHT = 0.148
        StraightOrSpinOdometry._CIRC_M = math.pi * StraightOrSpinOdometry._WHEEL_DIAM_M
    elif robot_type == "zero":
        StraightOrSpinOdometry._WHEEL_DIAM_M = 0.13
        StraightOrSpinOdometry._TRACK_M = 0.2
        StraightOrSpinOdometry._CAMERA_OFFSET_M = 0.06
        StraightOrSpinOdometry._CAMERA_HEIGHT = 0.561
        StraightOrSpinOdometry._CIRC_M = math.pi * StraightOrSpinOdometry._WHEEL_DIAM_M

# --------------  main odometry class ----------------------------------------
class StraightOrSpinOdometry:
    # --- geometry -----------------------------------------------------------
    # # mini
    # _WHEEL_DIAM_M = 0.095
    # _TRACK_M = 0.160
    # _CAMERA_OFFSET_M = 0.075

    # zero
    _WHEEL_DIAM_M = 0.13
    _TRACK_M = 0.2
    _CAMERA_OFFSET_M = 0.06
    _CAMERA_HEIGHT = 0.561

    _CIRC_M = math.pi * _WHEEL_DIAM_M
    _RPM_EQ_EPS = 5
    # --- feature / ray params ----------------------------------------------
    _FEATURES_MAX = 2000
    _MIN_MATCH_ANGLES = 15          # require this many usable ray pairs
    _RANSAC_THRESH_RAD = 0.05 # 3 degrees
    _RANSAC_ITERS = 150
    _ORB_BRUTEFORCE_LEVELS = [200, 1000, 2000, 5000]

    def __init__(
        self,
        robot_type: str = "mini", # "zero" | "mini"
        rpm_api: str = "http://localhost:8000/data",
        cam_api: str = "http://localhost:8000/v2/front",
        poll_s: float = 0.1,
        timeout_s: float = 2.0,
    ):
        set_default_params(robot_type)
        if robot_type == "mini":
            self._dirs = load_directions_dict("config/pixel_direction_dict_s.json")
        elif robot_type == "zero":
            self._dirs = load_directions_dict("config/pixel_direction_dict.json")

        # REST end‑points
        self._rpm_api, self._cam_api = rpm_api, cam_api
        self._poll_s, self._timeout = poll_s, timeout_s

        # pose state
        self._x = self._y = self._th = 0.0
        self._path: deque[Tuple[float, float]] = deque([(0.0, 0.0)], maxlen=2000)
        self._lock = threading.Lock()

        # integration bookkeeping
        self._prev_frame: np.ndarray | None = None
        self._prev_ts: float | None = None

        # thread control
        self._running = False
        self._thread: threading.Thread | None = None

    # --------------  public API  -------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, join: bool = True):
        self._running = False
        if join and self._thread:
            self._thread.join()

    def get_pose(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._x, self._y, self._th

    def get_frame_and_pose(self, resize = 512) -> Tuple[float, np.ndarray, lietorch.SE3]:

        # TODO handles no frame case
        with self._lock:
            frame = self._prev_frame
            pos = self._x, self._y
            yaw = self._th
        
        if frame is None:
            return None, None, None
        pose = pos_yaw_to_se3(pos, yaw)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Resize frame if needed
        h, w = frame.shape[:2]
        target_size = resize

        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame = frame[:,:,::-1]
        timestamp = time.time()
        return timestamp, frame / 255.0, pose
    # --------------  low‑level utils  --------------------------------------
    @staticmethod
    def _rpm_to_mps(rpm: float) -> float:
        return rpm / 60.0 * StraightOrSpinOdometry._CIRC_M

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi


    def _yaw_from_frames_OF(self, prev, cur)->float|None:
        g0=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
        g1=cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        p0=cv2.goodFeaturesToTrack(g0,self._FEATURES_MAX,0.01,8)
        if p0 is None or len(p0)<6: return None
        p0=p0.reshape(-1,2)
        p1,st,_=cv2.calcOpticalFlowPyrLK(g0,g1,p0,None)
        if p1 is None or st is None: return None
        p1=p1.reshape(-1,2); mask=st.ravel()==1
        if mask.sum()<6: return None
        g0p,g1p=p0[mask],p1[mask]
        h,w=g0.shape; c=np.array([w*0.5,h*0.5],np.float32)
        v0,v1=g0p-c, g1p-c
        cross=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0]
        dot=(v0*v1).sum(1)
        ang=np.arctan2(cross,dot)
        return float(np.median(ang))

    # --------------  new ray‑based yaw  ------------------------------------
    def _yaw_from_rays(self, prev: np.ndarray, cur: np.ndarray) -> float | None:
        g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cur,  cv2.COLOR_BGR2GRAY)

        for nfeat in self._ORB_BRUTEFORCE_LEVELS:
            k0, k1, matches = detect_and_match_orb(g0, g1, nfeat)
            if len(matches) < self._MIN_MATCH_ANGLES:
                continue

            angles: List[float] = []
            for m in matches:
                x0, y0 = map(int, map(round, k0[m.queryIdx].pt))
                x1, y1 = map(int, map(round, k1[m.trainIdx].pt))
                key0, key1 = f"{x0},{y0}", f"{x1},{y1}"
                if key0 in self._dirs and key1 in self._dirs:
                    v0, v1 = np.asarray(self._dirs[key0]), np.asarray(self._dirs[key1])
                    # a = angle_between(v0, v1)
                    a =  self._wrap(_bearing_xz(v1) - _bearing_xz(v0))
                    if not math.isnan(a):
                        angles.append(a)
                if len(angles) >= self._MIN_MATCH_ANGLES:
                    break  # no need to process more

            if len(angles) < 3:
                continue

            # robust aggregate
            best = ransac_mode(
                angles,
                iters=self._RANSAC_ITERS,
                thresh_rad=self._RANSAC_THRESH_RAD,
            )
            if best is None:
                best = statistics.median(angles)
            return best

        return None  # failed

    # --------------  main fusion loop  -------------------------------------
    def _loop(self):
        while self._running:
            # wheel RPMs ------------------------------------------------
            try:
                rpm_rows = (
                    requests.get(self._rpm_api, timeout=self._timeout)
                    .json()
                    .get("rpms", [])
                )
            except Exception as e:
                logger.error(f"Error getting RPMs: {e}")
                continue

            rpm_rows.sort(key=lambda r: r[4])  # by timestamp

            # camera frame (Base‑64 JPEG) ------------------------------
            try:
                b64_img = (
                    requests.get(self._cam_api, timeout=self._timeout)
                    .json()
                    .get("front_frame", "")
                )
                frame = (
                    cv2.imdecode(np.frombuffer(base64.b64decode(b64_img), np.uint8),
                                    cv2.IMREAD_COLOR)
                    if b64_img else None
                )
            except Exception as e:
                logger.error(f"Error getting camera frame: {e}")
                continue

            # first compute yaw from frames
            if frame is not None and self._prev_frame is not None:
                try:
                    dth = self._yaw_from_rays(self._prev_frame, frame)
                except Exception as e:
                    logger.error(f"Error computing yaw from frames: {e}")
                    dth = 0

                if dth is not None:
                    # with self._lock:
                    #     self._th = self._wrap(self._th + dth)
                    #     self._path.append((self._x, self._y))

                    with self._lock:
                        old_th = self._th
                        new_th = self._wrap(old_th + dth)

                        # account for the camera’s forward offset:
                        r = self._CAMERA_OFFSET_M
                        dy = r * (math.sin(new_th) - math.sin(old_th))
                        dx = r * (math.cos(new_th) - math.cos(old_th))

                        self._x += dx
                        self._y += dy
                        self._th = new_th
                        self._path.append((self._x, self._y))

            # integrate each RPM sample -------------------------------
            straight = False
            for r1, r2, r3, r4, ts in rpm_rows:
                if self._prev_ts is not None and ts <= self._prev_ts:
                    continue
                dt = 0.0 if self._prev_ts is None else ts - self._prev_ts
                self._prev_ts = ts

                rpm_l, rpm_r = 0.5 * (r1 + r3), 0.5 * (r2 + r4)
                same_sign = (rpm_l * rpm_r) > 0
                close_mag = abs(rpm_l - rpm_r) <= self._RPM_EQ_EPS
                straight = same_sign and close_mag

                if straight:
                    # only consider moving when the robot is moving straight
                    v = self._rpm_to_mps(rpm_l)
                    with self._lock:
                        self._x += v * math.cos(self._th) * dt
                        self._y += v * math.sin(self._th) * dt
                        self._path.append((self._x, self._y))

            if frame is not None:
                self._prev_frame = frame

            time.sleep(self._poll_s)

    # ----------------------  realtime visualiser  --------------------------
    def visualize(self, win="Odometry", scale_px=100, img_size=800):
        """
        Pop up an OpenCV window that shows the path and current heading.
        Press **q** to quit.
        
        Robot control:
        - **w** - move forward
        - **s** - move backward
        - **a** - rotate left
        - **d** - rotate right
        - **space** - stop
        """
        cv2.namedWindow(win)
        origin = (img_size // 2, img_size // 2)
        
        # Robot control settings
        robot_speed = 0.5
        last_key_time = 0
        key_repeat_interval = 0.2  # seconds between repeated commands
        key_states = {ord("w"): False, ord("a"): False, ord("s"): False, ord("d"): False}

        while True:
            canvas = np.zeros((img_size, img_size, 3), np.uint8)

            # axes
            cv2.line(canvas, (0, origin[1]), (img_size, origin[1]), (40, 40, 40), 1)
            cv2.line(canvas, (origin[0], 0), (origin[0], img_size), (40, 40, 40), 1)

            with self._lock:
                pts = list(self._path)  # Convert deque to list for slicing
                x, y, th = self._x, self._y, self._th

            # trajectory
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    xa, ya = pts[i]
                    xb, yb = pts[i + 1]
                    p1 = (int(origin[0] + xa * scale_px), int(origin[1] - ya * scale_px))
                    p2 = (int(origin[0] + xb * scale_px), int(origin[1] - yb * scale_px))
                    cv2.line(canvas, p1, p2, (0, 255, 0), 2)

            # heading arrow
            tip = (int(origin[0] + x * scale_px), int(origin[1] - y * scale_px))
            arrow = (
                int(tip[0] + 30 * math.cos(th)),
                int(tip[1] - 30 * math.sin(th)),
            )
            cv2.arrowedLine(canvas, tip, arrow, (0, 0, 255), 2, tipLength=0.3)
            
            # Add control instructions to the visualization
            instructions = [
                "Robot Control:",
                "w - forward",
                "s - backward", 
                "a - rotate left",
                "d - rotate right",
                "space - stop",
                f"Speed: {robot_speed:.1f}"
            ]
            
            for i, text in enumerate(instructions):
                cv2.putText(canvas, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                           
            # Display current camera frame if available
            if self._prev_frame is not None:
                # Create a smaller version of the frame to display in corner
                frame_height, frame_width = self._prev_frame.shape[:2]
                frame_display_width = img_size // 3
                frame_display_height = int(frame_height * (frame_display_width / frame_width))
                
                frame_resized = cv2.resize(self._prev_frame, 
                                          (frame_display_width, frame_display_height))
                
                # Position in top-right corner with padding
                x_offset = img_size - frame_display_width - 10
                y_offset = 10
                
                # Create a region of interest and overlay the frame
                roi = canvas[y_offset:y_offset+frame_display_height, 
                            x_offset:x_offset+frame_display_width]
                
                # Add a border around the frame
                cv2.rectangle(canvas, 
                             (x_offset-2, y_offset-2), 
                             (x_offset+frame_display_width+2, y_offset+frame_display_height+2), 
                             (100, 100, 100), 2)
                
                # Overlay the frame
                canvas[y_offset:y_offset+frame_display_height, 
                      x_offset:x_offset+frame_display_width] = frame_resized

            cv2.imshow(win, canvas)
            key = cv2.waitKey(30) & 0xFF
            
            # Handle key press events
            current_time = time.time()
            if key == ord("q"):
                break
            elif key == ord(" "):  # Space to stop
                self._send_robot_command("stop", 0)
                # Reset all key states
                for k in key_states:
                    key_states[k] = False
            elif key in key_states:
                key_states[key] = True
            elif key == ord("+") or key == ord("="):
                robot_speed = min(1.0, robot_speed + 0.1)
            elif key == ord("-"):
                robot_speed = max(0.1, robot_speed - 0.1)
                
            # Process currently pressed keys
            if current_time - last_key_time >= key_repeat_interval:
                if any(key_states.values()):
                    last_key_time = current_time
                    if key_states[ord("w")]:
                        self._send_robot_command("forward", robot_speed)
                    elif key_states[ord("s")]:
                        self._send_robot_command("backward", robot_speed)
                    elif key_states[ord("a")]:
                        self._send_robot_command("left", robot_speed)
                    elif key_states[ord("d")]:
                        self._send_robot_command("right", robot_speed)

        # Make sure to stop the robot when closing the window
        self._send_robot_command("stop", 0)
        cv2.destroyWindow(win)
        
    def _send_robot_command(self, command, speed=0.5):
        """
        Send a command to the robot using the control API.
        
        Args:
            command: Direction to move ("forward", "backward", "left", "right", "stop")
            speed: Speed factor between 0 and 1
        """
        linear, angular = 0, 0
        
        if command == "forward":
            linear = speed
        elif command == "backward":
            linear = -speed
        elif command == "left":
            angular = speed
        elif command == "right":
            angular = -speed
        # "stop" command keeps linear and angular at 0
        
        try:
            response = requests.post(
                'http://localhost:8000/control',
                json={'command': {'linear': linear, 'angular': angular}}
            )
            if response.status_code == 200:
                print(f"Robot command sent: {command} (linear={linear}, angular={angular})")
            else:
                print(f"Failed to send robot command: {response.status_code}")
        except Exception as e:
            print(f"Error sending robot command: {e}")


def record_odometry(data_path: str, duration_s: float = 60.0, poll_s: float = 0.1, robot_type: str = "mini"):
    """
    Record odometry data to a file for the specified duration while showing visualization.
    
    Args:
        data_path: Path to save the recorded data
        duration_s: Duration to record in seconds
        poll_s: Time between polls in seconds
    """
    
    data_path = data_path + f"_{robot_type}.pkl"
    logger.info(f"Recording odometry data to {data_path} for {duration_s} seconds")
    odo = StraightOrSpinOdometry(robot_type=robot_type)
    odo.start()
    
    # Start visualization in a separate thread
    vis_thread = threading.Thread(target=odo.visualize, daemon=True)
    vis_thread.start()
    
    try:
        data = []
        last_frame = None
        start_time = time.time()
        
        while time.time() - start_time < duration_s:
            # Get current timestamp, frame and pose
            timestamp, frame, pose = odo.get_frame_and_pose()
            if frame is None:
                continue
            
            # Only store if frame is different from the last one
            if last_frame is None or not np.array_equal(frame, last_frame):
                data.append({
                    'timestamp': timestamp,
                    'frame': frame,
                    'pose': pose,
                })
                last_frame = frame.copy()
                logger.info(f"Recorded frame at t={timestamp:.2f}, pose={pose}")
            
            time.sleep(poll_s)
        
        # Save data to file
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Recorded {len(data)} frames to {data_path}")
    
    finally:
        odo.stop()

def replay_odometry(data_path: str):
    """
    Load and replay odometry data from a file.
    
    Args:
        data_path: Path to the recorded data file
        
    Returns:
        List of dictionaries containing timestamp, frame, pose
    """
    
    logger.info(f"Loading odometry data from {data_path}")
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {len(data)} frames from {data_path}")
        return data
    
    except Exception as e:
        logger.error(f"Failed to load odometry data: {e}")
        return []

class OdometryData(torch.utils.data.Dataset):
    def __init__(self, data_path: str, wall_clock = False, use_odometry = False, **kwargs):
        self.data = replay_odometry(data_path)
        self.wall_clock = wall_clock
        self.use_odometry = use_odometry
        # For wall_clock mode
        self.last_real_time = None
        self.last_data_time = None
        self.current_idx = 0
        self.idx = 0
        self.robot_type = data_path.split("_")[-1].split(".")[0]
        set_default_params(self.robot_type)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def get_frame_and_pose(self):
        """
        Get frame and pose at the specified index or based on wall clock time.
        
        Args:
            idx: Index to retrieve (ignored if wall_time=True)
            
        Returns:
            timestamp, frame, pose tuple
        """
        if not self.wall_clock:
            # Simple indexed access mode
            if self.idx >= len(self.data):
                return None, None, None
            item = self.data[self.idx]
            self.idx += 1
            # frame = item['frame'][:,:,::-1]
            frame = item['frame']
            if self.use_odometry:
                return item['timestamp'], frame, item['pose']
            else:
                return item['timestamp'], frame, None
        
        # Wall clock time simulation mode
        current_real_time = time.time()
        
        # Initialize timing on first call
        if self.last_real_time is None:
            self.last_real_time = current_real_time
            self.last_data_time = self.data[0]['timestamp']
            self.current_idx = 0
            if self.use_odometry:
                return self.data[0]['timestamp'], self.data[0]['frame'], self.data[0]['pose']
            else:
                return self.data[0]['timestamp'], self.data[0]['frame'], None
        
        # Calculate elapsed time since last call
        real_elapsed = current_real_time - self.last_real_time
        target_data_time = self.last_data_time + real_elapsed
        
        # Find the frame with timestamp closest to but greater than target_data_time
        while self.current_idx < len(self.data) - 1:
            self.current_idx += 1
            if self.data[self.current_idx]['timestamp'] > target_data_time:
                break
        
        # Update timing state
        self.last_real_time = current_real_time
        self.last_data_time = self.data[self.current_idx]['timestamp']
        
        # Return the current frame and pose
        item = self.data[self.current_idx]
        # frame = item['frame'][:,:,::-1]
        frame = item['frame']
        if self.use_odometry:
            return item['timestamp'], frame, item['pose']
        else:
            return item['timestamp'], frame, None
        
# ---------------------------------------------------------------------------
# run standalone ------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # odo = StraightOrSpinOdometry(directions_json="config/pixel_direction_dict_s.json")
    # odo.start()
    # try:
    #     odo.visualize()
    # finally:
    #     odo.stop()

    record_odometry("datasets/recorded/outdoor.pkl", duration_s=300.0, poll_s=0.1, robot_type="mini")
    # data = replay_odometry("odometry.pkl")
    # print(data)