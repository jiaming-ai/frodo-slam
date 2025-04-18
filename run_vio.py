import argparse
import logging
import os
import pickle
import time
import yaml
import torch
import numpy as np
from mast3r_slam.config import load_config, config, set_global_config

from mast3r_slam.dataloader import load_dataset, Intrinsics
from mast3r_slam.vio import VIO

import sys
import requests
import base64
from PIL import Image
import io
import cv2 
import threading
from collections import deque

venv_bin = os.path.join(sys.prefix, 'bin')
os.environ['PATH'] = venv_bin + os.pathsep + os.environ['PATH']

def get_frame(resize=512):
    response = requests.get('http://localhost:8000/v2/front')
    if response.status_code == 200:
        data = response.json()
        base64_image = data.get('front_frame')
        if base64_image:
            # Decode base64 image
            image_bytes = base64.b64decode(base64_image)
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Convert to numpy array
            frame = np.array(pil_image)

            # Convert RGBA to RGB if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            # PIL already returns RGB, so no conversion needed for 3-channel images

            # Resize frame if needed
            h, w = frame.shape[:2]
            target_size = resize
            if max(h, w) > target_size:
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return frame / 255.0
    return None
    
def fetch_data(image_buffer, stop_event, interval=0.1):
    """Continuously fetch frames from the robot camera and store in buffer"""
    while not stop_event.is_set():
        frame = get_frame()
        if frame is not None:
            timestamp = time.time()
            # Add to the left (newest first)
            image_buffer.appendleft((timestamp, frame))
          
        time.sleep(interval)  # Wait before fetching next frame

def run_robot(args):
    # Load config
    load_config(args.config)
    
    # Get first frame to determine image size
    frame = get_frame(args.resize)
    if frame is None:
        print("Failed to get initial frame from robot camera")
        return
    
    h, w = frame.shape[:2]
    # Load calibration if provided
    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Create camera intrinsics
        from mast3r_slam.dataloader import Intrinsics
        camera_intrinsics = Intrinsics.from_calib(
            (h, w),
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )
        K = camera_intrinsics.K_frame
    
    # Initialize VIO
    vio = VIO(
        config=config,
        img_size=(h, w),
        calib=K,
        device=args.device,
        visualize=args.visualize
    )
    
    # Create thread-safe buffer for frames
    image_buffer = deque(maxlen=10)
    stop_event = threading.Event()
    
    # Start frame fetching thread
    fetch_thread = threading.Thread(
        target=fetch_data, 
        args=(image_buffer, stop_event, 0.1)  # Fetch every 0.1 seconds
    )
    fetch_thread.daemon = True
    fetch_thread.start()
    
    # Process frames continuously
    frame_count = 0
    last_processed_time = 0
    min_frame_interval = 0.03  # Minimum 30ms between frames (~33 FPS max)
    start_time = time.time()
    
    try:
        while True:
            # Check if we have frames in the buffer
            if not image_buffer:
                time.sleep(0.01)  # Short sleep if buffer is empty
                continue
            
            # Get the most recent frame
            timestamp, frame = image_buffer[0]
            
            # Check if enough time has passed since last processed frame
            if timestamp - last_processed_time < min_frame_interval:
                time.sleep(0.01)  # Wait a bit before checking again
                continue
                
            # Remove the frame we're about to process
            image_buffer.popleft()
            
            # Process frame
            success, pose, new_kf = vio.grab_rgb(frame, timestamp, None)
            last_processed_time = timestamp
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames at {fps:.2f} FPS")
            
    except KeyboardInterrupt:
        print(f"Interrupted after processing {frame_count} frames")
    finally:
        # Signal thread to stop and wait for it
        stop_event.set()
        fetch_thread.join(timeout=1.0)
        vio.terminate()
        print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")

def run_dataset(args):
    # Load config
    load_config(args.config)
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    # Load calibration if provided
    K = None
    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )
        K = dataset.camera_intrinsics.K_frame
    
    
    # Load odometry data if available
    odom = None
    odom_path = os.path.join(args.dataset, "traj_data.pkl")
    if os.path.exists(odom_path):
        odom = pickle.load(open(odom_path, "rb"))
    
    
    # Initialize VIO
    vio = VIO(
        config=config,
        img_size=(h, w),
        calib=K,
        device=args.device,
        visualize=args.visualize
    )
    
    # Process frames
    timestamps = []
    
    start_time = time.time()
    frame_count = 0
    
    for i in range(len(dataset)):
        timestamp, img = dataset[i]
        timestamps.append(timestamp)
        
        # Get odometry for current frame if available
        current_odom = None
        if odom is not None:
            current_odom = {
                'pos': odom['pos'][i],
                'yaw': odom['yaw'][i]
            }
        
        # Process frame
        success, pose, new_kf = vio.grab_rgb(img, timestamp, current_odom)
        
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"Processed {frame_count} frames at {fps:.2f} FPS")
    
    print(f"Processed {len(dataset)} frames in {time.time() - start_time:.2f} seconds")
    vio.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VIO on a dataset")
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk", 
                        help="Path to dataset")
    parser.add_argument("--config", default="config/base_no_fnn.yaml", 
                        help="Path to config file")
    parser.add_argument("--calib", default="", 
                        help="Path to calibration file")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize with rerun")
    parser.add_argument("--device", default="cuda:0", 
                        help="Device to run on")
    parser.add_argument("--resize", type=int, default=512,
                        help="Resize image to this size (long edge)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    
    run_dataset(args)
    # run_robot(args)
