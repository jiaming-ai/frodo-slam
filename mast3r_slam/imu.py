import requests
import time
import threading
from collections import deque
import numpy as np
import cv2
import math
import torch
import pypose as pp
import warnings
# Suppress specific pypose warnings if needed (optional)
warnings.filterwarnings("ignore", message=".*Converting non-supported dtype.*")
warnings.filterwarnings("ignore", message=".*torch.lstsq is deprecated.*")


# Constants
GRAVITY_MAGNITUDE = 9.81007 # Use a precise value if known for your location
DEG_TO_RAD = math.pi / 180.0
# Assume world frame is Z-up, gravity points along -Z
WORLD_GRAVITY_VECTOR = torch.tensor([0.0, 0.0, -GRAVITY_MAGNITUDE])

def _compute_rotation_between_vectors(u, v, eps=1e-8):
    """
    Computes the pypose.SO3 rotation R such that R*u aligns with v.
    Uses Rodrigues' rotation formula based on axis-angle representation.

    Args:
        u (torch.Tensor): The starting vector (3D). Should be normalized or have non-zero length.
        v (torch.Tensor): The target vector (3D). Should be normalized or have non-zero length.
        eps (float): Small epsilon for numerical stability checks (e.g., checking for zero vectors or collinearity).

    Returns:
        pypose.SO3: The rotation object representing the transformation from u's direction to v's direction.
    """
    device = u.device
    dtype = u.dtype # Use the dtype of the input vectors

    # Normalize vectors first (important for dot/cross product interpretations)
    u_norm_val = torch.linalg.norm(u)
    v_norm_val = torch.linalg.norm(v)

    if u_norm_val < eps or v_norm_val < eps:
        print(f"Warning: Zero vector encountered in alignment (u_norm={u_norm_val:.2e}, v_norm={v_norm_val:.2e}). Returning identity.")
        return pp.identity_SO3(device=device, dtype=dtype)

    u_normalized = u / u_norm_val
    v_normalized = v / v_norm_val

    # Calculate the rotation axis (normalized cross product)
    axis_raw = torch.cross(u_normalized, v_normalized, dim=0)
    axis_norm = torch.linalg.norm(axis_raw)

    # Calculate the cosine of the angle using the dot product
    # Clamp to avoid acos domain errors due to floating point inaccuracies
    cos_angle = torch.clamp(torch.dot(u_normalized, v_normalized), -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos_angle)

    if axis_norm < eps:
        # Vectors are collinear
        if cos_angle > 0: # Already aligned (angle approx 0)
            # Return identity rotation
            return pp.identity_SO3(device=device, dtype=dtype)
        else: # Anti-parallel (angle approx pi)
            # The cross product is zero, axis is ill-defined.
            # We need *any* axis perpendicular to u_normalized.
            # Find an arbitrary vector not parallel to u_normalized
            # Try crossing with [1, 0, 0], then [0, 1, 0] if the first is parallel.
            axis_tmp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            axis = torch.cross(u_normalized, axis_tmp)
            if torch.linalg.norm(axis) < eps: # u_normalized was parallel to axis_tmp (e.g., [1,0,0])
                axis_tmp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
                axis = torch.cross(u_normalized, axis_tmp)

            # Check if axis is still zero (shouldn't happen unless u is zero, handled above)
            axis_n = torch.linalg.norm(axis)
            if axis_n < eps:
                 print("Warning: Could not find perpendicular axis for anti-parallel alignment. Returning identity.")
                 return pp.identity_SO3(device=device, dtype=dtype)

            axis = axis / axis_n # Normalize the chosen perpendicular axis
            # Angle is pi
            angle = torch.tensor(math.pi, device=device, dtype=dtype)
            # Construct SO3 from axis-angle (rotation vector)
            rotation_vector = axis * angle
            return pp.SO3(rotation_vector).to(dtype=dtype)

    else:
        # General case: non-collinear vectors
        axis = axis_raw / axis_norm # Normalize the rotation axis
        # Construct SO3 from axis-angle (rotation vector)
        rotation_vector = axis * angle
        return pp.SO3(rotation_vector).to(dtype=dtype)

class RealTimeIMUIntegrator:
    def __init__(self, server_url, fetch_interval=0.1, device='cuda'):
        self.server_url = server_url
        self.fetch_interval = fetch_interval
        self.device = device
        # Ensure world gravity vector is on the correct device and dtype early
        self.world_gravity_vector = WORLD_GRAVITY_VECTOR.to(self.device).float() # Use float32 typically

        # --- Data Timestamp Tracking ---
        self.last_server_accel_ts = -1.0
        self.last_server_gyro_ts = -1.0

        # --- Calibration Data ---
        self.acc_bias = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.gyro_bias = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.init_rot = pp.identity_SO3(device=self.device, dtype=torch.float32) # Will be updated by calibration

        # --- IMU State (Managed by IMUPreintegrator internally) ---
        # Initialize later after calibration
        self.integrator = None

        # --- State Tracking for Integration Steps ---
        self.last_integration_ts = -1.0
        self.current_gyro = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.current_accel = torch.zeros(3, device=self.device, dtype=torch.float32)


        # --- Threading ---
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

        # --- Visualization ---
        self.vis_window_name = "IMU Pose Visualization"
        self.vis_img_size = 600
        self.vis_scale = 50 # Pixels per meter for position - adjust as needed

    def _initialize_integrator(self):
        """Initializes the IMUPreintegrator after calibration."""
        # Use estimated biases during initialization
        self.integrator = pp.module.IMUPreintegrator(
            gravity=GRAVITY_MAGNITUDE,
            # Provide initial bias estimates if the integrator uses them internally
            # Note: Check pypose documentation if these args exist and are used this way.
            # If not, you might need to subtract bias manually before passing data,
            # but it's better if the integrator handles it.
            # As of pypose ~0.4, biases are internal states managed by the module.
            # We will set them directly after initialization.
            prop_cov=True,
            reset=True 
        ).to(self.device).float() # Ensure float32

        # Set initial state AFTER creating the integrator object
        self.reset_imu_state(set_defaults=False) # Reset internal buffers without setting defaults
        # Now apply calibrated values
        with self._lock:
            self.integrator.rot = self.init_rot.unsqueeze(0).clone() # Set initial rotation (needs batch dim)
            self.integrator.gyro_bias = self.gyro_bias.unsqueeze(0).clone() # Set initial gyro bias (needs batch dim)
            self.integrator.acc_bias = self.acc_bias.unsqueeze(0).clone()   # Set initial acc bias (needs batch dim)
            # Ensure dtype consistency if needed
            self.integrator.pos = self.integrator.pos.to(self.integrator.rot.dtype)
            self.integrator.vel = self.integrator.vel.to(self.integrator.rot.dtype)
            self.integrator.cov = self.integrator.cov.to(self.integrator.rot.dtype)
            # Ensure initial current values match integrator state
            self.current_gyro.zero_() # Start assuming zero angular velocity
            self.current_accel = self.init_rot.Act(self.world_gravity_vector) # Expected accel at rest with init_rot

        print("IMUPreintegrator initialized with calibrated values.")
        print(f"  Initial Rotation (Quat): {self.integrator.rot.tensor()}")
        print(f"  Initial Gyro Bias: {self.integrator.gyro_bias}")
        print(f"  Initial Accel Bias: {self.integrator.acc_bias}")


    def _calibrate_static(self, duration=3.0):
        """Calibrates biases and initial orientation assuming static sensor."""

        print(f"Starting static calibration for {duration} seconds. Keep the sensor perfectly still...")
        start_time = time.monotonic()
        accel_readings = []
        gyro_readings = []
        last_ts = -1

        while time.monotonic() - start_time < duration:
            raw_data = self._fetch_data()
            if raw_data:
                # Only process, don't integrate yet
                current_batch_last_ts = -1 # Track last timestamp within this fetch
                if 'accels' in raw_data:
                    for acc in raw_data['accels']:
                        ts = acc[3]
                        if ts > last_ts: # Avoid duplicates across fetches
                           acc_sensor = torch.tensor(acc[:3], dtype=torch.float32, device=self.device) * GRAVITY_MAGNITUDE
                           accel_readings.append(acc_sensor)
                           current_batch_last_ts = max(current_batch_last_ts, ts)
                if 'gyros' in raw_data:
                     for gyro in raw_data['gyros']:
                         ts = gyro[3]
                         if ts > last_ts: # Avoid duplicates across fetches
                            gyro_sensor = torch.tensor(gyro[:3], dtype=torch.float32, device=self.device) * DEG_TO_RAD
                            gyro_readings.append(gyro_sensor)
                            current_batch_last_ts = max(current_batch_last_ts, ts)

                # Update last timestamp seen during calibration based on this fetch
                if current_batch_last_ts > last_ts:
                     last_ts = current_batch_last_ts

            time.sleep(self.fetch_interval / 2) # Fetch reasonably fast during calibration

        if not accel_readings or not gyro_readings:
            print("Calibration failed: No data received.")
            # Use default zero biases and identity rotation
            self.acc_bias.zero_()
            self.gyro_bias.zero_()
            self.init_rot = pp.identity_SO3(device=self.device, dtype=torch.float32)
            return False

        # --- Calculate Biases and Initial Orientation ---
        avg_acc = torch.stack(accel_readings).mean(dim=0)
        avg_gyro = torch.stack(gyro_readings).mean(dim=0)

        # Gyro bias is simply the average reading when static
        self.gyro_bias = avg_gyro

        # Estimate initial orientation by aligning measured gravity with world gravity
        measured_g_norm = torch.linalg.norm(avg_acc)
        if measured_g_norm < 1e-6: # Use a small epsilon check
            print("Calibration warning: Near-zero average acceleration detected. Using identity orientation.")
            self.init_rot = pp.identity_SO3(device=self.device, dtype=torch.float32)
        else:
            # Normalize measured gravity direction in body frame
            measured_g_dir_body = avg_acc / measured_g_norm
            # Normalize world gravity direction
            world_g_norm = torch.linalg.norm(self.world_gravity_vector)
            # Ensure world_gravity_vector isn't zero (shouldn't be)
            if world_g_norm < 1e-6:
                print("ERROR: World gravity vector magnitude is near zero. Check definition.")
                self.init_rot = pp.identity_SO3(device=self.device, dtype=torch.float32)
            else:
                world_g_dir = self.world_gravity_vector / world_g_norm

                # Find rotation R_wb such that: R_wb * world_g_dir = measured_g_dir_body
                # --- USE THE HELPER FUNCTION ---
                try:
                    # Use the custom function to compute the rotation
                    # Pass vectors with matching device and dtype
                    self.init_rot = _compute_rotation_between_vectors(
                        u=world_g_dir.to(self.device).to(avg_acc.dtype), # Ensure compatible inputs
                        v=measured_g_dir_body.to(self.device).to(avg_acc.dtype)
                    )
                    # Ensure the result is float32 if needed elsewhere, though helper uses input dtype
                    self.init_rot = self.init_rot.to(dtype=torch.float32)

                except Exception as e:
                    print(f"Error during _compute_rotation_between_vectors: {e}. Vectors were:")
                    print(f"  world_g_dir: {world_g_dir}")
                    print(f"  measured_g_dir_body: {measured_g_dir_body}")
                    print("Falling back to identity orientation.")
                    self.init_rot = pp.identity_SO3(device=self.device, dtype=torch.float32)


        # Accelerometer bias: avg_acc = R_wb_initial * g_world + bias_acc
        # bias_acc = avg_acc - R_wb_initial * g_world
        # Note: Use Act() for pypose SO3 action on vectors
        expected_g_body = self.init_rot.Act(self.world_gravity_vector.to(self.init_rot.dtype).to(self.init_rot.device))
        self.acc_bias = avg_acc - expected_g_body

        print("Calibration complete.")
        print(f"  Avg Accel ({len(accel_readings)} samples): {avg_acc.cpu().numpy()}")
        print(f"  Avg Gyro ({len(gyro_readings)} samples): {avg_gyro.cpu().numpy()}")
        print(f"  Estimated Accel Bias: {self.acc_bias.cpu().numpy()}")
        print(f"  Estimated Gyro Bias: {self.gyro_bias.cpu().numpy()}")
        print(f"  Estimated Initial Rotation (Quaternion): {self.init_rot.tensor().cpu().numpy()}")

        # Set the last server timestamps to avoid re-processing calibration data
        # Ensure last_ts was updated correctly during calibration loop
        if last_ts > 0:
            self.last_server_accel_ts = last_ts
            self.last_server_gyro_ts = last_ts
        else:
            print("Warning: last_ts not updated during calibration, timestamp tracking might be off.")
            # Fallback: try finding max ts from collected data if possible
            # This part needs careful checking based on your server data structure

        return True
    
    def _fetch_data(self):
        """Fetches data from the server."""
        try:
            response = requests.get(self.server_url, timeout=1.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
        except ValueError as e:
            print(f"Error decoding JSON: {e}")
            return None

    def _process_new_data(self, raw_data):
        """
        Filters new data, converts units, and returns a sorted list of new measurements.
        Measurements are kept in the SENSOR FRAME.
        Returns: List of tuples: [(timestamp, type, torch.Tensor(data_in_sensor_frame))]
        """
        new_measurements = []
        dtype = torch.float32 # Use float32

        # Process Accelerometer data
        if 'accels' in raw_data:
            for acc in raw_data['accels']:
                ts = acc[3]
                if ts > self.last_server_accel_ts:
                    self.last_server_accel_ts = ts
                    # Convert g to m/s^2. NO coordinate transform.
                    acc_sensor = torch.tensor(acc[:3], dtype=dtype, device=self.device) * GRAVITY_MAGNITUDE
                    new_measurements.append((ts, 'accel', acc_sensor)) # Store raw sensor frame data

        # Process Gyroscope data
        if 'gyros' in raw_data:
            for gyro in raw_data['gyros']:
                ts = gyro[3]
                if ts > self.last_server_gyro_ts:
                    self.last_server_gyro_ts = ts
                    # Convert deg/s to rad/s. NO coordinate transform.
                    gyro_sensor = torch.tensor(gyro[:3], dtype=dtype, device=self.device) * DEG_TO_RAD
                    new_measurements.append((ts, 'gyro', gyro_sensor)) # Store raw sensor frame data

        # Sort combined list by timestamp
        new_measurements.sort(key=lambda x: x[0])

        return new_measurements


    def _process_batch(self, new_measurements):
        """
        Processes a batch of sorted measurements to perform integration.
        Uses and updates the internal state of self.integrator.
        Feeds RAW sensor measurements; integrator uses internal bias estimates.
        """
        if not new_measurements or self.integrator is None: # Ensure integrator is initialized
            return False # No new data or not ready

        dt_list = []
        gyro_list = []
        accel_list = []
        updated = False
        dtype = self.integrator.rot.dtype # Match integrator dtype

        with self._lock: # Protect access to integrator state and tracking variables
            batch_start_ts = self.last_integration_ts

            # Use the current sensor readings *from the previous step*
            # These should represent the state *at* batch_start_ts
            batch_current_gyro = self.current_gyro.clone()
            batch_current_accel = self.current_accel.clone()

            # Handle the very first measurement batch after calibration
            if batch_start_ts < 0:
                 # Use the timestamp of the *first* measurement in the batch as the start
                 batch_start_ts = new_measurements[0][0]
                 # Find the accel/gyro readings *closest to but not after* this start time
                 # (This logic might need refinement depending on data timing)
                 # For simplicity, let's use the first measurement's values if they occur at batch_start_ts
                 found_accel = False
                 found_gyro = False
                 for ts, type, data in new_measurements:
                     if abs(ts - batch_start_ts) < 1e-6 :
                          if type == 'accel':
                              batch_current_accel = data
                              found_accel = True
                          elif type == 'gyro':
                              batch_current_gyro = data
                              found_gyro = True
                     if found_accel and found_gyro:
                         break
                 # If not found exactly at start_ts, the zeros from __init__ might suffice
                 # or we could extrapolate backwards (more complex). Let's stick with this for now.


            # Iterate through measurements to build integration steps
            last_step_ts = batch_start_ts
            for ts, type, data in new_measurements:
                if ts <= last_step_ts : # Skip duplicate timestamps or out-of-order
                    # Still update the 'current' reading if timestamp is same
                    if abs(ts - last_step_ts) < 1e-9:
                        if type == 'accel':
                           batch_current_accel = data
                        elif type == 'gyro':
                           batch_current_gyro = data
                    continue

                dt = ts - last_step_ts
                if dt < 1e-9: # Skip potentially zero dt steps (already handled above but keep for safety)
                    continue

                # Append state for the interval [last_step_ts, ts]
                # Use sensor readings valid *at the beginning* of the interval (last_step_ts)
                # The integrator expects RAW readings.
                dt_list.append(dt)
                gyro_list.append(batch_current_gyro) # Raw gyro at start of interval
                accel_list.append(batch_current_accel) # Raw accel at start of interval

                # Update the 'current' sensor reading *after* using the previous one
                if type == 'accel':
                    batch_current_accel = data
                elif type == 'gyro':
                    batch_current_gyro = data

                # Move timestamp forward
                last_step_ts = ts

            # Perform batch integration if any steps were generated
            if dt_list:
                # Ensure consistency in dtype for batch tensors
                dt_batch = torch.tensor(dt_list, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(-1) # (1, F, 1)
                gyro_batch = torch.stack(gyro_list, dim=0).unsqueeze(0).to(dtype) # (1, F, 3)
                accel_batch = torch.stack(accel_list, dim=0).unsqueeze(0).to(dtype) # (1, F, 3)

                # Call the integrator's forward method.
                # It uses its internal state (pos, rot, vel, biases, cov) and updates it.
                # We provide the RAW measurements.
                integration_output = self.integrator(
                    dt=dt_batch,
                    gyro=gyro_batch, # Raw Gyro Data
                    acc=accel_batch,  # Raw Accel Data
                )

                # Update tracking variables for the *next* batch
                self.last_integration_ts = last_step_ts
                # Store the latest RAW readings processed
                self.current_gyro = batch_current_gyro
                self.current_accel = batch_current_accel
                updated = True

        return updated


    def _run(self):
        """Background thread main loop."""
        print("IMU processing thread started.")
        while not self._stop_event.is_set():
            start_time = time.monotonic()

            # 1. Fetch Data
            raw_data = self._fetch_data()

            # 2. Process New Data into a sorted list
            new_measurements = []
            if raw_data:
                new_measurements = self._process_new_data(raw_data)

            # 3. Integrate if new measurements arrived
            if new_measurements:
               self._process_batch(new_measurements)

            # 4. Wait for next cycle
            elapsed_time = time.monotonic() - start_time
            sleep_time = self.fetch_interval - elapsed_time
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

        print("IMU processing thread stopped.")

    def reset_imu_state(self, set_defaults=True):
        """Resets integrator state. Option to skip setting defaults if called before calibration."""
        with self._lock:
            if self.integrator is not None:
                 if set_defaults:
                    # Reset biases to zero, rotation to identity only if requested
                    self.integrator.acc_bias.zero_()
                    self.integrator.gyro_bias.zero_()
                    self.integrator.rot.identity_() # pypose handles batch dim here
                 print("Integrator state reset.")
            else:
                 print("Integrator not initialized yet, cannot reset.")


    def start(self):
        """Calibrates, initializes, and starts the background processing thread."""
        if self._thread is None or not self._thread.is_alive():
            # 1. Perform Static Calibration
            if not self._calibrate_static(duration=5.0):
                print("WARNING: Calibration failed. Using default biases (0) and orientation (identity). Expect drift.")
                # Ensure defaults are set if calibration fails
                self.acc_bias.zero_()
                self.gyro_bias.zero_()
                self.init_rot = pp.identity_SO3(device=self.device, dtype=torch.float32)


            # 2. Initialize Integrator with Calibrated Values
            self._initialize_integrator() # This now sets initial rot/biases inside

            # 3. Reset Timestamps for Integration Start
            # Make sure the integration starts *after* the calibration data
            self.last_integration_ts = max(self.last_server_accel_ts, self.last_server_gyro_ts)
            if self.last_integration_ts < 0: # Handle case where calibration failed to get timestamps
                self.last_integration_ts = time.time() # Fallback to current time


            # 4. Start Processing Thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            print("Thread already running.")


    def stop(self):
        """Stops the background processing thread."""
        print("Stopping IMU processing thread...")
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        print("Thread stopped.")

    def get_current_pose(self):
        """Returns the current estimated pose (position, rotation). Thread-safe."""
        with self._lock:
            # Clone data from the integrator's internal buffers
            # .squeeze(0) removes the batch dimension (B=1) for external use
            pos = self.integrator.pos.clone().squeeze(0) # (1, 3)
            rot = self.integrator.rot.clone().squeeze(0) # SO3 as quaternion (1,4)
            #vel = self.integrator.vel.clone().squeeze(0).cpu().numpy() # Optional
        return pos, rot # , vel

    def visualize(self):
        """Runs the OpenCV visualization loop in the main thread."""
        print("Starting visualization. Press 'q' to quit.")
        cv2.namedWindow(self.vis_window_name)

        while True:
            # Create blank image
            img = np.zeros((self.vis_img_size, self.vis_img_size, 3), dtype=np.uint8)
            center_x, center_y = self.vis_img_size // 2, self.vis_img_size // 2

            # Get current pose safely
            pos_pp, rot_pp = self.get_current_pose()
            
            rot_mat_world_to_body = rot_pp.matrix().squeeze().cpu().numpy() # Get rotation matrix (3, 3)

            rot_mat_body_to_world = rot_mat_world_to_body.T
            pos_np = pos_pp.squeeze().cpu().numpy() # (3,)

            # --- Draw Position (Top-down view: World X-Y plane) ---
            vis_x = center_x + int(pos_np[0] * self.vis_scale) # X coordinate
            vis_y = center_y - int(pos_np[1] * self.vis_scale) # Y coordinate (inverted for display)
            cv2.circle(img, (vis_x, vis_y), 5, (255, 255, 255), -1) # White dot for position

            # --- Calculate heading angle on XY plane ---
            x_axis_body = np.array([1.0, 0.0, 0.0]) # Body X axis
            x_axis_world = rot_mat_body_to_world @ x_axis_body

            heading_vector = x_axis_world[:2] # Project onto XY plane
            heading_norm = np.linalg.norm(heading_vector)
            if heading_norm > 1e-6:
                heading_angle = np.arctan2(heading_vector[1], heading_vector[0])
                heading_deg = np.degrees(heading_angle)
            else:
                heading_angle = 0.0
                heading_deg = 0.0


            # --- Draw heading direction ---
            heading_length = 30
            heading_end_x = vis_x + int(heading_length * np.cos(heading_angle))
            heading_end_y = vis_y - int(heading_length * np.sin(heading_angle)) # Inverted Y for display
            cv2.line(img, (vis_x, vis_y), (heading_end_x, heading_end_y), (0, 255, 0), 2) # Green line

            # --- Display Text Info ---
            pos_str = f"Pos (X,Y,Z): ({pos_np[0]:.2f}, {pos_np[1]:.2f}, {pos_np[2]:.2f})"
            heading_str = f"Heading (XY): {heading_deg:.1f} deg"
            ts_str = f"Last Int TS: {self.last_integration_ts:.2f}"
            # Display biases
            acc_bias_str = f"Acc Bias: {self.acc_bias.cpu().numpy()}"
            gyro_bias_str = f"Gyro Bias: {self.gyro_bias.cpu().numpy()}"


            cv2.putText(img, "2D Top-Down View (XY Plane)", (10, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(img, pos_str, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, heading_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, ts_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, acc_bias_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(img, gyro_bias_str, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


            # Show image
            cv2.imshow(self.vis_window_name, img)

            # Exit condition
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Visualization stopped.")

if __name__ == '__main__':
    # Replace with your actual server URL
    SERVER_ENDPOINT = 'http://localhost:8000/data' # Make sure this server is running

    # Create the system instance
    # Use 'cpu' if CUDA is unavailable or causing issues
    integrator_system = RealTimeIMUIntegrator(SERVER_ENDPOINT, fetch_interval=0.05, device='cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Start includes calibration now
        integrator_system.start()

        # Run visualization in the main thread
        integrator_system.visualize()

    except KeyboardInterrupt:
        print("Keyboard interrupt detected.")
    finally:
        # Ensure the background thread is stopped cleanly
        integrator_system.stop()
        print("System shut down.")