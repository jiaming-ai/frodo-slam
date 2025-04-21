import threading
from collections import deque
from loguru import logger
import numpy as np
import copy
from scipy.spatial.transform import Rotation
import time

# Try to import Open3D, but provide fallback if visualization fails
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not available. Visualization will be disabled.")


class MapVisualizer:
    def __init__(self, visualization_enabled=True):
        self.vis = None
        self.vis_thread = None
        self.lock = threading.Lock()
        self.running = False
        self.visualization_enabled = visualization_enabled
        
        # Trajectory data
        self.vio_poses = []
        self.odom_poses = []
        self.origin_set = False
        self.origin_vio = None
        self.origin_odom = None
        
        # Visualization objects
        if self.visualization_enabled:
            self.vio_traj = o3d.geometry.LineSet()
            self.odom_traj = o3d.geometry.LineSet()
            self.vio_frames = []
            self.odom_frames = []
            
            # Try to start visualization in a separate thread
            self.start_visualization()
    
    def start_visualization(self):
        if not self.visualization_enabled:
            logger.warning("Visualization is disabled due to missing dependencies or OpenGL support.")
            return
            
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualization_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()
    
    def _visualization_loop(self):
        try:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            window_created = self.vis.create_window("Trajectory Visualization", width=1024, height=768)
            
            if not window_created:
                logger.error("Failed to create Open3D visualization window.")
                self.visualization_enabled = False
                self.running = False
                return
            
            # Add coordinate frame at origin
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            self.vis.add_geometry(origin)
            
            # Add initial empty trajectories
            self.vis.add_geometry(self.vio_traj)
            self.vis.add_geometry(self.odom_traj)
            
            # Set up view control
            view_control = self.vis.get_view_control()
            view_control.set_zoom(0.8)
            
            # Main visualization loop
            while self.running:
                with self.lock:
                    # Update geometries
                    self._update_trajectories()
                    
                    # Update all frames
                    for frame in self.vio_frames:
                        self.vis.update_geometry(frame)
                    for frame in self.odom_frames:
                        self.vis.update_geometry(frame)
                    
                    self.vis.update_geometry(self.vio_traj)
                    self.vis.update_geometry(self.odom_traj)
                
                try:
                    self.vis.poll_events()
                    self.vis.update_renderer()
                except Exception as e:
                    logger.error(f"Error in visualization loop: {e}")
                    break
                    
                time.sleep(0.05)  # Update at 20 Hz
            
            self.vis.destroy_window()
        except Exception as e:
            logger.error(f"Failed to initialize Open3D visualization: {e}")
            self.visualization_enabled = False
            self.running = False
    
    def _create_coordinate_frame(self, pose, size=0.2, is_vio=True):
        """Create a coordinate frame from a pose"""
        if not self.visualization_enabled:
            return None
            
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        
        # Extract translation and rotation from pose
        if is_vio:  # VIO pose (Sim3)
            # Extract translation, rotation quaternion, and scale
            t = pose.data[0, :3].cpu().numpy()
            q = pose.data[0, 3:7].cpu().numpy()  # [qx, qy, qz, qw]
            s = pose.data[0, 7].cpu().numpy()  # scale
            
            # Convert quaternion to rotation matrix (adjust order if needed)
            q_scipy = [q[3], q[0], q[1], q[2]]  # [qw, qx, qy, qz] for scipy
            R = Rotation.from_quat(q_scipy).as_matrix()
            
            # Apply scale to rotation matrix
            R = R * s
        else:  # Odometry pose (SE3)
            t = pose.data[0, :3].cpu().numpy()
            q = pose.data[0, 3:7].cpu().numpy()  # [qx, qy, qz, qw]
            
            # Convert quaternion to rotation matrix
            q_scipy = [q[3], q[0], q[1], q[2]]  # [qw, qx, qy, qz] for scipy
            R = Rotation.from_quat(q_scipy).as_matrix()
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        # Apply transformation
        frame.transform(transform)
        return frame
    
    def _update_trajectories(self):
        """Update trajectory visualizations"""
        if not self.visualization_enabled:
            return
            
        if len(self.vio_poses) > 1:
            # Create VIO trajectory line set
            points = np.array([pose.data[0, :3].cpu().numpy() for pose in self.vio_poses])
            lines = [[i, i+1] for i in range(len(points)-1)]
            colors = [[1, 0, 0] for _ in range(len(lines))]  # Red for VIO
            
            self.vio_traj.points = o3d.utility.Vector3dVector(points)
            self.vio_traj.lines = o3d.utility.Vector2iVector(lines)
            self.vio_traj.colors = o3d.utility.Vector3dVector(colors)
        
        if len(self.odom_poses) > 1:
            # Create odometry trajectory line set
            points = np.array([pose.data[0, :3].cpu().numpy() for pose in self.odom_poses])
            lines = [[i, i+1] for i in range(len(points)-1)]
            colors = [[0, 0, 1] for _ in range(len(lines))]  # Blue for odometry
            
            self.odom_traj.points = o3d.utility.Vector3dVector(points)
            self.odom_traj.lines = o3d.utility.Vector2iVector(lines)
            self.odom_traj.colors = o3d.utility.Vector3dVector(colors)
    
    def add_pose_vio(self, pose):
        """Add a VIO pose to the trajectory"""
        with self.lock:
            # Set origin if this is the first pose
            if not self.origin_set:
                self.origin_vio = copy.deepcopy(pose)
                self.origin_set = True
            
            # Store the pose
            self.vio_poses.append(copy.deepcopy(pose))
            
            # Calculate and display distance if there are at least 2 poses
            if len(self.vio_poses) >= 2:
                prev_pos = self.vio_poses[-2].data[0, :3].cpu().numpy()
                curr_pos = self.vio_poses[-1].data[0, :3].cpu().numpy()
                distance = np.linalg.norm(curr_pos - prev_pos)
                logger.debug(f"VIO step distance: {distance:.3f} m")
            
            # Create and add coordinate frame for this pose if visualization is enabled
            if self.visualization_enabled:
                frame = self._create_coordinate_frame(pose, size=0.1, is_vio=True)
                self.vio_frames.append(frame)
                
                # Add to visualizer if not already added
                if self.vis is not None and len(self.vio_frames) == len(self.vio_poses):
                    try:
                        self.vis.add_geometry(frame)
                    except Exception as e:
                        logger.error(f"Error adding VIO frame to visualizer: {e}")
    
    def add_pose_odometry(self, pose):
        """Add an odometry pose to the trajectory"""
        if pose is None:
            return
            
        with self.lock:
            # Store the pose
            self.odom_poses.append(copy.deepcopy(pose))
            
            # Calculate and display distance if there are at least 2 poses
            if len(self.odom_poses) >= 2:
                prev_pos = self.odom_poses[-2].data[0, :3].cpu().numpy()
                curr_pos = self.odom_poses[-1].data[0, :3].cpu().numpy()
                distance = np.linalg.norm(curr_pos - prev_pos)
                logger.debug(f"Odom step distance: {distance:.3f} m")
            
            # Create and add coordinate frame for this pose if visualization is enabled
            if self.visualization_enabled:
                frame = self._create_coordinate_frame(pose, size=0.1, is_vio=False)
                self.odom_frames.append(frame)
                
                # Add to visualizer if not already added
                if self.vis is not None and len(self.odom_frames) == len(self.odom_poses):
                    try:
                        self.vis.add_geometry(frame)
                    except Exception as e:
                        logger.error(f"Error adding odometry frame to visualizer: {e}")
    
    def visualize(self):
        """Ensure visualization is running"""
        if not self.running and self.vis_thread is None and self.visualization_enabled:
            self.start_visualization()
    
    def terminate(self):
        """Stop the visualization thread"""
        self.running = False
        if self.vis_thread is not None:
            self.vis_thread.join(timeout=1.0)