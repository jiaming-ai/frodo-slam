import cv2
import numpy as np
import pickle
import os
import time
import open3d as o3d

class RectanglePlaneEstimator:
    def __init__(
        self, 
        mask_file='config/rect_mask.pkl', 
        max_subsample=1000, 
        max_tilt_deg=5.0, 
        inliers_threshold=0.04,
        ransac_n=3,
        ransac_iterations=1000,
        min_inliers=100
    ):
        """
        mask_file: path to save/load the two corner points
        max_subsample: max number of points for plane fitting
        max_tilt_deg: if computed tilt exceeds this, height is considered invalid
        inliers_threshold: inliers threshold for plane fitting
        ransac_n: number of points for plane fitting
        ransac_iterations: number of iterations for plane fitting
        min_inliers: minimum number of inliers for plane fitting
        """
        self.mask_file = mask_file
        self.max_subsample = max_subsample
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.inliers_threshold = inliers_threshold
        self.min_inliers = min_inliers
        self.ransac_n = ransac_n
        self.ransac_iterations = ransac_iterations

        self.corners = None
        if os.path.exists(self.mask_file):
            try:
                with open(self.mask_file, 'rb') as f:
                    self.corners = pickle.load(f)
                print(f"[INIT] Loaded rectangle corners from '{self.mask_file}': {self.corners}")
            except Exception as e:
                print(f"[INIT] Failed to load corners: {e}")

    def run(
        self, 
        pointmap: np.ndarray, 
        image: np.ndarray = None, 
        image_size: tuple = None,
        show: bool = False
    ):
        """
        image:    H×W×3 uint8
        pointmap: (H*W)×3 float array of (X,Y,Z) per pixel, flattened row-major.
        show:     if False, skip visualization overlay
        """
        assert image is not None or image_size is not None, "image or image_size is required"
        if image is not None:
            h, w = image.shape[:2]
        else:
            h, w = image_size

        if image is not None:

            # 1) get rectangle corners if needed
            if self.corners is None:
                self._select_rectangle(image)
                if self.corners is None:
                    print("[RUN] Selection cancelled.")
                    return
                with open(self.mask_file, 'wb') as f:
                    pickle.dump(self.corners, f)
                print(f"[RUN] Saved corners to '{self.mask_file}': {self.corners}")

        (x0, y0), (x1, y1) = self.corners
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))

        # 2) compute flat indices for rectangle
        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        Xs, Ys = np.meshgrid(xs, ys)
        flat_idx = (Ys.ravel() * w + Xs.ravel())

        # 3) gather 3D points
        pts3d = pointmap[flat_idx]

        # 4) subsample
        N = pts3d.shape[0]
        if N > self.max_subsample:
            choice = np.random.choice(N, self.max_subsample, replace=False)
            pts3d_sub = pts3d[choice]
        else:
            pts3d_sub = pts3d

        # 5) plane segmentation with Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d_sub)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.inliers_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.ransac_iterations)

        if len(inliers) < self.min_inliers:
            print(f"[RUN] Not enough inliers: {len(inliers)} < {self.min_inliers}")
            return None

        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        # 6) compute tilt & height
        tilt = np.arccos(abs(b) / np.linalg.norm(normal))
        tilt_deg = np.degrees(tilt)
        height = None
        if tilt <= self.max_tilt and abs(b) > 1e-6:
            height = -d / b

        # # 7) print results & timings
        # print(f"[RESULT] Tilt: {tilt_deg:.2f}°")
        # if height is not None:
        #     print(f"[RESULT] Height: {height:.3f}")
        # else:
        #     print("[RESULT] Height: None (tilt too large or invalid)")

        # 8) optional overlay
        if show:
            overlay = image.copy()
            overlay[y_min:y_max+1, x_min:x_max+1] = (0, 255, 0)
            out = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(out, f"Tilt: {tilt_deg:.1f}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(out, f"Height: {height:.3f}" if height is not None else "Height: None",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow('Rectangle Estimator', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return height

    def _select_rectangle(self, image: np.ndarray):
        """User clicks two corners of an axis-aligned rectangle."""
        self.corners = []
        self.finished = False
        disp = image.copy()
        cv2.namedWindow('Select Rectangle')
        cv2.setMouseCallback('Select Rectangle', self._mouse_cb, disp)
        print("[SELECT] Click two corners of the rectangle, or ESC to cancel.")
        while True:
            cv2.imshow('Select Rectangle', disp)
            key = cv2.waitKey(1) & 0xFF
            if self.finished:
                break
            if key == 27:
                self.corners = None
                break
        cv2.destroyAllWindows()

    def _mouse_cb(self, event, x, y, flags, disp):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners.append((x, y))
            cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
            if len(self.corners) == 2:
                (x0, y0), (x1, y1) = self.corners
                cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 0, 0), 2)
                self.finished = True


if __name__ == '__main__':
    img = cv2.imread('path/to/image.png')
    h, w = img.shape[:2]
    # Flat pointmap: shape (H*W, 3)
    pointmap = np.random.randn(h*w, 3).astype(np.float32)

    estimator = RectanglePlaneEstimator(mask_file='rect_mask.pkl')
    estimator.run(img, pointmap, show=True)
