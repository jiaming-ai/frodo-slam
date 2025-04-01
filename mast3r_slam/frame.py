import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None
    odom: Optional[torch.Tensor] = None
    X_traversability: Optional[torch.Tensor] = None

    def fit_ground_plane(self):
        """
        fit a ground plane to the pointcloud
        return the plane equation in the camera frame
        most of the time the norm is (0,+-1,0) as y is vertical
        so the plane equation is z = d
        use RANSAC
        1. sample 3 points from traversible pointcloud (or X_canon if no traversible pointcloud)
        2. solve the plane equation from the 3 points
        3. project other traversible points to the normal vector, get the distance to the plane
        4. count the number of points that are within a certain threshold ds < threshold
        5. repeat above for N times, and select the plane with the highest inlier ratio
        """
        # TODO: implement this
        pass
    
    def pred_metric_scale(self):
        """
        First fit a ground plane to the pointcloud.
        Then use the known height of the camera (0.5xxm) h to compute the ratio s' = abs(h/ds) from the plane equation
        Note the s, which is the current scale T_wc.data[...,7]
        Then update the T_wc:
        T_wc.translation = T_wc.translation * s'
        T_wc.scale = T_wc.scale * s'

        """
        # TODO: implement this
        pass

    def pred_traversability(self):
        """
        pred the traversability of the pointcloud
        For each point, pred the traversability score (logit), or use sigmoid to get probability (0-1)
        """
        # TODO: implement this
        # self.X_traversability = 
        
    def get_pointcloud_world(self):
        """return the pointcloud in the world frame
        """
        # TODO: implement this
        # return self.X_canon @ self.T_WC.matrix().T
        pass

    
    
    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0", odom=None):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC, odom=odom)
    return frame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self):
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1


class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=200, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()
        # self.n_size = manager.Value("i", 0)
        self._idx = manager.Value("i", -1)

        self.h, self.w = h, w
        self.buffer_size = buffer
        self.dtype = dtype
        self.device = device  # Store the target device for when we return frames

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # Store everything in CPU memory instead of device
        self.dataset_idx = torch.zeros(buffer, dtype=torch.int, device="cpu").share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, dtype=dtype, device="cpu").share_memory_()
        self.uimg = torch.zeros(buffer, h, w, 3, dtype=dtype, device="cpu").share_memory_()
        self.img_shape = torch.zeros(buffer, 1, 2, dtype=torch.int, device="cpu").share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, dtype=torch.int, device="cpu").share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, dtype=dtype, device="cpu").share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, dtype=dtype, device="cpu").share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, dtype=dtype, device="cpu").share_memory_()
        self.N = torch.zeros(buffer, dtype=torch.int, device="cpu").share_memory_()
        self.N_updates = torch.zeros(buffer, dtype=torch.int, device="cpu").share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, dtype=dtype, device="cpu").share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, dtype=torch.long, device="cpu").share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, dtype=torch.bool, device="cpu").share_memory_()
        self.K = torch.zeros(3, 3, dtype=dtype, device="cpu").share_memory_()
        # fmt: on

    def get_frame_cpu(self, idx) -> Frame:
        with self.lock:
            # Wrap the index if needed
            idx = idx % self.buffer_size
            # Move data to the target device when returning a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],  # uimg stays on CPU
                lietorch.Sim3(self.T_WC[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K
            return kf       # return a frame from the cpu memory

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # Wrap the index if needed
            idx = idx % self.buffer_size
            # Move data to the target device when returning a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx].to(device=self.device),
                self.img_shape[idx].to(device=self.device),
                self.img_true_shape[idx].to(device=self.device),
                self.uimg[idx],  # uimg stays on CPU
                lietorch.Sim3(self.T_WC[idx].to(device=self.device)),
            )
            kf.X_canon = self.X[idx].to(device=self.device)
            kf.C = self.C[idx].to(device=self.device)
            kf.feat = self.feat[idx].to(device=self.device)
            kf.pos = self.pos[idx].to(device=self.device)
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K.to(device=self.device)
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            assert idx <= self._idx.value + 1
            self._idx.value = idx if idx > self._idx.value else self._idx.value
            idx = idx % self.buffer_size

            # Move data to CPU before storing
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img.cpu()
            self.uimg[idx] = value.uimg  # Already on CPU
            self.img_shape[idx] = value.img_shape.cpu()
            self.img_true_shape[idx] = value.img_true_shape.cpu()
            self.T_WC[idx] = value.T_WC.data.cpu()
            self.X[idx] = value.X_canon.cpu()
            self.C[idx] = value.C.cpu()
            self.feat[idx] = value.feat.cpu()
            self.pos[idx] = value.pos.cpu()
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True
            return idx

    def __len__(self):
        with self.lock:
            # Return the actual number of frames, capped by buffer size
            return min(self._idx.value + 1, self.buffer_size)

    def to_cpu(self):
        # Update device attribute
        self.device = "cpu"


    def append(self, value: Frame):
        with self.lock:
            self[self._idx.value + 1] = value
        return self._idx.value

    def pop_last(self):
        with self.lock:
            self._idx.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self._idx.value == -1:
                return None
            return self[self._idx.value]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            # Wrap the index if needed
            idx = idx % self.buffer_size
            self.T_WC[idx] = T_WCs.data.cpu()

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        with self.lock:
            assert config["use_calib"]
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
    
    
    
    # def append(self, value: Frame):
    #     with self.lock:
    #         self[self._idx.value] = value

    # def pop_last(self):
    #     with self.lock:
    #         self._idx.value -= 1

    # def last_keyframe(self) -> Optional[Frame]:
    #     with self.lock:
    #         if self.n_size.value == 0:
    #             return None
    #         return self[self.n_size.value - 1]

    # def update_T_WCs(self, T_WCs, idx) -> None:
    #     with self.lock:
    #         # Wrap the index if needed
    #         idx = idx % self.buffer_size if isinstance(idx, int) else idx % self.buffer_size
    #         self.T_WC[idx] = T_WCs.data.cpu()

    # def get_dirty_idx(self):
    #     with self.lock:
    #         idx = torch.where(self.is_dirty)[0]
    #         self.is_dirty[:] = False
    #         return idx

    # def set_intrinsics(self, K):
    #     assert config["use_calib"]
    #     with self.lock:
    #         self.K[:] = K.cpu()

    # def get_intrinsics(self):
    #     assert config["use_calib"]
    #     with self.lock:
    #         return self.K.to(device=self.device)



class KeyframesCuda:
    def __init__(self, h, w, buffer=200, dtype=torch.float32, device="cuda"):
        self.idx = -1

        self.h, self.w = h, w
        self.buffer_size = buffer
        self.dtype = dtype
        self.device = device  # Store the target device for when we return frames

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # Store everything in CPU memory instead of device
        self.dataset_idx = torch.zeros(buffer, dtype=torch.int, device=device)
        self.img = torch.zeros(buffer, 3, h, w, dtype=dtype, device=device)
        self.uimg = torch.zeros(buffer, h, w, 3, dtype=dtype, device=device)
        self.img_shape = torch.zeros(buffer, 1, 2, dtype=torch.int, device=device)
        self.img_true_shape = torch.zeros(buffer, 1, 2, dtype=torch.int, device=device)
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, dtype=dtype, device=device)
        self.X = torch.zeros(buffer, h * w, 3, dtype=dtype, device=device)
        self.C = torch.zeros(buffer, h * w, 1, dtype=dtype, device=device)
        self.N = torch.zeros(buffer, dtype=torch.int, device=device)
        self.N_updates = torch.zeros(buffer, dtype=torch.int, device=device)
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, dtype=dtype, device=device)
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, dtype=torch.long, device=device)
        self.is_dirty = torch.zeros(buffer, 1, dtype=torch.bool, device=device)
        self.K = torch.zeros(3, 3, dtype=dtype, device=device)
        # fmt: on

    def reset(self):
        self.idx = -1

    def __getitem__(self, idx) -> Frame:
        # NOTE: this is unsafe, idx is not checked. it can points to non-existing frames.

            # Wrap the index if needed
        idx = idx % self.buffer_size
        # Move data to the target device when returning a frame
        kf = Frame(
            int(self.dataset_idx[idx]),
            self.img[idx],
            self.img_shape[idx],
            self.img_true_shape[idx],
            self.uimg[idx],  # uimg stays on CPU
            lietorch.Sim3(self.T_WC[idx]),
        )
        kf.X_canon = self.X[idx]
        kf.C = self.C[idx]
        kf.feat = self.feat[idx]
        kf.pos = self.pos[idx]
        kf.N = int(self.N[idx])
        kf.N_updates = int(self.N_updates[idx])
        if config["use_calib"]:
            kf.K = self.K
        return kf

    def __setitem__(self, idx, value: Frame) -> None:
        # NOTE: we assume sequential insertion only, idx is not checked.
        # self.idx is always points to the last data in the buffer

        assert idx <= self.idx + 1
        self.idx = self.idx + 1 if idx > self.idx else self.idx
        idx = idx % self.buffer_size

        # if idx >= self.buffer_size:
        #     # Calculate the index to use within the buffer
        #     idx = idx % self.buffer_size
        #     # We're overwriting an existing frame, so we don't need to increment n_size
        #     # If we've filled the buffer, n_size should remain at buffer size
        #     self.n_size = min(self.buffer_size, self.n_size)
        # else:
        #     # Normal case - update n_size if needed
        #     self.n_size = max(idx + 1, self.n_size)

        self.dataset_idx[idx] = value.frame_id
        self.img[idx] = value.img
        self.uimg[idx] = value.uimg
        self.img_shape[idx] = value.img_shape
        self.img_true_shape[idx] = value.img_true_shape
        self.T_WC[idx] = value.T_WC.data
        self.X[idx] = value.X_canon
        self.C[idx] = value.C
        self.feat[idx] = value.feat
        self.pos[idx] = value.pos
        self.N[idx] = value.N
        self.N_updates[idx] = value.N_updates
        self.is_dirty[idx] = True
        return idx

    def __len__(self):
        return min(self.idx + 1, self.buffer_size)

    def append(self, value: Frame):
        self[self.idx + 1] = value
        return self.idx
    def pop_last(self):
        self.idx -= 1

    def last_keyframe(self) -> Optional[Frame]:
        if self.idx == -1:
            return None
        return self[self.idx]

    def update_T_WCs(self, T_WCs, idx) -> None:
        # Wrap the index if needed
        idx = idx % self.buffer_size
        self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        idx = torch.where(self.is_dirty)[0]
        self.is_dirty[:] = False
        return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        return self.K