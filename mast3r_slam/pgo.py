import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import pypose.optim as optim
from mast3r_slam.profile import timeit, timeblock, print_timing_registry

def pos_yaw_to_se3(pos, yaw):
    """
    Convert position and yaw to SE(3)
    """
    T = torch.eye(4)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    # reverse the yaw, otherwise it's z->x
    T[:3, :3 ] = torch.tensor([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0],
        [-sin_yaw, 0, cos_yaw]
    ])
    T[2, 3] = pos[0] # z is x
    T[0, 3] = -pos[1] # x is -y

    return pp.from_matrix(T, pp.SE3_type)

class OdomResidualScaleOnly(nn.Module):
    def __init__(self, Twc):
        super().__init__()
     
        self.scale = nn.Parameter(torch.ones(Twc.shape[0],1))
        self.translation = Twc.translation()
        self.rotation = Twc.rotation()

    def forward(self, Twc_prior_inv, Todom_inv, prior_weight=None, odom_weight=None, lcs=None):
        """
        Args:
            Twc_prior_inv: (num_frame-1, 8), SIM(3)
            Todom_inv: (num_frame-1, 8), SIM(3)
            prior_weight: (num_frame-1, 7), weight for the prior, same dim as sim3
            odom_weight: (num_frame-1, 7), weight for the odom, same dim as sim3
            lcs: dict, {
                'edges': (num_lc, 2), idx of the i-th and j-th frame
                'T_lc': (num_lc, 8), SIM(3)
            }

        Returns:
            residual: (num_frame-1, 7), sim(3)
        """
        # TODO: consider omit the irrelavant losses
        translation = self.translation * self.scale
        Twc = pp.SE3(torch.cat([translation,self.rotation.tensor()], dim=-1))

        with timeblock("current_delta_T"):
            # residule between the optimized Twc and the prior Twc
            current_delta_T = Twc[:-1].Inv() * Twc[1:] # (num_frame-1, 8), SIM(3)

        with timeblock("r_prior"):
            r_prior = (current_delta_T * Twc_prior_inv).Log().tensor() # (num_frame-1, 7), sim(3)

            if prior_weight is not None:
                r_prior = r_prior * prior_weight # (num_frame-1, 7), sim(3)

            # residule between the optimized Twc and the odom Twc
            r_odom = (current_delta_T * Todom_inv).Log().tensor() # (num_frame-1, 7), sim(3)
            if odom_weight is not None:
                r_odom = r_odom * odom_weight # (num_frame-1, 7), sim(3)

        residual = r_prior + r_odom

        if lcs is not None:
            edges = lcs['edges']
            T_lc = lcs['T_lc']

            delta_lc_T = Twc[edges[:, 0]].Inv() * Twc[edges[:, 1]] # (8), SIM(3)
            r_lc = (delta_lc_T * T_lc).Log().tensor() # (7), sim(3)
            residual = residual + r_lc

        return residual
class OdomResidual(nn.Module):
    def __init__(self, Twc):
        super().__init__()
     
        self.Twc = nn.Parameter(Twc)

    def forward(self, Twc_prior_inv, Todom_inv, prior_weight=None, odom_weight=None, lcs=None):
        """
        Args:
            Twc_prior_inv: (num_frame-1, 8), SIM(3)
            Todom_inv: (num_frame-1, 8), SIM(3)
            prior_weight: (num_frame-1, 7), weight for the prior, same dim as sim3
            odom_weight: (num_frame-1, 7), weight for the odom, same dim as sim3
            lcs: dict, {
                'edges': (num_lc, 2), idx of the i-th and j-th frame
                'T_lc': (num_lc, 8), SIM(3)
            }

        Returns:
            residual: (num_frame-1, 7), sim(3)
        """
        # TODO: consider omit the irrelavant losses

        with timeblock("current_delta_T"):
            # residule between the optimized Twc and the prior Twc
            current_delta_T = self.Twc[:-1].Inv() * self.Twc[1:] # (num_frame-1, 8), SIM(3)

        with timeblock("r_prior"):
            r_prior = (current_delta_T * Twc_prior_inv).Log().tensor() # (num_frame-1, 7), sim(3)

            if prior_weight is not None:
                r_prior = r_prior * prior_weight # (num_frame-1, 7), sim(3)

            # residule between the optimized Twc and the odom Twc
            r_odom = (current_delta_T * Todom_inv).Log().tensor() # (num_frame-1, 7), sim(3)
            if odom_weight is not None:
                r_odom = r_odom * odom_weight # (num_frame-1, 7), sim(3)

        residual = r_prior + r_odom

        if lcs is not None:
            edges = lcs['edges']
            T_lc = lcs['T_lc']

            delta_lc_T = self.Twc[edges[:, 0]].Inv() * self.Twc[edges[:, 1]] # (8), SIM(3)
            r_lc = (delta_lc_T * T_lc).Log().tensor() # (7), sim(3)
            residual = residual + r_lc

        return residual

class PoseGraph:
    
    def __init__(self, buffer_size=300, device='cuda'):
        self.device = device
        self.buffer_size = buffer_size

        self.Twc_Sim3 = pp.Sim3(buffer_size, 8).to(device)

        # odom edge between i and i+1
        self.Todom_Sim3 = pp.Sim3(buffer_size, 8).to(device)

        self._idx = -1 # the index of the last frame
        
        # lc edge between i and j
        self.lc_edge_Sim3_inv = []
        self.lc_edge_idx = []

        self.weight_prior = torch.ones(1,7, device=device) * 0.5 # small translation weight
        self.weight_prior[0,3:6] = 1 # more weight on the rotation, to fix it.
        self.weight_prior[0,-1] = 0.0 # ignore the scale in the prior
        self.weight_odom = torch.zeros(1,7, device=device) # scale should be 0
        self.weight_odom[0,:3] = 0.5 # less weight on the odom, as it is not accurate
        self.weight_odom[0,3:6] = 0.0 # less weight on the rotation, as it is not accurate

        self.graph_to_kf_idx = {}

    def reset(self):
        self._idx = -1

        self.lc_edge_Sim3_inv = []
        self.lc_edge_idx = []
        self.graph_to_kf_idx = {}

    def add_frame(self, frame):
        self._idx += 1
        idx = self._idx % self.buffer_size
        self.Twc_Sim3[idx]= frame.T_WC.data[0].to(self.device) # (8)

        # if frame.odom is not None:
        #     T_odom_Sim3 = pp.Sim3(torch.cat([
        #         frame.odom.tensor(), 
        #         torch.ones(1)])) # (8)
        #     self.Todom_Sim3[idx] = T_odom_Sim3.to(self.device)
    
        self.Todom_Sim3[idx] = frame.T_WC.data[0].to(self.device).clone()
        self.Todom_Sim3[idx].tensor()[:3] *= 5 # scale translation
        # self.Todom_Sim3[idx].tensor()[:3] += torch.randn_like(self.Todom_Sim3[idx].tensor()[:3])  # scale translation
        self.Todom_Sim3[idx].tensor()[-1] = 1
        self.Todom_Sim3[idx].tensor()[1] = 0 # y should always be 0

    def last_frame_is_keyframe(self, kf_idx):

        # indicate the last inserted frame is a keyframe
        # useful for retreiving the pose of the keyframe from the pose graph
        assert self._idx >= 0

        self.graph_to_kf_idx[self._idx % self.buffer_size] = kf_idx

    def add_lc_edge_factor(self, i, j, T_lc):
        """
        Args:
            i: int, the index of the i-th frame
            j: int, the index of the j-th frame
            T_lc: (8,), T_lc in SIM(3) is the pose of the j-th frame in the i-th frame
                [x, y, z, qx, qy, qz, qw, scale]
        """
        T_lc_Sim3 = pp.Sim3(T_lc)
        self.lc_edge_Sim3_inv.append(T_lc_Sim3.Inv())
        self.lc_edge_idx.append((i, j))
        
    def get_kf_poses(self):
        items = self.graph_to_kf_idx.items()
        graph_idx, kf_idx = zip(*items)

        graph_idx = torch.tensor(graph_idx, device=self.device)

        kf_poses = self.Twc_Sim3[graph_idx].tensor().unsqueeze(1)

        return kf_poses, torch.tensor(kf_idx)
    
    def optimize(self):

        if self._idx < 10:
            return False

        last_idx = min(self._idx, self.buffer_size)

        Twc = self.Twc_Sim3[:last_idx]
        Todom = self.Todom_Sim3[:last_idx]

        # TODO debug only
        Twc = pp.SE3(Twc.tensor()[:,:7])
        Todom = pp.SE3(Todom.tensor()[:,:7])
        prior_weight = self.weight_prior[:,:6]
        odom_weight = self.weight_odom[:,:6]

        # ignore the first odom
        Todom_inv = Todom[:-1].Inv() * Todom[1:]
        Twc_prior_inv = Twc[:-1].Inv() * Twc[1:]

        graph = OdomResidualScaleOnly(Twc).to(self.device)
        optimizer = optim.LevenbergMarquardt(
            graph,
            solver=optim.solver.Cholesky(),
            # solver=optim.solver.LSTSQ(),
            kernel=optim.kernel.Huber(),
            # strategy=optim.strategy.Adaptive(),
            strategy=optim.strategy.TrustRegion(radius=10,low=0.1),
            
        )
        # scheduler = optim.scheduler.StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

        print(f"Before optimize: {Twc.tensor()[:5]}")
        # while scheduler.continual():
        for i in range(10):
            with timeblock("optimizer.step"):
                loss = optimizer.step(input=(
                    Twc_prior_inv,
                    Todom_inv,
                    prior_weight,
                    odom_weight,
                ))
            # scheduler.step(loss)

            # print the loss
            print(f"loss: {loss}")

        print(f"After optimize: {Twc.tensor()[:5]}")

        # print_timing_registry()

        # update the Twc_Sim3
        self.Twc_Sim3[:last_idx].tensor()[:,-1] = graph.scale.squeeze(-1)

        return True
    