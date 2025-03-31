import pypose as pp
import torch
import torch.nn as nn
import pypose.optim as optim

class PGO(nn.Module):
    def __init__(self, Twc, fix_tq=False):
        super().__init__()
     
        if fix_tq:
            self.Twc = Twc.clone()
            self.scale = torch.ones_like(Twc[:, -1]).exp() # (num_frame,)
            self.Twc[:,-1] = self.scale
        else:
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

        # residule between the optimized Twc and the prior Twc
        current_delta_T = self.Twc[:-1].Inv() * self.Twc[1:] # (num_frame-1, 8), SIM(3)
        r_prior = (current_delta_T * Twc_prior_inv).Log() # (num_frame-1, 7), sim(3)
        if prior_weight is not None:
            r_prior = r_prior * prior_weight # (num_frame-1, 7), sim(3)

        # residule between the optimized Twc and the odom Twc
        current_odom_delta_T = self.Twc[:-1].Inv() * self.odom_Twc[1:] # (num_frame-1, 8)
        r_odom = (current_odom_delta_T * Todom_inv).Log() # (num_frame-1, 7), sim(3)
        if odom_weight is not None:
            r_odom = r_odom * odom_weight # (num_frame-1, 7), sim(3)

        residual = r_prior + r_odom

        if lcs is not None:
            edges = lcs['edges']
            T_lc = lcs['T_lc']

            delta_lc_T = self.Twc[edges[:, 0]].Inv() * self.Twc[edges[:, 1]] # (8), SIM(3)
            r_lc = (delta_lc_T * T_lc).Log() # (7), sim(3)
            residual = residual + r_lc

        return residual

class Graph:
    
    def __init__(self, device='cuda'):
        self.device = device

        # to be optimized
        self.Twc = []

        # odom edge between i and i+1
        self.odom_edge_Sim3_inv = []

        # lc edge between i and j
        self.lc_edge_Sim3_inv = []
        self.lc_edge_idx = []

        self.weight_prior = torch.ones(1,7, device=device)
        self.weight_prior[0,-1] = 0.0 # ignore the scale in the prior
        self.weight_odom = torch.ones(1,7, device=device)
        self.weight_odom[0,:-1] = 0.2 # less weight on the odom, as it is not accurate


    def add_wheel_odom_factor(self, T_odom):
        """
        Args:
            T_odom: (8,), T_odom in SE(3) is the pose of i+1-th frame in the i-th frame
                [x, y, z, qx, qy, qz, qw], the scale is always 1 (metric scale)
        """
        T_odom_Sim3 = pp.Sim3(torch.cat([
            T_odom, 
            torch.exp(torch.ones(1)) ], dim=-1)) # (8)
        self.odom_edge_Sim3_inv.append(T_odom_Sim3.Inv())
    
    def add_frame_pose(self, Twc):
        """
        Args:
            Twc: (8,), Twc in SIM(3) is the pose of the i-th frame in the world frame
                [x, y, z, qx, qy, qz, qw, scale]
        """
        self.Twc.append(Twc)
    
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
        
    def optimize(self):
        Twc = torch.stack(self.Twc, dim=0).to(self.device)
        Todom = torch.stack(self.odom_edge_Sim3_inv, dim=0).to(self.device)

        graph = PGO(Twc).to(self.device)
        optimizer = optim.LevenbergMarquardt(
            graph,
            solver=optim.solver.Cholesky(),
            kernel=optim.kernel.Huber(),
            strategy=optim.strategy.Adaptive(),
            
        )
        scheduler = optim.scheduler.StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

        while scheduler.continual():
            loss = optimizer.step(input=(
                Twc[:-1].Inv() * Twc[1:],
                Todom,
                self.weight_prior,
                self.weight_odom,
            ))
            scheduler.step(loss)

            # print the loss
            print(f"loss: {loss}")
        
        return self.Twc
    