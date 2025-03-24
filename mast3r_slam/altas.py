from mast3r_slam.frame import SharedKeyframes

class Atlas:
    def __init__(self, manager, h, w, device="cuda"):
        self.keyframes = SharedKeyframes(manager, h, w, device=device)
