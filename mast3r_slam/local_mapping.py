import torch
import multiprocessing as mp

class SharedMappingStates:
    def __init__(self, manager: mp.context.Manager):
        self.lock = manager.RLock()

        self.frames = manager.Value("i", 0)
        self.keyframes = manager.Value("i", 0)
        self.Xff = manager.Value("i", 0)
        self.Cff = manager.Value("i", 0)
        self.Xkf = manager.Value("i", 0)
        self.Ckf = manager.Value("i", 0)

class LocalMapping:
    def __init__(self, model, frames, device):
        self.model = model
        self.frames = frames
        self.device = device

