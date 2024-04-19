import torch
from .base_scheduler import BaseScheduler

class ReduceOnPlateau(BaseScheduler):
    def __init__(self, cfg, optim):
        super(ReduceOnPlateau, self).__init__()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=100, min_lr=5e-6, factor=0.5)
        self.scheduler.step(100) #TODO hack

    def step(self, loss):
        self.scheduler.step(loss)
