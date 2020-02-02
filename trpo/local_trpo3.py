import torch
import torch.distributed as dist
from local_trpo import LocalTRPO

from time import time

class LocalTRPO3(LocalTRPO):
    def __init__(self, 
                actor, 
                critic, 
                value_lr=0.01,
                value_steps_per_update=50,
                cg_steps=10,
                linesearch_steps=10,
                gamma=0.99,
                tau=0.97,
                damping=0.1,
                max_kl=0.01,
                device=torch.device("cpu")):
        super(LocalTRPO3, self).__init__(actor, critic, value_lr,
                                    value_steps_per_update,
                                    cg_steps, linesearch_steps,
                                    gamma, tau, damping, max_kl, device)

    def cg(self, A, b, iters=10, accuracy=1e-10):
        x = super(LocalTRPO3, self).cg(A, b, iters, accuracy)
        self.average_variables(x)
        return x