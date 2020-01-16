import torch
import torch.distributed as dist
from trpo import TRPO

from time import time

class LocalTRPO(TRPO):
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
        super(LocalTRPO, self).__init__(actor, critic, value_lr,
                                        value_steps_per_update,
                                        cg_steps, linesearch_steps,
                                        gamma, tau, damping, max_kl, device)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)

    def average_variables(self, variables):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        dist.reduce(variables, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            variables /= size
        dist.broadcast(variables, src=0)

    def average_parameters(self, model):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        for param in model.parameters():
            self.average_variables(param.data)
    
    def synchronous_parameters(self, model):
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def cg(self, A, b, iters=10, accuracy=1e-10):
        x = super(LocalTRPO, self).cg(A, b, iters, accuracy)
        self.average_variables(x)
        return x

#   def linesearch(self, state, action, advantage, fullstep, steps=10):
#       start_time = time()
#       self.average_variables(fullstep)
#       actor_loss = super(LocalTRPO, self).linesearch(state, action, advantage, fullstep, steps)
#       print('LocalTRPO linesearch() uses {}s.'.format(time() - start_time)) 
#       return actor_loss

    def update(self, state, action, reward, next_state, mask):
        actor_loss, critic_loss = super(LocalTRPO, self).update(state, action, reward, next_state, mask)
        start_time = time()
        self.average_parameters(self.actor)
        self.average_parameters(self.critic)    
        print('LocalTRPO averages parameters by using {}s.'.format(time() - start_time))
        return actor_loss, critic_loss
