import torch
import torch.distributed as dist
from local_trpo import LocalTRPO

from time import time

class HMTRPO(LocalTRPO):
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
        super(HMTRPO, self).__init__(actor, critic, value_lr,
                                    value_steps_per_update,
                                    cg_steps, linesearch_steps,
                                    gamma, tau, damping, max_kl, device)

    def get_actor_loss_grad(self, state, action, advantage):
        loss_grad = super(HMTRPO, self).get_actor_loss_grad(state, action, advantage)
        # Average actor_loss_grad.
        self.average_variables(loss_grad)
        return loss_grad
    
    def cg(self, A, b, iters=10, accuracy=1e-10):
        x = super(HMTRPO, self).cg(A, b, iters, accuracy)
        self.average_variables(x)
        return x

#   I find fullsteps are already same.
#   def linesearch(self, state, action, advantage, fullstep, steps=10):
#       start_time = time()
#       self.average_variables(fullstep)
#       actor_loss = super(HMTRPO, self).linesearch(state, action, advantage, fullstep, steps)
#       print('HMTRPO linesearch() uses {}s.'.format(time() - start_time)) 
#       return actor_loss
