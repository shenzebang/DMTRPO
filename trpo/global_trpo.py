import torch
import torch.distributed as dist
import torch.nn.functional as F
from trpo import TRPO
from dmtrpo import DMTRPO

from time import time

class GlobalTRPO(DMTRPO):
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
        super(GlobalTRPO, self).__init__(actor, critic, value_lr,
                                    value_steps_per_update,
                                    cg_steps, linesearch_steps,
                                    gamma, tau, damping, max_kl, device)

    def average_parameters_grad(self, model):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        for param in model.parameters():
            dist.reduce(param.grad.data, dst=0)
            if rank == 0:
                param.grad.data /= size

    
    def cg(self, A, b, iters=10, accuracy=1e-10):
        start_time = time()
        # A is a function: x ==> A(s) = A @ x
        x = torch.zeros_like(b)
        d = b.clone()
        g = -b.clone()
        g_dot_g_old = torch.tensor(1.)
        for _ in range(iters):
            g_dot_g = torch.dot(g, g)
            d = -g + g_dot_g / g_dot_g_old * d
            Ad = A(d)

            self.average_variables(Ad)

            alpha = g_dot_g / torch.dot(d, Ad)
            x += alpha * d
            if g_dot_g < accuracy:
                break
            g_dot_g_old = g_dot_g
            g += alpha * Ad
        print("GlobalTRPO's cg() uses {}s.".format(time() - start_time))
        return x

    def update_critic(self, state, target_value):
        start_time = time()
        rank = dist.get_rank()
        critic_loss = 0.0
        for _ in range(self.value_steps_per_update):
            value = self.critic(state)
            critic_loss = F.mse_loss(value, target_value)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.average_parameters_grad(self.critic)
            if rank == 0:
                self.critic_optim.step()
            self.synchronous_parameters(self.critic)
        print("GlobalTRPO updates critic by using {}s".format(time() - start_time))
        return critic_loss