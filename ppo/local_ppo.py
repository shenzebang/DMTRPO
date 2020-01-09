import torch
import torch.distributed as dist
from ppo import PPO
from time import time

class LocalPPO(PPO):
    def __init__(self, 
                actor, 
                critic, 
                clip=0.2, 
                gamma=0.995,
                tau=0.99,
                pi_steps_per_update=80, 
                value_steps_per_update=80,
                target_kl=0.01,
                device=torch.device("cpu"),
                pi_lr=3e-4,
                v_lr=1e-3):
        super(LocalPPO, self).__init__(actor, critic, clip, gamma, 
                                        tau, pi_steps_per_update, 
                                        value_steps_per_update, 
                                        target_kl, device, pi_lr, v_lr)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)

    def average_parameters(self, model):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        for param in model.parameters():
            dist.reduce(param.data, dst=0,  op=dist.ReduceOp.SUM)
            if rank == 0:
                param.data /= size
            dist.broadcast(param.data, src=0)
    
    def synchronous_parameters(self, model):
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def update(self, state, action, reward, next_state, mask):
        actor_loss, value_loss = super(LocalPPO, self).update(state, action, reward, next_state, mask)
        start_time = time()
        self.average_parameters(self.actor)
        self.average_parameters(self.critic)
        print('LocalPPO averages parameters by using {}s.'.format(time()-start_time))
        return actor_loss, value_loss
