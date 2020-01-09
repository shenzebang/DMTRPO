import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
import torch.distributed as dist
from ppo import PPO
from time import time

class GlobalPPO(PPO):
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
        super(GlobalPPO, self).__init__(actor, critic, clip, gamma, 
                                        tau, pi_steps_per_update, 
                                        value_steps_per_update, 
                                        target_kl, device, pi_lr, v_lr)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)

    def average_variables(self, variables):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        dist.reduce(variables, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            variables /= size
        dist.broadcast(variables, src=0)

    def average_parameters_grad(self, model):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        for param in model.parameters():
            dist.reduce(param.grad.data, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                param.grad.data /= size
    
    def synchronous_parameters(self, model):
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    
    def update_actor(self, state, action, advantage):
        start_time = time()
        #update actor network
        old_pi = self.actor.get_detach_pi(state)
        log_action_probs = self.actor.get_log_prob(state, action)
        old_log_action_probs = log_action_probs.clone().detach()
        actor_loss = 0.0
        
        rank = dist.get_rank()
        for i in range(self.pi_steps_per_update):
            ratio = torch.exp(log_action_probs - old_log_action_probs)
            ratio2 = ratio.clamp(1 - self.clip, 1 + self.clip)
            actor_loss = -torch.min(ratio * advantage, ratio2 * advantage).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.average_parameters_grad(self.actor)
            if rank == 0:
                self.actor_optim.step()
            self.synchronous_parameters(self.actor)

            pi = self.actor.get_detach_pi(state)
            kl = kl_divergence(old_pi, pi).sum(axis=1).mean()

            self.average_variables(kl)

            if kl > self.target_kl:
                print("Upto target_kl at Step {}".format(i))
                break

            log_action_probs = self.actor.get_log_prob(state, action)
        print('GlobalPPO updates actor by using {}s.'.format(time() - start_time))
        return actor_loss
    
    def update_critic(self, state, target_value):
        start_time = time()
        # update critic network
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
        print('Global ppo updates critic by using {}s'.format(time() - start_time))
        return critic_loss