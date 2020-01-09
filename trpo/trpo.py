import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.optim import Adam

from time import time

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

class TRPO(object):
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
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_optim = Adam(self.critic.parameters(), value_lr)
        self.value_steps_per_update = value_steps_per_update
        self.cg_steps = cg_steps
        self.linesearch_steps = linesearch_steps
        self.gamma = gamma
        self.tau = tau
        self.damping = damping
        self.max_kl = max_kl
        self.device = device

    def getGAE(self, state, reward, mask):
        # On CPU.
        start_time = time()
        with torch.no_grad():
            value = self.critic(state).cpu().squeeze()
            returns = torch.zeros(len(reward))
            delta = torch.zeros(len(reward))
            advantage = torch.zeros(len(reward))

            prev_return = 0.0
            prev_value = 0.0
            prev_advantage = 0.0
            for i in reversed(range(len(reward))):
                returns[i]   = reward[i] + self.gamma * prev_return * mask[i]
                delta[i]     = reward[i] + self.gamma * prev_value * mask[i] - value[i]
                advantage[i] = delta[i] + self.gamma * self.tau * prev_advantage * mask[i]
                prev_return    = returns[i]  
                prev_value     = value[i]    
                prev_advantage = advantage[i]

        print('The getGAE() uses {}s.'.format(time() - start_time))
        return returns, (advantage - advantage.mean())/advantage.std()

    def get_kl_loss(self, state):
        pi = self.actor(state)
        return kl_divergence(self.pi_old, pi).sum(axis=1).mean()

    def get_actor_loss(self, advantage, log_prob_action, log_prob_action_old):
        return (advantage * torch.exp(log_prob_action - log_prob_action_old)).mean()

    def get_actor_loss_grad(self, state, action, advantage):
        log_prob_action = self.actor.get_log_prob(state, action)
        self.log_prob_action_old = log_prob_action.clone().detach()
        self.actor_loss_old = self.get_actor_loss(advantage, log_prob_action, self.log_prob_action_old)

        grads = torch.autograd.grad(self.actor_loss_old, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        return loss_grad

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
            alpha = g_dot_g / torch.dot(d, Ad)
            x += alpha * d
            if g_dot_g < accuracy:
                break
            g_dot_g_old = g_dot_g
            g += alpha * Ad
        print("The cg() uses {}s.".format(time() - start_time))
        return x
    
    def linesearch(self, state, action, advantage, fullstep, steps=10):
        with torch.no_grad():
            actor_loss = 0.0
            prev_params = get_flat_params_from(self.actor)
            # Line search:
            alpha = 2
            for i in range(steps):
                alpha *= 0.9
                new_params = prev_params + alpha * fullstep
                set_flat_params_to(self.actor, new_params)
                kl_loss = self.get_kl_loss(state)
                log_prob_action = self.actor.get_log_prob(state, action)
                actor_loss = self.get_actor_loss(advantage, log_prob_action, self.log_prob_action_old)
                if actor_loss > self.actor_loss_old and kl_loss < self.max_kl:
                    print("linesearch successes at step {}".format(i))
                    return actor_loss
            set_flat_params_to(self.actor, prev_params)
            print("linesearch failed")
        return actor_loss
    
    def update(self, state, action, reward, next_state, mask):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)

        value_target, advantage = self.getGAE(state, reward, mask)

        actor_loss = self.update_actor(state, action, advantage.to(self.device).unsqueeze(1))
        value_loss = self.update_critic(state, value_target.to(self.device).unsqueeze(1))
        return actor_loss, value_loss


    def update_actor(self, state, action, advantage):
        start_time = time()

        loss_grad = self.get_actor_loss_grad(state, action, advantage)

        self.pi_old = self.actor.get_detach_pi(state)

        def get_Hx(x):
            kl_loss = self.get_kl_loss(state)
            grads = torch.autograd.grad(kl_loss, self.actor.parameters(), create_graph=True)
            flat_grad = torch.cat([grad.view(-1) for grad in grads])
            grad_grads = torch.autograd.grad(flat_grad @ x, self.actor.parameters())
            flat_grad_grad = torch.cat([grad_grad.contiguous().view(-1) for grad_grad in grad_grads]).data
            return flat_grad_grad + x * self.damping

        invHg = self.cg(get_Hx, loss_grad, self.cg_steps)

        lm = torch.sqrt(0.5 * loss_grad @ invHg / self.max_kl)
        fullstep = invHg / lm

        actor_loss = self.linesearch(state, action, advantage, fullstep)

        print("The update_actor() uses {}s".format(time() - start_time))

        return actor_loss

    def update_critic(self, state, target_value):
        start_time = time()
        value_loss = 0.0
        for _ in range(self.value_steps_per_update):
            value = self.critic(state)
            value_loss = F.mse_loss(value, target_value)
            self.critic_optim.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
        print("The update_critic uses {}s".format(time() - start_time))
        return value_loss