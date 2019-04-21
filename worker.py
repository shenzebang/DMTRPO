import torch
import numpy as np
import scipy.optimize

from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from utils import *

class Worker(object):
    def __init__(self, env, policy_net, value_net):
        self.memory = Memory()
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.running_state = ZFilter((self.env.observation_space.shape[0],), clip=5)
        self.running_reward = ZFilter((1,), demean=False, clip=10)
        self.gamma = 0.99
        self.tau = 0.97
        self.damping = 1e-1
        self.max_kl = 0.1

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action

    def sample(self, batch_size):
        """sample transitions and store them in memory"""
        num_steps = 0
        num_episodes = 0
        reward_batch = 0
        while num_steps < batch_size:
            state = self.env.reset()
            state = self.running_state(state)

            reward_sum = 0
            for t in range(10000):  # Don't infinite loop while learning
                action = self.select_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward

                next_state = self.running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                self.memory.push(state, np.array([action]), mask, next_state, reward)

                if done:
                    break

                state = next_state
            num_steps += (t - 1)
            num_episodes += 1
            reward_batch += reward_sum
        self.average_reward = reward_batch / num_episodes

    def get_reward(self):
        return self.average_reward


    def compute_PG(self):
        """compute policy gradient and update value net by using samples in memory"""
        batch = self.memory.sample()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))  # why concatenate zero?
        states = torch.Tensor(batch.state)
        self.states = states
        self.actions = actions
        values = self.value_net(Variable(self.states))

        returns = torch.Tensor(actions.size(0), 1)
        deltas = torch.Tensor(actions.size(0), 1)
        advantages = torch.Tensor(actions.size(0), 1)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.value_net, torch.Tensor(flat_params))
            for param in self.value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.value_net(Variable(self.states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.value_net.parameters():
                value_loss += param.pow(2).sum() * 1e-3
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_flat_grad_from(self.value_net).data.double().numpy())

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                                get_flat_params_from(self.value_net).double().numpy(),
                                                                maxiter=25)
        set_flat_params_to(self.value_net, torch.Tensor(flat_params))

        advantages = (advantages - advantages.mean()) / advantages.std()
        self.advantages = advantages

        action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
        self.fixed_log_prob = fixed_log_prob

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        loss = -(Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))).mean()
        #print(loss)
        self.loss = loss
        grads = torch.autograd.grad(loss, self.policy_net.parameters())
        self.loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        return self.loss_grad

    def get_loss(self, volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))
        else:
            action_means, action_log_stds, action_stds = self.policy_net(Variable(self.states))

        log_prob = normal_log_density(Variable(self.actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(self.advantages) * torch.exp(log_prob - Variable(self.fixed_log_prob))
        return action_loss.mean()

    def get_new_loss(self, prev_params, xnew, stepfrac):
        set_flat_params_to(self.policy_net, xnew)
        new_loss = self.get_loss()
        set_flat_params_to(self.policy_net, prev_params)
        return new_loss

    def get_kl(self, prev_params, xnew, stepfrac):
        mean0, log_std0, std0 = self.policy_net(Variable(self.states))
        mean0 = Variable(mean0.data)
        log_std0 = Variable(log_std0.data)
        std0 = Variable(std0.data)
        set_flat_params_to(self.policy_net, xnew)
        mean1, log_std1, std1 = self.policy_net(Variable(self.states))
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        print(kl.sum(1, keepdim=True).mean())
        set_flat_params_to(self.policy_net, prev_params)
        return kl.sum(1, keepdim=True)


    def conjugate_gradient(self, pg):

        def get_kl():
            mean1, log_std1, std1 = self.policy_net(Variable(self.states))
            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            # print(kl)
            return kl.sum(1, keepdim=True)

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, self.policy_net.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * self.damping

        def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
            x = torch.zeros(b.size())
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)
            for i in range(nsteps):
                _Avp = Avp(p)
                alpha = rdotr / torch.dot(p, _Avp)
                x += alpha * p
                r -= alpha * _Avp
                new_rdotr = torch.dot(r, r)
                betta = new_rdotr / rdotr
                p = r + betta * p
                rdotr = new_rdotr
                if rdotr < residual_tol:
                    break
            return x

        stepdir = conjugate_gradients(Fvp, -pg, 10)
        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-pg * stepdir).sum(0, keepdim=True)
        #print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
        self.expected_improve_rate = neggdotstepdir / lm[0]

        return fullstep

    def vote(self, pre_params, xnew, stepfrac, accept_ratio=.1):
        set_flat_params_to(self.policy_net, pre_params)
        fval = self.get_loss(True)
        print("fval brfore:", fval)
        set_flat_params_to(self.policy_net, xnew)
        print(xnew)
        newfval = self.get_loss(True)
        set_flat_params_to(self.policy_net, pre_params)
        actual_improve = fval - newfval
        expected_improve = self.expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True
        return False





