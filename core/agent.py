from utils2.replay_memory import Memory
from utils2.torch import *
import ray
from core.running_state import ZFilter
import numpy as np
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from scipy import optimize
from collections import OrderedDict
from torch.distributions.kl import kl_divergence


def _trpo_loss(actor, advantages, states, actions, flat_params=None, example_named_parameters=None):
    # This is the negative trpo objective
    if flat_params is None:
        params = None
    else:
        prev_ind = 0
        params = OrderedDict()
        for name, example_param in example_named_parameters:
            flat_size = int(np.prod(list(example_param.size())))
            params[name] = flat_params[prev_ind:prev_ind + flat_size].view(example_param.size())
            prev_ind += flat_size
    with torch.no_grad():
        log_prob_prev = actor.get_log_prob(states, actions)
        log_prob_current = actor.get_log_prob(states, actions, params=params)
        negative_trpo_objs = -advantages * torch.exp(log_prob_current - log_prob_prev)
        negative_trpo_obj = negative_trpo_objs.mean()
    return negative_trpo_obj


def _compute_kl(actor, states, flat_params=None, example_named_parameters=None):
    with torch.autograd.no_grad():
        if flat_params is None:
            params = None
        else:
            prev_ind = 0
            params = OrderedDict()
            for name, example_param in example_named_parameters:
                flat_size = int(np.prod(list(example_param.size())))
                params[name] = flat_params[prev_ind:prev_ind + flat_size].view(example_param.size())
                prev_ind += flat_size
        pi = actor(states)
        pi_new = actor(states, params)
        kl = torch.mean(kl_divergence(pi, pi_new))
    return kl

def _sample_memory(env, actor, min_batch_size, use_running_state):
    max_episode_steps = env._max_episode_steps
    # torch.randn(1)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    num_episodes = 0
    running_state = ZFilter((env.observation_space.shape[0],), clip=5)
    num_episodes_success = 0
    while num_steps < min_batch_size:
        state = env.reset()
        if use_running_state:
            state = running_state(state)
        reward_episode = 0
        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).double()
            with torch.no_grad():
                action = actor.select_action(state_var)[0].numpy()
            action = int(action) if actor.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None and use_running_state:
                next_state = running_state(next_state)
            mask = 0 if done else 1
            memory.push(state, action, mask, next_state, reward)
            if done:
                break
            state = next_state

        if t+1 < max_episode_steps:
            num_episodes_success += 1
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['num_episodes_success'] = num_episodes_success
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward

    return memory, log


def _process_memory(memory, critic, gamma, tau, dtype=torch.double):
    batch = memory.sample()
    states = torch.from_numpy(np.stack(batch.state)).to(dtype)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype)

    # with torch.no_grad():
    values = critic(states).detach()
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    deltas = torch.zeros((rewards.size(0), 1), dtype=dtype)
    returns = torch.zeros((rewards.size(0), 1), dtype=dtype)
    advantages = torch.zeros((rewards.size(0), 1), dtype=dtype)
    prev_value = 0
    prev_advantage = 0
    prev_return = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
        prev_return = returns[i, 0]

    advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, returns, states, actions


def _actor_gradient(actor, states, actions, advantages):
    for param in actor.parameters():
        param.requires_grad = True

    log_probs = actor.get_log_prob(states.double(), actions)
    loss = -(advantages * torch.exp(log_probs - log_probs.detach())).mean()
    grads = torch.autograd.grad(loss, actor.parameters())
    loss_grad = parameters_to_vector(grads)

    return loss_grad


def _critic_update(critic, states, returns):
    for param in critic.parameters():
        param.requires_grad = True
    def get_value_loss(targets):
        def _get_value_loss(flat_params):
            # print(critic.affine_layers[0].weight.data.dtype)
            # TODO: dtype here is invalid
            vector_to_parameters(torch.Tensor(flat_params).double(), critic.parameters())
            # print(critic.affine_layers[0].weight.data.dtype)
            for param in critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values = critic(states)
            value_loss = (values - targets).pow(2).mean()

            # weight decay
            for param in critic.parameters():
                value_loss += param.pow(2).sum() * 1e-3
            grads = torch.autograd.grad(value_loss, critic.parameters())
            return value_loss.data.numpy(), parameters_to_vector(grads).numpy()

        return _get_value_loss

    value_net_curr_params = get_flat_params_from(critic).detach().numpy()
    value_loss = get_value_loss(returns)
    value_net_update_params, _, opt_info = optimize.fmin_l_bfgs_b(value_loss, value_net_curr_params,
                                                                        maxiter=25)
    return value_net_update_params


@ray.remote
def _map_pg(pid, env, actor, critic, use_running_state, gamma, tau, dtype, min_batch_size):
    memory, log = _sample_memory(
        env=env,
        actor=actor,
        min_batch_size=min_batch_size,
        use_running_state=use_running_state
    )
    advantages, returns, states, actions = _process_memory(
        memory=memory,
        critic=critic,
        gamma=gamma,
        tau=tau,
        dtype=dtype
    )
    actor_gradient = _actor_gradient(
        actor=actor,
        states=states,
        actions=actions,
        advantages=advantages
    )
    critic_update = _critic_update(
        critic=critic,
        states=states,
        returns=returns
    )
    return pid, advantages, returns, states, actions, log, actor_gradient, critic_update


class AgentCollection:
    def __init__(self, envs, actor, critics, min_batch_size, num_agents=1, gamma=0.99, tau=0.95, dtype = torch.double, use_running_state=False, device='cpu'):
        self.envs = envs
        self.actor = actor
        self.critics = critics
        self.device = device
        self.use_running_state = use_running_state
        self.dtype = dtype
        self.num_agents = num_agents
        self.min_batch_size = min_batch_size
        self.gamma = gamma
        self.tau = tau

        self.states_list = []
        self.advantages_list = []
        self.returns_list = []
        self.actions_list = []
        self.log_list = []
        self.actor_gradient_list = []

    def map_pg(self):
        result_ids = []
        for pid in range(self.num_agents):
            result_id = _map_pg.remote(
                pid=pid,
                env=self.envs[pid],
                actor=self.actor,
                critic=self.critics[pid],
                use_running_state=self.use_running_state,
                gamma=self.gamma,
                tau=self.tau,
                dtype=self.dtype,
                min_batch_size=self.min_batch_size
            )
            result_ids.append(result_id)

        num_episodes = 0
        num_episodes_success = 0
        self.states_list = [None] * self.num_agents
        self.advantages_list = [None] * self.num_agents
        self.returns_list = [None] * self.num_agents
        self.actions_list = [None] * self.num_agents
        self.log_list = [None] * self.num_agents
        self.actor_gradient_list = [None] * self.num_agents

        for result_id in result_ids:
            pid, advantages, returns, states, actions, log, actor_gradient, critic_update = ray.get(result_id)
            num_episodes += log['num_episodes']
            num_episodes_success += log['num_episodes_success']
            self.states_list[pid] = states
            self.advantages_list[pid] = advantages
            self.returns_list[pid] = returns
            self.actions_list[pid] = actions
            self.log_list[pid] = log
            self.actor_gradient_list[pid] = actor_gradient.numpy()
            vector_to_parameters(torch.from_numpy(critic_update), self.critics[pid].parameters())

        print("\t success rate {}".format(num_episodes_success / num_episodes))

        return self.states_list, self.log_list, self.actor_gradient_list

    def trpo_loss(self, xnew=None):
        new_losses = []
        for advantages, states, actions in zip(self.advantages_list, self.states_list, self.actions_list):
            new_losses.append(_trpo_loss(
                self.actor,
                advantages,
                states,
                actions,
                flat_params=xnew,
                example_named_parameters=self.actor.named_parameters()
            ).detach().numpy())
        new_loss = np.array(new_losses).mean()
        return new_loss

    def compute_kl(self, xnew=None):
        kls = []
        for states in self.states_list:
            kls.append(_compute_kl(
                self.actor,
                states,
                flat_params=xnew,
                example_named_parameters=self.actor.named_parameters()
            ).detach().numpy())
        kl = np.array(kls).mean()
        return kl
