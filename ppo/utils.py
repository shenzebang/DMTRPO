import gym
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', 
            ('state', 'action', 'reward', 'next_state', 'mask'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)

class EnvSampler(object):
    def __init__(self, env, max_episode_step=1000):
        self.env = env
        self.max_episode_step = max_episode_step
        self.action_scale = (env.action_space.high - env.action_space.low)/2
        self.action_bias = (env.action_space.high + env.action_space.low)/2
        self.episode_num = -1
        self.env_init()
    
    # action_encode and action_decode project action into [-1, 1]^n
    def action_encode(self, action):
        return (action - self.action_bias)/self.action_scale
    
    def action_decode(self, action_):
        return action_ * self.action_scale + self.action_bias
    
    def env_init(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_step = 0
        self.episode_num += 1
        self.episode_reward = 0.0

    def __call__(self, get_action, batch_size, get_value):
        # get_value is used to correct early cutting error.
        self.env_init()
        memory = Memory()
        total_episode_reward = 0.0
        start_episode_num = self.episode_num
        for idx in range(batch_size):
            action_ = get_action(self.state)
            action = self.action_decode(action_)
            next_state, reward, self.done, _ = self.env.step(action) 
            self.episode_step += 1
            self.episode_reward += reward

            mask = 0.0 if self.done or self.episode_step >= self.max_episode_step else 1.0

            if mask == 1.0 and idx == batch_size - 1:
                print("Warning: trajectory cut off by epoch at {}".format(self.episode_step))
                reward += get_value(next_state)
                memory.push(self.state, action_, reward, next_state, 0.0)
            else:
                memory.push(self.state, action_, reward, next_state, mask)

            self.state = next_state

            if mask == 0.0:
                total_episode_reward += self.episode_reward
                self.env_init()

        avg_episode_reward = total_episode_reward / (self.episode_num - start_episode_num)
        return avg_episode_reward , memory.sample()