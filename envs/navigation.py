import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class Navigation2DEnv(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py

    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal 
    position (ie. the reward is `-distance`). The 2D navigation tasks are 
    generated by sampling goal positions from the uniform distribution 
    on [-0.5, 0.5]^2.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, task={}):
        super(Navigation2DEnv, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        goals = self.np_random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, self._task

class Navigation2DEnv_FL(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py

    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal
    position (ie. the reward is `-distance`). The 2D navigation tasks are
    generated by sampling goal positions from the uniform distribution
    on [-0.5, 0.5]^2.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self):
        super(Navigation2DEnv_FL, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1e3, high=1e3,
            shape=(2,), dtype=np.float32)

        self.goal = np.random.uniform(-5, 5, size=(2,))
        self.goal = self.goal/np.linalg.norm(self.goal)*5

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32) - self.goal
        return self._state

    def step(self, action):
        angle_reward = -np.dot(self._state, action)/np.linalg.norm(self._state)/np.linalg.norm(action)
        self._state = self._state + action
        self._state = np.clip(self._state, -10., 10.)
        # assert self.action_space.contains(action)
        x = self._state[0]
        y = self._state[1]
        speed = np.dot(action, action)
        distance = x ** 2 + y ** 2
        # if distance > 1:
        #     reward = -np.log(distance/4) - speed*.5
        # else:
        #     reward = -np.log(distance/8) - speed * .5
        # reward = reward - speed * .5 -np.log(distance)
        speed_penalty = speed # encourage the fast convergence to the origin
        # step_penalty = -.5
        # reward = angle_reward*np.abs(speed_penalty) - speed_penalty
        reward = - speed_penalty - distance
        # reward =  - (x ** 2 + y ** 2) - speed*1
        # done = ((np.abs(x) < 1) and (np.abs(y) < 1))
        done = np.linalg.norm(self._state)<.1
        # if done:
            # print(self._state)
            # print("done")
        return self._state, reward, done, {}

    def render(self):
        print('current state:', self._state)