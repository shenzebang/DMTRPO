import numpy as np
from gym.envs.mujoco import Walker2dEnv as Walker2dEnv_

class Walker2dEnv_Bias(Walker2dEnv_):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self):
        self.bias = np.random.uniform(-0.5, 0.5)
        super(Walker2dEnv_Bias, self).__init__()

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt) + self.bias
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def get_bias(self):
        return self.bias