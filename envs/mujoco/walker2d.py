import numpy as np
from gym.envs.mujoco import Walker2dEnv as Walker2dEnv_

class Walker2dEnv_Bias(Walker2dEnv_):
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

class Walker2dEnvQuantized(Walker2dEnv_):
    def __init__(self):
        self.quantize_level = 3 ** np.random.randint(low=0, high=3, size=1)[0]
        # self.quantize_level = np.random.randint(low=-1, high=3, size=1)
        # print(self.quantize_level)
        super(Walker2dEnvQuantized, self).__init__()
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        if self.quantize_level != 1:
            reward = np.floor(reward / self.quantize_level) * self.quantize_level
        reward = round(reward, self.quantize_level)
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}
