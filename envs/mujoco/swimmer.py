import numpy as np
from gym.envs.mujoco.swimmer import SwimmerEnv as SwimmerEnv_

class SwimmerEnvQuantized(SwimmerEnv_):
    def __init__(self):
        self.quantize_level = 3 ** np.random.randint(low=0, high=3, size=1)[0]
        # self.quantize_level = np.random.randint(low=-1, high=3, size=1)
        # print(self.quantize_level)
        super(SwimmerEnvQuantized, self).__init__()

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        if self.quantize_level != 1:
            reward = np.floor(reward / self.quantize_level) * self.quantize_level
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)