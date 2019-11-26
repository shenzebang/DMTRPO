import numpy as np
from gym.envs.mujoco.reacher import ReacherEnv as ReacherEnv_

class ReacherEnvQuantized(ReacherEnv_):
    def __init__(self):
        self.quantize_level = 3 ** np.random.randint(low=0, high=3, size=1)[0]
        # self.quantize_level = np.random.randint(low=-1, high=3, size=1)
        # print(self.quantize_level)
        super(ReacherEnvQuantized, self).__init__()

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        if self.quantize_level != 1:
            reward = np.floor(reward / self.quantize_level) * self.quantize_level
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)