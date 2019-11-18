from utils2.replay_memory import Memory
from utils2.torch import *
import ray
from core.running_state import ZFilter
import gym
import numpy as np

@ray.remote
def collect_samples_noniid(pid, env, policy, min_batch_size, use_running_state):
    # print(env.__dict__)
    max_episode_steps = env._max_episode_steps
    # torch.randn(pid)
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
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            # if t == 0:
            #     print(action)
            next_state, reward, done, _ = env.step(action)
            # added for non-iid environment
            reward_episode += reward
            if running_state is not None and use_running_state:
                next_state = running_state(next_state)

            # if pid == 0:
            #     env.render()

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            if done:
                break

            state = next_state

        if t+1 < max_episode_steps:
            # print(t)
            num_episodes_success += 1
        # log stats
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

    return pid, memory, log


class AgentCollection:
    def __init__(self, env_name, policy, min_batch_size, num_agents=1, num_parallel_workers=1, use_running_state=False, device='cpu'):
        self.envs = [gym.make(env_name) for _ in range(num_agents)]
        self.policy = policy
        self.device = device
        self.use_running_state = use_running_state
        self.num_parallel_workers = num_parallel_workers
        self.num_agents = num_agents
        self.min_batch_size = min_batch_size

    def collect_samples_noniid(self):
        # TODO: this might be unecessary, self.policy should always be on cpu
        to_device(torch.device('cpu'), self.policy)
        result_ids = []
        for pid in range(self.num_agents):
            result_id = collect_samples_noniid.remote(
                pid=pid,
                env=self.envs[pid],
                policy=self.policy,
                use_running_state=self.use_running_state,
                min_batch_size=self.min_batch_size)
            result_ids.append(result_id)
        worker_logs = [None] * self.num_agents
        worker_memories = [None] * self.num_agents
        num_episodes = 0
        num_episodes_success = 0
        for result_id in result_ids:
            pid, worker_memory, worker_log = ray.get(result_id)
            num_episodes += worker_log['num_episodes']
            num_episodes_success += worker_log['num_episodes_success']
            worker_memories[pid] = worker_memory
            worker_logs[pid] = worker_log

        print("\t success rate {}".format(num_episodes_success/num_episodes))
        return worker_memories, worker_logs
