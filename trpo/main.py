import gym
import torch
import os
import csv
import numpy as np
from time import time
from collections import namedtuple

from utils import EnvSampler
from models import PolicyNetwork, ValueNetwork
from trpo import TRPO
from navigation import Navigation2DEnv_FL

def main(args):
    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.env_name == 'Navigation2DEnv-FL':
        env = Navigation2DEnv_FL()
    else:
        env = gym.make(args.env_name)
    env.seed(args.seed)
    device = torch.device(args.device)

    # 2.Create actor, critic, EnvSampler() and TRPO.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, 
                hidden_sizes=args.hidden_sizes, init_std=args.init_std).to(device)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes).to(device)
    env_sampler = EnvSampler(env, args.max_episode_step, args.reward_step)
    trpo = TRPO(actor, 
                critic,
                args.value_lr,
                args.value_steps_per_update,
                args.cg_steps,
                args.linesearch_steps,
                args.gamma,
                args.tau,
                args.damping,
                args.max_kl,
                device)
        
    def get_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor.select_action(state)
        return action.cpu().numpy()[0]

    def get_value(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            value = critic(state)
        return value.cpu().numpy()[0, 0]

    total_step = 0
    for episode in range(1, args.episodes+1):
        episode_reward, samples = env_sampler(get_action, args.batch_size, get_value)
        actor_loss, value_loss = trpo.update(*samples)
        total_step += args.batch_size
        yield total_step, episode_reward, actor_loss, value_loss

# The properties of args:
# 1. env_name (default = 'HalfCheetah-v2')
# 2. device (default = 'cuda:0')
# 3. seed (default = 1)
# 4. hidden_sizes (default = (64, 64))
# 5. max_episode_step (default = 1000)
# 6. batch_size (default = 1000)
# 7. episodes (default = 1000)
# 8. value_lr (default = 1e-3)
# 9. value_steps_per_update (default=80)
# 10. cg_steps (default = 20)
# 11. linesearch_steps (default = 20)
# 12. gamma (default = 0.99)
# 13. tau (default = 0.97)
# 14. damping (default = 0.1)
# 15. max_kl (default = 0.01)
# 16. init_std (default = 1.0)
# 17. reward_step (default = 0)

Args = namedtuple('Args', 
                    ('env_name',
                    'device',
                    'seed',
                    'hidden_sizes',
                    'max_episode_step',
                    'batch_size',
                    'episodes',
                    'value_lr',
                    'value_steps_per_update',
                    'cg_steps',
                    'linesearch_steps',
                    'gamma',
                    'tau',
                    'damping',
                    'max_kl',
                    'init_std',
                    'reward_step'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--batch', type=int, default=1000, metavar='N',
                        help='number of batch size (default: 1000)')
    parser.add_argument('--episodes', type=int, default=1000, metavar='N',
                        help='number of eperiment episodes(default: 1000)')
    parser.add_argument('--reward_step', type=int, default=0, metavar='N',
                        help='the unit of reward step (default: 0)')
    parser.add_argument('--env_name', default='HalfCheetah-v2', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default cpu)')
    
    args = parser.parse_args()
    alg_args = Args(args.env_name,  # env_name
                args.device,        # device
                args.seed,          # seed
                (64, 64),           # hidden_sizes
                1000,               # max_episode_step
                args.batch,         # batch_size
                args.episodes,      # episodes
                1e-3,               # value_lr
                50,                 # value_steps_per_update
                20,                 # cg_steps
                20,                 # linesearch_steps
                0.99,               # gamma
                0.97,               # tau
                0.1,                # damping
                0.02,               # max_kl
                1.0,                # init_std
                args.reward_step)   # reward_step
                
    logdir = "./logs/alg_trpo/env_{}_reward_step_{}".format(alg_args.env_name, alg_args.reward_step)
    file_name = 'alg_trpo_env_{}_reward_step_{}_batch{}_seed{}_time{}.csv'.format(alg_args.env_name, alg_args.reward_step, alg_args.batch_size, alg_args.seed, time())
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    full_name = os.path.join(logdir, file_name)

    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'reward'])
    start_time = time()
    for step, reward, actor_loss, value_loss in main(alg_args):
        writer.writerow([step, reward])
        print("Step {}: Reward = {}, Actor_loss = {}, Value_loss = {}".format(step, reward, actor_loss, value_loss))
    print("Total time: {}s.".format(time() - start_time))
