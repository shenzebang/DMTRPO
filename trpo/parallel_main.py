import os
import gym
import torch
from torch.multiprocessing import Process
import torch.distributed as dist
from time import time
import csv
from collections import namedtuple
import numpy as np

from utils import EnvSampler
from models import PolicyNetwork, ValueNetwork
from local_trpo import LocalTRPO
from local_trpo2 import LocalTRPO2
from local_trpo3 import LocalTRPO3
from hmtrpo import HMTRPO
from global_trpo import GlobalTRPO
from navigation import Navigation2DEnv_FL

def run(rank, size, args):
    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.env_name == 'Navigation2DEnv-FL':
        env = Navigation2DEnv_FL()
    else:
        env = gym.make(args.env_name)
    env.seed(args.seed)

    device = args.device
    if device == 'cuda':
        device = 'cuda:{}'.format(rank % torch.cuda.device_count())
    device = torch.device(device)

    # 2.Create actor, critic, EnvSampler() and PPO.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, 
                hidden_sizes=args.hidden_sizes, init_std=args.init_std).to(device)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes).to(device)
    env_sampler = EnvSampler(env, args.max_episode_step, args.reward_step)
    trpo_args = {
        'actor': actor, 
        'critic': critic,
        'value_lr': args.value_lr,
        'value_steps_per_update': args.value_steps_per_update,
        'cg_steps': args.cg_steps,
        'linesearch_steps': args.linesearch_steps,
        'gamma': args.gamma,
        'tau': args.tau,
        'damping': args.damping,
        'max_kl': args.max_kl,
        'device': device
    }
    if args.alg_name == 'local_trpo':
        alg = LocalTRPO(**trpo_args)
    elif args.alg_name == 'local_trpo2':
        alg = LocalTRPO2(**trpo_args)
    elif args.alg_name == 'local_trpo3':
        alg = LocalTRPO3(**trpo_args)
    elif args.alg_name == 'hmtrpo':
        alg = HMTRPO(**trpo_args)
    elif args.alg_name  == 'global_trpo':
        alg = GlobalTRPO(**trpo_args)

    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor.select_action(state)
        return action.detach().cpu().numpy()[0]

    def get_value(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            value = critic(state)
        return value.cpu().numpy()[0, 0]

    # 3.Start training.
    total_step = 0
    for episode in range(1, args.episodes+1):
        episode_reward, samples = env_sampler(get_action, args.batch_size, get_value)
        actor_loss, value_loss = alg.update(*samples)
        total_step += args.batch_size
        model = None
        if rank == 0 and episode == args.episodes:
            model = actor
        yield total_step, episode_reward, actor_loss, value_loss, model

# The properties of args:
# 0. alg_name (default = 'hmtrpo')
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
                    ('alg_name',
                    'env_name',
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
                    'reward_step',
                    'master_addr',
                    'master_port'))

def parallel_run(start_time, rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group(backend, rank=rank, world_size=size)

    logdir = "./logs/alg_{}/env_{}/workers{}".format(args.alg_name, args.env_name, size)
    file_name = 'alg_{}_env_{}_reward_step_{}_worker{}_seed{}_time{}.csv'.format(args.alg_name, args.env_name, args.reward_step, rank, args.seed, start_time)
    full_name = os.path.join(logdir, file_name)

    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'reward'])

    model_dir = "./models/alg_{}/env_{}/workers{}".format(args.alg_name, args.env_name, size)
    model_file_name = 'alg_{}_env_{}_reward_step_{}_actor_seed{}_time{}.pth.tar'.format(args.alg_name, 
                        args.env_name, args.reward_step, args.seed, start_time)
    model_full_name = os.path.join(model_dir, model_file_name)

    for step, reward, actor_loss, value_loss, model in fn(rank, size, args):
        writer.writerow([step, reward])
        print('Rank {}, Step {}: Reward = {}, actor_loss = {}, value_loss = {}'.format(rank, step, reward, actor_loss, value_loss))
        if model:
            torch.save(model.state_dict(), model_full_name)

    csvfile.close()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--alg', default="local_trpo", metavar='G',
                        help='name of the algorithm to run (default: local_trpo)')
    parser.add_argument('--env_name', default="HalfCheetah-v2", metavar='G',
                        help='name of environment to run (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default: cpu)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--workers', type=int, default=8, metavar='N',
                        help='number of workers(default: 8)')
    parser.add_argument('--batch', type=int, default=1000, metavar='N',
                        help='number of batch size (default: 1000)')
    parser.add_argument('--episodes', type=int, default=1000, metavar='N',
                        help='number of experiment episodes(default: 1000)')
    parser.add_argument('--reward_step', type=int, nargs='+', default=(0,), metavar='N',
                        help='the unit of reward step (default: 0)')
    parser.add_argument('--master_addr', default='127.0.0.1', metavar='G',
                        help="master node's ip address")
    parser.add_argument('--master_port', type=str, default='29500', metavar='N',
                        help='master port')
    parser.add_argument('--node_size', type=int, default=1, metavar='N',
                        help='number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, metavar='N',
                        help='rank of this node')
    args = parser.parse_args()

    logdir = "./logs/alg_{}/env_{}/workers{}".format(args.alg, args.env_name, args.workers)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model_dir = "./models/alg_{}/env_{}/workers{}".format(args.alg, args.env_name, args.workers)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    size = args.workers // args.node_size * args.node_size
    processes = []
    start_time = time()
    backend = 'gloo' if args.device == 'cpu' else 'nccl'

    start_rank = args.node_rank * size // args.node_size
    end_rank = (args.node_rank + 1) * size // args.node_size

    num_reward_step = len(args.reward_step)

    for rank in range(start_rank, end_rank):
        reward_step = args.reward_step[rank % num_reward_step]
        seed = args.seed + rank
        alg_args = Args(args.alg,       # alg_name
                    args.env_name,      # env_name
                    args.device,        # device
                    seed,               # seed
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
                    reward_step,        # reward_step
                    args.master_addr,   # master_addr
                    args.master_port)   # master_port
        p = Process(target=parallel_run, args=(start_time, rank, size, run, alg_args, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() # wait all process stop.

    end_time = time()
    print("Total time: {}".format(end_time - start_time))