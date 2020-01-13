import os
import gym
import torch
from torch.multiprocessing import Process
import torch.distributed as dist
from time import time
import csv
from collections import namedtuple

from utils import EnvSampler
from models import PolicyNetwork, ValueNetwork
from local_ppo import LocalPPO
from global_ppo import GlobalPPO

# The properties of args:
# 1. env_name (default = 'HalfCheetah-v2')
# 2. device (default = "cuda:0")
# 3. seed (default = 1)
# 4. hidden_sizes (default = (64, 32))
# 5. episodes (default = 100. Not the number of trajectories, but the number of batches.)
# 6. max_episode_step (default = 1000)
# 7. batch_size (default = 4000)
# 8. gamma (default = 0.99)
# 9. tau (default = 0.97)
# 10. clip (default = 0.2)
# 11. max_kl (default =  0.01)
# 12. pi_steps_per_update (default = 80) 
# 13. value_steps_per_update (default = 80)
# 14. pi_lr (default = 3e-4)
# 15. value_lr (default = 1e-3)
def run(rank, size, args):
    env = gym.make(args.env_name)
    device = args.device
    if device == 'cuda':
        device = 'cuda:{}'.format(rank % torch.cuda.device_count())
    device = torch.device(device)

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.seed(args.seed)

    # 2.Create actor, critic, EnvSampler() and PPO.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, hidden_sizes=args.hidden_sizes).to(device)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes).to(device)
    env_sampler = EnvSampler(env, args.max_episode_step)
    ppo_args = {
        'actor': actor,
        'critic': critic,
        'clip': args.clip,
        'gamma': args.gamma,
        'tau': args.tau,
        'target_kl': args.target_kl,
        'device': device,
        'pi_steps_per_update': args.pi_steps_per_update,
        'value_steps_per_update': args.value_steps_per_update,
        'pi_lr': args.pi_lr,
        'v_lr': args.value_lr
    }
    if args.alg_name == 'local_ppo':
        alg = LocalPPO(**ppo_args)
    elif args.alg_name  == 'global_ppo':
        alg = GlobalPPO(**ppo_args)

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
        yield total_step, episode_reward, actor_loss, value_loss

Args = namedtuple('Args',
                ('alg_name',
                'env_name', 
                'device', 
                'seed', 
                'hidden_sizes', 
                'episodes', 
                'max_episode_step', 
                'batch_size', 
                'gamma', 
                'tau', 
                'clip', 
                'target_kl',
                'pi_steps_per_update',
                'value_steps_per_update',
                'pi_lr',
                'value_lr'))

def parallel_run(start_time, rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    logdir = "./logs/algo_{}/env_{}/workers{}".format(args.alg_name, args.env_name, size)
    file_name = 'worker{}_seed{}_time{}.csv'.format(rank, args.seed, start_time)
    full_name = os.path.join(logdir, file_name)

    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'reward'])

    for step, reward, actor_loss, value_loss in fn(rank, size, args):
        writer.writerow([step, reward])
        print('Rank {}, Step {}: Reward = {}, actor_loss = {}, value_loss = {}'.format(rank, step, reward, actor_loss, value_loss))

    csvfile.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--agent', type=int, default=8, metavar='N',
                        help='number of agents (default: 8)')
    parser.add_argument('--alg', default="local_ppo", metavar='G',
                        help='name of the algorithm to run (default: local_ppo)')
    parser.add_argument('--env_name', default="HalfCheetah-v2", metavar='G',
                        help='name of environment to run (default: HalfCheetah-v2)')
    parser.add_argument('--batch', type=int, default=1000, metavar='N',
                        help='number of batch size (default: 1000)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default: cpu)')
    args = parser.parse_args()

    logdir = "./logs/algo_{}/env_{}/workers{}".format(args.alg, args.env_name, args.agent)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    size = args.agent
    processes = []
    start_time = time()
    backend = 'gloo' if args.device == 'cpu' else 'nccl'
    for rank in range(size):
        alg_args = Args(args.alg,       # alg_name
                    args.env_name,      # env_name
                    args.device,        # device
                    args.seed+rank,     # seed
                    (64, 64),           # hidden_sizes
                    1000,               # episodes
                    1000,               # max_episode_step
                    args.batch,         # batch_size
                    0.99,               # gamma
                    0.97,               # tau
                    0.2,                # clip
                    0.015,              # target_kl
                    80,                 # pi_steps_per_update
                    50,                 # value_steps_per_update
                    3e-4,               # pi_lr
                    1e-3)               # value_lr
        p = Process(target=parallel_run, args=(start_time, rank, size, run, alg_args, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() # wait all process stop.

    end_time = time()
    print("Total time: {}".format(end_time - start_time))