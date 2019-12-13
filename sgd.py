from itertools import count
from time import time
import gym
import scipy.optimize
from tensorboardX import SummaryWriter

from core.models import *

from torch.autograd import Variable
from torch import Tensor
import torch.tensor as tensor
# from core.agent import AgentCollection
from core.agent_ray import AgentCollection
from utils.utils import *
from core.running_state import ZFilter
# from core.common import estimate_advantages_parallel
from core.common_ray import estimate_advantages_parallel
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import numpy as np
from torch.distributions.kl import kl_divergence
# from core.natural_gradient import conjugate_gradient_gloabl
from core.natural_gradient_ray import conjugate_gradient_global
from core.policy_gradient import compute_policy_gradient_parallel
from core.log_determinant import compute_log_determinant
# from envs.mujoco.half_cheetah import HalfCheetahVelEnv_FL
import ray
import os
import envs

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

def main(args):
    ray.init(num_cpus=args.num_workers, num_gpus=1)
    dtype = torch.double
    torch.set_default_dtype(dtype)
    env = gym.make(args.env_name)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    policy_net = Policy(num_inputs, num_actions, hidden_sizes = (args.hidden_size,) * args.num_layers)
    print("Network structure:")
    for name, param in policy_net.named_parameters():
        print("name: {}, size: {}".format(name, param.size()[0]))
    flat_param = parameters_to_vector(policy_net.parameters())
    matrix_dim = flat_param.size()[0]
    print("number of total parameters: {}".format(matrix_dim))
    value_net = Value(num_inputs)
    batch_size = args.batch_size
    running_state = ZFilter((env.observation_space.shape[0],), clip=5)
    lr = args.lr

    algo = "sgd"
    logdir = "./algo_{}/env_{}/batchsize_{}_nworkers_{}_seed_{}_time{}".format(algo, str(args.env_name), batch_size, args.agent_count, args.seed, time())
    writer = SummaryWriter(logdir)

    agents = AgentCollection(env, policy_net, 'cpu', running_state=running_state, render=args.render,
                             num_agents=args.agent_count, num_parallel_workers=args.num_workers)

    for i_episode in count(1):
        losses = []
        # Sample Trajectories
        print('Episode {}. Sampling trajectories...'.format(i_episode))
        time_begin = time()
        memories, logs = agents.collect_samples(batch_size)
        time_sample = time() - time_begin
        print('Episode {}. Sampling trajectories is done, using time {}.'.format(i_episode, time_sample))

        # Process Trajectories
        print('Episode {}. Processing trajectories...'.format(i_episode))
        time_begin = time()
        advantages_list, returns_list, states_list, actions_list = \
            estimate_advantages_parallel(memories, value_net, args.gamma, args.tau)
        time_process = time() - time_begin
        print('Episode {}. Processing trajectories is done, using time {}'.format(i_episode, time_process))

        # Computing Policy Gradient
        print('Episode {}. Computing policy gradients...'.format(i_episode))
        time_begin = time()
        policy_gradients, value_net_update_params = compute_policy_gradient_parallel(policy_net, value_net, states_list, actions_list, returns_list, advantages_list)
        pg = np.array(policy_gradients).mean(axis=0)
        pg = torch.from_numpy(pg)
        value_net_average_params = np.array(value_net_update_params).mean(axis=0)
        value_net_average_params = torch.from_numpy(value_net_average_params)
        vector_to_parameters(value_net_average_params, value_net.parameters())
        time_pg = time() - time_begin
        print('Episode {}. Computing policy gradients is done, using time {}.'.format(i_episode, time_pg))
        prev_params = get_flat_params_from(policy_net)
        xnew = prev_params - lr * pg
        set_flat_params_to(policy_net, xnew)

        rewards = [log['avg_reward'] for log in logs]
        average_reward = np.array(rewards).mean()

        if i_episode % args.log_interval == 0:
            print('Episode {}. Average reward {:.2f}'.format(
                i_episode, average_reward))
            writer.add_scalar("Avg_return", average_reward, i_episode*args.agent_count*batch_size)
        if i_episode > args.max_episode:
            break

if __name__ == '__main__':
    import argparse
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='SGD with iid Environment')

    # MDP
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--agent-count', type=int, default=1, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="2DNavigation-v1", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                        help='gae (default: 0.97)')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    # Optimization
    parser.add_argument('--max-episode', type=int, default=500, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=1e4, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')


    # Miscellaneous
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for parallel computing')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus for parallel computing log determinants')

    args = parser.parse_args()

    args.device = torch.device(args.device
                        if torch.cuda.is_available() else 'cpu')

    args.gpus = args.gpus if args.device == 'cuda' and torch.cuda.is_available() else 0

    main(args)
