import torch
from trpo_server import TRPOServer
from hmtrpo_server import HMTRPOServer
from localtrpo_server import LocalTRPOServer
import pandas as pd
from time import time
import numpy as np
import ray
import os
from utils.plot import plot

class Runner(object):
    def __init__(self, args):
        self.plot = args.plot
        self.num_repeat = args.num_repeat
        self.args = args
        assert args.algo in ['trpo', 'localtrpo', 'hmtrpo']
        if args.algo == 'trpo':
            self.server = TRPOServer(args=args)
        if args.algo == 'hmtrpo':
            self.server = HMTRPOServer(args=args)
        if args.algo == 'localtrpo':
            self.server = LocalTRPOServer(args=args)


    def train(self):
        steps = []
        avg_rewards = []
        for i_episode in range(self.args.max_episode):
            num_steps, reward = self.server.step(i_episode=i_episode)
            steps.append(num_steps)
            avg_rewards.append(reward)
        return steps, avg_rewards


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')

    # MDP
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--agent-count', type=int, default=10, metavar='N',
                        help='number of agents (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--env-name', default="2DNavigation-v1", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer (default 64)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--init-std', type=float, default=1.0,
                        help='initial std of Gaussian policy')

    # Optimization
    parser.add_argument('--algo', default="trpo", metavar='G',
                        help='name of the algorithm to run')
    parser.add_argument('--max-kl', type=float, default=1e-1, metavar='G',
                        help='max kl value (default: 1e-1)')
    parser.add_argument('--cg-damping', type=float, default=1e-2, metavar='G',
                        help='damping for conjugate gradient (default: 1e-2)')
    parser.add_argument('--cg-iter', type=int, default=10, metavar='G',
                        help='maximum iteration of conjugate gradient (default: 1e-1)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization parameter for critics (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='per-iteration batch size for each agent (default: 1000)')
    parser.add_argument('--use-running-state', action='store_true',
                        help='use running state to normalize states')
    parser.add_argument('--max-episode', type=int, default=1000, metavar='G',
                        help='maximum number of episodes')
    parser.add_argument('--step-size', type=float, default=1, metavar='G',
                        help='step size (default: 1)')

    # Miscellaneous
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='number of workers for parallel computing')
    parser.add_argument('--plot', action='store_true',
                        help='plot the experiment result')
    parser.add_argument('--num-repeat', type=int, default=1,
                        help='number of repeated experiments with different random seeds')

    args = parser.parse_args()

    args.device = torch.device(args.device
                        if torch.cuda.is_available() else 'cpu')
    results = pd.DataFrame()
    ray.init(num_cpus=args.num_workers, num_gpus=1)
    for num_trial in range(args.num_repeat):
        if num_trial != 0:
            args.seed = np.random.randint(10000)
        runner = Runner(args)
        steps, avg_rewards = runner.train()
        if num_trial == 0:
            results['steps'] = steps
        results['avg_rewards{}'.format(num_trial)] = avg_rewards

    # todo unite logdir with server
    logdir = "./logs/algo_{}/env_{}".format(args.algo, args.env_name)
    file_name = 'batchsize_{}_nworkers_{}_seed_{}_repeat_{}_time{}.csv'.format(
        args.batch_size, args.agent_count, args.seed, args.num_repeat, time())
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    full_name = os.path.join(logdir, file_name)
    results.to_csv(full_name)

    if args.plot:
        plot_file = 'batchsize_{}_nworkers_{}_seed_{}_repeat_{}_time{}.pdf'.format(
            args.batch_size, args.agent_count, args.seed, args.num_repeat, time())
        plot_name = os.path.join(logdir, plot_file)
        plot(results, plot_name, args.num_repeat)



