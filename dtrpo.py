import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from worker import Worker

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
"""
gamma: discount factor
env_name: 
tau: gae lamda
l2-reg: l2 regularization constant
max_kl: max kl_distance
damping: CG damping
seed:
batchsize:
render:
log-interval:
"""
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
env.seed(args.seed)
torch.manual_seed(args.seed)
policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
batch_size = args.batch_size

num_workers = 1
workers = []
for _ in range(num_workers):
    workers.append(Worker(env, policy_net, value_net))

for i_episode in count(1):
    num_steps = 0
    num_episodes = 0
    policy_gradients = [] # list of PGs
    stepdirs = [] # list of F^-1 * PG
    rewards = []

    for worker in workers:
        worker.sample(batch_size)
        policy_gradients.append(worker.compute_PG().numpy())
    pg = np.array(policy_gradients).mean(axis=0)
    pg = torch.from_numpy(pg)

    for worker in workers:
        stepdirs.append(worker.conjugate_gradient(pg).numpy())
    fullstep = np.array(stepdirs).mean(axis=0)
    fullstep = torch.from_numpy(fullstep)

    #linesearch
    prev_params = get_flat_params_from(policy_net)
    #print(policy_net is workers[0].policy_net)
    for (n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(10)):
        feedbacks = [] #list of acceptance or refusal
        xnew = prev_params + stepfrac * fullstep
        #set_flat_params_to(policy_net, xnew)
        for worker in workers:
            feedbacks.append(worker.vote(prev_params, xnew, stepfrac))
        #if not False in feedbacks:
        if not False in feedbacks:
            print("Step accepted!")
            set_flat_params_to(policy_net, xnew)
            break
        else:
            print("Backtrack:", n_backtracks + 1, "refused")

    for worker in workers:
        rewards.append(worker.get_reward())
    average_reward = np.array(rewards).mean()

    if i_episode % args.log_interval == 0:
        print('Episode {}\tAverage reward {:.2f}'.format(
            i_episode, average_reward))