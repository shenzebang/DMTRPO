from time import time
import gym
import envs
from tensorboardX import SummaryWriter

from core.models import *

from torch.autograd import Variable
from core.agent import AgentCollection
from utils.utils import *
import numpy as np
from torch.distributions.kl import kl_divergence
from core.natural_gradient_ray import conjugate_gradient_global
from torch.nn.utils.convert_parameters import vector_to_parameters
import ray

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


class TRPOServer:
    def __init__(self, args, dtype=torch.double):
        ray.init(num_cpus=args.num_workers, num_gpus=1)
        torch.set_default_dtype(dtype)
        dummy_env = gym.make(args.env_name)
        num_inputs = dummy_env.observation_space.shape[0]
        num_actions = dummy_env.action_space.shape[0]
        torch.manual_seed(args.seed)

        algorithm_name = "TRPO"
        logdir = "./logs/algo_{}/env_{}/batchsize_{}_nworkers_{}_seed_{}_time{}".format(
            algorithm_name, str(args.env_name), args.batch_size, args.agent_count, args.seed, time()
        )
        self.args = args
        self.envs = [gym.make(self.args.env_name) for _ in range(self.args.agent_count)]
        self.writer = SummaryWriter(logdir)
        self.actor = Policy(num_inputs, num_actions, hidden_sizes = (args.hidden_size,) * args.num_layers, init_std=1)
        self.critics = [Value(num_inputs) for _ in range(args.agent_count)]
        self.agents = AgentCollection(
            envs=self.envs,
            actor=self.actor,
            critics=self.critics,
            min_batch_size=args.batch_size,
            num_agents=args.agent_count,
            device=args.device,
            gamma=args.gamma,
            tau=args.tau,
            dtype=dtype,
            use_running_state=args.use_running_state
        )

    def step(self, i_episode):
        losses = []
        def trpo_loss(advantages, states, actions, params, params_trpo_ls):
            # This is the negative trpo objective
            with torch.no_grad():
                set_flat_params_to(self.actor, params)
                log_prob_prev = self.actor.get_log_prob(states, actions)
                set_flat_params_to(self.actor, params_trpo_ls)
                log_prob_current = self.actor.get_log_prob(states, actions)
                negative_trpo_objs = -advantages * torch.exp(log_prob_current - log_prob_prev)
                negative_trpo_obj = negative_trpo_objs.mean()
                set_flat_params_to(self.actor, params)
            return negative_trpo_obj

        def compute_kl(states, prev_params, xnew):
            with torch.autograd.no_grad():
                set_flat_params_to(self.actor, prev_params)
                pi = self.actor(Variable(states))
                set_flat_params_to(self.actor, xnew)
                pi_new = self.actor(Variable(states))
                set_flat_params_to(self.actor, prev_params)
                kl = torch.mean(kl_divergence(pi, pi_new))
            return kl

        print('Episode {}. map_pg...'.format(i_episode))
        time_begin = time()
        states_list, advantages_list, returns_list, actions_list, log_list, actor_gradient_list, critic_update_list \
            = self.agents.map_pg()
        self.log_list = log_list
        time_sample = time() - time_begin
        print('Episode {}. map_pg is done, using time {}.'.format(i_episode, time_sample))

        actor_gradient_list = np.array(actor_gradient_list).mean(axis=0)
        self.actor_gradient_average = torch.from_numpy(actor_gradient_list)
        for params, critic in zip(critic_update_list, self.critics):
            vector_to_parameters(torch.from_numpy(params), critic.parameters())


        print('Episode {}. map_cg...'.format(i_episode))
        time_begin = time()
        natural_gradient_direction = self.conjugate_gradient(states_list)
        time_ng = time() - time_begin
        print('Episode {}. map_cg is done, using time {}'
              .format(i_episode, time_ng))

        # Linear Search
        print('Episode {}. Linear search...'.format(i_episode))
        time_begin = time()
        prev_params = get_flat_params_from(self.actor)
        for advantages, states, actions in zip(advantages_list, states_list, actions_list):
            losses.append(trpo_loss(advantages, states, actions, prev_params, prev_params).detach().numpy())
        fval = np.array(losses).mean()

        ls_flag = False
        for (n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(10)):
            new_losses = []
            kls = []
            xnew = prev_params + stepfrac * natural_gradient_direction
            for advantages, states, actions in zip(advantages_list, states_list, actions_list):
                new_losses.append(trpo_loss(advantages, states, actions, prev_params, xnew).data)
                kls.append(compute_kl(states, prev_params, xnew).detach().numpy())
            new_loss = np.array(new_losses).mean()
            kl = np.array(kls).mean()
            if new_loss - fval < 0 and kl < args.max_kl:
                set_flat_params_to(self.actor, xnew)
                self.writer.add_scalar("n_backtracks", n_backtracks, i_episode)
                ls_flag = True
                break
        time_ls = time() - time_begin
        if ls_flag:
            print('Episode {}. Linear search is done in {} steps, using time {}'
                  .format(i_episode, n_backtracks, time_ls))
        else:
            print('Episode {}. Linear search is done but failed, using time {}'
                  .format(i_episode, time_ls))

        self.log(i_episode)

    def conjugate_gradient(self, states_list):
        conjugate_gradient_direction = conjugate_gradient_global(
            policy_net=self.actor,
            states_list=states_list,
            pg=self.actor_gradient_average,
            max_kl=self.args.max_kl,
            cg_damping=self.args.cg_damping,
            cg_iter=self.args.cg_iter,
            device=self.args.device
        )
        return conjugate_gradient_direction

    def log(self, i_episode):
        rewards = [log['avg_reward'] for log in self.log_list]
        average_reward = np.array(rewards).mean()

        if i_episode % self.args.log_interval == 0:
            print('Episode {}. Average reward {:.2f}'.format(
                i_episode, average_reward))
            self.writer.add_scalar("Avg_return", average_reward, i_episode * self.args.agent_count * self.args.batch_size)
        return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TRPO with non-iid Environment')

    # MDP
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--agent-count', type=int, default=10, metavar='N',
                        help='number of agents (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.995)')
    parser.add_argument('--env-name', default="2DNavigation-v1", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.97)')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    # Optimization
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

    # Miscellaneous
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='number of workers for parallel computing')

    args = parser.parse_args()

    args.device = torch.device(args.device
                        if torch.cuda.is_available() else 'cpu')

    trpo_server = TRPOServer(args=args)
    # TODO: write a runner class to conduct trials under different parameter settings
    for i_episode in range(args.max_episode):
        trpo_server.step(i_episode=i_episode)
