from time import time
import gym
import envs
from tensorboardX import SummaryWriter
from core.models import *
from core.agent import AgentCollection
from utils.utils import *
import numpy as np
from core.natural_gradient_ray import conjugate_gradient_global

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


class TRPOServer:
    def __init__(self, args, dtype=torch.double):
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
        self.actor = Policy(num_inputs, num_actions, hidden_sizes = (args.hidden_size,) * args.num_layers, init_std=args.init_std)
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
        self.log_list = []
        self.step_size = args.step_size

    def step(self, i_episode):
        _ = torch.randn(1)

        print('Episode {}. map_pg...'.format(i_episode))
        time_begin = time()
        states_list, log_list, actor_gradient_list = self.agents.map_pg()
        self.log_list = log_list
        time_sample = time() - time_begin
        print('Episode {}. map_pg is done, using time {}.'.format(i_episode, time_sample))

        print('Episode {}. map_cg...'.format(i_episode))
        time_begin = time()
        natural_gradient_direction = self.conjugate_gradient(actor_gradient_list, states_list)
        time_ng = time() - time_begin
        print('Episode {}. map_cg is done, using time {}'
              .format(i_episode, time_ng))

        print('Episode {}. Linear search...'.format(i_episode))
        time_begin = time()
        ls_flag, n_backtracks, xnew = self.line_search(i_episode, natural_gradient_direction)
        set_flat_params_to(self.actor, xnew)
        time_ls = time() - time_begin
        if ls_flag:
            print('Episode {}. Linear search succeeded in {} steps, using time {}'
                  .format(i_episode, n_backtracks, time_ls))
        else:
            print('Episode {}. Linear search failed, using time {}'
                  .format(i_episode, time_ls))

        return self.log(i_episode)

    def line_search(self, i_episode, natural_gradient_direction):
        prev_params = get_flat_params_from(self.actor)
        xnew = prev_params
        n_backtracks = 0
        fval = self.agents.trpo_loss()
        ls_flag = False
        for (n_backtracks, stepfrac) in enumerate(0.9 ** np.arange(-5, 20)):
            xnew = prev_params + self.step_size * stepfrac * natural_gradient_direction
            new_loss = self.agents.trpo_loss(xnew)
            kl = self.agents.compute_kl(xnew)
            if new_loss - fval < 0 and kl < self.args.max_kl:
                self.writer.add_scalar("n_backtracks", n_backtracks, i_episode)
                ls_flag = True
                break
        return ls_flag, n_backtracks, xnew

    def conjugate_gradient(self, actor_gradient_list, states_list):
        conjugate_gradient_direction = conjugate_gradient_global(
            policy_net=self.actor,
            states_list=states_list,
            pg_list=actor_gradient_list,
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
            print('Episode {}. Average reward {}'.format(
                i_episode, average_reward))
            self.writer.add_scalar("Avg_return", average_reward, i_episode * self.args.agent_count * self.args.batch_size)

        num_steps = i_episode * self.args.batch_size * self.args.agent_count

        return num_steps, average_reward
