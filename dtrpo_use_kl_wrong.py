import argparse
from itertools import count

import gym
import scipy.optimize
from tensorboardX import SummaryWriter

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from worker import Worker
from running_state import ZFilter
from replay_memory import Memory

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
parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
for args.batch_size, num_workers in [(2500,10), (5000,10), (500,20), (500,50), (2500, 20)]:
    for args.env_name in ["Reacher-v2", "Hopper-v2", "Ant-v2", "HalfCheetah-v2"]:
        for args.seed in [1, 11, 21]:
            gamma = args.gamma
            tau = args.tau
            damping = args.damping
            max_kl = args.max_kl
            env = gym.make(args.env_name)
            num_inputs = env.observation_space.shape[0]
            num_actions = env.action_space.shape[0]
            env.seed(args.seed)
            torch.manual_seed(args.seed)
            policy_net = Policy(num_inputs, num_actions)
            value_net = Value(num_inputs)
            batch_size = args.batch_size
            running_state = ZFilter((env.observation_space.shape[0],), clip=5)

            #num_workers = 10
            logdir = "./DTRPO_wrong/%s/batchsize_%d_nworkers_%d_%d"%(str(args.env_name), batch_size, num_workers, args.seed)
            writer = SummaryWriter(logdir)


            def select_action(state):
                state = torch.from_numpy(state).unsqueeze(0)
                action_mean, _, action_std = policy_net(Variable(state))
                action = torch.normal(action_mean, action_std)
                return action

            def sample(batch_size):
                """sample transitions and store them in memory"""
                memory = Memory()
                num_steps = 0
                num_episodes = 0
                reward_batch = 0
                while num_steps < batch_size:
                    state = env.reset()
                    state = running_state(state)

                    reward_sum = 0
                    for t in range(10000):  # Don't infinite loop while learning
                        action = select_action(state)
                        action = action.data[0].numpy()
                        next_state, reward, done, _ = env.step(action)
                        reward_sum += reward

                        next_state = running_state(next_state)

                        mask = 1
                        if done:
                            mask = 0

                        memory.push(state, np.array([action]), mask, next_state, reward)

                        if done:
                            break

                        state = next_state
                    num_steps += (t - 1)
                    num_episodes += 1
                    reward_batch += reward_sum
                average_reward = reward_batch / num_episodes
                return memory, average_reward

            def compute_PG(memory):
                """compute policy gradient and update value net by using samples in memory"""
                batch = memory.sample()
                rewards = torch.Tensor(batch.reward)
                masks = torch.Tensor(batch.mask)
                actions = torch.Tensor(np.concatenate(batch.action, 0))  # why concatenate zero?
                states = torch.Tensor(batch.state)
                values = value_net(Variable(states))

                returns = torch.Tensor(actions.size(0), 1)
                deltas = torch.Tensor(actions.size(0), 1)
                advantages = torch.Tensor(actions.size(0), 1)

                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                for i in reversed(range(rewards.size(0))):
                    returns[i] = rewards[i] + gamma * prev_return * masks[i]
                    deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
                    advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

                    prev_return = returns[i, 0]
                    prev_value = values.data[i, 0]
                    prev_advantage = advantages[i, 0]

                targets = Variable(returns)

                # Original code uses the same LBFGS to optimize the value loss
                def get_value_loss(flat_params):
                    set_flat_params_to(value_net, torch.Tensor(flat_params))
                    for param in value_net.parameters():
                        if param.grad is not None:
                            param.grad.data.fill_(0)

                    values_ = value_net(Variable(states))

                    value_loss = (values_ - targets).pow(2).mean()

                    # weight decay
                    for param in value_net.parameters():
                        value_loss += param.pow(2).sum() * 1e-3
                    value_loss.backward()
                    return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

                flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                                        get_flat_params_from(value_net).double().numpy(),
                                                                        maxiter=25)
                set_flat_params_to(value_net, torch.Tensor(flat_params))

                advantages = (advantages - advantages.mean()) / advantages.std()

                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

                log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                loss = -(Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))).mean()
                grads = torch.autograd.grad(loss, policy_net.parameters())
                loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
                return loss_grad

            def conjugate_gradient(memory, pg):
                batch = memory.sample()
                states = torch.Tensor(batch.state)
                def get_kl():
                    mean1, log_std1, std1 = policy_net(Variable(states))
                    mean0 = Variable(mean1.data)
                    log_std0 = Variable(log_std1.data)
                    std0 = Variable(std1.data)
                    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                    return kl.sum(1, keepdim=True)

                def Fvp(v):
                    kl = get_kl()
                    kl = kl.mean()

                    grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
                    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

                    kl_v = (flat_grad_kl * Variable(v)).sum()
                    grads = torch.autograd.grad(kl_v, policy_net.parameters())
                    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

                    return flat_grad_grad_kl + v * damping

                def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
                    x = torch.zeros(b.size())
                    r = b.clone()
                    p = b.clone()
                    rdotr = torch.dot(r, r)
                    for i in range(nsteps):
                        _Avp = Avp(p)
                        alpha = rdotr / torch.dot(p, _Avp)
                        x += alpha * p
                        r -= alpha * _Avp
                        new_rdotr = torch.dot(r, r)
                        betta = new_rdotr / rdotr
                        p = r + betta * p
                        rdotr = new_rdotr
                        if rdotr < residual_tol:
                            break
                    return x

                stepdir = conjugate_gradients(Fvp, -pg, 10)
                shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
                lm = torch.sqrt(shs / max_kl)
                fullstep = stepdir / lm[0]
                #print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
                return fullstep

            def compute_loss(memory, prev_params, params, volatile=False):
                batch = memory.sample()
                rewards = torch.Tensor(batch.reward)
                masks = torch.Tensor(batch.mask)
                actions = torch.Tensor(np.concatenate(batch.action, 0))  # why concatenate zero?
                states = torch.Tensor(batch.state)
                values = value_net(Variable(states))

                returns = torch.Tensor(actions.size(0), 1)
                deltas = torch.Tensor(actions.size(0), 1)
                advantages = torch.Tensor(actions.size(0), 1)

                prev_return = 0
                prev_value = 0
                prev_advantage = 0
                for i in reversed(range(rewards.size(0))):
                    returns[i] = rewards[i] + gamma * prev_return * masks[i]
                    deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
                    advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

                    prev_return = returns[i, 0]
                    prev_value = values.data[i, 0]
                    prev_advantage = advantages[i, 0]
                advantages = (advantages - advantages.mean()) / advantages.std()

                set_flat_params_to(policy_net, prev_params)
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = policy_net(Variable(states))
                else:
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
                fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
                set_flat_params_to(policy_net, params)
                if volatile:
                    with torch.no_grad():
                        action_means, action_log_stds, action_stds = policy_net(Variable(states))
                else:
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
                log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
                action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
                set_flat_params_to(policy_net, prev_params)
                return action_loss.mean()

            def compute_kl(memory, prev_params, xnew):
                batch = memory.sample()
                states = torch.Tensor(batch.state)
                set_flat_params_to(policy_net, prev_params)
                mean0, log_std0, std0 = policy_net(Variable(states))
                mean0 = Variable(mean0.data)
                log_std0 = Variable(log_std0.data)
                std0 = Variable(std0.data)
                set_flat_params_to(policy_net, xnew)
                mean1, log_std1, std1 = policy_net(Variable(states))
                kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                #print(kl.sum(1, keepdim=True).mean())
                set_flat_params_to(policy_net, prev_params)
                return kl.sum(1, keepdim=True).mean()

            for i_episode in count(1):
                num_steps = 0
                num_episodes = 0
                memories = []
                policy_gradients = [] # list of PGs
                stepdirs = [] # list of F^-1 * PG
                rewards = []
                losses = []

                for _ in range(num_workers):
                    batch, reward = sample(batch_size)
                    memories.append(batch)
                    rewards.append(reward)
                for memory in memories:
                    policy_gradients.append(compute_PG(memory))
                #pg = np.array(policy_gradients).mean(axis=0)
                #pg = torch.from_numpy(pg)

                for i in range(len(memories)):
                    stepdirs.append(conjugate_gradient(memories[i], policy_gradients[i]).numpy())
                fullstep = np.array(stepdirs).mean(axis=0)
                fullstep = torch.from_numpy(fullstep)

                # linesearch
                prev_params = get_flat_params_from(policy_net)
                for memory in memories:
                    losses.append(compute_loss(memory, prev_params, prev_params, True).detach().numpy())
                fval = np.array(losses).mean()

                for (n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(10)):
                    new_losses = []
                    kls = []
                    xnew = prev_params + stepfrac * fullstep
                    for memory in memories:
                        new_losses.append(compute_loss(memory, prev_params, xnew, True).data)
                        kls.append(compute_kl(memory, prev_params, xnew).detach().numpy())
                    new_loss = np.array(new_losses).mean()
                    kl = np.array(kls).mean()
                    print(new_loss - fval, kl)
                    if new_loss - fval < 0 and kl < 0.01:
                        print("Step accepted!")
                        set_flat_params_to(policy_net, xnew)
                        writer.add_scalar("n_backtracks", n_backtracks, i_episode)
                        break
                    else:
                        print("Backtrack:", n_backtracks + 1, "refused")

                average_reward = np.array(rewards).mean()

                if i_episode % args.log_interval == 0:
                    print('Episode {}\tAverage reward {:.2f}'.format(
                        i_episode, average_reward))
                    writer.add_scalar("Avg_return", average_reward, i_episode*num_workers*batch_size)
                if i_episode * num_workers * batch_size > 2e6:
                    break