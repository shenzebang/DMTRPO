import gym
import torch
from time import time
import os
import csv
from collections import namedtuple

from utils import EnvSampler
from models import PolicyNetwork, ValueNetwork
from ppo import PPO

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
def main(args):
    env = gym.make(args.env_name)
    device = torch.device(args.device)

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.seed(args.seed)

    # 2.Create actor, critic, EnvSampler() and PPO.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, hidden_sizes=args.hidden_sizes)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes)
    env_sampler = EnvSampler(env, args.max_episode_step)
    ppo = PPO(actor, 
              critic, 
              clip=args.clip, 
              gamma=args.gamma, 
              tau=args.tau, 
              target_kl=args.target_kl, 
              device=device,
              pi_steps_per_update=args.pi_steps_per_update,
              value_steps_per_update=args.value_steps_per_update,
              pi_lr=args.pi_lr,
              v_lr=args.value_lr)

    # 3.Start training.
    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor.select_action(state)
        return action.detach().cpu().numpy()[0]

    def get_value(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            value = critic(state)
        return value.cpu().numpy()[0, 0]

    total_step = 0
    for episode in range(1, args.episodes+1):
        episode_reward, samples = env_sampler(get_action, args.batch_size, get_value)
        actor_loss, value_loss = ppo.update(*samples)
        total_step += args.batch_size
        yield total_step, episode_reward, actor_loss, value_loss

Args = namedtuple('Args',
            ('env_name', 
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--batch', type=int, default=1000, metavar='N',
                        help='number of batch size (default: 1000)')
    parser.add_argument('--env_name', default='HalfCheetah-v2', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default cpu)')
    
    args = parser.parse_args()

    alg_args = Args(args.env_name,  # env_name
                args.device,        # device
                args.seed,          # seed
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

    logdir = "./logs/algo_ppo/env_{}".format(alg_args.env_name)
    file_name = 'batch{}_seed{}_time{}.csv'.format(alg_args.batch_size, alg_args.seed, time())
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    full_name = os.path.join(logdir, file_name)

    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'reward'])
    start_time = time()
    for step, reward, actor_loss, value_loss in main(alg_args):
        writer.writerow([step, reward])
        print('Step {}: Reward = {}, actor_loss = {}, value_loss = {}'.format(step, reward, actor_loss, value_loss))
    print("Total time: {}s.".format(time() - start_time))