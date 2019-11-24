"""
Run this script in /DMTRPO, change the env_name and algo_name to plot.

"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
import argparse

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.autolayout'] = True

parser = argparse.ArgumentParser(description='Plot experiment result')
parser.add_argument('--list', nargs='+', help='algorimthms to plot')
parser.add_argument('--env-name', type=str, help='env-name')
parser.add_argument('--num-repeat', type=int, help='num_repeat')

args = parser.parse_args()
algo_list = args.list
env_name = args.env_name
num_repeat = args.num_repeat
nums_algo = len(algo_list)
legent_name = {}
algo_pd_dict = {}
for i, algo in enumerate(algo_list):
    legent_name['algo{}'.format(i)] = algo
    algo_pd_dict[algo] = pd.DataFrame()

# load csv
for algo in algo_list:
    path = './logs/algo_{}/env_{}'.format(algo, env_name)
    file_list = os.listdir(path)
    # only includes csvs
    for i, csv in enumerate(file_list):
        csv_path = os.path.join(path, csv)
        data = pd.read_csv(csv_path)
        if i == 0:
            algo_pd_dict[algo]['Step'] = data['steps']
        algo_pd_dict[algo]['Run{}'.format(i)] = data['avg_rewards0']

loc_list = ['Run{}'.format(i) for i in range(num_repeat)]
for algo in algo_list:
    algo_pd_dict[algo]['reward_smooth'] = algo_pd_dict[algo].loc[:,loc_list].mean(1).ewm(span=3).mean()
    algo_pd_dict[algo]['std'] = algo_pd_dict[algo].loc[:,loc_list].std(1).ewm(span=3).mean()
    algo_pd_dict[algo]['low'] = algo_pd_dict[algo]['reward_smooth'] - algo_pd_dict[algo]['std']
    algo_pd_dict[algo]['high'] = algo_pd_dict[algo]['reward_smooth'] + algo_pd_dict[algo]['std']

#figure configuration
fig = plt.figure(figsize=(14, 7))
plt.title(env_name, fontsize=50)
plt.xlabel('System probes(state-action pair)', fontsize=35)
plt.ylabel('Average return', fontsize=35)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

#plot
for algo in algo_list:
    plt.plot(algo_pd_dict[algo]['Step'], algo_pd_dict[algo]['reward_smooth'], label=algo)
    plt.fill_between(algo_pd_dict[algo]['Step'], algo_pd_dict[algo]["low"] , algo_pd_dict[algo]["high"], alpha=0.2)

ax = plt.subplot(111)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.offsetText.set_fontsize(30)
plt.legend(fontsize = 'xx-large', loc = 'upper left')
plt.show()