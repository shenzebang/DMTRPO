"""
Run this script in /DMTRPO, change the env_name and algo_name to plot.

"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.autolayout'] = True

linestyles = {"hmtrpo":'-.', "ltrpo":':', "trpo":'-'}
legend_name = {"hmtrpo":'HMTRPO', "ltrpo":'LocalTRPO', "trpo":'TRPO'}

env_name = 'Walker2d-v2'
trpo = pd.DataFrame()
ltrpo = pd.DataFrame()
hmtrpo = pd.DataFrame()
algo_pd_dict = {'trpo':trpo, 'localtrpo':ltrpo, 'hmtrpo':hmtrpo}
num_repeat = 5

# load csv
for algo in ['hmtrpo', 'localtrpo', 'trpo']:
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
for algo in ['hmtrpo', 'localtrpo', 'trpo']:
    algo_pd_dict[algo]['reward_smooth'] = algo_pd_dict[algo].loc[:,loc_list].mean(1).ewm(span=3).mean()
    algo_pd_dict[algo]['std'] = algo_pd_dict[algo].loc[:,loc_list].std(1).ewm(span=3).mean()
    algo_pd_dict[algo]['low'] = algo_pd_dict[algo]['reward_smooth'] - algo_pd_dict[algo]['std']
    algo_pd_dict[algo]['high'] = algo_pd_dict[algo]['reward_smooth'] + algo_pd_dict[algo]['std']

fig = plt.figure(figsize=(14, 7))
plt.title(env_name, fontsize=50)
plt.xlabel('System probes(state-action pair)', fontsize=35)
plt.ylabel('Average return', fontsize=35)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.plot(trpo['Step'], trpo['reward_smooth'], label=legend_name['trpo'], color='b', linestyle=linestyles["trpo"])
plt.plot(hmtrpo['Step'], hmtrpo['reward_smooth'], label=legend_name['hmtrpo'], color='g', linestyle=linestyles["hmtrpo"])
plt.plot(ltrpo['Step'], ltrpo['reward_smooth'], label=legend_name['ltrpo'], color='r', linestyle=linestyles['ltrpo'])

plt.fill_between(trpo['Step'], trpo["low"] , trpo["high"], color='b', alpha=0.2)
plt.fill_between(hmtrpo['Step'], hmtrpo["low"], hmtrpo["high"], color='g', alpha=0.2)
plt.fill_between(ltrpo['Step'], ltrpo["low"], ltrpo["high"], color='r', alpha=0.2)

ax = plt.subplot(111)
#define ticks
#plt.xticks([0, 5e6, 1e7, 15e6, 2e7], fontsize=30)
#plt.yticks([-1000, 0, 1000, 2000], fontsize=30)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.offsetText.set_fontsize(30)
plt.legend(fontsize = 'xx-large', loc = 'upper left')
plt.show()