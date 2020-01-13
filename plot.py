import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
import argparse
import scikits.bootstrap as sci
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.autolayout'] = True
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='Plot experiment results')
parser.add_argument('--alg-list', nargs='+', help='algorimthms to plot')
parser.add_argument('--env-name', default='Hopper-v2', type=str, help='env-name')
parser.add_argument('--workers', default=2, type=int, help='workers')
parser.add_argument('--exp-num', default=0, type=int, help='The number of experiment.(0 means all.)')

args = parser.parse_args()

alg_list = args.alg_list
env_name = args.env_name
num_alg = len(alg_list)
legent_name = {}
alg_pd_dict = {}
for i, alg in enumerate(alg_list):
    legent_name['alg{}'.format(i)] = alg
    alg_pd_dict[alg] = pd.DataFrame()

start, end = 0, 1
# load csv
for alg in alg_list:
    if alg == 'trpo':
        path = './trpo/logs/algo_{}/env_{}'.format(alg, env_name)
    elif alg == 'ppo':
        path = './ppo/logs/algo_{}/env_{}'.format(alg, env_name)
    elif 'trpo' in alg:
        path = './trpo/logs/algo_{}/env_{}/workers{}'.format(alg, env_name, args.workers)
    else:
        path = './ppo/logs/algo_{}/env_{}/workers{}'.format(alg, env_name, args.workers)

    file_list = os.listdir(path)
    file_list.sort(key=lambda x:x[x.find('time'):])
    if args.exp_num == 0:
        start = 0
        end = len(file_list)
    else:
        start = args.workers * (args.exp_num - 1)
        end = args.workers * args.exp_num
    # only includes csvs
    for i, csv in enumerate(file_list):
        csv_path = os.path.join(path, csv)
        data = pd.read_csv(csv_path)
        if i == 0:
            alg_pd_dict[alg]['step'] = data['step']
        alg_pd_dict[alg]['Run{}'.format(i)] = data['reward']

loc_list = ['Run{}'.format(i) for i in range(start, end)]
for alg in alg_list:
    alg_pd_dict[alg]['reward_smooth'] = alg_pd_dict[alg].loc[:,loc_list].mean(1).ewm(span=3).mean()
    ci = sci.ci(alg_pd_dict[alg].loc[:,loc_list].T, alpha=0.1, statfunction=lambda x: np.average(x, axis=0))
    alg_pd_dict[alg]['low'] = ci[0]
    alg_pd_dict[alg]['high'] = ci[1]
    
#figure configuration
fig = plt.figure(figsize=(14, 7))
plt.title(env_name, fontsize=50)
plt.xlabel('System probes(state-action pair)', fontsize=35)
plt.ylabel('Average return', fontsize=35)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

#plot
for alg in alg_list:
    alg_label = 'hmtrpo' if alg == 'dmtrpo' else alg
        
    plt.plot(alg_pd_dict[alg]['step'], alg_pd_dict[alg]['reward_smooth'], label=alg_label)
    plt.fill_between(alg_pd_dict[alg]['step'], alg_pd_dict[alg]["low"] , alg_pd_dict[alg]["high"], alpha=0.2)

ax = plt.subplot(111)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.offsetText.set_fontsize(30)
plt.legend(fontsize = 'xx-large', loc = 'upper left')
plt.savefig('./{}.pdf'.format(env_name))
#plt.show()
