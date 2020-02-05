import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
import argparse
import scikits.bootstrap as sci
import numpy as np

def load_dir(path, keyword='', keyword2=''):
    file_list = os.listdir(path)
    pd_dict = pd.DataFrame()
    # Dir only includes csvs
    i = 0
    for file in file_list:
        if keyword in file and keyword2 in file:
            full_path = os.path.join(path, file)
            data = pd.read_csv(full_path)
            if i == 0:
                pd_dict['step'] = data['step']
            pd_dict['Run{}'.format(i)] = data['reward']
            i += 1
    pd_dict['average_reward'] = pd_dict.iloc[:, 1:i+1].mean(1).ewm(span=3).mean()
    ci = sci.ci(pd_dict.iloc[:, 1:i+1].T, alpha=0.1, statfunction=lambda x: np.average(x, axis=0))
    pd_dict['low'] = ci[0]
    pd_dict['high'] = ci[1]
    return pd_dict


parser = argparse.ArgumentParser(description='Plot experiment results')
parser.add_argument('--alg_list', default='sac', nargs='+', help='algorimthms to plot')
parser.add_argument('--env_name', default='HalfCheetah-v2', type=str, help='env-name')
parser.add_argument('--keywords', default=('env',), type=str, nargs='+', help="keywords of file's name")
parser.add_argument('--keyword', default='', type=str, help="keywords for all algorithms")
parser.add_argument('--workers', default=(2,), type=int, nargs='+', help="the number of workers")


args = parser.parse_args()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.autolayout'] = True
plt.switch_backend('agg')
#figure configuration
fig = plt.figure(figsize=(14, 7))
plt.title(args.env_name, fontsize=50)
plt.xlabel('System probes(state-action pair)', fontsize=35)
plt.ylabel('Average return', fontsize=35)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

workers_tag = ''
for workers in args.workers:
    workers_tag += '_workers{}'.format(workers)
    keyword2 = args.keyword
    for keyword in args.keywords:
        alg_pd_dict = {}
        for alg in args.alg_list:
            if 'trpo' in alg:
                path = './trpo/logs/alg_{}/env_{}/workers{}'.format(alg, args.env_name, workers)
            elif 'ppo' in alg:
                path = './ppo/logs/alg_{}/env_{}/workers{}'.format(alg, args.env_name, workers)
            alg_pd_dict[alg] = load_dir(path, keyword, keyword2)
        #plot
        for alg in args.alg_list:
            alg_label = alg
            if alg_label == 'local_trpo3':
                alg_label = 'local_trpo'
            plt.plot(alg_pd_dict[alg]['step'], alg_pd_dict[alg]['average_reward'], label=alg_label+'_'+keyword+'_workers{}'.format(workers))
            plt.fill_between(alg_pd_dict[alg]['step'], alg_pd_dict[alg]["low"] , alg_pd_dict[alg]["high"], alpha=0.2)

ax = plt.subplot(111)
ax.xaxis.offsetText.set_fontsize(30)
ax.yaxis.offsetText.set_fontsize(30)
plt.legend(fontsize = 'xx-large', loc = 'lower right')
plt.savefig('./{}.pdf'.format(args.env_name+'_' + '_'.join(args.alg_list) +'_'+'_'.join(args.keywords)+workers_tag))
#plt.show()
