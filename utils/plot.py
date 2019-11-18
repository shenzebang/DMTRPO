import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scikits.bootstrap as sci
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.autolayout'] = True
plt.switch_backend('agg')

def plot(dataframe, file_name, num_repeat):
    plot_df = pd.DataFrame()
    plot_df['steps'] = dataframe['steps']
    results = pd.DataFrame()
    for i in range(num_repeat):
        results["Run{}".format(i)] = dataframe["avg_rewards{}".format(i)]

    loc_list = ['Run{}'.format(i) for i in range(num_repeat)]
    plot_df["reward_smooth"] = results.loc[:,loc_list].mean(1).ewm(span=5).mean()

    fig = plt.figure(figsize=(14, 7))
    plt.title('exp_result', fontsize=50)
    plt.xlabel('System probes(state-action pair)', fontsize=35)
    plt.ylabel('Average return', fontsize=35)

    ci = sci.ci(results.T, alpha=0.1, statfunction=lambda x: np.average(x, axis=0))
    plot_df["low"] = ci[0]
    plot_df["high"] = ci[1]
    plot_df["low"] = plot_df["low"].ewm(span=5).mean()
    plot_df["high"] = plot_df["high"].ewm(span=3).mean()

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.plot(plot_df['steps'], plot_df['reward_smooth'], label='reward', color='b')
    plt.fill_between(plot_df['steps'], plot_df["low"], plot_df["high"], color='b', alpha=0.2)
    ax = plt.subplot(111)
    plt.xticks(fontsize=30)
    ax.xaxis.offsetText.set_fontsize(30)
    ax.yaxis.offsetText.set_fontsize(30)
    plt.legend(fontsize = 'xx-large', loc = 'upper left')

    plt.savefig(file_name)