import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def smooth(data, weight=0.8):
    smooth_data = []
    last = data[0]
    for point in data:
        point = point * (1 - weight) + last * weight
        smooth_data.append(point)
        last = point
    return np.array(smooth_data)


def load_data(remark, seeds):
    data = {}
    min_length = 100
    min_length_seed = 10
    for seed in seeds:
        data[seed] = pd.read_csv(filepath_or_buffer='results/{}_seed_{}.csv'.format(remark, seed))
        if len(data[seed]) < min_length:
            min_length = len(data[seed])
            min_length_seed = seed

    time = data[min_length_seed]['Step'].tolist()
    data = pd.concat([data[10]['Value'][0:min_length], data[20]['Value'][0:min_length], data[30]['Value'][0:min_length], data[40]['Value'][0:min_length], data[50]['Value'][0:min_length]], axis=1).T.values
    for i, d in enumerate(data):
        data[i] = smooth(d, weight=0.6)

    assert len(time) == len(data[0])

    return time, data


def plot_figure():
    seeds = [10, 20, 30, 40, 50]
    remarks =  ['DDPG', 'TD3', 'PPO']
    colors = ['r', 'g', 'b']
    labels = ['DDPG', 'TD3', 'PPO']

    plt.figure()
    for remark, color, label in zip(remarks, colors, labels):
        time, data = load_data(remark= remark, seeds=seeds)
        sns.tsplot(time=time, data=data, color=color, condition=label, ci='sd', err_style='ci_band')

    plt.grid()
    plt.legend( fontsize='x-large')
    plt.xlabel('Total steps',  fontsize='x-large')
    plt.ylabel('Average reward',  fontsize='x-large')
    plt.show()


def plot_single(seed=10):
    remarks =  ['DDPG', 'TD3', 'PPO']
    colors = ['r', 'g', 'b']
    labels = ['DDPG', 'TD3', 'PPO']

    for remark, color, label in zip(remarks, colors, labels):
        DATA = pd.read_csv(filepath_or_buffer='results/{}_seed_{}.csv'.format(remark, seed))
        time, data = DATA['Step'], DATA['Value']
        data_std = data.std()
        data =smooth(data.tolist(), weight=0.4)
        plt.plot(time, data, label=label, color=color)
        plt.fill_between(time, data-0.5*data_std, data+0.5*data_std, color=color, alpha=0.15)

    plt.grid()
    plt.title("Seed {}".format(seed))
    plt.xlabel('Total steps')
    plt.ylabel('Average reward')


if __name__ == '__main__':
    plot_figure()

    #single seed plot
    plt.figure()
    for i, seed in enumerate([10, 20, 30, 40, 50]):
        plt.subplot(2, 3, i+1)
        plot_single(seed=seed)
        if i == 4:
            plt.legend(loc=7, bbox_to_anchor=(1.55, 0.5), fontsize='x-large', labelspacing=1.2, fancybox=True)
    plt.show()
