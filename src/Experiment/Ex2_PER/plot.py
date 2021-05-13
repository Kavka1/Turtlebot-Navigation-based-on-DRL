import seaborn as sns
import pandas as pd
import os
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

def load_data(remark):
    data = {}
    min_length = 100
    min_length_seed = 10
    for seed in [10, 30, 50]:
        data[seed] = pd.read_csv(filepath_or_buffer='results/{}_seed_{}.csv'.format(remark, seed))
        if len(data[seed]) < min_length:
            min_length = len(data[seed])
            min_length_seed = seed

    time = data[min_length_seed]['Step'].tolist()
    data = pd.concat([data[10]['Value'][0:min_length], data[30]['Value'][0:min_length], data[50]['Value'][0:min_length]], axis=1).T.values

    assert len(time) == len(data[0])

    return time, smooth(data)

def plot_figure():
    remarks =  ['Normal_DDPG', 'PER_DDPG', 'Normal_TD3', 'PER_TD3']
    colors = ['r', 'g', 'b', 'black']
    labels = ['DDPG', 'DDPG+PER', 'TD3', 'TD3+PER']

    plt.figure()
    for remark, color, label in zip(remarks, colors, labels):
        time, data = load_data(remark=remark)
        sns.tsplot(time=time, data=data, color=color, condition=label)

    plt.grid()
    plt.xlabel('Total steps', fontsize='x-large')
    plt.ylabel('Average reward', fontsize='x-large')
    plt.show()


def plot_single(seed, remarks, labels, colors):
    for remark, color, label in zip(remarks, colors, labels):
        DATA = pd.read_csv(filepath_or_buffer='results/{}_seed_{}.csv'.format(remark, seed))
        time, data = DATA['Step'], DATA['Value']
        data_std = data.std()
        data = smooth(data.tolist(), weight=0.75)
        plt.plot(time, data, label=label, color=color)
        plt.fill_between(time, data - 0.3 * data_std, data + 0.3 * data_std, color=color, alpha=0.15)

    plt.grid()
    plt.title("Seed {}".format(seed))
    plt.xlabel('Total steps')
    plt.ylabel('Average reward')


if __name__ == '__main__':
    #plot_figure()


    plt.figure()
    for i, seed in enumerate([10, 30, 50, 10, 30, 50]):
        ax = plt.subplot(2, 3, i + 1)
        if i<=2:
            remarks=['Normal_DDPG', 'PER_DDPG']
            labels=['DDPG', 'DDPG+PER']
            colors = ['r', 'g']
        else:
            remarks = ['Normal_TD3', 'PER_TD3']
            labels = ['TD3', 'TD3+PER']
            colors = ['blue', 'orange']
        plot_single(seed=seed, remarks=remarks, labels=labels, colors=colors)
        if seed == 50:
            plt.legend(loc=4, fontsize='x-large', labelspacing=1., fancybox=True)
    plt.show()
