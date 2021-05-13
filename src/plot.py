import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(data, weight):
    smooth_data = []
    last = data[0]
    for point in data:
        point = point * (1 - weight) + last * weight
        smooth_data.append(point)
        last = point
    return np.array(smooth_data)

def smooth_and_fill_between(x, y, label, smooth_weight, width, alpha):
    y = smooth(y, smooth_weight)
    y_std = y.std()
    plt.plot(x, y, label=label)
    plt.fill_between(x, y - width * y_std, y + width * y_std, alpha=alpha)

def plot_average_reward():
    ppo_data = pd.read_csv(filepath_or_buffer='results/PPO_2021-03-29_01-11-51/run-PPO_2021-03-29_01-11-51-tag-average_reward_per_20_episode.csv')
    td3_data = pd.read_csv(filepath_or_buffer='results/TD3_2021-03-16_00-26-40/run-TD3_2021-03-16_00-26-40-tag-average_reward_per_20_episode.csv')
    ddpg_data = pd.read_csv(filepath_or_buffer='results/DDPG_2021-03-12_10-38-39/run-DDPG_2021-03-12_10-38-39-tag-average_reward_per_20_episode.csv')

    smooth_and_fill_between(td3_data['Step'], td3_data['Value'], label='TD3', smooth_weight=0.75, width=0.5, alpha=0.2)
    smooth_and_fill_between(ppo_data['Step'], ppo_data['Value'], label='PPO', smooth_weight=0.75, width=0.5, alpha=0.2)
    smooth_and_fill_between(ddpg_data['Step'], ddpg_data['Value'], label='DDPG', smooth_weight=0.75, width=0.5, alpha=0.2)

    plt.grid()
    plt.legend(loc='lower right', bbox_to_anchor=(0., 1., 1, 0.), ncol=3, fontsize=14)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Average reward', fontsize=14)
    plt.show()

def plot_q_loss():
    ddpg = pd.read_csv(filepath_or_buffer='results/DDPG_2021-03-12_10-38-39/run-DDPG_2021-03-12_10-38-39-tag-Loss_q_loss.csv')
    td3_1 = pd.read_csv(filepath_or_buffer='results/TD3_2021-03-16_00-26-40/run-TD3_2021-03-16_00-26-40-tag-Loss_q1_loss.csv')
    td3_2 = pd.read_csv(filepath_or_buffer='results/TD3_2021-03-16_00-26-40/run-TD3_2021-03-16_00-26-40-tag-Loss_q2_loss.csv')

    smooth_and_fill_between(ddpg['Step'], ddpg['Value'], label='DDPG', smooth_weight=0.75, width=1, alpha=0.3)
    smooth_and_fill_between(td3_1['Step'], td3_1['Value'], label='TD3_q1', smooth_weight=0.75, width=1, alpha=0.3)
    smooth_and_fill_between(td3_2['Step'], td3_2['Value'], label='TD3_q2', smooth_weight=0.75, width=1, alpha=0.3)

    plt.grid()
    plt.legend(loc='lower right', bbox_to_anchor=(0., 1., 1, 0.), ncol=3, fontsize=14)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Q loss', fontsize=14)
    plt.show()

if __name__ == '__main__':
    plot_q_loss()