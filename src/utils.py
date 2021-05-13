import torch
import numpy as np

def gae_estimator(rewards, is_terminals, values, size, gamma, lamda):
    returns = np.zeros(shape=[size], dtype=np.float32)
    deltas = np.zeros(shape=[size], dtype=np.float32)
    advantages = np.zeros(shape=[size], dtype=np.float32)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(size)):
        returns[i] = rewards[i] + gamma * prev_return * (1 - is_terminals[i])
        deltas[i] = rewards[i] + gamma * prev_value * (1 - is_terminals[i]) - values.data[i]
        advantages[i] = deltas[i] + gamma * lamda * prev_advantage * (1 - is_terminals[i])

        prev_return = returns[i]
        prev_value = values.data[i]
        prev_advantage = advantages[i]

    return returns, deltas, advantages

def compute_weight(p, N, alpha=0.6, beta=0.8):
    P = p ** alpha
    P = P / P.sum()
    w = (N * P) ** (-beta)
    w = w / w.max()
    return w