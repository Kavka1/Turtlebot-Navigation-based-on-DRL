import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VFunction, GaussianPolicy
import os
import numpy as np
from utils import gae_estimator

class replay_buffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.is_terminals[:]

class PPOAgent():
    def __init__(self,s_dim, a_dim, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr = args.lr_ppo
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.action_std = args.action_std
        self.K_epoch = args.K_epoch
        self.epsilon_clip = args.epsilon_clip
        self.device = torch.device(args.device)

        self.policy = GaussianPolicy(s_dim, a_dim, action_std=self.action_std, device=self.device).to(self.device)
        self.V = VFunction(s_dim).to(self.device)
        self.policy_old = GaussianPolicy(s_dim, a_dim, action_std=self.action_std, device=self.device).to(self.device)
        self.V_old = VFunction(s_dim).to(self.device)
        self.memory = replay_buffer()

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_V = optim.Adam(self.V.parameters(), lr=self.lr)

        self.hard_update()

    def hard_update(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.V_old.load_state_dict(self.V.state_dict())

    def choose_action(self, s):
        self.memory.states.append(s)

        s = torch.tensor(s).to(self.device).float().unsqueeze(dim=0)
        action, dist = self.policy_old(s)
        self.memory.logprobs.append(dist.log_prob(action).cpu().item())

        action = action[0].cpu().tolist()
        self.memory.actions.append(action)

        return action

    def learn(self):
        old_states = torch.tensor(self.memory.states).to(self.device).detach()
        old_actions = torch.tensor(self.memory.actions).to(self.device).detach()
        old_log_probs = torch.tensor(self.memory.logprobs).to(self.device).detach()

        returns, deltas, advantages = gae_estimator(self.memory.rewards, self.memory.is_terminals, self.V_old(old_states).detach(), len(self.memory.rewards),self.gamma, self.lamda)
        returns = torch.from_numpy(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = torch.from_numpy(advantages).to(self.device)

        for i in range(self.K_epoch):
            actions, dist = self.policy(old_states)
            log_probs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_value = self.V(old_states).squeeze(dim=1)

            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            loss_pi = (- torch.min(surr2, surr1) - 0.01*dist_entropy).mean()
            loss_v = 0.5*F.mse_loss(state_value, returns)

            self.optimizer_pi.zero_grad()
            self.optimizer_V.zero_grad()
            loss_pi.backward()
            loss_v.backward()
            self.optimizer_pi.step()
            self.optimizer_V.step()

        self.hard_update()

    def save_model(self, remarks='', pi_path=None, v_path=None):
        if not os.path.exists('pretrained_models_PPO/'):
            os.mkdir('pretrained_models_PPO/')

        if pi_path == None:
            pi_path = 'pretrained_models_PPO/policy_{}'.format(remarks)
        if v_path == None:
            v_path = 'pretrained_models_PPO/v_{}'.format(remarks)
        print('Saving model to {} and {}'.format(pi_path, v_path))
        torch.save(self.policy.state_dict(), pi_path)
        torch.save(self.V.state_dict(), v_path)

    def load_model(self, pi_path, v_path):
        print('Loading models from {} and {}'.format(pi_path, v_path))
        self.policy.load_state_dict(torch.load(pi_path))
        self.V.load_state_dict(torch.load(v_path))