import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import QFunction, DeterministicPolicy
import os
from utils import *
import numpy as np

class TD3Agent():
    def __init__(self,s_dim, a_dim, args):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr_pi = args.lr_pi
        self.lr_q = args.lr_q
        self.gamma = args.gamma
        self.tau = args.tau
        self.noise_std = args.noise_std_td3
        self.noise_clip = args.noise_clip
        self.batch_size = args.batch_size
        self.policy_update_interval = args.policy_update_interval
        self.PER = args.PER
        self.device = torch.device(args.device)
        self.policy_loss_log = torch.tensor(0.).to(self.device)

        self.policy = DeterministicPolicy(s_dim, a_dim).to(self.device)
        self.policy_target = DeterministicPolicy(s_dim, a_dim).to(self.device)
        self.Q1 = QFunction(s_dim, a_dim).to(self.device)
        self.Q1_target = QFunction(s_dim, a_dim).to(self.device)
        self.Q2 = QFunction(s_dim, a_dim).to(self.device)
        self.Q2_target = QFunction(s_dim, a_dim).to(self.device)
        self.hard_update_target()

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr_pi)
        self.optimizer_q1 = optim.Adam(self.Q1.parameters(), lr=self.lr_q)
        self.optimizer_q2 = optim.Adam(self.Q2.parameters(), lr=self.lr_q)

    def hard_update_target(self):
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def soft_update_target(self):
        for param, param_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
        for param, param_target in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def choose_action(self, s):
        s = torch.tensor(s).to(self.device).float()
        return self.policy(s).cpu().tolist()

    def learn(self, memory, total_step):
        batch = memory.sample_batch(self.batch_size)
        if self.PER:
            s, a, r, s_, done, p = batch
            p = torch.from_numpy(p).to(self.device)
            w = compute_weight(p, self.batch_size)
        else:
            s, a, r, s_, done = batch
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device).unsqueeze(dim=1)
        s_ = torch.from_numpy(s_).to(self.device)
        done = torch.from_numpy(done).to(self.device).unsqueeze(dim=1)

        noise = (torch.randn_like(a) * self.noise_std).clamp(-self.noise_clip, self.noise_clip)
        a_target_next = self.policy_target(s_) + noise
        q1_next = self.Q1_target(s_, a_target_next)
        q2_next =self.Q2_target(s_, a_target_next)
        q_next_min = torch.min(q1_next, q2_next)
        q_loss_target = r + (1 - done) * self.gamma * q_next_min

        #update q1
        q1_loss_pred = self.Q1(s, a)
        if self.PER:
            q1_loss = (w * (q1_loss_pred - q_loss_target.detach())**2).mean()
        else:
            q1_loss = F.mse_loss(q1_loss_pred, q_loss_target.detach()).mean()
        self.optimizer_q1.zero_grad()
        q1_loss.backward()
        self.optimizer_q1.step()

        #update q2
        q2_loss_pred = self.Q2(s, a)
        if self.PER:
            q2_loss = (w * (q2_loss_pred - q_loss_target.detach())**2).mean()
        else:
            q2_loss = F.mse_loss(q2_loss_pred, q_loss_target.detach()).mean()
        self.optimizer_q2.zero_grad()
        q2_loss.backward()
        self.optimizer_q2.step()

        #delay upodate policy
        if total_step % self.policy_update_interval == 0:
            policy_loss = - self.Q1(s, self.policy(s)).mean()
            self.optimizer_pi.zero_grad()
            policy_loss.backward()
            self.optimizer_pi.step()
            self.soft_update_target()

            self.policy_loss_log = policy_loss

        return q1_loss.item(), q2_loss.item(), self.policy_loss_log.item()

    def compute_priority(self, s, a, r, s_):
        s, s_, a = torch.tensor(s).to(self.device).float().unsqueeze(0), torch.tensor(s_).to(self.device).float().unsqueeze(0), torch.tensor(a).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            a_ = self.policy_target(s_)
            p = abs(r + self.gamma * self.Q1_target(s_, a_) - self.Q1(s, a))
        return p.cpu().item()

    def save_model(self, remarks='', pi_path=None, q1_path=None, q2_path=None):
        if not os.path.exists('pretrained_models_TD3/'):
            os.mkdir('pretrained_models_TD3/')

        if pi_path == None:
            pi_path = 'pretrained_models_TD3/policy_{}'.format(remarks)
        if q1_path == None:
            q1_path = 'pretrained_models_TD3/q1_{}'.format(remarks)
        if q2_path == None:
            q2_path = 'pretrained_models_TD3/q2_{}'.format(remarks)
        print('Saving model to {} , {} and {}'.format(pi_path, q1_path, q2_path))
        torch.save(self.policy.state_dict(), pi_path)
        torch.save(self.Q1.state_dict(), q1_path)
        torch.save(self.Q2.state_dict(), q2_path)

    def load_model(self, pi_path, q1_path, q2_path):
        print('Loading models from {} , {} and {}'.format(pi_path, q1_path, q2_path))
        self.policy.load_state_dict(torch.load(pi_path))
        self.Q1.load_state_dict(torch.load(q1_path))
        self.Q2.load_state_dict(torch.load(q2_path))