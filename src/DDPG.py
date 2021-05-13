import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import os
from model import DeterministicPolicy, QFunction
from utils import *

class DDPGAgent(nn.Module):
    def __init__(self, s_dim, a_dim, args):
        super(DDPGAgent, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr_pi = args.lr_pi
        self.lr_q = args.lr_q
        self.weight_decay_q = args.weight_decay_q
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.PER = args.PER
        self.device = torch.device(args.device)

        self.policy = DeterministicPolicy(s_dim, a_dim).to(self.device)
        self.policy_target = DeterministicPolicy(s_dim, a_dim).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.Q = QFunction(s_dim, a_dim).to(self.device)
        self.Q_target = QFunction(s_dim, a_dim).to(self.device)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.optimizer_pi = optim.Adam(self.policy.parameters(), lr=self.lr_pi)
        self.optimizer_q = optim.Adam(self.Q.parameters(), lr=self.lr_q, weight_decay=self.weight_decay_q)

    def choose_action(self, s):
        s = torch.tensor(s).to(self.device).float()
        return self.policy(s).cpu().tolist()

    def learn(self, memory):
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

        #train q
        action_next_target = self.policy_target(s_)
        q_loss_target = r + (1 - done) * self.gamma * self.Q_target(s_, action_next_target)
        q_loss_eval = self.Q(s, a)
        if self.PER:
            q_loss = (w * (q_loss_eval - q_loss_target.detach())**2).mean()
        else:
            q_loss = F.mse_loss(q_loss_eval, q_loss_target.detach()).mean()
        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        #train policy
        pi_loss = - self.Q(s, self.policy(s)).mean()
        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        #update the target net
        self.soft_update()

        return q_loss.cpu().item(), pi_loss.cpu().item()

    def soft_update(self):
        for param_eval, param_target in zip(self.policy.parameters(), self.policy_target.parameters()):
            param_target.data.copy_((1 - self.tau) * param_target + self.tau * param_eval)
        for param_eval, param_target in zip(self.Q.parameters(), self.Q_target.parameters()):
            param_target.data.copy_((1 - self.tau) * param_target + self.tau * param_eval)

    def compute_priority(self, s, a, r, s_):
        s, s_, a = torch.tensor(s).to(self.device).float().unsqueeze(0), torch.tensor(s_).to(self.device).float().unsqueeze(0), torch.tensor(a).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            p = abs(r + self.gamma * self.Q_target(s_, self.policy_target(s_)) - self.Q(s, a))
        return p.cpu().item()

    def save_model(self, remarks, pi_path=None, q_path=None):
        if not os.path.exists('pretrained_models_DDPG/'):
            os.mkdir('pretrained_models_DDPG/')

        if pi_path == None:
            pi_path = 'pretrained_models_DDPG/policy_{}'.format(remarks)
        if q_path == None:
            q_path = 'pretrained_models_DDPG/q_{}'.format(remarks)
        print('Saving model to {} and {}'.format(pi_path, q_path))
        torch.save(self.policy.state_dict(), pi_path)
        torch.save(self.Q.state_dict(), q_path)

    def load_model(self, pi_path, q_path):
        print('Loading models from {} and {}'.format(pi_path, q_path))
        self.policy.load_state_dict(torch.load(pi_path))
        self.Q.load_state_dict(torch.load(q_path))