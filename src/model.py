import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import math

class GaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, action_std, device, hidden_units=[128, 64]):
        super(GaussianPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], a_dim)
        )
        action_var = torch.full(size=[a_dim, ], fill_value=action_std*action_std).to(device)
        self.cov_mat = torch.diag(action_var)

    def forward(self, s):
        action_mean = self.model(s)
        action_mean[:, 0] = torch.sigmoid(action_mean[:, 0])
        action_mean[:, 1] = torch.tanh(action_mean[:, 1])

        dist = MultivariateNormal(loc=action_mean, covariance_matrix=self.cov_mat)
        action  = dist.sample()
        return action, dist


class DeterministicPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_units=[128, 64]):
        super(DeterministicPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], a_dim),
        )

    def forward(self, s):
        preactivate = self.model(s)
        if preactivate.size() == torch.Size([2]):
            linear = F.sigmoid(preactivate[0])
            angular = F.tanh(preactivate[1])
            return torch.tensor([linear, angular])
        else:
            linear = F.sigmoid(preactivate[:, 0]).unsqueeze(dim=1)
            angular = F.tanh(preactivate[:, 1]).unsqueeze(dim=1)
            return torch.cat([linear, angular], dim=1)

class QFunction(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_units=[128, 64]):
        super(QFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, s, a):
        return self.model(torch.cat([s, a], dim=1))

class VFunction(nn.Module):
    def __init__(self, s_dim, hidden_units = [128, 64]):
        super(VFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(s_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, s):
        return self.model(s)
