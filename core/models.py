import torch
import torch.autograd as autograd
import torch.nn as nn
from utils2.math import *
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from collections import OrderedDict


def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
        # module.weight.data.zero_()
class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=torch.relu, init_std=1.0, min_std=1e-6):

        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        self.is_disc_action = False

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(_weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        #print(mu.data, scale.data)

        return Normal(loc=mu, scale=scale)

    def get_log_prob(self, x, actions, params=None):
        pi = self.forward(x, params)
        return pi.log_prob(actions).sum(1, keepdim=True)


    def select_action(self, state):
        pi = self.forward(state)
        action = pi.sample()
        return action

    def mean_action(self, state):
        pi = self.forward(state)
        return pi.loc


# class Policy(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(num_inputs, 64)
#         self.affine2 = nn.Linear(64, 64)
#
#         self.action_mean = nn.Linear(64, num_outputs)
#
#         self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
#
#         self.saved_actions = []
#         self.rewards = []
#         self.final_value = 0
#         self.is_disc_action = False
#
#     def forward(self, x, params=None):
#
#         x = torch.tanh(self.affine1(x))
#         x = torch.tanh(self.affine2(x))
#
#         action_mean = self.action_mean(x)
#         action_log_std = self.action_log_std.expand_as(action_mean)
#         action_std = torch.exp(action_log_std)
#
#         return Normal(loc=action_mean, scale=action_std)
#
#     def select_action(self, state):
#         pi = self.forward(state)
#         action = pi.sample()
#         return action
#
#     def get_log_prob(self, x, actions):
#         pi = self.forward(x)
#         return pi.log_prob(actions)

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            l = nn.Linear(last_dim, nh).double()
            # print(l.weight.data.dtype)
            self.affine_layers.append(l)
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1).double()

        # self.value_head.weight.data.mul_(0.1)
        # self.value_head.bias.data.mul_(0.0)
        self.value_head.apply(_weight_init)
    def forward(self, x):
        for affine in self.affine_layers:
            # print(affine.weight.data.dtype)
            # print(0)
            x = affine(x)
            x = self.activation(x)

        value = self.value_head(x)
        return value

def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution