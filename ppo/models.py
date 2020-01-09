import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from collections import OrderedDict
import math

def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=torch.tanh, output_activation=None):
        super().__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        pre_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(pre_size, size))
            pre_size = size
        self.output_layer = nn.Linear(pre_size, output_size)
        self.apply(_weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

# We require that action is in [-1, 1]^n
class PolicyNetwork(Network):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64),
                 activation=torch.tanh, output_activation=None, init_std=1.0, min_std=1e-6):

        super(PolicyNetwork, self).__init__(input_size, output_size, hidden_sizes, activation, output_activation)

        self.min_log_std = math.log(min_std)

        self.log_std = nn.Parameter(torch.full((1, output_size), math.log(init_std)))
        self.apply(_weight_init)

    def forward(self, x):
        mean = super(PolicyNetwork, self).forward(x)
        std = self.log_std.clamp(min=self.min_log_std).exp()
        return Normal(loc=mean, scale=std)
    
    def get_detach_pi(self, x):
        with torch.no_grad():
            pi = self.forward(x)
        return pi

    def get_log_prob(self, x, actions):
        pi = self.forward(x)
        return pi.log_prob(actions).sum(1, keepdim=True)

    def select_action(self, state):
        with torch.no_grad():
            pi = self.forward(state)
            action = pi.sample()
        return action

class ValueNetwork(Network):
    def __init__(self, input_size, hidden_sizes=(64, 64), activation=torch.tanh, output_activation=None):
        super(ValueNetwork, self).__init__(input_size, 1, hidden_sizes, activation, output_activation)

if __name__ == '__main__':
    policy = PolicyNetwork(10, 4, (10, 10))
    state = torch.randn((64, 10))
    action = policy(state)
    print(action.scale)
    print(list(policy.parameters()))

