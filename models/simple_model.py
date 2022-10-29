import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.distributions import Categorical, Normal
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, TopKPooling, global_max_pool, SAGPooling, global_mean_pool
import os
import sys

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from utils.gnn_util import *
from params import *

class GraphEncode(nn.Module):
    def __init__(self, in_channels, hidden ):
        super(GraphEncode, self).__init__()
        self.gnn_conv1 = GINConv(Seq(Lin(in_channels, hidden), nn.ReLU(), Lin(hidden, hidden)))
        self.gnn_pool1 = TopKPooling(in_channels)
        self.gnn_conv2 = GINConv(Seq(Lin(hidden, hidden), nn.ReLU(), 
        Lin(hidden, hidden)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out = F.relu(self.gnn_conv1(x, edge_index))
        out, edge_index, _, batch, perm, score = self.gnn_pool1(
            out, edge_index, None, batch, attn=x)
        out = F.relu(self.gnn_conv2(out, edge_index))
        out = global_mean_pool(out, batch)
        return out

class DiscreteHead(nn.Module):
    def __init__(self, input_size, action_size):
        super(DiscreteHead, self).__init__()
        hidden_size = 8
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        logit = self.head(x)
        prob = F.softmax(logit, dim=-1)
        return Categorical(prob)

class ContiHead(nn.Module):
    def __init__(self, input_size):
        super(ContiHead, self).__init__()
        hidden_size = 8
        self.share_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            # nn.Tanh() # balance distribution
        )
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Softplus() # std >= 0
        )

    def forward(self, x):
        base_out = self.share_layer(x)
        mu = self.mean_layer(base_out)
        std = self.std_layer(base_out)
        return Normal(mu, std)

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PolicyNet, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        head_input_size = 16
        self.fc = nn.Linear(hidden_size, head_input_size)
        ''' register customized head for each parameter '''
        self.heads = nn.ModuleDict()
        for parameter, values in discrete_values.items():
            head_name = parameter.replace('.', '_') # key cannot include '.'
            self.heads[head_name] = DiscreteHead(head_input_size, len(values))
        for parameter, values in conti_values.items():
            head_name = parameter.replace('.', '_')
            self.heads[head_name] = ContiHead(head_input_size)

    def forward(self, x, h_in, parameters):
        x = torch.unsqueeze(x, 0) # (length, batch, input_size)
        output, h_out = self.rnn(x, h_in) # output: (length, batch, hidden_size)
        output = output.view(1, -1)
        output = torch.tanh(self.fc(output))
        action_dists = []
        for parameter in parameters:
            head_name = parameter.replace('.', '_')
            action_dist = self.heads[head_name](output)
            action_dists.append(action_dist)
        return action_dists, h_out

    def sample(self, state, h_in, parameters):
        action_dists, h_out = self.forward(state, h_in, parameters) # discrete: prob, conti: dist parameters
        actions = {}
        total_log_prob = 0
        for parameter, action_dist in zip(parameters, action_dists):
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            total_log_prob += log_prob
            if parameter in discrete_values:
                action = action.item()
            else: # map continuous action to [min, max]
                [val_min, val_max] = conti_values[parameter]
                action = (val_max + val_min) / 2 + np.tanh(action.item()) * (val_max - val_min) / 2
            actions[parameter] = action
        return actions, total_log_prob, h_out