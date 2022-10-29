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

class ControllerNet(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_size):
        super(ControllerNet, self).__init__()
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        len_action = sum(map(len, parameters.values()))
        self.start_index = {}
        cur_index = 0
        for param in parameters:
            self.start_index[param] = cur_index
            cur_index += len(parameters[param])

        self.embedding = nn.Embedding(len_action, embedding_size)
        ''' register customized head for each parameter '''
        self.heads = nn.ModuleDict()
        for parameter, values in parameters.items():
            head_name = parameter.replace('.', '_') # key cannot include '.'
            self.heads[head_name] = nn.Linear(hidden_size, len(values))

    def forward(self, x, h_in, parameter):
        # x = torch.unsqueeze(x, 0) # (batch, length, input_size)
        x = self.embedding(x)
        x = torch.unsqueeze(x, 0)
        output, h_out = self.rnn(x, h_in) # output: (batch, length, hidden_size)
        output = output.view(1, -1)
        head_name = parameter.replace('.', '_')
        logits = self.heads[head_name](output)
        probs = F.softmax(logits, dim=-1)
        action_dist = Categorical(probs)
        return action_dist, h_out

    def sample(self, h_in):
        actions = {}
        log_probs = []
        probs = []
        last_param = [*parameters.keys()][-1]
        dummy_index = self.start_index[last_param]
        _input = torch.LongTensor([dummy_index]) # first(dummy) input
        for parameter in parameters:
            action_dist, h_out = self.forward(_input, h_in, parameter) 
            action = action_dist.sample() # tensor([1])

            log_prob = action_dist.log_prob(action) # tensor([-0.7466], grad_fn=<SqueezeBackward1>)
            log_probs.append(log_prob.detach())

            prob = action_dist.probs[0][action] # action_dist.probs: tensor([[0.5260, 0.4740]], grad_fn=<SoftmaxBackward>)
            probs.append(prob.detach())

            action = action.item()
            actions[parameter] = action # action index
            
            h_in = h_out.detach()
            _input = torch.LongTensor([action + self.start_index[parameter]])

            print("Parameter: {}".format(parameter))
            print("values:", parameters[parameter])
            print("probs:", [p.item() for p in action_dist.probs[0]])
            print()

        return actions, torch.stack(probs).view(-1), torch.stack(log_probs).view(-1), h_out.detach()
    
    def evaluate(self, h_in, actions):
        log_probs = []
        probs = []
        last_param = [*parameters.keys()][-1]
        dummy_index = self.start_index[last_param]
        _input = torch.LongTensor([dummy_index]) # first(dummy) input
        for parameter, action in actions.items():
            action_dist, h_out = self.forward(_input, h_in, parameter)
            log_prob = action_dist.log_prob(torch.LongTensor([action]))
            log_probs.append(log_prob)
            prob = action_dist.probs[0][action]
            probs.append(prob)
            h_in = h_out.detach()
            _input = torch.LongTensor([action + self.start_index[parameter]])

        return torch.stack(probs).view(-1), torch.stack(log_probs).view(-1), h_out.detach()



