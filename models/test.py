import os
import sys
root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from utils.gnn_util import *
from models.model import *

class Combine_Model(nn.Module):
    def __init__(self, in_channels, hidden, input_size, hidden_size, num_layers):
        super(Combine_Model, self).__init__()
        self.gnn = GraphEncode(in_channels, hidden)
        self.policy_net = PolicyNet(input_size, hidden_size, num_layers)
    
    def forward(self, stage, data, h_in, c_in):
        gnn_out = self.gnn(data)
        rep_feats = torch.rand(3)
        state = torch.cat( (gnn_out, rep_feats.view((1, -1))), axis=1 )
        parameters = stage_parameters[stage]
        action_dists, h_out, c_out = self.policy_net(state, h_in, c_in, parameters)
        return action_dists, h_out, c_out
