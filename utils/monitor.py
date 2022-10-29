import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.autograd import Variable
from torch.distributions import Categorical, Normal
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, TopKPooling, global_max_pool, SAGPooling, global_mean_pool
import pickle
import os
from pprint import pprint
import numpy as np
import scipy.signal
import argparse
from distutils.dir_util import copy_tree
import subprocess
from params import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

def log_rewards(process_dir: str, rewards):
    with open(os.path.join(process_dir, "reward.txt"), 'w') as f:
        f.write(' '.join(str(reward) for reward in rewards))
        f.write('\n')

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm