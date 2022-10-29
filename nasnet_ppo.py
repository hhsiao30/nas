import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.ppo_controller import *
from params import *
from utils.gnn_util import *
from utils.get_report import *
from utils.monitor import *
from utils.script_util import *

import os
import sys
import datetime
import subprocess
import numpy as np
from pprint import pprint
import pickle
import scipy.signal
import argparse
from distutils.dir_util import copy_tree

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        default="ldpc",
        choices=["ldpc", "aes", "vga"],
        help="The design to use.",
    )
    parser.add_argument(
        "--dirname",
        default=None,
        help="Training Directory. Use design name if unspecified",
    )
    parser.add_argument('--no-hidden-reuse', action='store_true', help="Whether NOT to reuse hidden state across episodes")
    parser.add_argument('--world-size', default=1, type=int, help='number of processes')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--freq', default=None, type=float, help='frequency')
    parser.add_argument('--port', default='12355', help='port of processes')
    parser.add_argument('--restore-dir', default=None, help='restore from checkpoint directory')
    args = parser.parse_args()
    return args

def step(rank, process_dir: str, stage: str, actions: dict):
    action_path = os.path.join(process_dir, "{}_actions.txt".format(stage))
    print("Process {}: Preidicted {} actions: ".format(rank, stage))
    with open(action_path, "w") as action_file:
        for parameter, action in actions.items():
            value = parameters[parameter][action]
            print(parameter, value)
            action_file.write("{} {}\n".format(parameter, value))
    print("\nProcess {}: Running {} with predicted actions...".format(rank, stage))

    # calling corresponding tcl script
    cur_dir = os.path.abspath(os.getcwd())
    os.chdir(process_dir)
    print("Process {}: working on {}".format(rank, process_dir))
    name = process_dir.replace('/', '_')
    log_dir = os.path.join(*[cur_dir, process_dir, "{}_output.log".format(stage)])
    cmd = "/tools/software/synopsys/icc2/latest/bin/icc2_shell -f {}.tcl > {}".format(stage, log_dir)
    # os.system(cmd)
    try:
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
    except:
        process.kill()
    subprocess.call(["cp", "icc2_output.txt", "{}_output2.txt".format(stage)])
    os.chdir(cur_dir)

    print("Process {}: {} finished.".format(rank, stage))

def discount_cumsum(x, gamma):
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]

def get_graph_data(rank, process_dir, cur_stage):
    stage = convert_stage_name[cur_stage]
    folder = process_dir + '/'
    data = None
    print("\nProcess {}: Generating graph {}".format(rank, stage))
    if os.path.exists( folder + stage + ".def" ) and os.path.exists( folder + stage + "_node_features.txt" ):
        cell2feat = get_cell2feat( folder + stage + "_node_features.txt" )
        data = genGraph( folder + stage, cell2feat )
    return data

def get_reward(report, init_report, next_report):
    return ( next_report["Setup TNS"] - report["Setup TNS"] ) / abs(init_report["Setup TNS"])

def setup_new_iter(rank, base_dir, design, _iter, freq):
    process_dir = "{}/iter_{}/process_{}".format(base_dir, _iter, rank)
    ref_dir = "ref/{}_ref".format(design)
    os.makedirs(process_dir, exist_ok=True)
    copy_tree(ref_dir, process_dir)
    ref_sdc = os.path.join("ref", "{}_ref.sdc".format(design))
    new_sdc = os.path.join(process_dir, "{}.sdc".format(design))
    write_sdc(ref_sdc, new_sdc, freq, design)
    return process_dir

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=10800), rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def cal_loss(clip_epsilon, old_probs, old_log_probs, probs, log_probs, advantage):
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantage
    surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
    
    loss = -(torch.min(surr1, surr2)).mean()
    return loss


def train(rank, args):
    # training setup
    design = args.design
    dirname = args.dirname
    restore_dir = args.restore_dir
    if restore_dir is not None:
        base_dir = dirname if dirname is not None else os.path.dirname(restore_dir.rstrip('/'))
    else:
        base_dir = dirname if dirname is not None else design
        os.makedirs(base_dir, exist_ok=True)
    print("Process {}: Set base_dir as {}".format(rank, base_dir))
    setup(rank, args.world_size, args.port)
    start_iter = 0
    freq = args.freq if args.freq is not None else default_freqs[design]

    # training parameters
    NUM_ITERS = 1000
    BATCH_SIZE = 1 # episodes per batch
    GAMMA = 1 # disable discount since too few stages & reward fluctuates
    rnn_num_layers = 1 # lstm or gru
    rnn_hidden_size = 100
    embedding_size = 32
    baseline_weight = 0.95
    baseline = None
    min_baseline = -100
    clip_epsilon = 0.2
    n_update_per_iteration = 8
    

    controller_net = ControllerNet(hidden_size=rnn_hidden_size, num_layers=rnn_num_layers, embedding_size=embedding_size)

    controller_net = DDP(controller_net).train()
    optimizer = torch.optim.Adam( list(controller_net.parameters()), lr=args.lr)
    h_in = torch.zeros(rnn_num_layers, 1, rnn_hidden_size) # hidden state: (num_layers, batch , hidden)

    # restore checkpoint
    if (restore_dir is not None):
        checkpoint_path = os.path.join(restore_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            if rank == 0:
                try:
                    print("Restore from checkpoint {}".format(checkpoint_path))
                    checkpoint = torch.load(checkpoint_path)
                    controller_net.load_state_dict(checkpoint["controller_net_state_dict"])
                    start_iter = checkpoint["epoch"] + 1
                    print("Successfully restore from checkpoint {}".format(checkpoint_path))
                except Exception as e:
                    sys.exit("Restore from checkpoint fail.\n{}".format(e))

            process_dir = os.path.join(restore_dir, "process_{}".format(rank))
            process_cpt_path = os.path.join(process_dir, "process.pt")
            if os.path.exists(process_cpt_path):
                process_cpt = torch.load(process_cpt_path)
                optimizer.load_state_dict(process_cpt["optimizer_state_dict"])
                if not args.no_hidden_reuse:
                    h_in = process_cpt["h_n"]
                    print("Process {}: Successfully restore hidden state from {}".format(rank, process_cpt_path))
        else:
            sys.exit("Restore from checkpoint fail.")

    # create tensorboard SummaryWriter
    log_dir = os.path.join(base_dir, "log/process_{}".format(rank))
    os.makedirs(log_dir, exist_ok=True)
    #tb = SummaryWriter(log_dir=log_dir)
    #print("Process {}: tensorboard log directory: {}".format(rank, log_dir))

    for _iter in range(start_iter, NUM_ITERS):
        if rank == 0:
            print("***********************************")
            print("* Iteration {}".format(_iter))
            print("***********************************")
            checkpoint_dir = "{}/iter_{}".format(base_dir, _iter)
        
        process_dir = setup_new_iter(rank, base_dir, design, _iter, freq) # <checkpoint_dir>/process_i
        actions, old_probs, old_log_probs, h_out = controller_net.module.sample(h_in)
        print("Process {}: old_log_probs: {}".format(rank, old_log_probs))
        # initial state
        report = get_all_report(process_dir, "final_opto") # parse report files
        init_report = report

        stage = "clock_opt"
        # run icc2 with predicted actions
        step(rank, process_dir, stage, actions)
        next_report = get_all_report(process_dir, convert_stage_name[stage])
        reward = get_reward(report, init_report, next_report)
        if baseline == None:
            baseline = max(min_baseline, reward)
        else:
            baseline = baseline * baseline_weight + reward * (1 - baseline_weight)
        advantage = reward - max(min_baseline, baseline)
        
        print("Process {}: reward: {}".format(rank, reward))
        print("Process {}: advantage: {}".format(rank, advantage))
        print("Process {}: baseline: {}".format(rank, baseline))
        print("Process {}: TNS: {}".format(rank, next_report["Setup TNS"]))
        print()


        for ppo_iter in range(n_update_per_iteration):
            probs, log_probs, h_out = controller_net.module.evaluate(h_in, actions)
            loss = cal_loss(clip_epsilon, old_probs, old_log_probs, probs, log_probs, advantage)
            print("Process {}: ppo_iter: {}".format(rank, ppo_iter))
            print("Process {}: log_probs: {}".format(rank, log_probs.detach()))
            print("Process {}: loss: {}".format(rank, loss.item()))
            print()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save checkpoint
        if rank == 0:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            print("Save checkpoint: {}\n\n".format(checkpoint_path))
            torch.save({
                "epoch": _iter,
                "controller_net_state_dict": controller_net.state_dict(),
            }, checkpoint_path)

        # save process-wise checkpoint
        process_cpt_path = os.path.join(process_dir, "process.pt")
        print("Save process checkpoint: {}\n\n".format(process_cpt_path))
        torch.save({
            "h_n": h_in,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "init_report": init_report,
            "final_report": report,
        }, process_cpt_path)

        if args.no_hidden_reuse:
            h_in = torch.zeros(rnn_num_layers, 1, rnn_hidden_size) # hidden state: (num_layers, batch , hidden)

        # synchronize process
        dist.barrier()
    cleanup()
    #tb.close()

if __name__ == "__main__":
    args = get_args()
    print(args)
    mp.spawn(train, args=(args, ), nprocs=args.world_size)

