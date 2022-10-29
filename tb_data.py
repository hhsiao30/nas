from utils.get_report import *
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter

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
import shutil

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
    parser.add_argument(
        "--tbdir",
        default=None,
        help="Tensorboard Directory. Use base name if unspecified",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    dirname = args.dirname
    design = args.design
    trial_names = [trial_name for trial_name in os.listdir(dirname)]
    trial_names.sort(key=lambda t: (len(t), t))
    tb_dir = args.tbdir
    if tb_dir is None:
        tb_dir = os.path.join("tb", os.path.basename(dirname.rstrip('/')))
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
    print(tb_dir)
    os.makedirs(tb_dir, exist_ok=True)
    for trial_name in trial_names:
        if trial_name == "ignore":
            continue
        trail_dir = os.path.join(dirname, trial_name)
        tb_trail_dir = os.path.join(tb_dir, trial_name)
        os.makedirs(tb_trail_dir, exist_ok=True)
        iter_names = [iter_dir for iter_dir in os.listdir(trail_dir) if iter_dir.startswith("iter_")]
        iter_names.sort(key=lambda t: (len(t), t))
        tb = SummaryWriter(log_dir=tb_trail_dir)
        for _iter, iter_name in enumerate(iter_names):
            proc_dir = os.path.join(*[trail_dir, iter_name, "process_0"])
            timing_file = os.path.join(proc_dir, "clock_final_opto.timing")
            if not os.path.exists(timing_file):
                break
            print(proc_dir)
            report = get_all_report(proc_dir, "clock_final_opto")
            n_iter = _iter + 1
            tb.add_scalar("TNS(ns)", report["Setup TNS"], n_iter)
            tb.add_scalar("WNS(ns)", report["Setup WNS"], n_iter)
            tb.add_scalar("Power(mW)", report["Total Power"], n_iter)
            tb.add_scalar("net_length", report["net_length"], n_iter)
            tb.flush()
        tb.close()
    
    '''
    for proc in range(nprocs):
        df[proc] = pd.DataFrame(columns=["name"])
        for _iter, iter_name in enumerate(iter_names):
            iter_dir = os.path.join(dirname, iter_name)
            proc_dir = os.path.join(iter_dir, "process_{}".format(proc))
            timing_file = os.path.join(proc_dir, "route_opt3.timing")
            if not os.path.exists(timing_file):
                break
            data = {"name": iter_name}
            timings = get_timing(timing_file)
            data = {**data, "TNS(ns)": timings["Setup TNS"]}
            power_file = os.path.join(proc_dir, "route_opt3.power")
            powers = get_power(power_file)
            data = {**data, "Total Power(uW)": powers["Total Power"]}
            df[proc] = df[proc].append(data, ignore_index=True)
    
    for proc in range(nprocs):
        print("\nprocess {}".format(proc))
        print(df[proc])
    
    metrics = ["TNS(ns)", "Total Power(uW)"]
    for metric in metrics:
        fig = plt.figure()
        for proc in range(nprocs):
            plt.xlabel("iteration")
            plt.ylabel(metric)
            y = df[proc][metric]
            x = np.arange(len(y))
            plt.plot(x, y, label="process{}".format(proc))
        plt.legend()
        figname = "{}.png".format(metric).replace('/', '|')
        figpath = os.path.join(dirname+'/fig', figname)
        os.makedirs(dirname+'/fig', exist_ok=True)
        plt.savefig(figpath)
        plt.close()
    
    selected_proc = np.arange(nprocs)
    for metric in metrics:
        fig = plt.figure()
        for proc in selected_proc:
            plt.xlabel("iteration")
            plt.ylabel(metric)
            y = df[proc][metric]
            x = np.arange(len(y))
            plt.plot(x, y)
            figname = "{}.png".format(metric).replace('/', '|')
            fig_dir = dirname+'/fig/select/process_{}'.format(proc)
            figpath = os.path.join(fig_dir, figname)
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(figpath)
            plt.close()
    '''
    
    

