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
from scipy.interpolate import make_interp_spline

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        default="ldpc",
        choices=["ldpc", "aes", "vga"],
        help="The design to use.",
    )
    parser.add_argument(
        "--trial-dir",
        default=None,
        help="Tensorboard Directory. Use base name if unspecified",
    )
    parser.add_argument(
        "--baseline",
        default=None,
    )
    parser.add_argument(
        "--stage",
        default="clock_final_opto",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    design = args.design
    trial_dir = args.trial_dir
    stage = args.stage
    trial_dir = trial_dir.rstrip('/')
    trial_splits = trial_dir.split('/')
    trial_name = trial_splits[-2] + '_' + trial_splits[-1]
    # trial_name = os.path.basename(trial_dir.rstrip('/'))
    os.makedirs("fig", exist_ok=True)
    fig_path = os.path.join("fig", "{}_{}.png".format(trial_name, stage))
    iter_names = [iter_dir for iter_dir in os.listdir(trial_dir) if iter_dir.startswith("iter_")]
    iter_names.sort(key=lambda t: (len(t), t))
    iters = []
    tns = []
    ignore_iters = []
    for _iter, iter_name in enumerate(iter_names):
        if _iter in ignore_iters:
            continue
        proc_dir = os.path.join(*[trial_dir, iter_name, "process_0"])
        timing_file = os.path.join(proc_dir, "{}.timing".format(stage))
        if not os.path.exists(timing_file):
            break
        print(proc_dir)
        report = get_all_report(proc_dir, stage)
        iters.append((_iter+1)*1)
        tns.append(report["Setup TNS"])
        # tb.add_scalar("TNS(ns)", report["Setup TNS"], (_iter+1)*2)
        # tb.flush()
    spl_i = make_interp_spline(iters, tns)
    xs = np.arange(1, iters[-1]+1)
    fig = plt.figure()
    plt.xlabel("iteration")
    plt.ylabel("TNS(ns)")
    print(iters)
    print((tns))
    print(xs)
    print(spl_i(xs))
    #plt.plot(xs, spl_i(xs), label="RL")
    plt.plot(iters, tns, label="RL")
    if args.baseline is not None:
        report = get_all_report(args.baseline, stage)
        # tns = [report["Setup TNS"]]*len(iters)
        tns = [-173.41]*len(iters) # ldpc 17.5
        # tns = [-18]*len(iters)
        plt.title(args.design.upper())
        plt.plot(iters, tns, label="baseline")
        plt.legend()
    plt.savefig(fig_path)
    plt.close()

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
