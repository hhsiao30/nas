import os
import pandas as pd
import numpy as np

def get_qor(path):
    with open(path, 'r') as f:
        data = dict()
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("Net Length:"):
                data["net_length"] = float(line.split()[-1])
            if line.startswith("Buf/Inv Cell Count"):
                data["Buf/Inv Cell Count"] = float(line.split()[-1])
            if line.startswith("Cell Area (netlist):"):
                data["Cell Area"] = float(line.split()[-1])
            if line.startswith("Total Number of Nets:"):
                data["num_nets"] = float(line.split()[-1])
            if line.startswith("Leaf Cell Count:"):
                data["Leaf Cell Count"] = float(line.split()[-1])

        # data["DRV"] = line[-1].strip()
    return data

def get_timing(path):
    with open(path, 'r') as f:
        data = dict()
        prefix = ""
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("Setup violations"):
                prefix = "Setup "
            if line.startswith("Hold violations"):
                prefix = "Hold "
            if line.startswith("WNS"):
                data[prefix+"WNS"] = float(line.split()[1])
            if line.startswith("TNS"):
                data[prefix+"TNS"] = float(line.split()[1])
            if line.startswith("NUM"):
                data[prefix+"NUM"] = float(line.split()[1])
    return data


def get_power(path):
    with open(path, 'r') as f:
        data = dict()
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("Total"):
                line_splits = line.split()
                if (len(line_splits) == 9):
                    data["Internal Power"] = float(line_splits[1])
                    data["Switching Power"] = float(line_splits[3])
                    data["Leakage Power"] = float(line_splits[5])
                    data["Total Power"] = float(line_splits[7])
    return data

def get_all_report(report_dir, report_name):
    data = {}
    qors = get_qor(os.path.join(report_dir, "{}.qor".format(report_name)))
    data = {**data, **qors}
    timings = get_timing(os.path.join(report_dir, "{}.timing".format(report_name)))
    data = {**data, **timings}
    powers = get_power(os.path.join(report_dir, "{}.power".format(report_name)))
    data = {**data, **powers}
    return data
