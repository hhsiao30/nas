import os
import glob
import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
import collections
from torch_geometric.data import Data
import torch

def getPower( stage ):
    with open( stage + ".power", 'r' ) as r:
        for line in r.readlines():
            line = line.strip().split()
            if line and line[0] == "Total" and line[1] != "Dynamic":
                return float(line[1]) + float(line[3]) + float(line[5])
    return None

def get_one_stage_data( dataset_dir, G=None ):
    if not G:
        G = json_graph.node_link_graph(json.load(open(dataset_dir + "-G.json")))
    x = np.load(dataset_dir + "-feat.npy")
    edge_index = np.array(list(G.edges))
    edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
    batch = np.array([0] * len(x))
    batch = torch.from_numpy(batch).long()
    x = torch.from_numpy(x).float()
    edge_index = torch.from_numpy(edge_index).long().permute(1,0)
    power = getPower( dataset_dir )
    assert power != None
    data = Data( x=x, edge_index=edge_index, batch=batch, y=torch.tensor(power).float() )
    return data

def genGraph( dataset_dir, cell_2_feats ):
    with open( dataset_dir+".def", 'r' ) as r:
        G = nx.Graph()
        read_unit = True
        read_comp = False
        read_net = False
        node_id = 0
        net_id = 0
        nets = []
        name2id = {}
        feats = []
        while 1:
            line = r.readline()
            if not line: break
            if read_unit and "UNITS DISTANCE MICRONS" in line:
                unit = float( line.strip().split()[-2] )
                read_unit = False
                continue
            elif "COMPONENTS" in line and "END" not in line:
                read_comp = True
                continue
            elif "COMPONENTS" in line and "END" in line:
                read_comp = False
                continue
            elif read_comp:
                line = line.strip().split()
                name = line[1]
                if name in cell_2_feats:
                    G.add_node( node_id )
                    name2id[ name ] = node_id
                    feats.append( cell_2_feats[name] )
                    node_id += 1
                continue
            elif "NETS" in line and "SPECIAL" not in line and "END" not in line:
                read_net = True
                gates = []
                continue
            elif "NETS" in line and "SPECIAL" not in line and "END" in line:
                read_net = False
                if len(gates) > 1 and gates[0] != "PIN" and gates[0] in name2id:
                    if net_id and net_id % 1000 == 0:
                        print("parsed {} nets".format(net_id) )
                    for gate in gates[1:]:
                        if gate in name2id:
                            G.add_edge( name2id[gates[0]], name2id[gate])
                gates = []
            elif read_net:
                line = line.strip().split()
                if '-' == line[0]:
                    assert len(line) == 2
                    if len(gates) > 1 and gates[0] != "PIN" and gates[0] in name2id:
                        if net_id and net_id % 1000 == 0:
                            print("parsed {} nets".format(net_id) )
                        for gate in gates[1:]:
                            if gate in name2id:
                                G.add_edge( name2id[gates[0]], name2id[gate])
                    gates = []
                else:
                    for i in range( len(line) ):
                        if '(' == line[i] and ')' == line[i+3]:
                            gates.append( line[i+1] )
                continue

        feats = np.array( feats )
        print( "Netlist Structure: {} nodes, {} edges".format(G.number_of_nodes(), G.number_of_edges()))
        print(" feature shape: {}".format( feats.shape ) )
        assert G.number_of_nodes() == feats.shape[0]

        np.save( dataset_dir + "-feat", feats )

        with open( dataset_dir + "-G.json", 'w' ) as w:
            json.dump( json_graph.node_link_data(G), w )

        with open( dataset_dir + '-name2id.json', 'w') as w:
            json.dump(name2id, w)

        data = get_one_stage_data( dataset_dir, G )
        return data

def get_cell2feat( featFile ):
    cell_2_feats = {}
    with open( featFile, 'r' ) as r:
        while 1:
            line = r.readline()
            if not line: break
            line = line.strip().split(',')
            if "" in line: continue
            line[4] = "1" if line[4] == "true" else "0"
            feat = list(map(float, line[2:]))
            assert line[0] not in cell_2_feats
            cell_2_feats[line[0]] = feat
    return cell_2_feats

