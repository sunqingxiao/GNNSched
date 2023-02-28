import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from task.common import *


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_feats, n_hidden)) # input layer
        for i in range(n_layers - 2): # hidden layers
            self.layers.append(GCNConv(n_hidden, n_hidden))
        self.layers.append(GCNConv(n_hidden, n_classes)) # output layer

    def forward(self, features, edge_index):
        x = features
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, edge_index))

        return F.log_softmax(x, dim=-1)


def partition_model(model):
    group_list = [[child] for child in model.children()]
    return group_list


def get_comp_size(graph, para_shape_list):
    num_nodes, num_edges, feat_length = graph.features.shape[0], graph.edge_index.shape[1], graph.features.shape[1]

    edge_index_size = calc_pad(graph.edge_index.nelement() * graph.edge_index.element_size())
    edge_weight_size = calc_pad(num_edges * 4) # float32
    feat_size = calc_pad(graph.features.nelement() * graph.features.element_size())
    label_size = calc_pad(graph.labels.nelement() * graph.labels.element_size())
    
    # add self loops from PyG
    add_loop_num_edges = num_nodes + num_edges
    loop_edge_index_size = calc_pad(edge_index_size * add_loop_num_edges / num_edges)
    loop_edge_weight_size = calc_pad(edge_weight_size * add_loop_num_edges / num_edges)
    loop_edge_size = loop_edge_index_size + loop_edge_weight_size

    comp_active_bytes = edge_index_size + edge_weight_size + feat_size + label_size + loop_edge_size
    comp_peak_bytes = comp_active_bytes
    
    for shape in para_shape_list[0]: # assuming float32 type (output/features)
        if len(shape) == 2: # 2-D weights
            output_size = calc_pad(num_nodes * shape[1] * 4)
            comp_active_bytes += output_size  # after GCN layer

            # propagate node features to edges
            edge_feat_size = calc_pad(add_loop_num_edges * 2 * shape[1] * 4)

            comp_active_bytes += edge_feat_size + output_size # during propagation
            if comp_active_bytes > comp_peak_bytes:
                comp_peak_bytes = comp_active_bytes
            comp_active_bytes -= (edge_feat_size + 2 * output_size) # only save features

    # multiply the threshold to ensure safe memory
    comp_peak_bytes = calc_pad(int(comp_peak_bytes * comp_ratio))

    return comp_peak_bytes
