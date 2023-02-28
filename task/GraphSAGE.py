import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from task.common import *


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden)) # input layer
        for i in range(n_layers - 2): # hidden layers
            self.layers.append(SAGEConv(n_hidden, n_hidden))
        self.layers.append(SAGEConv(n_hidden, n_classes)) # output layer

    def forward(self, features, edge_index):
        x = features
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, edge_index))

        return F.log_softmax(x, dim=-1)


def partition_model(model):
    group_list = [[child] for child in model.children()]
    return group_list


def get_comp_size(graph, para_shape_list):
    num_nodes, num_edges = graph.features.shape[0], graph.edge_index.shape[1]

    edge_index_size = calc_pad(graph.edge_index.nelement() * graph.edge_index.element_size())
    feat_size = calc_pad(graph.features.nelement() * graph.features.element_size())
    label_size = calc_pad(graph.labels.nelement() * graph.labels.element_size())

    comp_active_bytes = edge_index_size + feat_size + label_size
    comp_peak_bytes = comp_active_bytes
   
    sage_counter = 0
    for shape in para_shape_list[0]: # assuming float32 type (output/features)
        if len(shape) == 2 and sage_counter % 3 == 0: # 2-D weights
            aggr_output_size = calc_pad(num_nodes * shape[1] * 4) # features after aggregation

            # propagate node features to edges
            edge_feat_size = calc_pad(num_edges * shape[1] * 4)
            comp_active_bytes += edge_feat_size + aggr_output_size 

            if comp_active_bytes > comp_peak_bytes:
                comp_peak_bytes = comp_active_bytes
            comp_active_bytes -= edge_feat_size # only save features

            # lin_r and lin_l (linear layers) output
            lin_output_size = calc_pad(num_nodes * shape[0] * 4)

            comp_active_bytes += lin_output_size * 2
            if comp_active_bytes > comp_peak_bytes:
                comp_peak_bytes = comp_active_bytes
            comp_active_bytes -= (lin_output_size * 2 + aggr_output_size)
        sage_counter += 1

    # multiply the threshold to ensure safe memory
    comp_peak_bytes = calc_pad(int(comp_peak_bytes * comp_ratio))

    return comp_peak_bytes
