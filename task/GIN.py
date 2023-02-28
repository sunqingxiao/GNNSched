import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from task.common import *


class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(GIN, self).__init__()
        self.n_layers = n_layers
        self.convlayers = nn.ModuleList()
        self.bnlayers = nn.ModuleList()

        # input layer
        self.convlayers.append(GINConv(Sequential(Linear(in_feats, n_hidden), ReLU(), Linear(n_hidden, n_hidden))))
        self.bnlayers.append(torch.nn.BatchNorm1d(n_hidden))

        for i in range(n_layers - 2): # hidden layers
            self.convlayers.append(GINConv(Sequential(Linear(n_hidden, n_hidden), ReLU(), Linear(n_hidden, n_hidden))))
            self.bnlayers.append(torch.nn.BatchNorm1d(n_hidden))

        # output layer
        self.fc1 = Linear(n_hidden, n_hidden)
        self.fc2 = Linear(n_hidden, n_classes)

    def forward(self, features, edge_index):
        x = features
        for i, layer in enumerate(self.convlayers):  
            x = F.relu(layer(x, edge_index))
            x = self.bnlayers[i](x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)


def partition_model(model):
    group_list = []
    group_list = [[child] for child in model.children()]
    return group_list


def get_comp_size(graph, para_shape_list):
    num_nodes, num_edges = graph.features.shape[0], graph.edge_index.shape[1]

    edge_index_size = calc_pad(graph.edge_index.nelement() * graph.edge_index.element_size())
    feat_size = calc_pad(graph.features.nelement() * graph.features.element_size())
    label_size = calc_pad(graph.labels.nelement() * graph.labels.element_size())

    comp_active_bytes = edge_index_size + feat_size + label_size
    comp_peak_bytes = comp_active_bytes

    gin_counter = 0
    for shape in para_shape_list[0]: # GINConv Layers
        if len(shape) == 2 and gin_counter % 4 == 0: # 2-D weights
            aggr_output_size = calc_pad(num_nodes * shape[1] * 4) # features after aggregation

            # propagate node features to edges
            edge_feat_size = calc_pad(num_edges * shape[1] * 4)
            comp_active_bytes += edge_feat_size + aggr_output_size
            if comp_active_bytes > comp_peak_bytes:
                comp_peak_bytes = comp_active_bytes
            comp_active_bytes -= (edge_feat_size + aggr_output_size) # only save features

            # outputs - Sequential(Linear, Relu, Linear)
            lin_output_size = calc_pad(num_nodes * shape[0] * 4)

            if gin_counter == 0: 
                comp_active_bytes += lin_output_size * 3
            else:  comp_active_bytes += lin_output_size * 2

            # BatchNorm output
            comp_active_bytes += lin_output_size
            if comp_active_bytes > comp_peak_bytes:
                comp_peak_bytes = comp_active_bytes

            if gin_counter == 0: 
                comp_active_bytes -= lin_output_size * 4
            else:  comp_active_bytes -= lin_output_size * 3

        gin_counter += 1

    # output - fc1 layer
    fc1_output_size = calc_pad(num_nodes * para_shape_list[2][0][0] * 4)

    # output - fc2 layer
    fc2_output_size = calc_pad(num_nodes * para_shape_list[3][0][0] * 4)
    
    comp_active_bytes += (fc1_output_size  + fc2_output_size) * 2
    if comp_active_bytes > comp_peak_bytes:
        comp_peak_bytes = comp_active_bytes

    comp_active_bytes -= (fc1_output_size  + fc2_output_size) * 2
   
    # multiply the threshold to ensure safe memory
    comp_peak_bytes = calc_pad(int(comp_peak_bytes * comp_ratio))

    return comp_peak_bytes
