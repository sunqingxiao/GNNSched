import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch_geometric.nn import GCNConv

from task.common import *
from task.GCN import *

def import_func(model, graph):
    def test(model, graph):
        print(threading.currentThread().getName(),
            'GCN inference {} >>>>>>>>>>'.format(graph.name), time.time())
        model = model.cuda()
        features = graph.features.cuda(non_blocking=True)
        labels = graph.labels.cuda(non_blocking=True)
        edge_index = graph.edge_index.to(gpu, non_blocking=True)
        start = time.time()
        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index)
        infertime = time.time() - start
        return infertime

    return test


def import_model(data, graph):
    # feats and classes of graph data
    feat_dim, num_classes = 32, 2
    for idata, ifeat, iclasses in gnndatasets:
        if idata == data:
            feat_dim, num_classes = ifeat, iclasses
            break

    # load graph data
    name, edge_index, features, labels = load_graph_data(data, graph)
    graph = GraphSummary(name, edge_index, features, labels)

    model = GCN(feat_dim, hidden_dim, num_classes, num_layers, 0.5)
    model.load_state_dict(torch.load('../data/model/GCN_{}_weights.pth'.format(name)))

    # set full name to disguish
    FULL_NAME = 'GCN_{}'.format(data)
    set_fullname(model, FULL_NAME)

    return model, graph


def import_task(data, graph):
    model, graph = import_model(data, graph)
    func = import_func(model, graph)
    group_list = partition_model(model)

    return model, func, graph


def import_parameters(data, graph):
    model, graph = import_model(data, graph)
    group_list = partition_model(model)
    para_shape_list = [group_to_para_shape(group) for group in group_list]
    comp_total_bytes = get_comp_size(graph, para_shape_list)
    return comp_total_bytes


def get_para_size(data, graph):
    model, graph = import_model(data, graph)
    group_list = partition_model(model)
    batch_list = [group_to_batch(group) for group in group_list]
    para_shape_list = [group_to_para_shape(group) for group in group_list]
    
    para_total_bytes = 0
    for param, mod_list in batch_list:
        if param is None:
            continue
        else:
            para_total_bytes += param.element_size() * param.nelement()
    comp_total_bytes = get_comp_size(graph, para_shape_list)

    return para_total_bytes, comp_total_bytes
