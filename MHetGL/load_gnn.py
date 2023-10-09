import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
import math

"""
Load model from PyG: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
"""

class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, drop):
        super().__init__()

        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=True))

        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True))

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            # The last layer has no activation
            if i != self.num_layers - 1:
                x = self.act(x)

        return x

class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, drop):
        super().__init__()

        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr='mean', root_weight=True))

        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean', root_weight=True))

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            # The last layer has no activation
            if i != self.num_layers - 1:
                x = self.act(x)

        return x

class GAT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, drop):
        super().__init__()

        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, add_self_loops=True, heads=1))

        for _ in range(1, num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, add_self_loops=True, heads=1))

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            # The last layer has no activation
            if i != self.num_layers - 1:
                x = self.act(x)

        return x

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = nn.Parameter(torch.FloatTensor(1))
        stdv_eps = 0.21 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)
    def forward(self, adj, x):
        v = (self.eps) * torch.diag(adj)
        mask = torch.diag(torch.ones_like(v))
        adj = mask * torch.diag(v) + (1. - mask) * adj
        x = torch.mm(adj, x)
        return x

class DualGAT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, drop):
        super().__init__()

        self.num_layers = num_layers

        self.curv_trans = nn.ModuleList()
        for _ in range(num_layers): self.curv_trans.append(Transform())
        # self.curv_tran = CurvTransform()
        self.gdv_trans = nn.ModuleList()
        for _ in range(num_layers): self.gdv_trans.append(Transform())
        
        self.convs_1 = nn.ModuleList()
        self.convs_1.append(GATConv(input_dim, hidden_dim, add_self_loops=True, heads=1))

        for _ in range(1, num_layers):
            self.convs_1.append(GATConv(hidden_dim, hidden_dim, add_self_loops=True, heads=1))

        self.convs_2 = nn.ModuleList()
        self.convs_2.append(GATConv(input_dim, hidden_dim, add_self_loops=True, heads=1))

        for _ in range(1, num_layers):
            self.convs_2.append(GATConv(hidden_dim, hidden_dim, add_self_loops=True, heads=1))

        # self.fc = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fcs = nn.ModuleList()
        for _ in range(num_layers):
            self.fcs.append(nn.Linear(2 * hidden_dim, hidden_dim))

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, data):

        x, edge_index, gdv_edge_index = data.x, data.edge_index, data.gdv_edge_index
        curvs_adj, gdv_adj = data.curvs, data.gdv_mat

        for i in range(self.num_layers):
            x_1 = self.act(self.convs_1[i](self.gdv_trans[i](gdv_adj, x), gdv_edge_index))
            x_2 = self.act(self.convs_2[i](self.curv_trans[i](curvs_adj, x), edge_index))

            x = torch.cat([x_1, x_2], dim=-1)
            # x = self.act(self.fcs[i](x))
            x = self.fcs[i](x)

        return x



def init_model(model_name: str, num_layers: int, input_dim:int, hidden_dim: int, mode='vanilla', drop=0.):

    # Load vanilla GNN
    if mode == 'vanilla':
        if model_name == 'GCN':
            model = GCN(num_layers, input_dim, hidden_dim, drop)
        elif model_name == 'GraphSAGE':
            model = GraphSAGE(num_layers, input_dim, hidden_dim, drop)
        elif model_name == 'GAT':
            model = GAT(num_layers, input_dim, hidden_dim, drop)
        else: raise ValueError('Model {} not supported.'.format(model_name))
    elif mode == 'embed':
        if model_name == 'GCN':
            model = DualGCN(num_layers, input_dim, hidden_dim, drop)
        elif model_name == 'GraphSAGE':
            model = DualGraphSAGE(num_layers, input_dim, hidden_dim, drop)
        elif model_name == 'GAT':
            model = DualGAT(num_layers, input_dim, hidden_dim, drop)
        else: raise ValueError('Model {} not supported.'.format(model_name))
    else: raise ValueError('Mode {} not supported.'.format(mode))

    return model