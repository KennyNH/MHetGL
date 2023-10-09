import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as scio
from scipy.linalg import fractional_matrix_power
from pathlib import Path
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Flickr, CitationFull
from torch_geometric.data import Data
from torch_geometric.utils import contains_self_loops, add_remaining_self_loops, to_undirected, is_undirected, contains_isolated_nodes, to_networkx
from pygod.utils import load_data
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

"""
Load data from PyGOD or PyG
"""

def get_weibo(path):

    # pyg_data = load_data('weibo') # <class 'torch_geometric.data.data.Data'>
    pyg_data = torch.load(path + 'Weibo/data.pt')
    data_dict = pyg_data.to_dict()
    node_feats = data_dict['x']
    edge_index = data_dict['edge_index']
    labels = data_dict['y'] 

    return node_feats, edge_index, labels

def get_book():

    pyg_data = load_data('book') # <class 'torch_geometric.data.data.Data'>
    data_dict = pyg_data.to_dict()
    node_feats = data_dict['x']
    edge_index = data_dict['edge_index']
    labels = data_dict['y'] 

    return node_feats, edge_index, labels

def get_reddit(path):

    # pyg_data = load_data('reddit') # <class 'torch_geometric.data.data.Data'>
    pyg_data = torch.load(path + 'Reddit/data.pt')
    data_dict = pyg_data.to_dict()
    node_feats = data_dict['x']
    edge_index = data_dict['edge_index']
    labels = data_dict['y'].type(torch.int64)

    return node_feats, edge_index, labels

def get_cora(path, flag, seed):
    
    pyg_data = Planetoid(path + 'Cora', 'Cora', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        data_dict = pyg_data.to_dict()
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = data_dict['y'] 
        ano_mask = labels == 6
        labels[ano_mask] = 1
        labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=15, n=15, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=225, k=100, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=150, k=100, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=12, n=12, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_amazon(path, flag, seed):
    
    pyg_data = Amazon(path + 'Amazon', 'Computers', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        data_dict = pyg_data.to_dict()
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = data_dict['y'] 
        ano_mask = labels == 9
        labels[ano_mask] = 1
        labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=32, n=32, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=1024, k=100, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=500, k=100, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=25, n=25, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_citeseer(path, flag, seed):
    
    pyg_data = Planetoid(path + 'Citeseer', 'CiteSeer', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        data_dict = pyg_data.to_dict()
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = data_dict['y'] 
        ano_mask = labels == 0
        labels[ano_mask] = 1
        labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=18, n=18, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=324, k=100, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=150, k=100, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=13, n=13, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_flickr(path, flag, seed):
    
    pyg_data = Flickr(path + 'Flickr', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        data_dict = pyg_data.to_dict()
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = data_dict['y'] 
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=80, n=80, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=6400, k=100, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_pubmed(path, flag, seed):

    pyg_data = Planetoid(path + 'PubMed', 'PubMed', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        data_dict = pyg_data.to_dict()
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = data_dict['y'] 
        ano_mask = labels == 6
        labels[ano_mask] = 1
        labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=44, n=44, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=1971, k=500, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=985, k=300, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=31, n=31, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_photo(path, flag, seed):
    
    pyg_data = Amazon(path + 'Photo', 'Photo', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        pass
        # data_dict = pyg_data.to_dict()
        # node_feats = data_dict['x']
        # edge_index = data_dict['edge_index']
        # labels = data_dict['y'] 
        # ano_mask = labels == 9
        # labels[ano_mask] = 1
        # labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=27, n=27, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=750, k=75, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=325, k=75, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=18, n=18, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_ml(path, flag, seed):
    
    pyg_data = CitationFull(path + 'ML', 'Cora_ML', transform=T.NormalizeFeatures())[0]

    if flag == 'raw': 
        pass
        # data_dict = pyg_data.to_dict()
        # node_feats = data_dict['x']
        # edge_index = data_dict['edge_index']
        # labels = data_dict['y'] 
        # ano_mask = labels == 9
        # labels[ano_mask] = 1
        # labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=16, n=16, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=256, k=100, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=150, k=75, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=12, n=12, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def get_blog(path, flag, seed):
 
    dataFile = path + 'BlogCatalog/BlogCatalog.mat'
    data = scio.loadmat(dataFile)
    x, adj = data['Attributes'].toarray(), data['Network'].tocoo()
    edge_index = np.vstack([adj.row, adj.col])
    pyg_data = Data(x=torch.from_numpy(x).float(), edge_index=torch.from_numpy(edge_index))

    if flag == 'raw': 
        pass
        # data_dict = pyg_data.to_dict()
        # node_feats = data_dict['x']
        # edge_index = data_dict['edge_index']
        # labels = data_dict['y'] 
        # ano_mask = labels == 9
        # labels[ano_mask] = 1
        # labels[~ano_mask] = 0
    elif flag == 'structural':
        data_dict, ys = gen_structural_outliers(pyg_data, m=22, n=22, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ys
    elif flag == 'contextual':
        data_dict, ya = gen_contextual_outliers(pyg_data, n=500, k=100, random_state=seed)
        node_feats = data_dict['x']
        edge_index = data_dict['edge_index']
        labels = ya
    elif flag == 'syn':
        data_dict_contex, y_contex = gen_contextual_outliers(pyg_data, n=256, k=75, random_state=seed)
        data_dict_struct, y_struct = gen_structural_outliers(pyg_data, m=16, n=16, random_state=seed)
        node_feats = data_dict_contex['x']
        edge_index = data_dict_struct['edge_index']
        labels = y_contex + y_struct
        labels[labels > 1] = 1
    else: raise ValueError('Flag {} not supported.'.format(flag))

    return node_feats, edge_index, labels

def sim(z1, z2):
    """
    Calculate pair-wise similarity.
    """
    assert len(z1.shape) == 2 and len(z2.shape) == 2
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)
    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t())
    sim_mat = dot_numerator / dot_denominator # shape: [num_z1, num_z2]

    sim_mat = torch.where(torch.isinf(sim_mat), torch.full_like(sim_mat, 0), sim_mat)
    sim_mat = torch.where(torch.isnan(sim_mat), torch.full_like(sim_mat, 0), sim_mat)

    return sim_mat

def split_data(labels, split_ratio_list, ano_ratio, seed):
    """
    Split normal data into train, val and test. 
    Select anomalies for val and test and maintain the anomaly ratio of entire graph.
    """

    # filter normal and anomalous nodes
    nor_ids = torch.where(labels==0)[0].numpy() # indices of normal nodes 
    ano_ids = torch.where(labels==1)[0].numpy() # indices of anomalous nodes
    
    # shuffle ids
    np.random.shuffle(nor_ids)
    np.random.shuffle(ano_ids)

    # initialize mask
    num_nodes = len(labels)
    train_mask = np.zeros(num_nodes, dtype='bool')
    val_mask = np.zeros(num_nodes, dtype='bool')
    test_mask = np.zeros(num_nodes, dtype='bool')

    # split normal nodes
    num_nor_nodes = len(nor_ids)
    num_val = int(num_nor_nodes * split_ratio_list[1])
    num_test = int(num_nor_nodes * split_ratio_list[2])

    val_mask[nor_ids[:num_val]] = 1
    test_mask[nor_ids[num_val:num_val+num_test]] = 1
    train_mask[nor_ids[num_val+num_test:]] = 1

    # select anomalous nodes
    num_val = int(num_val * ano_ratio) + 1
    num_test = int(num_test * ano_ratio) + 1

    val_mask[ano_ids[:num_val]] = 1
    test_mask[ano_ids[num_val:num_val+num_test]] = 1

    # numpy to tensor
    train_mask = torch.from_numpy(train_mask)
    val_mask = torch.from_numpy(val_mask)
    test_mask = torch.from_numpy(test_mask)

    return train_mask, val_mask, test_mask

def get_curvature(path, data_name, edge_index, num_nodes, device, mode):
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True, remove_self_loops=True)

    orc = OllivierRicci(G, alpha=0.5, verbose="TRACE")
    orc.compute_ricci_curvature()
    G_orc = orc.G.copy()  # save an intermediate result
    ricci_curvtures = list(nx.get_edge_attributes(G_orc, 'ricciCurvature').values())
    nx_edges = list(G_orc.edges())

    if mode == 'v1':
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        curvs += torch.diag_embed(torch.ones(num_nodes))
    elif mode == 'v2':
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs.flatten()[softmax_mask.flatten()] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))
    elif mode == 'v3':
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))
    elif mode == 'v4':
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs.flatten()[softmax_mask.flatten()] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1)) 
    elif mode == 'v5':
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

    elif mode == 'v6': # sum-mean
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(A_tilde_hat)

    elif mode == 'v7': # softmax
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        # curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(A_tilde_hat)

    elif mode == 'v8': # no gauss prob
        ricci_curvtures = torch.tensor(ricci_curvtures)
        # miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        # ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        # curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(A_tilde_hat)

    elif mode == 'v9': # cancel normalization, softmax
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        # curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        # row_sum = np.array(np.sum(weight, axis=1))
        # degree_matrix = np.matrix(np.diag(row_sum+1))

        # D = fractional_matrix_power(degree_matrix, -0.5)
        # A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)

    elif mode == 'v10': # cancel normalization, sum-mean
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        softmax_mask = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense())) == 0
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        # row_sum = np.array(np.sum(weight, axis=1))
        # degree_matrix = np.matrix(np.diag(row_sum+1))

        # D = fractional_matrix_power(degree_matrix, -0.5)
        # A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)

    elif mode == 'v11': # add nei_num
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T)
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)
    
    elif mode == 'v12': # add nei_num, normalize adj instead of curv_adj
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T)
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(_A.detach().numpy(), axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)
    
    elif mode == 'v13': # add nei_num coeff=2
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)
    
    elif mode == 'v14': # add nei_num coeff=0.5
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 0.5
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)

    elif mode == 'v15': # prob function 1-
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = (1 - torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2)) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)
    
    elif mode == 'v-3': # sigmoid
        ricci_curvtures = torch.tensor(ricci_curvtures)
        ricci_curvtures = torch.sigmoid(ricci_curvtures)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        curvs += torch.diag_embed(torch.ones(num_nodes))

    elif mode == 'v-4': # sigmoid -> gauss in v13
        ricci_curvtures = torch.tensor(ricci_curvtures)
        ricci_curvtures = torch.sigmoid(ricci_curvtures)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)

    elif mode == 'v-5': # softmax -> gauss in v13
        ricci_curvtures = torch.tensor(ricci_curvtures)
        ricci_curvtures = torch.softmax(ricci_curvtures, -1)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)

    elif mode == 'v-6': # v13 correct
        ricci_curvtures = torch.tensor(ricci_curvtures)
        miu, std = torch.mean(ricci_curvtures), torch.std(ricci_curvtures)
        ricci_curvtures = torch.exp(- (ricci_curvtures - miu) ** 2 / 2 / std ** 2) / (math.sqrt(2 * math.pi) * std)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        curvs = curvs * nei_num_matrix
        # curvs[softmax_mask] = -1e15 # share storage
        # curvs = torch.softmax(curvs, dim=-1) # softmax
        curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        D = np.nan_to_num(D, nan=0)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(A_tilde_hat)
    
    elif mode == 'v-7': # node-wise softmax
        ricci_curvtures = torch.tensor(ricci_curvtures)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        curvs = curvs * nei_num_matrix
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        # curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        D = np.nan_to_num(D, nan=0)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)

    elif mode == 'v-8': # del nei num in v-7
        ricci_curvtures = torch.tensor(ricci_curvtures)
        curvs = torch.zeros(num_nodes, num_nodes)
        for i in range(len(nx_edges)):
            curvs[nx_edges[i][0], nx_edges[i][1]] = ricci_curvtures[i]
            curvs[nx_edges[i][1], nx_edges[i][0]] = ricci_curvtures[i]
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(G).todense()))
        softmax_mask = _A == 0
        _A_loop = _A + torch.diag(torch.ones(len(_A)))
        nei_num_matrix = torch.mm(_A_loop, _A_loop.T) ** 2.
        # curvs = curvs * nei_num_matrix
        curvs[softmax_mask] = -1e15 # share storage
        curvs = torch.softmax(curvs, dim=-1) # softmax
        # curvs /= curvs.sum(dim=-1, keepdim=True)
        curvs[~softmax_mask] += 1.
        curvs += torch.diag_embed(torch.sum(curvs, dim=-1))

        weight = curvs.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        D = np.nan_to_num(D, nan=0)
        A_tilde_hat = D.dot(weight).dot(D)

        curvs = torch.FloatTensor(weight)


    return curvs

def init_data(path: str, data_name: str, setting: str, split_ratio_list: list, device: str, seed=3407, curv_mode='v13'):

    # Load organic data
    if data_name == 'weibo':
        node_feats, edge_index, labels = get_weibo(path)
    elif data_name == 'reddit':
        node_feats, edge_index, labels = get_reddit(path)
    elif data_name == 'book':
        node_feats, edge_index, labels = get_book()
    
    # Load injected data

    # Cora
    elif data_name == 'cora-contex':
        node_feats, edge_index, labels = get_cora(path, flag='contextual', seed=seed)
    elif data_name == 'cora-struct':
        node_feats, edge_index, labels = get_cora(path, flag='structural', seed=seed)
    elif data_name == 'cora-raw':
        node_feats, edge_index, labels = get_cora(path, flag='raw', seed=seed)
    elif data_name == 'cora-syn':
        node_feats, edge_index, labels = get_cora(path, flag='syn', seed=seed)
    
    # Amazon (Computers)
    elif data_name == 'amazon-contex':
        node_feats, edge_index, labels = get_amazon(path, flag='contextual', seed=seed)
    elif data_name == 'amazon-struct':
        node_feats, edge_index, labels = get_amazon(path, flag='structural', seed=seed)
    elif data_name == 'amazon-raw':
        node_feats, edge_index, labels = get_amazon(path, flag='raw', seed=seed)
    elif data_name == 'amazon-syn':
        node_feats, edge_index, labels = get_amazon(path, flag='syn', seed=seed)

    # Flickr
    elif data_name == 'flickr-contex':
        node_feats, edge_index, labels = get_flickr(path, flag='contextual', seed=seed)
    elif data_name == 'flickr-struct':
        node_feats, edge_index, labels = get_flickr(path, flag='structural', seed=seed)
    elif data_name == 'flickr-raw':
        node_feats, edge_index, labels = get_flickr(path, flag='raw', seed=seed)

    # Citeseer
    elif data_name == 'citeseer-contex':
        node_feats, edge_index, labels = get_citeseer(path, flag='contextual', seed=seed)
    elif data_name == 'citeseer-struct':
        node_feats, edge_index, labels = get_citeseer(path, flag='structural', seed=seed)
    elif data_name == 'citeseer-raw':
        node_feats, edge_index, labels = get_citeseer(path, flag='raw', seed=seed)
    elif data_name == 'citeseer-syn':
        node_feats, edge_index, labels = get_citeseer(path, flag='syn', seed=seed)

    # ML
    elif data_name == 'ml-contex':
        node_feats, edge_index, labels = get_ml(path, flag='contextual', seed=seed)
    elif data_name == 'ml-struct':
        node_feats, edge_index, labels = get_ml(path, flag='structural', seed=seed)
    elif data_name == 'ml-raw':
        node_feats, edge_index, labels = get_ml(path, flag='raw', seed=seed)
    elif data_name == 'ml-syn':
        node_feats, edge_index, labels = get_ml(path, flag='syn', seed=seed)

    # Photo
    elif data_name == 'photo-contex':
        node_feats, edge_index, labels = get_photo(path, flag='contextual', seed=seed)
    elif data_name == 'photo-struct':
        node_feats, edge_index, labels = get_photo(path, flag='structural', seed=seed)
    elif data_name == 'photo-raw':
        node_feats, edge_index, labels = get_photo(path, flag='raw', seed=seed)
    elif data_name == 'photo-syn':
        node_feats, edge_index, labels = get_photo(path, flag='syn', seed=seed)

    # BlogCatalog
    elif data_name == 'blog-contex':
        node_feats, edge_index, labels = get_blog(path, flag='contextual', seed=seed)
    elif data_name == 'blog-struct':
        node_feats, edge_index, labels = get_blog(path, flag='structural', seed=seed)
    elif data_name == 'blog-raw':
        node_feats, edge_index, labels = get_blog(path, flag='raw', seed=seed)
    elif data_name == 'blog-syn':
        node_feats, edge_index, labels = get_blog(path, flag='syn', seed=seed)

    # PubMed
    elif data_name == 'pubmed-contex':
        node_feats, edge_index, labels = get_pubmed(path, flag='contextual', seed=seed)
    elif data_name == 'pubmed-struct':
        node_feats, edge_index, labels = get_pubmed(path, flag='structural', seed=seed)
    elif data_name == 'pubmed-raw':
        node_feats, edge_index, labels = get_pubmed(path, flag='raw', seed=seed)
    elif data_name == 'pubmed-syn':
        node_feats, edge_index, labels = get_pubmed(path, flag='syn', seed=seed)

    else: raise ValueError('Dataset {} not supported.'.format(data_name))

    # Preprocess
    edge_index, _ = add_remaining_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    # Display some basic info
    print('######## Basic of {} ########'.format(data_name))
    assert node_feats.dtype is torch.float32 and edge_index.dtype is torch.int64 and labels.dtype is torch.int64, \
        "Only torch.float32 and torch.int64 are supported {}-{}-{}.".format(node_feats.dtype, edge_index.dtype, labels.dtype)
    print('Num of Nodes:', node_feats.shape[0])
    print('Num of edges:', edge_index.shape[-1])
    print('Node feat dim:', node_feats.shape[-1])

    # print('Whether the graph contain self-loops:', contains_self_loops(edge_index))
    # print('Whether the graph is undirected:', is_undirected(edge_index))
    # print('Whether the graph contain isolated nodes:', contains_isolated_nodes(edge_index))

    assert len(torch.unique(labels)) == 2 and labels.max() == 1 and labels.min() == 0, \
        "Labels: {}-{}".format(labels.max(), labels.min())
    # print('Node label:', list(torch.unique(labels).numpy()))
    print('Num of anomalies:', labels.sum().item())
    ano_ratio = labels.sum() / node_feats.shape[0]
    print('Anomaly ratio: {:.2f}%'.format(ano_ratio * 100))
    print('###### Loading {} done ######\n'.format(data_name))

    # Evaluation Setting
    print('######## Preprocessing ########')
    print('Evaluation setting:', setting)
    if setting == 'clean':
        train_mask, val_mask, test_mask = split_data(labels, split_ratio_list, ano_ratio=ano_ratio, seed=seed)
    elif setting == 'polluted':
        train_mask = torch.from_numpy(np.ones(len(labels), dtype='bool'))
        val_mask = torch.from_numpy(np.ones(len(labels), dtype='bool'))
        test_mask = torch.from_numpy(np.ones(len(labels), dtype='bool'))
    elif setting == 'few-shot':
        pass
    elif setting == 'semi-supervised':
        pass
    else: raise ValueError('Setting {} not supported.'.format(setting))

    assert train_mask.dtype is torch.bool
    print('Num of train-val-test instances: {}-{}-{}'.format(train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()))
    print('############ Done #############\n')

    # Calculate Graphlet Degree Vectors (GDV) with orca (C++)
    nx_graph = to_networkx(Data(edge_index=edge_index, num_nodes=node_feats.shape[0]), to_undirected=True, remove_self_loops=True)
    print(nx_graph, type(nx_graph))
    gdv_path = path + 'gdv/' + data_name + '/'
    orca_path = path + 'gdv/orca/'
    edge_list_path = gdv_path + "{}.edgelist".format(data_name)
    if not os.path.exists(gdv_path + '{}.gdv'.format(data_name)):
        Path(gdv_path).mkdir(parents=True, exist_ok=True)
        nx.write_edgelist(nx_graph, edge_list_path, data=False)
        os.system("sed -i '1i\{} {}' ".format(nx_graph.number_of_nodes(), nx_graph.number_of_edges()) + edge_list_path)
        os.system(orca_path + 'orca.exe 5 ' + edge_list_path + ' ' + gdv_path + '{}.gdv'.format(data_name))
    gdvs = torch.from_numpy(np.loadtxt(gdv_path + '{}.gdv'.format(data_name), encoding="ascii")).float() # [num_nodes, 73]

    # Construct GDV-based Adjacency matrix 
    # if not os.path.exists(gdv_path + 'edge_index_weighted_3.pt'):
    #     threshold = 1 if data_name == 'reddit' else 0.9999
    #     gdv_sim_mat = sim(gdvs, gdvs) # [num_nodes, num_nodes]
    #     gdv_norms = torch.norm(gdvs, dim=-1) # [num_nodes,]
    #     reset_inds = torch.diag_embed((gdv_norms > 1.).float()) # [num_nodes, num_nodes]
    #     gdv_sim_mat = torch.mm(torch.mm(reset_inds, gdv_sim_mat), reset_inds) # delete elements related to nodes with small norms
    #     gdv_sim_mat[gdv_sim_mat < threshold] = 0.
    #     gdv_sim_mat -= torch.diag_embed(torch.diag(gdv_sim_mat)) # delete diag
    #     gdv_sim_mat += torch.diag_embed(torch.ones(gdv_sim_mat.shape[0])) # set diag as 1
    #     gdv_edge_index = gdv_sim_mat.to_sparse().indices()
    #     n_n, n_e = gdv_sim_mat.shape[0], gdv_edge_index.shape[1]

    #     _A = torch.from_numpy(np.array(nx.adjacency_matrix(nx_graph).todense()))
    #     _D = _A.sum(dim=-1)
    #     _D_pair = _D.unsqueeze(-1) + _D.unsqueeze(0)
    #     gdv_sim_mat = _D_pair * gdv_sim_mat

    #     softmax_mask = _A == 0
    #     # gdv_sim_mat[softmax_mask] = -1e15 # share storage
    #     # gdv_sim_mat = torch.softmax(gdv_sim_mat, dim=-1) # softmax
    #     gdv_sim_mat = gdv_sim_mat / gdv_sim_mat.sum(dim=-1, keepdim=True)
    #     gdv_sim_mat[~softmax_mask] += 1.
    #     gdv_sim_mat -= torch.diag_embed(torch.diag(gdv_sim_mat)) # delete diag
    #     gdv_sim_mat += torch.diag_embed(torch.sum(gdv_sim_mat, dim=-1))

    #     weight = gdv_sim_mat.detach().numpy()
    #     weight = np.nan_to_num(weight, nan=0)

    #     row_sum = np.array(np.sum(weight, axis=1))
    #     degree_matrix = np.matrix(np.diag(row_sum+1))

    #     D = fractional_matrix_power(degree_matrix, -0.5)
    #     D = np.nan_to_num(D, nan=0)
    #     A_tilde_hat = D.dot(weight).dot(D)

    #     gdv_mat = torch.FloatTensor(A_tilde_hat)
        
    #     torch.save(gdv_edge_index, gdv_path + 'edge_index_weighted_3.pt')
    #     torch.save(gdv_mat, gdv_path + 'gdv_mat_3.pt')
    #     torch.save(torch.tensor([(n_e - n_n) / 2, (n_e - n_n) / 2 / n_n ** 2]), gdv_path + 'basic_3.pt')
    # gdv_edge_index = torch.load(gdv_path + 'edge_index_weighted_3.pt')
    # gdv_mat = torch.load(gdv_path + 'gdv_mat_3.pt')
    # gdv_basics = torch.load(gdv_path + 'basic_3.pt')
    # print('Num of GDV-based edges:', gdv_basics[0].item())
    # print('Sparsity of GDV-based Adj:', gdv_basics[1].item())
    if not os.path.exists(gdv_path + 'edge_index_weighted_2.pt'):
        threshold = 1 if data_name == 'reddit' else 0.9999
        gdv_sim_mat = sim(gdvs, gdvs) # [num_nodes, num_nodes]
        gdv_norms = torch.norm(gdvs, dim=-1) # [num_nodes,]
        reset_inds = torch.diag_embed((gdv_norms > 1.).float()) # [num_nodes, num_nodes]
        gdv_sim_mat = torch.mm(torch.mm(reset_inds, gdv_sim_mat), reset_inds) # delete elements related to nodes with small norms
        gdv_sim_mat[gdv_sim_mat < threshold] = 0.
        gdv_sim_mat -= torch.diag_embed(torch.diag(gdv_sim_mat)) # delete diag
        gdv_sim_mat += torch.diag_embed(torch.ones(gdv_sim_mat.shape[0])) # set diag as 1
        gdv_edge_index = gdv_sim_mat.to_sparse().indices()
        n_n, n_e = gdv_sim_mat.shape[0], gdv_edge_index.shape[1]

        _A = torch.from_numpy(np.array(nx.adjacency_matrix(nx_graph).todense()))
        _D = _A.sum(dim=-1)
        _D_pair = _D.unsqueeze(-1) + _D.unsqueeze(0)
        gdv_sim_mat = _D_pair * gdv_sim_mat

        softmax_mask = _A == 0
        # gdv_sim_mat[softmax_mask] = -1e15 # share storage
        # gdv_sim_mat = torch.softmax(gdv_sim_mat, dim=-1) # softmax
        gdv_sim_mat = gdv_sim_mat / gdv_sim_mat.sum(dim=-1, keepdim=True)
        gdv_sim_mat[~softmax_mask] += 1.
        gdv_sim_mat -= torch.diag_embed(torch.diag(gdv_sim_mat)) # delete diag
        gdv_sim_mat += torch.diag_embed(torch.sum(gdv_sim_mat, dim=-1))

        weight = gdv_sim_mat.detach().numpy()
        weight = np.nan_to_num(weight, nan=0)

        row_sum = np.array(np.sum(weight, axis=1))
        degree_matrix = np.matrix(np.diag(row_sum+1))

        D = fractional_matrix_power(degree_matrix, -0.5)
        D = np.nan_to_num(D, nan=0)
        A_tilde_hat = D.dot(weight).dot(D)

        gdv_mat = torch.FloatTensor(weight)
        
        torch.save(gdv_edge_index, gdv_path + 'edge_index_weighted_2.pt')
        torch.save(gdv_mat, gdv_path + 'gdv_mat_2.pt')
        torch.save(torch.tensor([(n_e - n_n) / 2, (n_e - n_n) / 2 / n_n ** 2]), gdv_path + 'basic_2.pt')
    gdv_edge_index = torch.load(gdv_path + 'edge_index_weighted_2.pt')
    gdv_mat = torch.load(gdv_path + 'gdv_mat_2.pt')
    gdv_basics = torch.load(gdv_path + 'basic_2.pt')
    print('Num of GDV-based edges:', gdv_basics[0].item())
    print('Sparsity of GDV-based Adj:', gdv_basics[1].item())

    # Compute curvature weights
    mode = curv_mode
    curv_path = path + 'curv_weights/' + data_name + '/'
    if not os.path.exists(curv_path + mode + '.pt'):
        curvs = get_curvature(path, data_name, edge_index, node_feats.shape[0], device, mode) # [num_nodes, num_nodes]
        curv_weights = torch.tensor([curvs[edge_index[0][i], edge_index[1][i]] for i in range(edge_index.shape[1])])
        Path(curv_path).mkdir(parents=True, exist_ok=True)
        torch.save(curv_weights, curv_path + mode + '.pt')
        torch.save(curvs, curv_path + mode + '_adj.pt')
    curv_weights = torch.load(curv_path + mode + '.pt')
    curvs = torch.load(curv_path + mode + '_adj.pt')
    
    _A = torch.from_numpy(np.array(nx.adjacency_matrix(nx_graph).todense()))
    _D = _A.sum(dim=-1)

    # Change device
    node_feats = node_feats.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    gdv_edge_index = gdv_edge_index.to(device)
    gdv_mat = gdv_mat.to(device)
    curvs = curvs.to(device)
    _D = _D.to(device)

    print()

    return Data(x=node_feats, edge_index=edge_index, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, 
                gdv_edge_index=gdv_edge_index, gdv_mat=gdv_mat, curvs=curvs, nx_graph=nx_graph, degs=_D)


if __name__ == '__main__':
    # # Debug
    # init_data('./', 'cora-struct', 0, 'polluted', [0.6, 0.1, 0.3])
    # init_data('./', 'weibo', 0, 'polluted', [0.6, 0.1, 0.3])
    # init_data('./', 'reddit', 0, 'polluted', [0.6, 0.1, 0.3])
    # init_data('./', 'amazon-struct', 0, 'polluted', [0.6, 0.1, 0.3])
    # init_data('./', 'amazon-raw', 0, 'polluted', [0.6, 0.1, 0.3])
    # init_data('./', 'cora-raw', 0, 'polluted', [0.6, 0.1, 0.3])

    # # Global substructure similarity property of anomalies
    # data = init_data('../../Data/', 'reddit', 'clean', [0.6, 0.1, 0.3], 'cpu')
    # edge_index = data.edge_index
    # true_labels = data.y.to(torch.bool)
    # ano_indices = torch.tensor(range(len(true_labels)))[true_labels]
    # nx_graph = data.nx_graph
    # _A = torch.from_numpy(np.array(nx.adjacency_matrix(nx_graph).todense()))

    # # pow2 = True
    # # for i in ano_indices:
    # #     labels = torch.zeros(len(true_labels), dtype=torch.bool)
    # #     labels[i] = True

    # #     indices = _A[labels].sum(0) + labels
    # #     indices[indices > 1] = 1
    # #     indices = indices.to(torch.bool)
    # #     if pow2:
    # #         indices = _A[indices].sum(0) + indices
    # #         indices[indices > 1] = 1
    # #         indices = indices.to(torch.bool)        
    # #     print(indices, indices.sum())
    # #     if indices.sum() > 100: continue
    # #     _A_filter = _A[indices][:, indices].numpy()
    # #     print(_A_filter, _A_filter.shape, _A_filter.sum())

    # #     scores = torch.zeros(len(labels))
    # #     scores[indices] = -1
    # #     scores[labels] = 1
    # #     scores = scores[indices]

    # #     induced_nx_graph = nx.from_numpy_array(_A_filter)

    # #     pos = nx.nx_pydot.pydot_layout(induced_nx_graph)
    # #     nx.draw(induced_nx_graph, pos, cmap=plt.cm.coolwarm, node_color=scores, node_size = 20, width=0.15, edgecolors='black', linewidths=0.5)
    # #     # nx.draw(induced_nx_graph, pos, cmap=plt.cm.coolwarm, node_color=label_values, node_size = 5, width=0.05)
    # #     sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    # #     sm.set_array([])
    # #     plt.colorbar(sm, shrink=0.8, label='Reddit', location='left')
    # #     plt.savefig('../../pictures/case_reddit_2/{}.png'.format(i), bbox_inches='tight', dpi=300)
    # #     plt.clf()


    # Curvature distribution and heterophily analysis 
    # dataset_list = ['cora-struct', 'cora-syn', 'citeseer-struct', 'citeseer-syn', 
    #                 'ml-struct', 'ml-syn', 'pubmed-struct', 'pubmed-syn', 'reddit']
    dataset_list = ['citeseer-syn', 'cora-syn', 'ml-syn', 'pubmed-syn', 'reddit']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams.update({'font.size': 9}) # loss weights
    plt.rcParams.update({'font.size': 13}) # others
    plt.rcParams["axes.labelweight"] = "bold"
    dpi = 500
    curv_mode = 'v1'
    for dataset in dataset_list:

        data = init_data('../../Data/', dataset, 'clean', [0.6, 0.1, 0.3], 'cpu', curv_mode=curv_mode)
        gdvs = torch.from_numpy(np.loadtxt('../../Data/gdv/' + dataset + '/' + '{}.gdv'.format(dataset), encoding="ascii")).float() # [num_nodes, 73]
        edge_index = data.edge_index
        true_labels = data.y.to(torch.bool)
        ano_indices = torch.tensor(range(len(true_labels)))[true_labels]
        nx_graph = data.nx_graph
        _A = torch.from_numpy(np.array(nx.adjacency_matrix(nx_graph).todense()))
        gdv_edge_index = data.gdv_edge_index
        curvs = data.curvs
        degs = _A.sum(-1).numpy()

        # nor_curvs = []
        # ano_curvs = []
        # flag = torch.zeros(curvs.shape)
        # heterophily_num = 0
        # total_num = 0 
        # for i in range(edge_index.shape[1]):
        #     v1, v2 = edge_index[0][i], edge_index[1][i]
        #     if flag[v1, v2]: continue
        #     flag[v1, v2] = 1
        #     flag[v2, v1] = 1
        #     if not true_labels[v1] and not true_labels[v2]: nor_curvs.append(curvs[v1, v2].item())
        #     else: 
        #         ano_curvs.append(curvs[v1, v2].item())
        #         if not (true_labels[v1] and true_labels[v2]): heterophily_num += 1
        #         total_num += 1
        # print('Raw heterophily ratio: {}/{}={}'.format(heterophily_num, total_num, heterophily_num / total_num))

        # flag = torch.zeros(curvs.shape)
        # heterophily_num = 0
        # total_num = 0 
        # for i in range(gdv_edge_index.shape[1]):
        #     v1, v2 = gdv_edge_index[0][i], gdv_edge_index[1][i]
        #     if flag[v1, v2]: continue
        #     flag[v1, v2] = 1
        #     flag[v2, v1] = 1
        #     if not true_labels[v1] and not true_labels[v2]: continue
        #     if true_labels[v1] != true_labels[v2]: heterophily_num += 1
        #     total_num += 1       
        # print('Constructed heterophily ratio: {}/{}={}'.format(heterophily_num, total_num, heterophily_num / total_num))

        # print(len(nor_curvs), len(ano_curvs), type(ano_curvs[0]))

        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # sns.distplot(nor_curvs)
        # plt.title(dataset + '_normal')
        # plt.subplot(1, 2, 2)
        # sns.distplot(ano_curvs)
        # plt.title(dataset + '_abnormal')
        # plt.savefig('../../pictures/curvs_compare/curvs_compare_{}.png'.format(dataset), dpi=300)
        # plt.clf()

        

        # """
        # 3 types: nor-nor, ano-nor, ano-ano
        # """

        # flag = torch.zeros(curvs.shape)
        # heterophily_num = 0
        # total_num = 0 
        # for i in range(edge_index.shape[1]):
        #     v1, v2 = edge_index[0][i], edge_index[1][i]
        #     if flag[v1, v2]: continue
        #     flag[v1, v2] = 1
        #     flag[v2, v1] = 1
        #     if not true_labels[v1] and not true_labels[v2]: continue
        #     if true_labels[v1] != true_labels[v2]: heterophily_num += curvs[v1, v2].item()
        #     total_num += curvs[v1, v2].item()      
        # print('Constructed heterophily ratio: {}/{}={}'.format(heterophily_num, total_num, heterophily_num / total_num))

        # nor_nor_curvs = []
        # ano_nor_curvs = []
        # ano_ano_curvs = []
        # flag = torch.zeros(curvs.shape)
        # for i in range(edge_index.shape[1]):
        #     v1, v2 = edge_index[0][i], edge_index[1][i]
        #     if flag[v1, v2]: continue
        #     flag[v1, v2] = 1
        #     flag[v2, v1] = 1
        #     if not true_labels[v1] and not true_labels[v2]: nor_nor_curvs.append(curvs[v1, v2].item())
        #     elif true_labels[v1] and true_labels[v2]: ano_ano_curvs.append(curvs[v1, v2].item())
        #     else: ano_nor_curvs.append(curvs[v1, v2].item())

        # plt.figure(figsize=(18, 4), dpi=dpi)
        # plt.subplot(1, 3, 1)
        # fig = sns.distplot(nor_nor_curvs)
        # plt.title(dataset + '_nor_nor')
        # plt.subplot(1, 3, 2)
        # fig = sns.distplot(ano_nor_curvs)
        # plt.subplot(1, 3, 3)
        # fig = sns.distplot(ano_ano_curvs)
        # plt.title(dataset + '_ano_ano')    
        # plt.savefig('../../pictures/curvs_compare_3/curvs_compare_{}_{}.png'.format(dataset, curv_mode), bbox_inches='tight')
        # plt.clf()

        # # plt.figure(figsize=(18, 4))
        # # plt.subplot(1, 3, 1)
        # # fig = sns.distplot(nor_nor_curvs)
        # # plt.title(dataset + '_nor_nor')
        # # plt.subplot(1, 3, 2)
        # # fig = sns.distplot(ano_nor_curvs)
        # # plt.title(dataset + '_ano_nor')
        # # plt.subplot(1, 3, 3)
        # # fig = sns.distplot(ano_ano_curvs, )
        # # plt.title(dataset + '_ano_ano')    
        # # plt.savefig('../../pictures/curvs_compare/curvs_compare_{}_{}.png'.format(dataset, curv_mode), dpi=300)
        # # plt.clf()



        # gdv_sim_mat = sim(gdvs, gdvs).numpy()
        # label_map = (data.y.reshape(-1, 1) + data.y.reshape(1, -1)).numpy()

        # nor_nor_sim = []
        # ano_nor_sim = []
        # ano_ano_sim = []
        # for i in range(label_map.shape[0]):
        #     for j in range(label_map.shape[1]):
        #         if j >= i: break
        #         if label_map[i, j] == 2: ano_ano_sim.append(gdv_sim_mat[i, j])
        #         elif label_map[i, j] == 1: ano_nor_sim.append(gdv_sim_mat[i, j])
        #         elif label_map[i, j] == 0: nor_nor_sim.append(gdv_sim_mat[i, j])
        #         else:
        #             print('Wrong!!!!!')
        #             exit()

        # plt.figure(figsize=(18, 4), dpi=dpi)
        # plt.subplot(1, 3, 1)
        # sns.distplot(nor_nor_sim)
        # plt.title(dataset + '_nor_nor')
        # plt.subplot(1, 3, 2)
        # sns.distplot(ano_nor_sim)
        # plt.title(dataset + '_ano_nor')
        # plt.subplot(1, 3, 3)
        # sns.distplot(ano_ano_sim)
        # plt.title(dataset + '_ano_ano')    
        # plt.savefig('../../pictures/gdv_sim_3/gdv_sim_{}.png'.format(dataset), bbox_inches='tight')
        # plt.clf()



        # # nor_nor_curvs = []
        # # ano_nor_curvs = []
        # # ano_ano_curvs = []
        # # flag = torch.zeros(curvs.shape)
        # # for i in range(edge_index.shape[1]):
        # #     v1, v2 = edge_index[0][i], edge_index[1][i]
        # #     if flag[v1, v2]: continue
        # #     flag[v1, v2] = 1
        # #     flag[v2, v1] = 1
        # #     if not true_labels[v1] and not true_labels[v2]: nor_nor_curvs.append(degs[v1] + degs[v2])
        # #     elif true_labels[v1] and true_labels[v2]: ano_ano_curvs.append(degs[v1] + degs[v2])
        # #     else: ano_nor_curvs.append(degs[v1] + degs[v2])

        # # plt.figure(figsize=(18, 4))
        # # plt.subplot(1, 3, 1)
        # # sns.distplot(nor_nor_curvs)
        # # plt.title(dataset + '_nor_nor')
        # # plt.subplot(1, 3, 2)
        # # sns.distplot(ano_nor_curvs)
        # # plt.title(dataset + '_ano_nor')
        # # plt.subplot(1, 3, 3)
        # # sns.distplot(ano_ano_curvs)
        # # plt.title(dataset + '_ano_ano')    
        # # plt.savefig('../../pictures/degree_compare/degree_edge_{}.png'.format(dataset), dpi=300)
        # # plt.clf()

        # # label_map = (data.y.reshape(-1, 1) + data.y.reshape(1, -1)).numpy()
        # # nor_nor_sim = []
        # # ano_nor_sim = []
        # # ano_ano_sim = []
        # # for i in range(label_map.shape[0]):
        # #     for j in range(label_map.shape[1]):
        # #         if j >= i: break
        # #         if label_map[i, j] == 2: ano_ano_sim.append(degs[i] + degs[j])
        # #         elif label_map[i, j] == 1: ano_nor_sim.append(degs[i] + degs[j])
        # #         elif label_map[i, j] == 0: nor_nor_sim.append(degs[i] + degs[j])
        # #         else:
        # #             print('Wrong!!!!!')
        # #             exit()

        # # plt.figure(figsize=(18, 4))
        # # plt.subplot(1, 3, 1)
        # # sns.distplot(nor_nor_sim)
        # # plt.title(dataset + '_nor_nor')
        # # plt.subplot(1, 3, 2)
        # # sns.distplot(ano_nor_sim)
        # # plt.title(dataset + '_ano_nor')
        # # plt.subplot(1, 3, 3)
        # # sns.distplot(ano_ano_sim)
        # # plt.title(dataset + '_ano_ano')    
        # # plt.savefig('../../pictures/degree_compare/degree_all_{}.png'.format(dataset), dpi=300)
        # # plt.clf()



        """
        2 types: ano-nor, ano-ano
        """

        flag = torch.zeros(curvs.shape)
        heterophily_num = 0
        total_num = 0 
        for i in range(edge_index.shape[1]):
            v1, v2 = edge_index[0][i], edge_index[1][i]
            if flag[v1, v2]: continue
            flag[v1, v2] = 1
            flag[v2, v1] = 1
            if not true_labels[v1] and not true_labels[v2]: continue
            if true_labels[v1] != true_labels[v2]: heterophily_num += curvs[v1, v2].item()
            total_num += curvs[v1, v2].item()      
        print('Constructed heterophily ratio: {}/{}={}'.format(heterophily_num, total_num, heterophily_num / total_num))

        nor_nor_curvs = []
        ano_nor_curvs = []
        ano_ano_curvs = []
        flag = torch.zeros(curvs.shape)
        for i in range(edge_index.shape[1]):
            v1, v2 = edge_index[0][i], edge_index[1][i]
            if flag[v1, v2]: continue
            flag[v1, v2] = 1
            flag[v2, v1] = 1
            if not true_labels[v1] and not true_labels[v2]: nor_nor_curvs.append(curvs[v1, v2].item())
            elif true_labels[v1] and true_labels[v2]: ano_ano_curvs.append(curvs[v1, v2].item())
            else: ano_nor_curvs.append(curvs[v1, v2].item())

        plt.figure(figsize=(12, 4), dpi=dpi)
        plt.subplot(1, 2, 1)
        sns.distplot(ano_nor_curvs, color='#4351C8')
        plt.title((dataset[:-4] if dataset.endswith('syn') else dataset) + ': anomalous-normal')
        plt.subplot(1, 2, 2)
        sns.distplot(ano_ano_curvs, color='#AF172B')
        plt.title((dataset[:-4] if dataset.endswith('syn') else dataset) + ': anomalous-anomalous')    
        plt.savefig('../../pictures/curvs_compare_2/curvs_compare_{}_{}.png'.format(dataset, curv_mode), bbox_inches='tight')
        plt.clf()



        gdv_sim_mat = sim(gdvs, gdvs).numpy()
        label_map = (data.y.reshape(-1, 1) + data.y.reshape(1, -1)).numpy()

        nor_nor_sim = []
        ano_nor_sim = []
        ano_ano_sim = []
        for i in range(label_map.shape[0]):
            for j in range(label_map.shape[1]):
                if j >= i: break
                if label_map[i, j] == 2: ano_ano_sim.append(gdv_sim_mat[i, j])
                elif label_map[i, j] == 1: ano_nor_sim.append(gdv_sim_mat[i, j])
                elif label_map[i, j] == 0: nor_nor_sim.append(gdv_sim_mat[i, j])
                else:
                    print('Wrong!!!!!')
                    exit()

        plt.figure(figsize=(12, 4), dpi=dpi)
        plt.subplot(1, 2, 1)
        sns.distplot(ano_nor_sim, color='#4351C8')
        plt.title((dataset[:-4] if dataset.endswith('syn') else dataset) + ': anomalous-normal')
        plt.subplot(1, 2, 2)
        sns.distplot(ano_ano_sim, color='#AF172B')
        plt.title((dataset[:-4] if dataset.endswith('syn') else dataset) + ': anomalous-anomalous')    
        plt.savefig('../../pictures/gdv_sim_2/gdv_sim_{}.png'.format(dataset), bbox_inches='tight')
        plt.clf()



        # # nor_nor_curvs = []
        # # ano_nor_curvs = []
        # # ano_ano_curvs = []
        # # flag = torch.zeros(curvs.shape)
        # # for i in range(edge_index.shape[1]):
        # #     v1, v2 = edge_index[0][i], edge_index[1][i]
        # #     if flag[v1, v2]: continue
        # #     flag[v1, v2] = 1
        # #     flag[v2, v1] = 1
        # #     if not true_labels[v1] and not true_labels[v2]: nor_nor_curvs.append(degs[v1] + degs[v2])
        # #     elif true_labels[v1] and true_labels[v2]: ano_ano_curvs.append(degs[v1] + degs[v2])
        # #     else: ano_nor_curvs.append(degs[v1] + degs[v2])

        # # plt.figure(figsize=(12, 4))
        # # plt.subplot(1, 2, 1)
        # # sns.distplot(ano_nor_curvs)
        # # plt.title(dataset + '_ano_nor')
        # # plt.subplot(1, 2, 2)
        # # sns.distplot(ano_ano_curvs)
        # # plt.title(dataset + '_ano_ano')    
        # # plt.savefig('../../pictures/degree_compare/degree_edge_{}.png'.format(dataset), dpi=300)
        # # plt.clf()

        # # label_map = (data.y.reshape(-1, 1) + data.y.reshape(1, -1)).numpy()
        # # nor_nor_sim = []
        # # ano_nor_sim = []
        # # ano_ano_sim = []
        # # for i in range(label_map.shape[0]):
        # #     for j in range(label_map.shape[1]):
        # #         if j >= i: break
        # #         if label_map[i, j] == 2: ano_ano_sim.append(degs[i] + degs[j])
        # #         elif label_map[i, j] == 1: ano_nor_sim.append(degs[i] + degs[j])
        # #         elif label_map[i, j] == 0: nor_nor_sim.append(degs[i] + degs[j])
        # #         else:
        # #             print('Wrong!!!!!')
        # #             exit()

        # # plt.figure(figsize=(12, 4))
        # # plt.subplot(1, 2, 1)
        # # sns.distplot(ano_nor_sim)
        # # plt.title(dataset + '_ano_nor')
        # # plt.subplot(1, 2, 2)
        # # sns.distplot(ano_ano_sim)
        # # plt.title(dataset + '_ano_ano')    
        # # plt.savefig('../../pictures/degree_compare/degree_all_{}.png'.format(dataset), dpi=300)
        # # plt.clf()