import os
import argparse
import logging
import pandas as pd
from brokenaxes import brokenaxes
from matplotlib.ticker import MultipleLocator, IndexLocator, FixedLocator
import seaborn as sns
from pathlib import Path
import time
import torch
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_auc_score, roc_curve

# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# sns.set_theme(font='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams.update({'font.size': 13})
plt.rcParams["axes.labelweight"] = "bold"
dpi = 500

def evaluate(ano_scores: torch.Tensor, labels: torch.Tensor, case_study: bool, nx_graph=None, seed=None, mask=None, threshold=None) -> dict:
    

    masked_ano_scores = ano_scores.cpu().numpy()
    masked_labels = labels.cpu().numpy()

    # AUC & AUPR
    auc = roc_auc_score(masked_labels, masked_ano_scores)
    aupr = average_precision_score(masked_labels, masked_ano_scores)
    # fpr, tpr, _ = roc_curve(masked_labels, masked_ano_scores)
    fpr = 0
    tpr = 0

    # Given a threshold
    if threshold is not None:
        preds = (masked_ano_scores > threshold)
        acc = accuracy_score(masked_labels, preds)
        recall = recall_score(masked_labels, preds)
        precision = precision_score(masked_labels, preds)
        f1 = f1_score(masked_labels, preds)
    else:
        acc, recall, precision, f1 = 0., 0., 0., 0.

    if case_study:
        normal_scores =  masked_ano_scores[masked_labels == 0]
        abnormal_scores = masked_ano_scores[masked_labels == 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), dpi=dpi)
    
        # violins = ax1.violinplot(np.log(normal_scores), showmedians=True)
        # for pc in violins['bodies']:
        #     pc.set_facecolor('#4351C8')
        #     pc.set_edgecolor('black')
        # violins['cmedians'].set_color('#7F7F7F')
        # violins['cmins'].set_color('#7F7F7F')
        # violins['cmaxes'].set_color('#7F7F7F')
        # violins['cbars'].set_color('#7F7F7F')
        # ax1.set(ylabel='Log of Anomaly Score')
        # ax1.set_xticks(np.arange(1, 2), labels=['Normal'])
        # ax1.set_ylim(top=14.5, bottom=8.1)    
        # ax1.yaxis.set_major_locator(FixedLocator([8.3, 10.3, 12.3, 14.3]))

        # violins = ax2.violinplot(np.log(abnormal_scores), showmedians=True)
        # for pc in violins['bodies']:
        #     pc.set_facecolor('#AF172B')
        #     pc.set_edgecolor('black')
        # violins['cmedians'].set_color('#7F7F7F')
        # violins['cmins'].set_color('#7F7F7F')
        # violins['cmaxes'].set_color('#7F7F7F')
        # violins['cbars'].set_color('#7F7F7F')
        # ax2.set_xticks(np.arange(1, 2), labels=['Abnormal'])
        # ax2.set_ylim(bottom=14.2, top=16.8)    
        # ax2.yaxis.set_major_locator(FixedLocator([14.3, 15.1, 15.9, 16.7]))

        # plt.savefig('../../pictures/case_1.png', bbox_inches='tight')

        sns.violinplot(np.log(normal_scores), cut=0, color='#4351C8', ax=ax1)
        ax1.set(ylabel='Log of Anomaly Score')
        ax1.set_xticks(np.arange(1), labels=['Normal'])
        ax1.set_ylim(top=14.5, bottom=8.1)    
        ax1.yaxis.set_major_locator(FixedLocator([8.3, 10.3, 12.3, 14.3]))

        sns.violinplot(np.log(abnormal_scores), cut=0, color='#AF172B', ax=ax2)
        ax2.set_ylim(bottom=14.2, top=16.8)    
        ax2.set_xticks(np.arange(1), labels=['Abnormal'])
        ax2.yaxis.set_major_locator(FixedLocator([14.3, 15.1, 15.9, 16.7]))

        plt.savefig('../../pictures/case_3.png', bbox_inches='tight')
        plt.clf()

        
        masked_ids = np.arange(len(mask))[mask]
        normal_ids = masked_ids[masked_labels == 0]
        abnormal_ids = masked_ids[masked_labels == 1]
        part_normal_ids = normal_ids[:int(1.5*len(abnormal_ids))]
        induced_root_ids = np.hstack((part_normal_ids, abnormal_ids))
        induced_nx_graph = nx.induced_subgraph(nx_graph, induced_root_ids)
        induced_nx_graph = nx.Graph(induced_nx_graph)
        induced_nx_graph.remove_nodes_from(list(nx.isolates(induced_nx_graph)))
        induced_ids = np.array(list(induced_nx_graph.nodes))
        print(len(normal_ids), len(abnormal_ids), len(induced_ids))

        cut_thred = 3.5e6
        masked_ano_scores[masked_ano_scores > cut_thred] = cut_thred
        nor_max, nor_min, ano_max, ano_min = masked_ano_scores[masked_labels == 0].max(), masked_ano_scores[masked_labels == 0].min(), \
                                            masked_ano_scores[masked_labels == 1].max(), masked_ano_scores[masked_labels == 1].min()
        masked_ano_scores[masked_labels == 0] =  (masked_ano_scores[masked_labels == 0] - nor_max) / (nor_max - nor_min)
        masked_ano_scores[masked_labels == 1] =  (masked_ano_scores[masked_labels == 1] - ano_min) / (ano_max - ano_min)
        score_map = dict(zip(masked_ids, masked_ano_scores))
        label_map = dict(zip(masked_ids, masked_labels))
        score_values = np.array([score_map[node] for node in induced_nx_graph.nodes()])
        # cut_thred = 5e6
        # score_values[score_values > cut_thred] = cut_thred
        # score_values = (score_values - score_values.min()) / (score_values.max() - score_values.min()) # normalize
        label_values = np.array([label_map[node] for node in induced_nx_graph.nodes()])
        
        # pos = nx.kamada_kawai_layout(induced_nx_graph)
        pos = nx.nx_pydot.pydot_layout(induced_nx_graph)
        print(score_values.min(), score_values.max(), score_values.mean())
        nx.draw(induced_nx_graph, pos, cmap=plt.cm.coolwarm, node_color=score_values, node_size = 20, width=0.15, edgecolors='black', linewidths=0.5)
        # nx.draw(induced_nx_graph, pos, cmap=plt.cm.coolwarm, node_color=label_values, node_size = 5, width=0.05)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, shrink=0.8, label='Anomaly Score (Normalized)', location='left')
        plt.savefig('../../pictures/case_2.png', bbox_inches='tight', dpi=dpi)

        print(normal_scores.min(), normal_scores.max(), abnormal_scores.min(), abnormal_scores.max())
        print(nor_max, nor_min, ano_max, ano_min)



    return {'auc': auc, 'aupr': aupr, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1, 'fpr': fpr, 'tpr': tpr}

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # torch.use_deterministic_algorithms(True)

def get_args():
    parser = argparse.ArgumentParser('OCGNN')

    # Basic
    parser.add_argument("--tune", action='store_true',
            help="whether tune hyper-parameters or not. Default to be False")
    parser.add_argument("--dataset", type=str, default='cora-struct',
            help="cora,amazon,reddit,weibo")
    parser.add_argument("--setting", type=str, default='clean',
            help="clean/polluted")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu id")
    parser.add_argument("--seed", type=int, default=3407,
            help="random seed")
    parser.add_argument("--use_shell_args", action='store_true', 
            help="whether use args in shell. Default to be False")

    # Path
    parser.add_argument("--data_path", type=str, default='../Data/',
            help="root directory of data")
    parser.add_argument("--model_path", type=str, default='./best_model/',
            help="root directory of trained model")
    parser.add_argument("--log_path", type=str, default='./log/',
            help="root directory of log")
    parser.add_argument("--tune_dict_path", type=str, default='./tune_records/',
            help="root directory of tuning results")

    # Train & Test
    parser.add_argument("--no_train", action='store_true',
            help="whether skip the training step. Default to be False")    
    parser.add_argument("--no_test", action='store_true',
            help="whether skip the testing step. Default to be False") 
    parser.add_argument("--case_study", action='store_true',
            help="whether conduct case study. Default to be False") 

    # Learning
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
            help="weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.,
            help="dropout probability")
    parser.add_argument('--batch_size', type=int,default=128,
            help='batch size')
    parser.add_argument("--n_epochs", type=int, default=10000,
            help="number of training epochs")
    parser.add_argument("--patience", type=int, default=1000,
            help="earlystop patience")
        

    """
    Model
    """
    parser.add_argument("--model", type=str, default='DualSphere',
            help="DualSphere/ConDualSphere/ComDualSphere")

    # GNN
    parser.add_argument("--gnn", type=str, default='GAT',
            help="GCN/GAT/GraphSAGE")
    parser.add_argument("--num_layers", type=int, default=2,
            help="number of hidden gnn layers")
    parser.add_argument("--hidden_dim", type=int, default=32,
            help="number of hidden gnn units")
    parser.add_argument("--curv_mode", type=str, default='v-7',
            help="v6-15")

    # Hypersphere
    parser.add_argument("--beta", type=float, default=0.1,
            help="soft radius")
    parser.add_argument("--center_mode", type=str, default='init',
            help="init/update/train. Note that angular center cannot use update mode.")
    parser.add_argument("--radius_mode", type=str, default='cut',
            help="cut/none. Note that angular center cannot use cut mode.")
    
    # Clustering
    parser.add_argument("--cluster_model", type=str, default='CL',
            help="GMM/reduced_GMM/CL")
    parser.add_argument("--num_estimation_layers", type=int, default=1,
            help="number of hidden estimation layers")
    parser.add_argument("--num_clusters", type=int, default=5,
            help="number of clusters")      
    parser.add_argument("--con_center_mode", type=str, default='none',
            help="none/detach/train: how to incorporate center into contrastive-based clustering")   

    # MultiSphere
    parser.add_argument("--soft", type=int, default=0,
            help="whether use soft loss considering all the communities into the loss. Default to be False")    
    parser.add_argument("--norm", type=str, default='none',
            help="soft/hard/none: how normalize the distance between cluster center and members. Default to be soft")
    parser.add_argument("--mul_center_mode", type=str, default='train',
            help="detach/train: how to incorporate center into multi-sphere learning")     

    # Loss balance
    parser.add_argument("--lamda_angle", type=float, default=0,
            help="balance the direction-aware hypersphere")
    parser.add_argument("--lamda_reg_cluster", type=float, default=0,
            help="regularize clustering")
    parser.add_argument("--lamda_cluster", type=float, default=1,
            help="balance the clustering model")
    parser.add_argument("--lamda_local", type=float, default=1,
            help="balance multiple local hyperspheres")

    args = parser.parse_args()
    
    return args

def get_default_args(args):

    if args.use_shell_args: return args
    
    # fix some hyper-params
    assert args.num_clusters is not None and args.soft is not None and args.lamda_cluster is not None and args.lamda_local is not None
    assert args.lamda_reg_cluster is not None
    assert args.center_mode is not None and args.radius_mode is not None and args.con_center_mode is not None
    assert args.mul_center_mode is not None and args.num_estimation_layers is not None

    # cora-struct
    if args.dataset == 'cora-struct':

        args.lamda_cluster = 1
        args.lamda_local = 0.1
        args.lamda_reg_cluster = 0.01
        args.num_clusters = 8
        args.center_mode = 'update'
        args.radius_mode = 'none'
        args.mul_center_mode = 'train'
        args.lr = 0.001
        args.hidden_dim = 32

    # cora-contex
    elif args.dataset == 'cora-contex':
        pass

    # cora-raw
    elif args.dataset == 'cora-raw':
        pass

    # cora-syn
    elif args.dataset == 'cora-syn':

        args.lamda_cluster = 0.1
        args.lamda_local = 1
        args.lamda_reg_cluster = 0.01
        args.num_clusters = 8
        args.center_mode = 'init'
        args.radius_mode = 'cut'
        args.mul_center_mode = 'train'
        args.lr = 0.001
        args.hidden_dim = 32

    # amazon-struct
    elif args.dataset == 'amazon-struct':
        pass

    # amazon-contex
    elif args.dataset == 'amazon-contex':
        pass

    # amazon-raw
    elif args.dataset == 'amazon-raw':
        pass

    # amazon-syn
    elif args.dataset == 'amazon-syn':
        pass
    
    # citeseer-struct
    elif args.dataset == 'citeseer-struct':

        args.lamda_cluster = 10
        args.lamda_local = 10
        args.lamda_reg_cluster = 0.0001
        args.num_clusters = 6
        args.center_mode = 'update'
        args.radius_mode = 'none'
        args.mul_center_mode = 'detach'
        args.lr = 0.001
        args.hidden_dim = 256

    # citeseer-contex
    elif args.dataset == 'citeseer-contex':
        pass

    # citeseer-raw
    elif args.dataset == 'citeseer-raw':
        pass

    # citeseer-syn
    elif args.dataset == 'citeseer-syn':

        args.lamda_cluster = 10
        args.lamda_local = 0.001
        args.lamda_reg_cluster = 0.0001
        args.num_clusters = 6
        args.center_mode = 'update'
        args.radius_mode = 'none'
        args.mul_center_mode = 'detach'
        args.lr = 0.01
        args.hidden_dim = 32

    # reddit
    elif args.dataset == 'reddit':

        args.lamda_cluster = 10
        args.lamda_local = 1
        args.lamda_reg_cluster = 0.01
        args.num_clusters = 8
        args.center_mode = 'train'
        args.radius_mode = 'cut'
        args.mul_center_mode = 'detach'
        args.lr = 0.001
        args.hidden_dim = 32

    # weibo
    elif args.dataset == 'weibo':
        args.num_clusters = 8

        args.lamda_cluster = 0.001
        args.lamda_local = 0.01
        args.lamda_reg_cluster = 0.001

    # blog-struct
    elif args.dataset == 'blog-struct': 
        args.num_clusters = 4

        args.lamda_cluster = 10
        args.lamda_local = 10
        args.lamda_reg_cluster = 0

    # blog-contex
    elif args.dataset == 'blog-contex': pass

    # blog-syn
    elif args.dataset == 'blog-syn': 
        args.num_clusters = 4

        args.lamda_cluster = 10
        args.lamda_local = 10
        args.lamda_reg_cluster = 0

    # ml-struct
    elif args.dataset == 'ml-struct': 

        args.lamda_cluster = 0.001
        args.lamda_local = 10
        args.lamda_reg_cluster = 0.01
        args.num_clusters = 10
        args.center_mode = 'update'
        args.radius_mode = 'none'
        args.mul_center_mode = 'train'
        args.lr = 0.0001
        args.hidden_dim = 256

    # ml-contex
    elif args.dataset == 'ml-contex': pass

    # ml-syn
    elif args.dataset == 'ml-syn': 

        args.lamda_cluster = 10
        args.lamda_local = 1
        args.lamda_reg_cluster = 0.0001
        args.num_clusters = 10
        args.center_mode = 'init'
        args.radius_mode = 'none'
        args.mul_center_mode = 'detach'
        args.lr = 0.0001
        args.hidden_dim = 256

    # photo-struct
    elif args.dataset == 'photo-struct': 
        args.num_clusters = 4

        args.lamda_cluster = 0.01
        args.lamda_local = 10
        args.lamda_reg_cluster = 0

    # photo-contex
    elif args.dataset == 'photo-contex': pass

    # photo-syn
    elif args.dataset == 'photo-syn': 
        args.num_clusters = 6

        args.lamda_cluster = 0.001
        args.lamda_local = 0.001
        args.lamda_reg_cluster = 0

    # pubmed-struct
    elif args.dataset == 'pubmed-struct':

        args.lamda_cluster = 0.001
        args.lamda_local = 1
        args.lamda_reg_cluster = 0.01
        args.num_clusters = 4
        args.center_mode = 'init'
        args.radius_mode = 'cut'
        args.mul_center_mode = 'train'
        args.lr = 0.1
        args.hidden_dim = 16

    # pubmed-contex
    elif args.dataset == 'pubmed-contex': pass

    # pubmed-syn
    elif args.dataset == 'pubmed-syn': 

        args.lamda_cluster = 0.01
        args.lamda_local = 0.01
        args.lamda_reg_cluster = 0
        args.num_clusters = 4
        args.center_mode = 'update'
        args.radius_mode = 'none'
        args.mul_center_mode = 'train'
        args.lr = 0.001
        args.hidden_dim = 32

    else: raise ValueError('Dataset {} not supported.'.format(args.dataset))

    return args

def get_logger(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    # fh = logging.FileHandler(args.log_path + '{}_{}_{}_{}_{}.log'.format(args.dataset, args.center_mode, args.radius_mode, args.mul_center_mode, str(time.time())))
    # fh = logging.FileHandler(args.log_path + '{}_{:.1e}_{:.1e}_{}.log'.format(args.dataset, args.lamda_cluster, args.lamda_local, str(time.time())))
    # fh = logging.FileHandler(args.log_path + '{}_{:.1e}_{}_{}.log'.format(args.dataset, args.lamda_reg_cluster, args.num_clusters, str(time.time())))
    fh = logging.FileHandler(args.log_path + '{}_{:.1e}_{}_{}.log'.format(args.dataset, args.lr, args.hidden_dim, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, fh, ch

class EarlyStop:
    def __init__(self, logger, args):
        super(EarlyStop, self).__init__()

        self.best_loss = 1e10
        self.best_metric = 0

        self.logger = logger
        self.patience = args.patience
        self.cur = 0 
        
        self.args = args

        self.path = args.model_path
        Path(self.path).mkdir(parents=True, exist_ok=True)
    
    def test_loss(self, new_loss, model, epoch):
        if new_loss < self.best_loss:
            self.cur = 0
            self.best_loss = new_loss
            self.logger.info('Save model at epoch {}'.format(epoch))
            torch.save({'epoch': epoch, 'model': model}, self.path + 'model_{}_{}.pth.tar'.format(self.args.dataset, self.args.setting))
        else: 
            self.cur += 1
            self.logger.info('EarlyStop {}/{}'.format(self.cur, self.patience))

        if self.cur >= self.patience: return True # require earlystopping
        else: return False
    
    def test_metric(self, new_metric, model, epoch):
        if new_metric > self.best_metric:
            self.cur = 0
            self.best_metric = new_metric
            self.logger.info('Save model at epoch {}'.format(epoch))
            torch.save({'epoch': epoch, 'model': model}, self.path + 'model_{}_{}.pth.tar'.format(self.args.dataset, self.args.setting))
        else: 
            self.cur += 1
            self.logger.info('EarlyStop {}/{}'.format(self.cur, self.patience))

        if self.cur >= self.patience: return True # require earlystopping
        else: return False    


# ax = fig.add_subplot()
# ax = brokenaxes(
#          ylims=((0, 1500000), (1800000,14000000)),
#          hspace=0.25,             
#          despine=False,
#          diag_color='r'             
#         )

# plt.hist(normal_scores, bins=50, color='#00B8B8', alpha=1, label='Normal Nodes')
# plt.hist(abnormal_scores, bins=50, color='r', alpha=0.2, label='Abnormal Nodes')
# ax.set(xlabel='Anomaly Score', ylabel='Number')
# ax.set_xscale('log')
# ax.legend(loc='best')

# ax.boxplot(x=(normal_scores, abnormal_scores),labels=('Normal', 'Abnormal'))
# ax.set(ylabel='Anomaly Score')
# ax.set_yscale('log')

# labels=('Normal', 'Abnormal')
# ax.violinplot((normal_scores, abnormal_scores), quantiles=([0.25, 0.5, 0.75], [0.25, 0.5, 0.75]))
# ax.set(ylabel='Anomaly Score')
# ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
# ax.set_yscale('log')
