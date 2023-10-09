import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.transforms import RootedEgoNets, RootedRWSubgraph
from torch_geometric.utils import to_scipy_sparse_matrix, add_remaining_self_loops

from load_gnn import init_model
from load_cluster import init_clustering_model
from load_contrast import NodeContrastive

class MultiSphere(nn.Module):

    def __init__(self, args, input_dim, data, device, logger):
        super(MultiSphere, self).__init__()

        self.logger = logger

        self.center_mode = args.center_mode
        self.radius_mode = args.radius_mode

        # Init center and radius for distance-aware hypersphere
        self.center = None
        self.radius = torch.tensor(0., device=device)
        self.beta = args.beta

        # Init center and radius for direction-aware hypersphere
        self.center_proj = None
        
        # Loss coefficients
        self.lamda_angle = args.lamda_angle
        self.lamda_cluster = args.lamda_cluster
        self.lamda_local = args.lamda_local

        # Init GNN base model
        self.gnn = init_model(model_name=args.gnn, num_layers=args.num_layers, input_dim=input_dim, hidden_dim=args.hidden_dim, mode='embed')

        # Init clustering model
        self.cluster_model = init_clustering_model(args=args, device=device)
        self.cluster_model_name = args.cluster_model
        self.num_clusters = args.num_clusters
        self.con_center_mode = args.con_center_mode

        self.soft = args.soft
        self.norm = args.norm
        self.mul_center_mode = args.mul_center_mode

        self.device = device

    def forward(self, data, mask):

        """
        GNN Representation
        """
        self.embeds = self.gnn(data)
        self.masked_embeds = self.embeds[mask]

        """
        Distance-aware Hypersphere
        """
        if self.center_mode != 'train': assert self.center.requires_grad is False
        assert self.radius.requires_grad is False
        self.distances = torch.sum((self.masked_embeds - self.center) ** 2, dim=-1)
        
        if self.radius_mode == 'cut':
            dis_loss = torch.mean(torch.max(torch.zeros_like(self.distances), self.distances - self.radius ** 2))
        elif self.radius_mode == 'none':
            dis_loss = torch.mean(self.distances)
        else: raise ValueError('Mode {} not supported.')

        """
        Direction-aware Hypersphere
        """
        if self.center_mode != 'train': assert self.center_proj.requires_grad is False
        self.cos_angles = self.cos_sim(self.masked_embeds - self.center, self.center_proj.unsqueeze(0))
        
        dir_loss = -torch.mean(self.cos_angles)

        """
        Clustering
        """
        if self.con_center_mode == 'train':
            clu_loss, assign_probs, cluster_indices, cluster_embeds = self.cluster_model(self.embeds, data.edge_index, center=self.center)
        elif self.con_center_mode == 'detach':
            clu_loss, assign_probs, cluster_indices, cluster_embeds = self.cluster_model(self.embeds, data.edge_index, center=self.center.detach())
        elif self.con_center_mode == 'none':
            clu_loss, assign_probs, cluster_indices, cluster_embeds = self.cluster_model(self.embeds, data.edge_index)
        else: raise ValueError('Contrastive center mode {} not supported.'.format(self.con_center_mode))
        # print('Cluster Distribution: ', assign_probs.t().mean(dim=-1))

        if self.mul_center_mode == 'detach': cluster_embeds = cluster_embeds.detach()
        elif self.mul_center_mode != 'train': raise ValueError('Multisphere center mode {} not supported.'.format(self.mul_center_mode))

        """
        Multiple local Hyperspheres
        """
        if self.norm == 'soft': # compute the soft norm of distance for each cluster (use the entire graph without mask)
            pairwise_distances_inv = torch.sum((cluster_embeds.unsqueeze(1) - self.embeds.unsqueeze(0)) ** 2, dim=-1) # [num_clusters, num_nodes]
            center_norms = torch.mean(assign_probs.t() * pairwise_distances_inv / torch.sum(assign_probs.t(), dim=-1, keepdim=True), dim=-1) # [num_clusters]
            center_norms = F.softmax(center_norms, dim=-1).detach()
        elif self.norm == 'hard': # compute the hard norm of distance for each cluster (use the hard members of each cluster)
            hard_mem_inds = [cluster_indices == i for i in range(self.num_clusters)] 
            hard_mem_inds = torch.stack(hard_mem_inds) # bool:[num_clusters, num_nodes]
            pairwise_distances_inv = torch.sum((cluster_embeds.unsqueeze(1) - self.embeds.unsqueeze(0)) ** 2, dim=-1) # [num_clusters, num_nodes]
            center_norms = torch.stack([torch.mean(pairwise_distances_inv[i][hard_mem_inds[i]], dim=-1) for i in range(self.num_clusters)]) # [num_clusters]
            center_norms = F.softmax(center_norms, dim=-1).detach()
        elif self.norm == 'none': center_norms = torch.ones(self.num_clusters).to(self.device)
        else: raise ValueError('Norm {} not supported.'.format(self.norm))
        # print(center_norms)

        # compute hard multi-sphere distances
        hard_center_embeds = cluster_embeds[cluster_indices[mask]] # [num_masked_nodes, hidden_dim]
        hard_center_norms = center_norms[cluster_indices[mask]] # [num_masked_nodes]
        self.hard_multi_distances = torch.sum((self.masked_embeds - hard_center_embeds) ** 2 / hard_center_norms.unsqueeze(1), dim=-1) # [num_masked_nodes]
        
        if self.soft: # compute soft multi-sphere distances (only works during training)
            pairwise_distances = torch.sum((self.masked_embeds.unsqueeze(1) - cluster_embeds.unsqueeze(0)) ** 2 / center_norms.view(1, -1, 1), dim=-1) # [num_masked_nodes, num_clusters]
            soft_multi_distances = torch.sum(assign_probs[mask] * pairwise_distances, dim=-1) # [num_masked_nodes]
            multi_distances = soft_multi_distances
        else: multi_distances = self.hard_multi_distances
        multi_dis_loss = torch.mean(multi_distances)

        self.logger.info('{} {} {} {}'.format(self.distances.mean().item(), self.cos_angles.mean().item(), self.hard_multi_distances.mean().item(), assign_probs.t().mean(dim=-1)))

        """
        Merge loss and anomalous scores.
        """
        loss = dis_loss + self.lamda_angle * dir_loss + self.lamda_cluster * clu_loss + self.lamda_local * multi_dis_loss
        ano_scores = self.distances - self.lamda_angle * self.cos_angles + self.lamda_local * self.hard_multi_distances

        return loss, ano_scores, dis_loss, dir_loss, clu_loss, multi_dis_loss

    def init_center(self, data):

        embeds = self.gnn(data)
        assert len(embeds.shape) == 2
        if self.center_mode == 'train':

            self.center = nn.Parameter(torch.mean(embeds, dim=0).detach()) # Distance-aware
            self.center_proj = nn.Parameter(torch.mean(embeds, dim=0).detach()) # Direction-aware

            assert self.center.requires_grad is True and self.center.is_leaf is True
            assert self.center_proj.requires_grad is True and self.center_proj.is_leaf is True

        else: 
            self.center = torch.mean(embeds, dim=0).detach() # Distance-aware
            self.center_proj = self.center # Direction-aware, set a candidate

    def update_center(self):

        assert len(self.embeds.shape) == 2
        self.center = torch.mean(self.embeds, dim=0).detach() # Distance-aware

    def update_radius(self):

        self.radius = torch.tensor(np.quantile(np.sqrt(self.distances.detach().cpu().numpy()), 1 - self.beta), device=self.device)

    def cos_sim(self, z1, z2):
        assert len(z1.shape) == 2 and len(z2.shape) == 2
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        cos_angles = (dot_numerator / dot_denominator).squeeze()
        return cos_angles
