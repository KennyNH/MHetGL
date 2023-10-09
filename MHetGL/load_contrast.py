import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClusterContrastive(nn.Module):
    def __init__(self, args, device):
        super(ClusterContrastive, self).__init__()

        self.sim_mode = 'cosine'
        self.loss_mode = 'JS'
        self.device = device

    
    def forward(self, tar_embeds, con_embeds):

        num = tar_embeds.shape[0]
        dim = tar_embeds.shape[1]
        assert num == con_embeds.shape[0]
        identity = torch.eye(num).to(self.device)
        pos_mask = identity == 1
        neg_mask = identity == 0
        if self.sim_mode == 'cosine':
            pair_sim_mat = self.sim(tar_embeds, con_embeds) # [num, num]
            pos_scores = torch.sum(pair_sim_mat * pos_mask, dim=-1) # [num]
            neg_scores = torch.sum(pair_sim_mat * neg_mask, dim=-1) # [num]

            loss = self.compute(pos_scores, neg_scores)
        
        elif self.sim_mode == 'kl':
            pass
        
        return loss

    def compute(self, pos_scores, neg_scores):
        
        if self.loss_mode == 'JS':
            loss = -torch.log(pos_scores / (pos_scores + neg_scores)) # [num]
            loss = loss.mean()

        return loss
        
    def sim(self, z1, z2):
        """
        Calculate pair-wise similarity.
        """
        assert len(z1.shape) == 2 and len(z2.shape) == 2
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_mat = torch.exp(dot_numerator / dot_denominator) # shape: [num_z1, num_z2]
        return sim_mat
        
class NodeContrastive(nn.Module):
    def __init__(self, args, device):
        super(NodeContrastive, self).__init__()

        self.sim_mode = args.sim_mode
        self.loss_mode = 'JS'
        self.device = device

    
    def forward(self, node_embeds, cluster_embeds, cluster_indices):

        num_nodes = node_embeds.shape[0]
        num_clusters = cluster_embeds.shape[0]
        dim = node_embeds.shape[1]
        assert dim == cluster_embeds.shape[1]

        # Construct masks
        sp_inds = torch.stack([torch.arange(num_nodes).to(self.device), cluster_indices])
        sp_vals = torch.ones(num_nodes).to(self.device)
        mat = torch.sparse_coo_tensor(sp_inds, sp_vals, (num_nodes, num_clusters)).to_dense()
        pos_mask = mat == 1
        neg_mask = mat == 0
        if self.sim_mode == 'cosine':
            pair_sim_mat = self.sim(node_embeds, cluster_embeds) # [num_nodes, num_clusters]
            pos_scores = torch.sum(pair_sim_mat * pos_mask, dim=-1) # [num_nodes]
            neg_scores = torch.sum(pair_sim_mat * neg_mask, dim=-1) # [num_nodes]

            loss = self.compute(pos_scores, neg_scores)
        
        return loss

    def compute(self, pos_scores, neg_scores):
        
        if self.loss_mode == 'JS':
            loss = -torch.log(pos_scores / (pos_scores + neg_scores)) # [num_nodes]

        return loss
        
    def sim(self, z1, z2):
        """
        Calculate pair-wise similarity.
        """
        assert len(z1.shape) == 2 and len(z2.shape) == 2
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_mat = torch.exp(dot_numerator / dot_denominator) # shape: [num_z1, num_z2]
        return sim_mat
        
