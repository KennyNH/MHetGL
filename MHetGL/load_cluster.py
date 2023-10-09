import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data

from load_gnn import init_model
from load_contrast import ClusterContrastive

class GMM(nn.Module):
    """
    To avoid singular problem, we change it into an unnormalized variant and let covariance matrix trainable.
    """
    def __init__(self, args, input_dim, device):
        super(GMM, self).__init__()

        self.num_layers = args.num_estimation_layers
        self.num_clusters = args.num_clusters
        self.device = device

        # Estimation 
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, self.num_clusters))
        for _ in range(1, self.num_layers):
            self.linears.append(nn.Linear(self.num_clusters, self.num_clusters))
        self.act = nn.LeakyReLU()
        self.approx = nn.Linear(input_dim, input_dim)
    
    def estimate(self, embeds):
        """
        embeds: [num_nodes, hidden_dim]
        
        Return
        gamma: [num_nodes, num_clusters] 
        cluster_indices: [num_nodes]
        """
        x = embeds
        for i in range(self.num_layers):
            x = self.linears[i](x)
            
            # The last layer has no activation
            if i != self.num_layers - 1:
                x = self.act(x)
        
        # Softmax
        gamma = F.softmax(x, dim=-1)

        # Clustering
        cluster_indices = torch.argmax(gamma, dim=-1)

        return gamma, cluster_indices
    
    def maximize(self, embeds, gamma):
        """
        alpha: Mixture probability [num_clusters]
        mean: Mean of Gaussian components [num_clusters, hidden_dim]
        cov: Covariance matrix of Gaussian components [num_clusters, hidden_dim, hidden_dim]  
        """

        alpha = torch.sum(gamma, dim=0)/gamma.shape[0]

        mean = torch.sum(embeds.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mean /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mean = self.approx(embeds.unsqueeze(1) - mean.unsqueeze(0))
        z_mean_z_mean_t = z_mean.unsqueeze(-1) * z_mean.unsqueeze(-2)
        
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mean_z_mean_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        self.alpha = alpha
        self.mean = mean
        self.cov = cov
    
    def compute_energy(self, embeds):
        """Compute the likelihood/energy."""
        z_mean = (embeds.unsqueeze(1)- self.mean.unsqueeze(0))

        eps = 1e-20
        
        cov_inverse = self.cov # [num_clusters, hidden_dim, hidden_dim]

        E_z = -0.5 * torch.sum(torch.sum(z_mean.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mean, dim=-1) # [num_nodes, num_clusters]
        E_z = torch.exp(E_z)
        loss = -torch.log(torch.sum(self.alpha.unsqueeze(0) * E_z, dim=1) + eps) # [num_nodes]

        return loss

    def forward(self, embeds):

        gamma, cluster_indices = self.estimate(embeds)
        self.maximize(embeds, gamma)
        energy = self.compute_energy(embeds)
        loss = torch.mean(energy)

        return loss, cluster_indices, self.mean, energy

class ContrastiveClustering(nn.Module):
    """
    Without data augmentation.
    """
    def __init__(self, args, input_dim, device):
        super(ContrastiveClustering, self).__init__()

        self.num_layers = args.num_estimation_layers
        self.num_clusters = args.num_clusters
        self.device = device

        # # Estimation -- Linear
        # self.linears = nn.ModuleList()
        # self.linears.append(nn.Linear(input_dim, self.num_clusters))
        # for _ in range(1, self.num_layers):
        #     self.linears.append(nn.Linear(self.num_clusters, self.num_clusters))
        # self.act = nn.LeakyReLU()

        # Estimation -- GNN
        self.gnn = init_model(model_name=args.gnn, num_layers=self.num_layers, input_dim=input_dim, hidden_dim=self.num_clusters)

        # Contrastive 
        self.contrast = ClusterContrastive(args=args, device=device)

        self.lamda_reg_cluster = args.lamda_reg_cluster

    def estimate(self, embeds, edge_index):
        """
        embeds: [num_nodes, hidden_dim]
        
        Return
        gamma: [num_nodes, num_clusters] 
        cluster_indices: [num_nodes]
        """
        x = self.gnn(Data(x=embeds, edge_index=edge_index))
        
        # Softmax
        gamma = F.softmax(x, dim=-1)

        # Clustering
        cluster_indices = torch.argmax(gamma, dim=-1)

        return gamma, cluster_indices

    def forward(self, embeds, edge_index, center=None):

        gamma, cluster_indices = self.estimate(embeds, edge_index) # [num_nodes, num_clusters]
        # gamma_t = gamma.t() # [num_clusters, num_nodes]
        target_gamma = F.softmax(gamma ** 2 / torch.sum(gamma, dim=0), dim=-1) # [num_nodes, num_clusters] (augmentation)
        # target_gamma_t = target_gamma.t() # [num_clusters, num_nodes]

        miu = torch.sum(embeds.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        miu /= torch.sum(gamma, dim=0).unsqueeze(-1) # [num_clusters, hidden_dim]

        target_miu = torch.sum(embeds.unsqueeze(1) * target_gamma.unsqueeze(-1), dim=0)
        target_miu /= torch.sum(target_gamma, dim=0).unsqueeze(-1) # [num_clusters, hidden_dim]
        
        if center is not None: cl_loss = self.contrast(miu - center, target_miu - center)
        else: cl_loss = self.contrast(miu, target_miu)

        regularization = torch.mean(torch.sum(gamma, dim=0) ** 2) * self.lamda_reg_cluster

        return cl_loss + regularization, gamma.detach(), cluster_indices, miu

class KLDivClustering(nn.Module):
    """
    Without data augmentation.
    """
    def __init__(self, args, input_dim, device):
        super(KLDivClustering, self).__init__()

        self.num_layers = args.num_estimation_layers
        self.num_clusters = args.num_clusters
        self.device = device

        # Estimation 
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, self.num_clusters))
        for _ in range(1, self.num_layers):
            self.linears.append(nn.Linear(self.num_clusters, self.num_clusters))
        self.act = nn.LeakyReLU()

    def estimate(self, embeds):
        """
        embeds: [num_nodes, hidden_dim]
        
        Return
        gamma: [num_nodes, num_clusters] 
        cluster_indices: [num_nodes]
        """
        x = embeds
        for i in range(self.num_layers):
            x = self.linears[i](x)
            
            # The last layer has no activation
            if i != self.num_layers - 1:
                x = self.act(x)
        
        # Softmax
        gamma = F.softmax(x, dim=-1)

        # Clustering
        cluster_indices = torch.argmax(gamma, dim=-1)

        return gamma, cluster_indices

    def forward(self, embeds):

        gamma, cluster_indices = self.estimate(embeds)
        target_gamma = F.softmax(gamma ** 2 / torch.sum(gamma, dim=0), dim=-1) # [num_nodes, num_clusters]

        mean = torch.sum(embeds.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mean /= torch.sum(gamma, dim=0).unsqueeze(-1) # [num_clusters, hidden_dim]
    
        kl_loss = F.kl_div(target_gamma.log(), gamma, reduction='sum')

        return kl_loss, cluster_indices, mean, None

def init_clustering_model(args, device):

    # Load GNN
    if args.cluster_model == 'GMM':
        model = GMM(args=args, input_dim=args.hidden_dim, device=device)
    elif args.cluster_model == 'CL':
        model = ContrastiveClustering(args=args, input_dim=args.hidden_dim, device=device)
    elif args.cluster_model == 'KL':
        model = KLDivClustering(args=args, input_dim=args.hidden_dim, device=device)
    else: raise ValueError('Model {} not supported.'.format(args.cluster_model))

    return model