# Funciones de augmentación y corrupción
# utils/augmentations.py
import torch

def augment(x, edge_index, drop_feat_p=0.1, drop_edge_p=0.2):
    """ Realiza dropout de características y aristas. """
    # Dropout de características
    feat_mask = torch.rand(x.size(1)) > drop_feat_p
    x_aug = x.clone()
    x_aug[:, ~feat_mask] = 0

    # Dropout de aristas
    edge_mask = torch.rand(edge_index.size(1)) > drop_edge_p
    edge_index_aug = edge_index[:, edge_mask]

    return x_aug, edge_index_aug

def corrupt(x, edge_index):
    """ Crea una vista corrompida con aristas aleatorias y features permutadas. 
        Implementa la idea de SHUFFLEFEAT+RANDOMEDGE del paper. 
       
    """
    num_nodes = x.size(0)
    num_edges = edge_index.size(1) // 2
    
    # Aristas aleatorias
    rand_u = torch.randint(0, num_nodes, (num_edges,))
    rand_v = torch.randint(0, num_nodes, (num_edges,))
    corrupt_edge_index = torch.stack(
        [torch.cat([rand_u, rand_v]), torch.cat([rand_v, rand_u])], dim=0
    )
    
    # Permutación de features
    permuted_indices = torch.randperm(num_nodes)
    corrupt_x = x[permuted_indices]
    
    return corrupt_x, corrupt_edge_index