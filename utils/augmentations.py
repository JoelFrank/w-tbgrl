# utils/augmentations.py
import torch
import torch.nn.functional as F

def augment(x, edge_index, edge_weight, nodes_U_idx, nodes_V_idx, drop_feat_p=0.1, drop_edge_p=0.2):
    """
    Aumenta el grafo aplicando drop de características y drop de aristas conscientes del peso.

    Args:
        x (torch.Tensor): Tensor de características de nodos.
        edge_index (torch.Tensor): Tensor de PyTorch Geometric con los enlaces del grafo (2, num_edges).
        edge_weight (torch.Tensor): Tensor de pesos de las aristas (num_edges,).
        nodes_U_idx (torch.Tensor): Índices numéricos de los nodos del conjunto U.
        nodes_V_idx (torch.Tensor): Índices numéricos de los nodos del conjunto V.
        drop_feat_p (float): Probabilidad de drop de características.
        drop_edge_p (float): Probabilidad base de drop de aristas.

    Returns:
        tuple: x_aug, edge_index_aug, edge_weight_aug
    """
    # 1. Feature dropping
    x_aug = x.clone()
    feat_mask = torch.rand(x_aug.size(), device=x.device) > drop_feat_p
    x_aug = x_aug * feat_mask.long() # Asumiendo que x son IDs de tipos, no floats. Si son floats, usar .float()

    # 2. Edge dropping (Consciente del peso)
    if edge_index.numel() == 0: # Si no hay aristas
        return x_aug, edge_index.clone(), edge_weight.clone()

    min_weight = edge_weight.min()
    max_weight = edge_weight.max()

    if max_weight == min_weight: # Todos los pesos son iguales
        # Probabilidad uniforme de mantener basada en drop_edge_p
        keep_probabilities = torch.full_like(edge_weight, 1.0 - drop_edge_p)
    else:
        # Escalar los pesos a un rango [0, 1] y usar como probabilidad de mantener.
        # Ajuste para que 'drop_edge_p' sea una base y los pesos influyan.
        # Por ejemplo, si drop_edge_p=0.2, queremos mantener al menos el 80% de los enlaces muy pesados.
        # Y un porcentaje menor de los enlaces muy ligeros.
        # Una función logística o simplemente un escalado lineal puede funcionar.
        
        # Una forma: Probabilidad de drop inversamente proporcional al peso.
        # Donde un peso alto significa baja probabilidad de drop.
        # scaled_weights = (edge_weight - min_weight) / (max_weight - min_weight)
        # Aquí, 1 - drop_edge_p es el mínimo de probabilidad de mantener (para el peso más bajo)
        # Y 1.0 es el máximo (para el peso más alto).
        
        # Linear scaling from (1 - drop_edge_p) to 1.0
        # keep_probabilities = (1.0 - drop_edge_p) + scaled_weights * drop_edge_p
        
        # Una estrategia más simple y robusta: la probabilidad de mantener es proporcional al peso normalizado,
        # pero con un umbral mínimo.
        normalized_weights = edge_weight / max_weight # Escalar entre 0 y 1 (o valor máximo real)
        # Esto hace que los enlaces con peso 1 tengan una probabilidad 'base' de mantenerse,
        # y los enlaces con pesos más altos, más probabilidad.
        keep_probabilities = normalized_weights # Probabilidad de mantener = peso normalizado

        # Ajustar para asegurar que la probabilidad de mantener sea al menos 1 - drop_edge_p
        # y que no exceda 1.0
        keep_probabilities = torch.clamp(keep_probabilities, min=1.0 - drop_edge_p, max=1.0)


    edge_random_mask = torch.rand(edge_index.size(1), device=edge_index.device)
    edge_keep_mask = edge_random_mask < keep_probabilities
    
    edge_index_aug = edge_index[:, edge_keep_mask]
    edge_weight_aug = edge_weight[edge_keep_mask]

    return x_aug, edge_index_aug, edge_weight_aug

def corrupt(x, edge_index, edge_weight_original, nodes_U_idx, nodes_V_idx, min_corrupted_edge_weight=0.01):
    """
    Corrompe el grafo generando una nueva vista con características aleatorias y aristas aleatorias bipartitas.

    Args:
        x (torch.Tensor): Tensor de características de nodos.
        edge_index (torch.Tensor): Tensor de PyTorch Geometric con los enlaces del grafo original.
        edge_weight_original (torch.Tensor): Tensor de pesos de las aristas del grafo original.
        nodes_U_idx (torch.Tensor): Índices numéricos de los nodos del conjunto U.
        nodes_V_idx (torch.Tensor): Índices numéricos de los nodos del conjunto V.
        min_corrupted_edge_weight (float): Peso bajo asignado a las aristas corruptas.

    Returns:
        tuple: x_corrupted, edge_index_corrupted, edge_weight_corrupted
    """
    # 1. Feature shuffling (SHUFFLEROWS(X))
    x_corrupted = x[torch.randperm(x.size(0), device=x.device)]

    # 2. Random Edge Generation (RANDOMEDGE_BIPARTITE_WEIGHTED)
    # Generar nuevas aristas aleatorias respetando la bipartición.
    # El número de aristas corruptas se mantiene similar al número de aristas originales.
    num_corrupt_edges = edge_index.size(1)

    if num_corrupt_edges == 0: # Si no hay aristas originales, no generes corruptas
        return x_corrupted, torch.empty((2, 0), dtype=torch.long, device=x.device), torch.empty((0,), dtype=torch.float, device=x.device)

    # Muestrear nodos de U y V para crear aristas aleatorias
    idx_u = torch.randint(0, len(nodes_U_idx), (num_corrupt_edges,), device=x.device)
    idx_v = torch.randint(0, len(nodes_V_idx), (num_corrupt_edges,), device=x.device)

    random_edges_u = nodes_U_idx[idx_u]
    random_edges_v = nodes_V_idx[idx_v]

    edge_index_corrupted = torch.stack([random_edges_u, random_edges_v], dim=0)

    # Asignar un peso bajo a las aristas corruptas.
    edge_weight_corrupted = torch.full((num_corrupt_edges,), min_corrupted_edge_weight, device=x.device)

    return x_corrupted, edge_index_corrupted, edge_weight_corrupted