# Evaluación Hits@K y ROC-AUC
# utils/evaluation.py
import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_auc_score, average_precision_score

@torch.no_grad()
def evaluate(decoder, embeddings, pos_edges, all_edges_set, nodes_A_idx, nodes_B_idx, k, idx2id):
    """
    ### CORRECCIÓN ###
    Realiza una evaluación con muestreo de negativos consciente de la estructura bipartita.
    """
    decoder.eval()
    if not pos_edges: 
        return {'Hits@K': 0, 'ROC-AUC': 0, 'AP': 0}, []

    # --- Muestreo de Negativos Bipartito ---
    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        # Muestrear un nodo de cada tipo
        u = random.choice(nodes_A_idx)
        v = random.choice(nodes_B_idx)
        # Asegurarse de que el enlace no exista en ninguna parte del grafo
        if (u, v) not in all_edges_set and (v, u) not in all_edges_set:
            neg_edges.append((u, v))
            
    pos_edges_t = torch.tensor(pos_edges, dtype=torch.long).t()
    neg_edges_t = torch.tensor(neg_edges, dtype=torch.long).t()

    # --- Cálculo de Scores y Creación del CSV ---
    all_edges_t = torch.cat([pos_edges_t, neg_edges_t], dim=1)
    labels = torch.cat([torch.ones(pos_edges_t.size(1)), torch.zeros(neg_edges_t.size(1))])
    scores = decoder(embeddings[all_edges_t[0]], embeddings[all_edges_t[1]])
    
    predictions_list = [{
        'u': idx2id[int(u_idx)], 'v': idx2id[int(v_idx)], 'label': l.item(), 'score': torch.sigmoid(s).item()
    } for u_idx, v_idx, l, s in zip(all_edges_t[0], all_edges_t[1], labels, scores)]

    # --- Cálculo de Métricas ---
    roc = roc_auc_score(labels.numpy(), torch.sigmoid(scores).numpy())
    ap = average_precision_score(labels.numpy(), torch.sigmoid(scores).numpy()) #evaluacion sigmoide 

    # --- Cálculo de Hits@K ---
    hits = []
    for u, v in pos_edges:
        #neg_samples_v = torch.tensor(random.sample(nodes_B_idx, k * 2), dtype=torch.long)
        num_samples = min(k * 2, len(nodes_B_idx))
        if num_samples > 0:
            neg_samples_v = torch.tensor(random.sample(nodes_B_idx, num_samples), dtype=torch.long)
        else:
            neg_samples_v = torch.tensor([], dtype=torch.long)

        all_targets_v = torch.cat([torch.tensor([v]), neg_samples_v])
        source_rep = torch.tensor([u]).repeat(all_targets_v.size(0))
        
        all_scores = decoder(embeddings[source_rep], embeddings[all_targets_v])
        rank = (all_scores >= all_scores[0]).sum().item()
        hits.append(1 if rank <= k else 0)

    metrics = {f'Hits@{k}': np.mean(hits), 'ROC-AUC': roc, 'AP': ap}
    
    return metrics, predictions_list