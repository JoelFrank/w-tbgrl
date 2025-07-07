# Carga y preparación del grafo desde tus DataFrames
# data/dataset_loader.py
import torch
import networkx as nx
import pandas as pd

def construir_grafo_bipartito(df_list):
    """Construye un único grafo a partir de una lista de DataFrames."""
    G = nx.Graph()
    for df in df_list:
        for _, row in df.iterrows():
            nodo_a, nodo_b, tipo_a = row['OBSERVAÇÃO PATRIMÔNIO'], row['LOCALIZAÇÃO'], row['TIPO DO EQUIPAMENTO']
            if not G.has_node(nodo_a): G.add_node(nodo_a, tipo=tipo_a, node_type='A')
            if not G.has_node(nodo_b): G.add_node(nodo_b, tipo=-1, node_type='B')
            G.add_edge(nodo_a, nodo_b)
    return G

def preparar_dataset_desde_splits(df_train, df_val, df_test):
    """Prepara todos los tensores necesarios a partir de los DataFrames ya divididos."""
    print("Preparando dataset desde los splits de dataframes...")
    G_full = construir_grafo_bipartito([df_train, df_val, df_test])
    
    all_nodes = list(G_full.nodes)
    id2idx = {nid: i for i, nid in enumerate(all_nodes)}
    idx2id = {i: nid for nid, i in id2idx.items()}

    tipo_unicos = list(pd.concat([df_train, df_val, df_test])['TIPO DO EQUIPAMENTO'].unique())
    tipo2id = {t: i + 1 for i, t in enumerate(tipo_unicos)}
    tipo_ids = torch.tensor([tipo2id.get(G_full.nodes[n]['tipo'], 0) for n in all_nodes])
    mask_embed = torch.tensor([1 if G_full.nodes[n]['tipo'] != -1 else 0 for n in all_nodes])

    def extraer_aristas(df, id2idx_map):
        return [(id2idx_map[u], id2idx_map[v]) for _, (u, v) in df[['OBSERVAÇÃO PATRIMÔNIO', 'LOCALIZAÇÃO']].iterrows() if u in id2idx_map and v in id2idx_map]

    train_edges = extraer_aristas(df_train, id2idx)
    val_edges = extraer_aristas(df_val, id2idx)
    test_edges = extraer_aristas(df_test, id2idx)
    
    dataset = {
        'num_nodes': len(all_nodes),
        'num_tipos': len(tipo2id) + 1,
        'tipo_ids': tipo_ids,
        'mask_embed': mask_embed,
        'G_full': G_full,
        'id2idx': id2idx
    }
    edges = {'train': train_edges, 'val': val_edges, 'test': test_edges}
    
    print("Preparación de datos completa.")
    return dataset, edges