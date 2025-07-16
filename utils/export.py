import networkx as nx
import logging

def export_splits_to_gephi(edges_and_weights_dict, G_full, idx2id):
    """
    Crea un grafo para cada split (train, val, test) con aristas ponderadas
    y lo exporta a un archivo .gexf para Gephi.
    
    Args:
        edges_and_weights_dict (dict): Diccionario que contiene 'train_edge_index',
                                       'train_edge_weight', 'val_edge_index', etc.
                                       Los edge_index son tensores PyTorch.
        G_full (nx.Graph): El grafo completo NetworkX con todos los nodos y sus atributos.
        idx2id (dict): Mapeo de los índices de nodos numéricos a sus nombres originales.
    """
    logging.info("Iniciando exportación de grafos de splits para Gephi (con pesos)...")

    # Mapeo de nombres de split a las claves en edges_and_weights_dict
    split_keys = {
        'train': ('train_edge_index', 'train_edge_weight'),
        'val': ('val_edge_index', 'val_edge_weight'),
        'test': ('test_edge_index', 'test_edge_weight')
    }

    for split_name, keys in split_keys.items():
        edge_index_key, edge_weight_key = keys
        
        edge_index_tensor = edges_and_weights_dict.get(edge_index_key)
        edge_weight_tensor = edges_and_weights_dict.get(edge_weight_key)

        # Verificar si el tensor de edge_index está vacío o no existe
        if edge_index_tensor is None or edge_index_tensor.numel() == 0:
            logging.warning(f"No hay enlaces para el split '{split_name}'. No se generará el grafo.")
            continue

        # Convertir tensores PyTorch a listas de Python
        edge_indices = edge_index_tensor.t().tolist() # Transponer para obtener [(u,v), ...] y luego a lista
        edge_weights = edge_weight_tensor.tolist()

        # 1. Crear un nuevo grafo vacío para este split
        G_split = nx.Graph()

        # 2. Añadir los enlaces con sus pesos usando los nombres originales de los nodos
        for i, (u_idx, v_idx) in enumerate(edge_indices):
            u_orig = idx2id[u_idx]
            v_orig = idx2id[v_idx]
            weight = edge_weights[i]
            G_split.add_edge(u_orig, v_orig, weight=weight)

        # 3. Copiar los atributos de los nodos desde el grafo completo
        # Esto asegura que 'tipo', 'tipo_id', y 'node_type' (entre otros) se incluyan
        for node_orig_id in G_split.nodes():
            if node_orig_id in G_full:
                G_split.nodes[node_orig_id].update(G_full.nodes[node_orig_id])

        # 4. Exportar el grafo a un archivo .gexf
        output_path = f"grafo_{split_name}.gexf"
        try:
            nx.write_gexf(G_split, output_path)
            logging.info(f"Grafo del split '{split_name}' (ponderado) exportado en: {output_path}")
        except Exception as e:
            logging.error(f"No se pudo exportar el grafo '{split_name}': {e}")
            
    logging.info("Exportación de grafos completada.")