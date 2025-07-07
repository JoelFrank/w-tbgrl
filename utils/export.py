# utils/export.py
import networkx as nx
import logging

def export_splits_to_gephi(edges, G_full, idx2id):
    """
    Crea un grafo para cada split (train, val, test) y lo exporta a un archivo .gexf para Gephi.
    
    Args:
        edges (dict): Diccionario con las listas de enlaces, ej: {'train': [...], 'val': [...]}
        G_full (nx.Graph): El grafo completo con todos los nodos y sus atributos.
        idx2id (dict): Mapeo de los índices de nodos a sus nombres originales.
    """
    logging.info("Iniciando exportación de grafos de splits para Gephi...")

    for split_name, edge_list in edges.items():
        if not edge_list:
            logging.warning(f"La lista de enlaces para el split '{split_name}' está vacía. No se generará el grafo.")
            continue

        # 1. Crear un nuevo grafo vacío para este split
        G_split = nx.Graph()

        # 2. Añadir los enlaces usando los nombres originales de los nodos
        for u_idx, v_idx in edge_list:
            G_split.add_edge(idx2id[u_idx], idx2id[v_idx])

        # 3. Copiar los atributos de los nodos desde el grafo completo
        for node in G_split.nodes():
            if node in G_full:
                G_split.nodes[node].update(G_full.nodes[node])

        # 4. Exportar el grafo a un archivo .gexf
        output_path = f"grafo_{split_name}.gexf"
        try:
            nx.write_gexf(G_split, output_path)
            logging.info(f"Grafo del split '{split_name}' exportado en: {output_path}")
        except Exception as e:
            logging.error(f"No se pudo exportar el grafo '{split_name}': {e}")
            
    logging.info("Exportación de grafos completada.")