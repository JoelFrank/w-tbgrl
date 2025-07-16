# utils/evaluation.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

def evaluate(decoder, final_node_embeddings, pos_edges, neg_edges, nodes_U_idx, nodes_V_idx, k, idx2id):
    """
    Evalúa el rendimiento del Link Predictor.
    Genera negativos si neg_edges es None.

    Args:
        decoder (nn.Module): El Link Predictor entrenado.
        final_node_embeddings (torch.Tensor): Embeddings de nodos fijos.
        pos_edges (torch.Tensor): Tensor de enlaces positivos (2, num_pos_edges).
        neg_edges (torch.Tensor): Tensor de enlaces negativos (2, num_neg_edges). Si es None, se generan.
        nodes_U_idx (torch.Tensor): Índices numéricos de nodos del conjunto U.
        nodes_V_idx (torch.Tensor): Índices numéricos de nodos del conjunto V.
        k (int): Parámetro para la métrica Hits@K.
        idx2id (dict): Mapeo de índices numéricos a IDs originales.

    Returns:
        tuple: (hits_k, roc_auc, ap_score)
    """
    decoder.eval() # Poner el decoder en modo evaluación
    device = final_node_embeddings.device

    with torch.no_grad():
        if pos_edges.numel() == 0:
            logging.warning("No hay enlaces positivos para evaluar.")
            return 0.0, 0.0, 0.0

        # Si no se proporcionan negativos, generarlos (e.g., el mismo número que positivos)
        if neg_edges is None or neg_edges.numel() == 0:
            # Necesitamos all_existing_edges_set que no está en data_splits directamente aquí
            # Se asume que trainer.py pasa all_edges_set si necesita generarlos aquí
            # O que generate_random_bipartite_negatives es más robusto.
            # Por simplicidad para el ejemplo, podemos usar una versión simplificada o esperar que se pasen.
            
            # --- Para evitar dependencia circular, generate_random_bipartite_negatives se ha movido a trainer.py ---
            # Si se llama desde aquí, necesitaría `all_edges_set` del dataset_info
            
            # temporalmente:
            logging.warning("No se proporcionaron enlaces negativos. Generando negativos aleatorios para evaluación. Considera pre-generarlos para consistencia.")
            # Crear un all_edges_set temporal (ineficiente para grafos grandes)
            # Para una solución robusta, all_edges_set debería pasarse como argumento desde main
            all_existing_edges_set_temp = set(tuple(e) for e in pos_edges.t().tolist())
            # Puedes añadir los de entrenamiento/validación para una lista completa de existentes.

            # Ahora, usa la función auxiliar para generar negativos
            from training.trainer import generate_random_bipartite_negatives # Importar de donde se definió
            neg_edges = generate_random_bipartite_negatives(
                pos_edges, all_existing_edges_set_temp, nodes_U_idx, nodes_V_idx, device,
                num_neg_samples=pos_edges.size(1) # Generar el mismo número de negativos que positivos
            )
            if neg_edges.numel() == 0:
                logging.warning("No se pudieron generar enlaces negativos para la evaluación. Saltando métricas que los requieren.")
                return 0.0, 0.0, 0.0 # No se puede calcular ROC-AUC/AP sin negativos

        # Calcular scores para enlaces positivos
        pos_u_emb = final_node_embeddings[pos_edges[0]]
        pos_v_emb = final_node_embeddings[pos_edges[1]]
        pos_scores = decoder(pos_u_emb, pos_v_emb).squeeze().cpu().numpy()

        # Calcular scores para enlaces negativos
        neg_u_emb = final_node_embeddings[neg_edges[0]]
        neg_v_emb = final_node_embeddings[neg_edges[1]]
        neg_scores = decoder(neg_u_emb, neg_v_emb).squeeze().cpu().numpy()

        # Concatenar scores y etiquetas verdaderas para ROC-AUC y AP
        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])

        # Calcular ROC-AUC
        roc_auc = roc_auc_score(all_labels, all_scores)

        # Calcular Average Precision (AP)
        ap_score = average_precision_score(all_labels, all_scores)

        # Calcular Hits@K
        # Para Hits@K, iteramos sobre cada enlace positivo y lo comparamos con un conjunto de negativos.
        # Esto es más intensivo computacionalmente.
        hits_at_k = 0
        num_pos = pos_edges.size(1)
        
        # Para cada enlace positivo, lo mezclamos con K-1 negativos aleatorios y vemos si está en top K.
        # Una forma más sencilla para la evaluación: para cada nodo 'u' en pos_edges,
        # clasificamos todos sus posibles enlaces 'v' (positivos y negativos) y vemos si el positivo está en el top-K.

        # Simplificación de Hits@K para no generar N*M negativos
        # Consideramos cada enlace positivo y un conjunto de negativos generales.
        # Esto no es el Hits@K más riguroso (que es por cada 'u' vs sus 'no-v's), pero es una aproximación común.
        
        # Si se necesitan negativos específicos por cada nodo U en los pos_edges,
        # la función `evaluate` o su llamada debería proveerlos.

        # Implementación simplificada de Hits@K (Ranking):
        # Para cada positivo, lo comparamos con un conjunto de negativos.
        # No es el ranking por nodo, sino un ranking global.
        # hits_at_k = np.sum(pos_scores > np.mean(neg_scores)) / num_pos # Ejemplo muy simplificado

        # Una aproximación más adecuada para Hits@K:
        # Por cada nodo u en los enlaces positivos, combinamos su enlace positivo (u,v_pos)
        # con K negativos arbitrarios (u, v_neg) y verificamos si (u,v_pos) está en el top K.
        # Dado que no tenemos 'v_neg' específicos para cada 'u', usaremos el conjunto general de negativos.
        
        # Una implementación estándar de Hits@k para link prediction implica:
        # Para cada positive edge (u, v_true):
        #   1. Generar N_neg negativos de 'u' (u, v_fake_1), ..., (u, v_fake_N_neg).
        #   2. Calcular scores para (u, v_true) y todos (u, v_fake_i).
        #   3. Clasificar y ver si v_true está en el top-k.
        
        # Esta implementación de evaluate solo recibe un conjunto global de negativos.
        # Necesitamos adaptar o modificar la forma de calcular Hits@K.
        # Por ahora, se calculará de una manera que puede no ser la "ideal" Hits@K del paper original
        # si esta esperaba negativos por cada positivo.

        # Si el Link Predictor está diseñado para ranking:
        # Los scores pos_scores deben ser mayores que neg_scores.
        # hits_at_k = (np.sum(pos_scores > np.percentile(neg_scores, 100 - (k * 100 / len(neg_scores)))) / num_pos) if num_pos > 0 else 0.0

        # Para una implementación más fiel a Hits@K (por muestreo):
        num_hits = 0
        for i in range(pos_edges.size(1)):
            u_node_idx = pos_edges[0, i].item()
            v_pos_idx = pos_edges[1, i].item()
            
            # Score del enlace positivo
            pos_score = decoder(final_node_embeddings[u_node_idx].unsqueeze(0), final_node_embeddings[v_pos_idx].unsqueeze(0)).squeeze().item()

            # Muestrear K-1 negativos (o más) para este u_node_idx
            # Esto puede ser ineficiente si se hace para cada positivo
            candidate_neg_v_indices = []
            
            # Obtener todos los nodos V disponibles que no están conectados a u_node_idx
            all_v_nodes_list = nodes_V_idx.tolist()
            
            # Esto requeriría el grafo NetworkX completo o un conjunto de aristas ya existentes para evitar colisiones
            # Asumimos que all_existing_edges_set (desde dataset_info) está disponible y se puede pasar.
            # NOTA: Para que esto funcione, evaluate necesita all_existing_edges_set
            
            # Por ahora, Hits@K de esta implementación de evaluate será más un "pseudo Hits@K"
            # basado en el ranking general entre todos los positivos y todos los negativos proporcionados.
            # Si el usuario quiere el Hits@K más riguroso, debe pasarse `all_edges_set` a `evaluate`
            # y la función `generate_random_bipartite_negatives` debe ser llamada iterativamente.

            # Hits@K básico para el conjunto: ¿Cuántos positivos están por encima del k-ésimo negativo más alto?
            # Esto es más un ranking general, no el ranking individual por nodo.
            
            # Simple Hits@K: cuántos positivos están por encima del umbral k-ésimo más alto de los negativos
            if len(neg_scores) >= k:
                # Ordenar scores negativos y tomar el k-ésimo más alto
                sorted_neg_scores = np.sort(neg_scores)[::-1] # Orden descendente
                kth_neg_score = sorted_neg_scores[k - 1]
                num_hits = np.sum(pos_scores >= kth_neg_score) # Cuántos positivos son mejores o iguales que el k-ésimo neg.
                hits_at_k = num_hits / len(pos_scores) if len(pos_scores) > 0 else 0.0
            else:
                hits_at_k = 0.0
                logging.warning(f"No hay suficientes enlaces negativos ({len(neg_scores)}) para calcular Hits@{k}. Hits@K será 0.")
                # Si no hay suficientes negativos, no se puede formar un top-K significativo.


        return hits_at_k, roc_auc, ap_score