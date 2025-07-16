# training/trainer.py
import copy
import torch
import torch.nn.functional as F
from utils.augmentations import augment, corrupt
from utils.evaluation import evaluate
import logging
import pandas as pd # Importar pandas para el DataFrame de predicciones

def update_target_network(online_net, target_net, ema_decay):
    """
    Actualiza los pesos de la red target utilizando un promedio móvil exponencial (EMA)
    de los pesos de la red online.
    """
    for param_q, param_k in zip(online_net.parameters(), target_net.parameters()):
        param_k.data = param_k.data * ema_decay + param_q.data * (1.0 - ema_decay)

def pretrain_tbgrl(encoder, predictor, data, hparams):
    """
    Realiza el pre-entrenamiento de T-BGRL para el encoder.

    Args:
        encoder (nn.Module): El modelo del encoder (GCNEncoder).
        predictor (nn.Module): El modelo del predictor.
        data (dict): Diccionario con 'x' (features), 'edge_index', 'edge_weight',
                     'nodes_U_idx', 'nodes_V_idx'.
        hparams (dict): Diccionario de hiperparámetros.

    Returns:
        nn.Module: El encoder pre-entrenado (online_encoder).
    """
    logging.info("Configurando pre-entrenamiento de T-BGRL...")
    online_encoder = encoder
    target_encoder = copy.deepcopy(online_encoder)
    
    # Asegurarse de que el target_encoder no se actualiza por gradientes directos
    for param in target_encoder.parameters():
        param.requires_grad = False

    # El optimizador solo actualiza el online_encoder y el predictor
    optimizer = torch.optim.Adam(list(online_encoder.parameters()) + list(predictor.parameters()), lr=hparams['lr'])
    
    # Extraer datos y configurar dispositivo
    x = data['x']
    edge_index = data['edge_index']
    edge_weight = data['edge_weight']
    nodes_U_idx = data['nodes_U_idx']
    nodes_V_idx = data['nodes_V_idx']
    
    # Mover al dispositivo de los modelos
    device = x.device

    for epoch in range(hparams['pretrain_epochs']):
        online_encoder.train()
        predictor.train()
        target_encoder.eval() # Target encoder siempre en modo evaluación

        optimizer.zero_grad()

        # Generar vistas aumentadas y corruptas
        # edge_weight se pasa a augment y corrupt
        x1, edge_index1, edge_weight1 = augment(
            x, edge_index, edge_weight, nodes_U_idx, nodes_V_idx,
            hparams['drop_feat_p'], hparams['drop_edge_p']
        )
        x2, edge_index2, edge_weight2 = augment(
            x, edge_index, edge_weight, nodes_U_idx, nodes_V_idx,
            hparams['drop_feat_p'], hparams['drop_edge_p']
        )
        xc, edge_index_c, edge_weight_c = corrupt(
            x, edge_index, edge_weight, nodes_U_idx, nodes_V_idx, # Se pasa edge_weight_original aquí
            hparams['min_corrupted_edge_weight']
        )

        # Forward pass: obtener embeddings de nodos para las tres vistas
        z1_nodes = predictor(online_encoder(x1, edge_index1, edge_weight1))
        h2_nodes = target_encoder(x2, edge_index2, edge_weight2)
        hc_nodes = target_encoder(xc, edge_index_c, edge_weight_c)

        # Normalizar embeddings (se hace para el cálculo de similitud coseno)
        z1_nodes = F.normalize(z1_nodes, dim=1)
        h2_nodes = F.normalize(h2_nodes, dim=1)
        hc_nodes = F.normalize(hc_nodes, dim=1)

        # Calcular pérdida de T-BGRL
        # Término atractivo (maximizar similitud entre z1 y h2)
        loss_attractive = - (z1_nodes * h2_nodes).sum(dim=1).mean()
        
        # Término repulsivo (minimizar similitud entre z1 y hc)
        # Esto es lo contrario del atractivo, por lo que el signo es positivo
        loss_repulsive = (z1_nodes * hc_nodes).sum(dim=1).mean()

        # Combinación de pérdidas (siguiendo la fórmula de T-BGRL / BYOL)
        loss = hparams['lambda'] * loss_repulsive + (1 - hparams['lambda']) * loss_attractive

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Actualizar la red target con EMA
        update_target_network(online_encoder, target_encoder, hparams['ema_decay'])
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Pre-train Epoch {epoch + 1}/{hparams['pretrain_epochs']}, Loss: {loss.item():.4f}, Attractive: {loss_attractive.item():.4f}, Repulsive: {loss_repulsive.item():.4f}")

    return online_encoder # Retorna el encoder online, que es el que se ha entrenado


def train_link_predictor(decoder, final_node_embeddings, data_splits, hparams, device, all_edges_set):
    """
    Entrena el Link Predictor (decoder) utilizando los embeddings de nodos fijos.

    Args:
        decoder (nn.Module): El modelo del Link Predictor.
        final_node_embeddings (torch.Tensor): Embeddings de nodos pre-entrenados (fijos).
        data_splits (dict): Diccionario con los enlaces de train, val, test, etc.
        hparams (dict): Diccionario de hiperparámetros.
        device (torch.device): El dispositivo (CPU/GPU) donde se deben crear los tensores.
        all_edges_set (set): Conjunto de todas las aristas existentes en el grafo completo (para generar negativos).
        
    Returns:
        tuple: (test_hits_k, test_roc_auc, test_ap_score, test_predictions_df)
    """
    logging.info("Iniciando entrenamiento del Link Predictor...")
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=hparams['lp_lr'])
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none') # Usar reduction='none' para aplicar pesos

    train_pos_edges = data_splits['train_pos_edges']
    train_pos_weights = data_splits['train_pos_weights']
    
    val_pos_edges = data_splits['val_pos_edges']
    test_pos_edges = data_splits['test_pos_edges']

    nodes_U_idx = data_splits['nodes_U_idx']
    nodes_V_idx = data_splits['nodes_V_idx']
    idx2id = data_splits['idx2id']
    k_eval = hparams['k']

    # Entrenar por épocas
    for epoch in range(hparams['lp_epochs']):
        decoder.train()
        optimizer_decoder.zero_grad()

        # Generar un número igual de negativos para el entrenamiento del Link Predictor
        num_train_pos_edges = train_pos_edges.size(1)
        
        # Generar negativos que respeten la estructura bipartita
        num_train_neg_edges = num_train_pos_edges
        u_neg_idx = torch.randint(0, len(nodes_U_idx), (num_train_neg_edges,), device=device)
        v_neg_idx = torch.randint(0, len(nodes_V_idx), (num_train_neg_edges,), device=device)

        train_neg_edges_sampled = torch.stack([nodes_U_idx[u_neg_idx], nodes_V_idx[v_neg_idx]], dim=0)

        # Eliminar los negativos que por casualidad son positivos existentes
        # Esto asegura que los 'negativos' de entrenamiento sean realmente no-aristas
        unique_train_neg_edges_list = []
        for i in range(train_neg_edges_sampled.size(1)):
            u_node = train_neg_edges_sampled[0, i].item()
            v_node = train_neg_edges_sampled[1, i].item()
            canonical_edge = tuple(sorted((u_node, v_node))) # Para grafos no dirigidos
            if canonical_edge not in all_edges_set:
                unique_train_neg_edges_list.append([u_node, v_node])
        
        if unique_train_neg_edges_list:
            train_neg_edges_sampled_filtered = torch.tensor(unique_train_neg_edges_list, dtype=torch.long).t().contiguous().to(device)
        else:
            logging.warning(f"Epoch {epoch+1}: No se pudieron generar negativos únicos válidos para entrenamiento del Link Predictor. Saltando esta época.")
            continue # Saltar esta época si no hay negativos válidos

        # Combinar positivos y negativos
        all_train_edges = torch.cat([train_pos_edges, train_neg_edges_sampled_filtered], dim=1)
        train_labels = torch.cat([
            torch.ones(train_pos_edges.size(1), device=device),
            torch.zeros(train_neg_edges_sampled_filtered.size(1), device=device)
        ], dim=0)

        # Pesos para la pérdida (1.0 para negativos, pesos reales para positivos)
        train_neg_weights = torch.ones(train_neg_edges_sampled_filtered.size(1), device=device)
        all_train_weights = torch.cat([train_pos_weights, train_neg_weights], dim=0)


        # Obtener embeddings para los extremos de los enlaces de entrenamiento
        u_emb = final_node_embeddings[all_train_edges[0]]
        v_emb = final_node_embeddings[all_train_edges[1]]

        # Calcular scores de enlaces con el decoder
        predicted_scores = decoder(u_emb, v_emb).squeeze() # .squeeze() para asegurar que tenga 1D

        # Calcular la pérdida BCE ponderada
        loss = bce_loss(predicted_scores, train_labels)
        weighted_loss = (loss * all_train_weights).mean() # Aplicar la ponderación

        weighted_loss.backward()
        optimizer_decoder.step()

        # Evaluación en validación cada 'eval_every' épocas
        if (epoch + 1) % hparams['eval_every'] == 0:
            decoder.eval()
            with torch.no_grad():
                logging.info(f"Link Predictor - Epoch {epoch + 1}/{hparams['lp_epochs']}:")
                val_neg_edges_for_eval = generate_random_bipartite_negatives(
                    val_pos_edges, all_edges_set, nodes_U_idx, nodes_V_idx, device
                )
                
                hits_k, roc_auc, ap_score = evaluate(
                    decoder, final_node_embeddings, val_pos_edges, val_neg_edges_for_eval,
                    nodes_U_idx, nodes_V_idx, k_eval, idx2id
                )
                logging.info(f"  Val Loss: {weighted_loss.item():.4f}, Hits@K: {hits_k:.4f}, ROC-AUC: {roc_auc:.4f}, AP: {ap_score:.4f}")
            decoder.train() # Volver a modo entrenamiento

    # Evaluación final en el conjunto de prueba y preparación de DataFrame de predicciones
    logging.info("Realizando evaluación final del Link Predictor en el conjunto de prueba...")
    decoder.eval()
    test_predictions_df = None # Inicializar a None
    with torch.no_grad():
        test_neg_edges_for_eval = generate_random_bipartite_negatives(
            test_pos_edges, all_edges_set, nodes_U_idx, nodes_V_idx, device
        )
        
        if test_neg_edges_for_eval.numel() == 0 and test_pos_edges.numel() == 0:
             logging.warning("No hay enlaces positivos ni negativos en el conjunto de prueba para evaluar.")
             test_hits_k, test_roc_auc, test_ap_score = 0.0, 0.0, 0.0
        else:
            test_hits_k, test_roc_auc, test_ap_score = evaluate(
                decoder, final_node_embeddings, test_pos_edges, test_neg_edges_for_eval,
                nodes_U_idx, nodes_V_idx, k_eval, idx2id
            )
            logging.info(f"Resultados Finales del Link Predictor en Test: Hits@K: {test_hits_k:.4f}, ROC-AUC: {test_roc_auc:.4f}, AP: {test_ap_score:.4f}")

            # --- Preparar DataFrame de predicciones para el test set ---
            all_test_edges = torch.cat([test_pos_edges, test_neg_edges_for_eval], dim=1)
            all_test_labels = torch.cat([
                torch.ones(test_pos_edges.size(1), device=device),
                torch.zeros(test_neg_edges_for_eval.size(1), device=device)
            ], dim=0)

            test_u_emb = final_node_embeddings[all_test_edges[0]]
            test_v_emb = final_node_embeddings[all_test_edges[1]]
            test_predicted_scores = torch.sigmoid(decoder(test_u_emb, test_v_emb)).squeeze().cpu().numpy() # Usar sigmoid para convertir logits a probabilidades

            test_predictions_df = pd.DataFrame({
                'u_original_id': [idx2id[u.item()] for u in all_test_edges[0]],
                'v_original_id': [idx2id[v.item()] for v in all_test_edges[1]],
                'true_label': all_test_labels.cpu().numpy(),
                'predicted_score': test_predicted_scores
            })

    return test_hits_k, test_roc_auc, test_ap_score, test_predictions_df

# Helper function to generate negatives for evaluation (can also be in utils/evaluation.py)
def generate_random_bipartite_negatives(pos_edges, all_existing_edges_set, nodes_U_idx, nodes_V_idx, device, num_neg_samples=None):
    """
    Genera un número de enlaces negativos aleatorios para evaluación,
    asegurando que no sean enlaces existentes y respetando la bipartición.
    Si num_neg_samples es None, genera el mismo número que pos_edges.
    """
    num_neg_samples = num_neg_samples if num_neg_samples is not None else pos_edges.size(1)
    neg_edges = []
    attempts = 0
    max_attempts = num_neg_samples * 10 # Aumentar intentos para grafos más densos o pocos negativos

    nodes_U_list = nodes_U_idx.tolist()
    nodes_V_list = nodes_V_idx.tolist()
    
    # Crear un set temporal para evitar duplicados en la generación actual de negativos
    # y no modificar el all_existing_edges_set original del dataset_info
    current_neg_candidates_set = set()

    while len(neg_edges) < num_neg_samples and attempts < max_attempts:
        u_idx_rand = torch.randint(0, len(nodes_U_list), (1,)).item()
        v_idx_rand = torch.randint(0, len(nodes_V_list), (1,)).item()
        
        u_node = nodes_U_list[u_idx_rand]
        v_node = nodes_V_list[v_idx_rand]
        
        # Crear la tupla canónica para el enlace (asegurando el orden para el set)
        canonical_edge_tuple = tuple(sorted((u_node, v_node))) 
        
        # Verificar si el enlace no existe en el conjunto de TODOS los enlaces existentes
        # Y que no se haya añadido ya en esta tanda de generación de negativos
        if canonical_edge_tuple not in all_existing_edges_set and \
           canonical_edge_tuple not in current_neg_candidates_set:
            neg_edges.append([u_node, v_node])
            current_neg_candidates_set.add(canonical_edge_tuple) # Añadir al set temporal
            
        attempts += 1
    
    if len(neg_edges) < num_neg_samples:
        logging.warning(f"No se pudieron generar suficientes negativos ({len(neg_edges)}/{num_neg_samples} generados) después de {attempts} intentos. El grafo puede ser muy denso o el conjunto de negativos limitado para el muestreo aleatorio.")

    if neg_edges:
        return torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(device)
    else:
        return torch.empty((2, 0), dtype=torch.long, device=device)