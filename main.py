# main.py
import yaml
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import copy
import os
import logging
import time

# Importar los módulos del proyecto
from data.dataset_loader import preparar_dataset_desde_splits
from models.gcn_encoder import GCNEncoder
from models.predictor import Predictor
from models.link_predictor import LinkPredictor
from training.trainer import pretrain_tbgrl, train_link_predictor
from utils.evaluation import evaluate
from utils.export import export_splits_to_gephi # <-- Nueva importación


# --- Bloque para guardar las ejecuciones ---
# incluirá el tiempo que tardó cada una de las 5 ejecuciones, y al final, un resumen con el tiempo total y el promedio.
log_file_path = 'experimento.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', # Se ha simplificado el formato para mayor claridad
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)

def get_edge_index_from_edges(edges):
    """Crea un tensor de edge_index no dirigido a partir de una lista de enlaces."""
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    tensor_edges = torch.tensor(edges, dtype=torch.long).t()
    return torch.cat([tensor_edges, tensor_edges.flip(0)], dim=1)

if __name__ == "__main__":
    # === 1. Cargar Configuración ===
    with open("configs/tbgrl_config.yaml", "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)
    logging.info(f"--- Iniciando Experimento: MODO {hparams['split_method'].upper()} ---")

    # === 2. Cargar y Dividir Datos Cronológicamente ===
    file_path = "data/raw/SISMETRO-Exportação-SS-2021-2024_P1(OK PATRIMONIO).xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo de datos no encontrado en: {file_path}")
        
    df_total = pd.read_excel(file_path)
    
    df_total['DATA DE ABERTURA'] = pd.to_datetime(df_total['DATA DE ABERTURA'])
    df_total = df_total.sort_values(by='DATA DE ABERTURA').reset_index(drop=True)

    n = len(df_total)
    test_size = int(hparams['test_pct'] * n)
    val_size = int(hparams['val_pct'] * n)
    train_size = n - val_size - test_size

    df_train = df_total.iloc[:train_size].copy()
    df_val = df_total.iloc[train_size:train_size + val_size].copy()
    df_test = df_total.iloc[train_size + val_size:].copy()
    
    logging.info(f"División Temporal: {len(df_train)} train | {len(df_val)} val | {len(df_test)} test")

    # === 3. Preparar Grafo y Tensores ===
    dataset, edges = preparar_dataset_desde_splits(df_train, df_val, df_test)
    idx2id = {v: k for k, v in dataset['id2idx'].items()}

    # Llama a la función modularizada para exportar los grafos
    export_splits_to_gephi(edges, dataset['G_full'], idx2id)
    # --- FIN DEL BLOQUE ---

    # === 4. Bucle de Múltiples Ejecuciones ===
    # --- ### AÑADIDO: Identificar nodos por tipo para el muestreo bipartito ### ---
    G_full = dataset['G_full']
    nodes_A_idx = [dataset['id2idx'][n] for n, d in G_full.nodes(data=True) if d['node_type'] == 'A']
    nodes_B_idx = [dataset['id2idx'][n] for n, d in G_full.nodes(data=True) if d['node_type'] == 'B']
    all_edges_set = set(map(tuple, map(sorted, dataset['G_full'].edges())))

    all_test_results = []
    final_predictions_df = None

    # ### AÑADIDO ### Iniciar temporizador general del experimento
    overall_start_time = time.time()

    for run in range(hparams['num_runs']):
        logging.info(f"\n{'='*25} INICIANDO EJECUCIÓN {run + 1}/{hparams['num_runs']} {'='*25}")
        # ### AÑADIDO ### Iniciar temporizador para esta ejecución específica
        run_start_time = time.time()
        
        encoder = GCNEncoder(dataset['num_tipos'], hparams['emb_dim'], hparams['hidden_dim'], hparams['out_dim'])
        predictor = Predictor(in_dim=hparams['out_dim'])
        decoder = LinkPredictor(in_dim=hparams['out_dim'])
        
        if hparams['split_method'] == 'temporal_transductive':
            logging.info("Modo Transductivo: El encoder ve toda la estructura del grafo para aprender embeddings.")
            all_known_edges = edges['train'] + edges['val'] + edges['test']
            edge_index_encoder = get_edge_index_from_edges(all_known_edges)
        elif hparams['split_method'] == 'temporal_inductive':
            logging.info("Modo Inductivo: El encoder solo ve la estructura de entrenamiento (cold-start).")
            edge_index_encoder = get_edge_index_from_edges(edges['train'])
        else:
            raise ValueError("split_method no reconocido en el archivo de configuración.")

        dataset['features'] = dataset['tipo_ids'].float().unsqueeze(1) 
        data_for_encoder = {**dataset, 'edge_index': edge_index_encoder}
        
        encoder = pretrain_tbgrl(encoder, predictor, data_for_encoder, hparams)
        
        decoder_dataset = {
            **data_for_encoder,
            'train_edges': edges['train'],
            'val_edges': edges['val'],
            'idx2id': idx2id,
            'nodes_A_idx': nodes_A_idx,   # <-- Agrega esto
            'nodes_B_idx': nodes_B_idx,    # <-- Y esto
            'all_edges_set': all_edges_set  # <-- Agrega esto
        }
        best_decoder = train_link_predictor(encoder, decoder, decoder_dataset, hparams)
        
        edge_index_inference = get_edge_index_from_edges(edges['train'] + edges['val'])
        with torch.no_grad():
            embeddings_inference = encoder(edge_index_inference, dataset['tipo_ids'], dataset['mask_embed'])
        
        test_results, predictions = evaluate(
            best_decoder,
            embeddings_inference,
            edges['test'],
            all_edges_set, # Pasar el set de todos los enlaces para evitar falsos negativos
            nodes_A_idx,   # Pasar la lista de nodos A
            nodes_B_idx,   # Pasar la lista de nodos B
            k=hparams['k_hits'],
            idx2id=idx2id
            )
        all_test_results.append(test_results)
        
        logging.info("\nResultados de la Ejecución en Test:")
        for k, v in test_results.items(): logging.info(f"{k}: {v:.4f}")
        
        if run == hparams['num_runs'] - 1:
            final_predictions_df = pd.DataFrame(predictions)

        # ### AÑADIDO ### Calcular y mostrar el tiempo de la ejecución actual
        run_end_time = time.time()
        logging.info(f"--> Tiempo de la Ejecución {run + 1}: {run_end_time - run_start_time:.2f} segundos")

    # === 5. Agregación y Guardado de Resultados Finales ===
    final_metrics = defaultdict(list)
    for res in all_test_results:
        for metric, value in res.items(): final_metrics[metric].append(value)
            
    logging.info(f"\n{'='*20} RESULTADO FINAL AGREGADO ({hparams['split_method'].upper()}) {'='*20}")
    for metric, values in final_metrics.items():
        mean, std = np.mean(values), np.std(values)
        logging.info(f"{metric}: {mean:.4f} ± {std:.4f}")
    
    # ### AÑADIDO ### Calcular y mostrar tiempos totales y promedios
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    avg_duration = total_duration / hparams['num_runs']
    logging.info("\n--- Tiempos de Ejecución ---")
    logging.info(f"Tiempo total para {hparams['num_runs']} ejecuciones: {total_duration:.2f} segundos ({total_duration/60:.2f} minutos)")
    logging.info(f"Tiempo promedio por ejecución: {avg_duration:.2f} segundos")
    
    if final_predictions_df is not None:
        output_path = "link_predictions_test_set.csv"
        final_predictions_df.sort_values(by='score', ascending=False, inplace=True)
        final_predictions_df.to_csv(output_path, index=False, float_format='%.4f')
        logging.info(f"\nPredicciones detalladas del Test Set guardadas en: {output_path}")