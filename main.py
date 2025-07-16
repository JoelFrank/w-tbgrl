# main.py
import torch
import yaml
import logging
import time # Importar time para medir tiempos
import numpy as np # Importar numpy para cálculos de media y std

# Importar las clases y funciones actualizadas
from data.dataset_loader import cargar_y_preparar_datos
from models.gcn_encoder import GCNEncoder
from models.predictor import Predictor
from models.link_predictor import LinkPredictor
from training.trainer import pretrain_tbgrl, train_link_predictor
from utils.evaluation import evaluate # evaluate para la evaluación final
from utils.export import export_splits_to_gephi

# --- MODIFICACIÓN AQUÍ: Configurar logging para escribir en un archivo y en la consola ---
# Crear un logger personalizado para poder añadir múltiples handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler para la consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Handler para el archivo de log
file_handler = logging.FileHandler('experiment_log.log', mode='w', encoding='utf-8') # Guarda los logs en 'experiment_log.log'
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Reemplazar logging.info/error con logger.info/error en el resto del script
# Esto no es estrictamente necesario si ya se usa logging.info, pero es una buena práctica
# cuando se configuran múltiples handlers. Por ahora, los logging.info existentes seguirán funcionando
# con el root logger, que ahora también tiene los handlers añadidos.


def main():
    logger.info("Iniciando el pipeline de WBT-GL para Link Prediction.") # Usar logger.info

    # 1. Cargar configuración de hiperparámetros
    try:
        with open('configs/tbgrl_config.yaml', 'r', encoding='utf-8') as f:
            hparams = yaml.safe_load(f)
        logger.info("Hiperparámetros cargados exitosamente.") # Usar logger.info
    except FileNotFoundError:
        logger.error("configs/tbgrl_config.yaml no encontrado. Asegúrate de que el archivo existe.") # Usar logger.error
        return
    except yaml.YAMLError as e:
        logger.error(f"Error al parsear el archivo de configuración YAML: {e}") # Usar logger.error
        return

    # 2. Configurar dispositivo (CPU/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}") # Usar logger.info

    # 3. Cargar y preparar los datos (Fuera del bucle de ejecución para que los datos sean los mismos en todas las ejecuciones)
    file_paths = ['data/raw/SISMETRO-Exportação-SS-2021-2024_P1(OK PATRIMONIO).xlsx']
    
    try:
        dataset_info, splits = cargar_y_preparar_datos(file_paths,
                                                      train_size=hparams['train_size'],
                                                      val_size=hparams['val_size'],
                                                      random_state=hparams['random_state'])
        logger.info("Datos cargados y preparados exitosamente.") # Usar logger.info
        # Log de división temporal
        logger.info(f"División Temporal: {len(dataset_info['G_full'].edges)} total enlaces. Train: {splits['train_edge_index'].size(1)} | Val: {splits['val_edge_index'].size(1)} | Test: {splits['test_edge_index'].size(1)}")
    except Exception as e:
        logger.error(f"Error al cargar o preparar los datos: {e}") # Usar logger.error
        return

    # Exportar los grafos de los splits para Gephi (Solo una vez)
    logger.info("Iniciando exportación de grafos de splits para Gephi...") # Usar logger.info
    export_splits_to_gephi(splits, dataset_info['G_full'], dataset_info['idx2id'])
    logger.info("Exportación de grafos completada.") # Usar logger.info

    # Extraer información relevante una vez para todas las ejecuciones
    features = dataset_info['tipo_ids'].to(device)
    num_node_feature_types = dataset_info['num_node_feature_types'] 
    nodes_U_idx = dataset_info['nodes_U_idx'].to(device)
    nodes_V_idx = dataset_info['nodes_V_idx'].to(device)
    all_edges_set = dataset_info['all_edges_set']

    # Convertir los edge_index y edge_weight a tensores PyTorch y mover al dispositivo
    for key in splits:
        if isinstance(splits[key], torch.Tensor):
            splits[key] = splits[key].to(device)
        elif isinstance(splits[key], tuple) and all(isinstance(t, torch.Tensor) for t in splits[key]):
            splits[key] = tuple(t.to(device) for t in splits[key])

    # Listas para almacenar resultados de múltiples ejecuciones
    all_hits_at_k = []
    all_roc_auc = []
    all_ap_score = []
    all_run_times = []

    num_executions = hparams.get('num_executions', 5) # Obtener num_executions del config, por defecto 5

    for run_idx in range(1, num_executions + 1):
        logger.info(f"\n========================= INICIANDO EJECUCIÓN {run_idx}/{num_executions} =========================") # Usar logger.info
        start_run_time = time.time() # Iniciar contador de tiempo para esta ejecución

        # Re-inicializar modelos para cada ejecución para asegurar independencia
        encoder = GCNEncoder(
            num_node_types=num_node_feature_types, 
            emb_dim=hparams['encoder_emb_dim'],
            hidden_dim=hparams['encoder_hidden_dim'],
            out_dim=hparams['encoder_out_dim']
        ).to(device)
        
        predictor = Predictor(
            in_dim=hparams['encoder_out_dim'],
            hidden_dim=hparams['predictor_hidden_dim'],
            out_dim=hparams['predictor_out_dim']
        ).to(device)

        decoder = LinkPredictor(
            in_dim=hparams['encoder_out_dim'],
            hidden_dim=hparams.get('link_predictor_hidden_dim', None)
        ).to(device)

        logger.info("Modelos inicializados para esta ejecución.") # Usar logger.info
        logger.info("Modo Inductivo: El encoder solo ve la estructura de entrenamiento (cold-start).") # Mensaje para el log

        # 5. Pre-entrenamiento de T-BGRL (Encoder)
        logger.info("Iniciando pre-entrenamiento de T-BGRL para el Encoder...") # Usar logger.info
        pre_trained_encoder = pretrain_tbgrl(
            encoder=encoder,
            predictor=predictor,
            data={
                'x': features, 
                'edge_index': splits['train_edge_index'], 
                'edge_weight': splits['train_edge_weight'],
                'nodes_U_idx': nodes_U_idx, 
                'nodes_V_idx': nodes_V_idx
            },
            hparams=hparams
        )
        logger.info("Pre-entrenamiento del Encoder completado.") # Usar logger.info

        # Obtener los embeddings finales del encoder pre-entrenado
        pre_trained_encoder.eval()
        with torch.no_grad():
            final_node_embeddings = pre_trained_encoder(
                features, 
                splits['train_edge_index'],
                splits['train_edge_weight']
            )
        logger.info(f"Embeddings finales de nodos generados. Forma: {final_node_embeddings.shape}") # Usar logger.info

        # 6. Entrenamiento del Link Predictor (Decoder) y Evaluación en Test
        logger.info("Iniciando entrenamiento del Link Predictor (Decoder)...") # Usar logger.info
        # train_link_predictor ahora devuelve las métricas del test y las predicciones
        test_hits_k, test_roc_auc, test_ap_score, test_predictions_df = train_link_predictor(
            decoder=decoder,
            final_node_embeddings=final_node_embeddings,
            data_splits={
                'train_pos_edges': splits['train_edge_index'],
                'train_pos_weights': splits['train_edge_weight'], 
                'val_pos_edges': splits['val_edge_index'],
                'test_pos_edges': splits['test_edge_index'],
                'nodes_U_idx': nodes_U_idx, 
                'nodes_V_idx': nodes_V_idx, 
                'idx2id': dataset_info['idx2id'], 
                'k': hparams['k']
            },
            hparams=hparams,
            device=device,
            all_edges_set=all_edges_set
        )
        logger.info("Entrenamiento del Link Predictor completado.") # Usar logger.info

        # Almacenar resultados de esta ejecución
        all_hits_at_k.append(test_hits_k)
        all_roc_auc.append(test_roc_auc)
        all_ap_score.append(test_ap_score)

        end_run_time = time.time() # Finalizar contador de tiempo para esta ejecución
        run_duration = end_run_time - start_run_time
        all_run_times.append(run_duration)

        logger.info("\nResultados de la Ejecución en Test:") # Usar logger.info
        logger.info(f"Hits@{hparams['k']}: {test_hits_k:.4f}") # Usar logger.info
        logger.info(f"ROC-AUC: {test_roc_auc:.4f}") # Usar logger.info
        logger.info(f"AP: {test_ap_score:.4f}") # Usar logger.info
        logger.info(f"--> Tiempo de la Ejecución {run_idx}: {run_duration:.2f} segundos") # Usar logger.info
        
        # Guardar predicciones detalladas del Test Set para cada ejecución (si se desea)
        if test_predictions_df is not None:
             output_predictions_path = f"link_predictions_test_set_run_{run_idx}.csv"
             test_predictions_df.to_csv(output_predictions_path, index=False)
             logger.info(f"Predicciones detalladas del Test Set para la Ejecución {run_idx} guardadas en: {output_predictions_path}") # Usar logger.info


    # 7. Resultados Finales Agregados
    logger.info("\n==================== RESULTADO FINAL AGREGADO (TEMPORAL_INDUCTIVE) ====================") # Usar logger.info
    logger.info(f"Hits@{hparams['k']}: {np.mean(all_hits_at_k):.4f} ± {np.std(all_hits_at_k):.4f}") # Usar logger.info
    logger.info(f"ROC-AUC: {np.mean(all_roc_auc):.4f} ± {np.std(all_roc_auc):.4f}") # Usar logger.info
    logger.info(f"AP: {np.mean(all_ap_score):.4f} ± {np.std(all_ap_score):.4f}") # Usar logger.info
    
    logger.info("\n--- Tiempos de Ejecución ---") # Usar logger.info
    logger.info(f"Tiempo total para {num_executions} ejecuciones: {np.sum(all_run_times):.2f} segundos ({np.sum(all_run_times)/60:.2f} minutos)") # Usar logger.info
    logger.info(f"Tiempo promedio por ejecución: {np.mean(all_run_times):.2f} segundos") # Usar logger.info

    logger.info("\nPipeline de WBT-GL completado.") # Usar logger.info


if __name__ == '__main__':
    main()