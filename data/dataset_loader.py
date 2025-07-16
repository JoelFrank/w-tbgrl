# data/dataset_loader.py
import torch
import networkx as nx
import pandas as pd
import logging

# Configurar logging para mensajes informativos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def construir_grafo_bipartito(df_list):
    """
    Construye un único grafo bipartito a partir de una lista de DataFrames,
    donde el peso de la arista representa la frecuencia de aparición.
    Los nodos 'OBSERVAÇÃO PATRIMÔNIO' tienen un atributo 'tipo' (TIPO DO EQUIPAMENTO),
    y los nodos 'LOCALIZAÇÃO' tienen 'tipo=-1' para diferenciarlos.
    """
    logging.info("Construyendo grafo bipartito NetworkX...")
    G = nx.Graph()
    
    # Pre-procesamiento de todos los dataframes para asegurar tipos de nodos consistentes
    all_obs_patrimonio = set()
    all_localizacao = set()
    all_tipo_equipamento = set()

    for df in df_list:
        for _, row in df.iterrows():
            obs_patrimonio = str(row['OBSERVAÇÃO PATRIMÔNIO']).strip()
            localizacao = str(row['LOCALIZAÇÃO']).strip()
            tipo_equipamento = str(row['TIPO DO EQUIPAMENTO']).strip()

            all_obs_patrimonio.add(obs_patrimonio)
            all_localizacao.add(localizacao)
            if tipo_equipamento: # Asegurarse de que no sea una cadena vacía
                all_tipo_equipamento.add(tipo_equipamento)

    # Mapeo de tipos de equipo a IDs numéricos para características de nodos
    tipo_unicos = sorted(list(all_tipo_equipamento))
    tipo2id = {t: i + 1 for i, t in enumerate(tipo_unicos)} # 0 reservado para 'desconocido' o nodos tipo B
    
    # Asignar nodos y atributos al grafo
    for df in df_list:
        for _, row in df.iterrows():
            nodo_a_orig = str(row['OBSERVAÇÃO PATRIMÔNIO']).strip()
            tipo_a_orig = str(row['TIPO DO EQUIPAMENTO']).strip()
            nodo_b_orig = str(row['LOCALIZAÇÃO']).strip()
            
            # Añadir nodos si no existen
            if not G.has_node(nodo_a_orig): 
                # Si el tipo_a_orig no está en tipo2id (ej. valor atípico o nulo pre-tratado), usar 0
                G.add_node(nodo_a_orig, tipo=tipo_a_orig, tipo_id=tipo2id.get(tipo_a_orig, 0), node_type='A')
            
            if not G.has_node(nodo_b_orig): 
                G.add_node(nodo_b_orig, tipo=-1, tipo_id=0, node_type='B') # tipo_id 0 para nodos B

            # Añadir o actualizar la arista ponderada por frecuencia
            if G.has_edge(nodo_a_orig, nodo_b_orig):
                G[nodo_a_orig][nodo_b_orig]['weight'] += 1
            else:
                G.add_edge(nodo_a_orig, nodo_b_orig, weight=1)

    logging.info(f"Grafo NetworkX construido con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
    return G, tipo2id

def preparar_dataset_desde_splits(df_train, df_val, df_test):
    """
    Prepara todos los tensores necesarios a partir de los DataFrames ya divididos,
    incluyendo los pesos de las aristas y los índices de los nodos bipartitos.
    """
    logging.info("Preparando tensores del dataset desde los DataFrames divididos...")
    
    # Construir el grafo completo con pesos de frecuencia
    G_full, tipo2id = construir_grafo_bipartito([df_train, df_val, df_test])
    
    all_nodes = list(G_full.nodes)
    id2idx = {nid: i for i, nid in enumerate(all_nodes)}
    idx2id = {i: nid for nid, i in id2idx.items()}

    # Identificar índices de nodos para los conjuntos A y B
    nodes_U_idx = torch.tensor([id2idx[n] for n in all_nodes if G_full.nodes[n]['node_type'] == 'A'], dtype=torch.long)
    nodes_V_idx = torch.tensor([id2idx[n] for n in all_nodes if G_full.nodes[n]['node_type'] == 'B'], dtype=torch.long)

    # Preparar características de nodos (tipo_ids y mask_embed)
    num_node_types = len(tipo2id) + 1 # +1 para el ID 0 (desconocido/nodos B)
    
    # Crear tensor de tipo_ids para todos los nodos
    # Usamos G_full.nodes[n]['tipo_id'] que ya fue establecido en construir_grafo_bipartito
    tipo_ids = torch.tensor([G_full.nodes[n]['tipo_id'] for n in all_nodes], dtype=torch.long)
    
    # mask_embed para identificar nodos con características reales (tipo != -1)
    mask_embed = torch.tensor([1 if G_full.nodes[n]['tipo'] != -1 else 0 
                               for n in all_nodes], dtype=torch.bool)
    
    # --- NUEVA ADICIÓN: Crear all_edges_set para verificación de negativos ---
    all_edges_set = set()
    for u_orig, v_orig in G_full.edges():
        u_num = id2idx[u_orig]
        v_num = id2idx[v_orig]
        # Almacenar en un formato canónico (ej. u_num, v_num con u_num < v_num) para grafos no dirigidos
        all_edges_set.add(tuple(sorted((u_num, v_num)))) 
    logging.info(f"Se identificaron {len(all_edges_set)} enlaces únicos en el grafo completo para la verificación de negativos.")

    def extraer_aristas_y_pesos(df, id2idx_map, nx_graph):
        edge_list = []
        weight_list = []
        # Utilizar df completo para asegurar que se consideran todas las ocurrencias en el split
        for _, row in df[['OBSERVAÇÃO PATRIMÔNIO', 'LOCALIZAÇÃO']].iterrows():
            u_orig, v_orig = str(row['OBSERVAÇÃO PATRIMÔNIO']).strip(), str(row['LOCALIZAÇÃO']).strip()
            if u_orig in id2idx_map and v_orig in id2idx_map:
                u_idx, v_idx = id2idx_map[u_orig], id2idx_map[v_orig]
                # Asegurarse de que la arista existe en el grafo completo para obtener su peso
                if nx_graph.has_edge(u_orig, v_orig):
                    # Aquí la clave: obtenemos el peso *final* del grafo completo G_full
                    edge_list.append((u_idx, v_idx))
                    weight_list.append(nx_graph[u_orig][v_orig]['weight'])
        
        # Convertir a tensores de PyTorch
        if edge_list:
            # Eliminar aristas duplicadas que pueden surgir de la extracción de filas de DF
            # Si el grafo es no dirigido, las aristas (u,v) y (v,u) son la misma.
            # Convertir a un formato canónico para eliminar duplicados de manera efectiva.
            unique_edges_with_weights = {} # Usar un dict para manejar unicidad y el peso
            for edge, weight in zip(edge_list, weight_list):
                # Usar una tupla ordenada para la clave, ya que el grafo es no dirigido
                canonical_edge = tuple(sorted(edge))
                unique_edges_with_weights[canonical_edge] = weight # El peso ya debería ser el final del G_full

            final_edge_list = [list(edge) for edge in unique_edges_with_weights.keys()]
            final_weight_list = [unique_edges_with_weights[edge] for edge in unique_edges_with_weights.keys()]

            if final_edge_list:
                edge_index = torch.tensor(final_edge_list, dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(final_weight_list, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weight = torch.empty((0,), dtype=torch.float)
        else: # Manejar caso sin aristas para evitar errores
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)
            
        return edge_index, edge_weight

    # Extraer aristas y pesos para cada split
    train_edge_index, train_edge_weight = extraer_aristas_y_pesos(df_train, id2idx, G_full)
    val_edge_index, val_edge_weight = extraer_aristas_y_pesos(df_val, id2idx, G_full)
    test_edge_index, test_edge_weight = extraer_aristas_y_pesos(df_test, id2idx, G_full)
    
    dataset = {
        'num_nodes': len(all_nodes),
        'num_node_feature_types': num_node_types, # Renombrado para mayor claridad
        'tipo_ids': tipo_ids,
        'mask_embed': mask_embed,
        'G_full': G_full, 
        'id2idx': id2idx,
        'idx2id': idx2id,
        'nodes_U_idx': nodes_U_idx,
        'nodes_V_idx': nodes_V_idx,
        'all_edges_set': all_edges_set # <--- ¡CLAVE AÑADIDA AQUÍ!
    }
    
    splits = {
        'train_edge_index': train_edge_index,
        'train_edge_weight': train_edge_weight,
        'val_edge_index': val_edge_index,
        'val_edge_weight': val_edge_weight,
        'test_edge_index': test_edge_index,
        'test_edge_weight': test_edge_weight
    }
    
    logging.info("Preparación de datos completa.")
    return dataset, splits


def cargar_y_preparar_datos(file_paths, train_size=0.8, val_size=0.1, random_state=42):
    """
    Carga datos desde múltiples archivos CSV/XLSX, los combina y los divide
    en conjuntos de entrenamiento, validación y prueba de forma temporal
    usando la columna 'DATA DE ABERTURA'. Luego construye y prepara el grafo bipartito.
    """
    logging.info(f"Cargando y combinando datos desde {len(file_paths)} archivos...")
    all_dataframes = []
    for fp in file_paths:
        try:
            if fp.endswith('.csv'):
                # Intentar leer CSV con varias codificaciones y delimitadores
                # Por defecto, pd.read_csv intenta ',', si falla, se podría probar ';' o '\t'
                try:
                    df = pd.read_csv(fp, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(fp, encoding='cp1252')
                except pd.errors.ParserError: # Si falla el tokenizing con ',', intentar con ';'
                    logging.warning(f"Error de tokenizing con ',' en {fp}. Intentando con ';'.")
                    try:
                        df = pd.read_csv(fp, encoding='utf-8', sep=';')
                    except UnicodeDecodeError:
                        df = pd.read_csv(fp, encoding='cp1252', sep=';')
                    except pd.errors.ParserError:
                        logging.error(f"Fallo al leer {fp} con ',' o ';'. Considera verificar el delimitador manualmente.")
                        raise
            elif fp.endswith('.xlsx') or fp.endswith('.xls'):
                df = pd.read_excel(fp)
            else:
                logging.warning(f"Formato de archivo no reconocido para {fp}. Saltando este archivo.")
                continue # Saltar al siguiente archivo si el formato no es reconocido
        except Exception as e:
            logging.error(f"Error al leer el archivo {fp}: {e}")
            raise # Volver a lanzar la excepción para detener la ejecución si hay un problema crítico
        all_dataframes.append(df)
    
    if not all_dataframes:
        logging.error("No se pudieron cargar DataFrames válidos. Verifica las rutas y formatos de archivo.")
        raise ValueError("No hay DataFrames cargados para procesar.")

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Asegurarse de que las columnas clave existen y convertir a string para limpieza
    required_cols = ['OBSERVAÇÃO PATRIMÔNIO', 'LOCALIZAÇÃO', 'TIPO DO EQUIPAMENTO', 'DATA DE ABERTURA']
    for col in required_cols:
        if col not in combined_df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en los datos combinados. Columnas disponibles: {combined_df.columns.tolist()}")
        # Limpieza de valores nulos y espacios en blanco
        # Rellenar NaN con cadena vacía antes de strip()
        combined_df[col] = combined_df[col].fillna('').astype(str).str.strip()

    # Filtrar filas donde 'OBSERVAÇÃO PATRIMÔNIO' o 'LOCALIZAÇÃO' estén vacíos después de limpieza
    initial_rows = len(combined_df)
    combined_df = combined_df[combined_df['OBSERVAÇÃO PATRIMÔNIO'] != '']
    combined_df = combined_df[combined_df['LOCALIZAÇÃO'] != '']
    if len(combined_df) < initial_rows:
        logging.warning(f"Se eliminaron {initial_rows - len(combined_df)} filas debido a valores vacíos en 'OBSERVAÇÃO PATRIMÔNIO' o 'LOCALIZAÇÃO' después de la limpieza.")

    # Convertir la columna 'DATA DE ABERTURA' a datetime para la división temporal
    combined_df['DATA DE ABERTURA'] = pd.to_datetime(combined_df['DATA DE ABERTURA'])
    logging.info("Ordenando datos por 'DATA DE ABERTURA' para división temporal...")
    combined_df = combined_df.sort_values(by='DATA DE ABERTURA').reset_index(drop=True)

    # División temporal
    n = len(combined_df)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    # Ajustar para asegurar que los splits no estén vacíos
    # n_test es el resto
    n_test = n - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test <= 0:
        logging.error(f"Los tamaños de los splits resultaron en conjuntos vacíos o prueba nula. Total: {n} filas. Train: {n_train}, Val: {n_val}, Test: {n_test}. Ajusta train_size/val_size o verifica el tamaño total de los datos.")
        raise ValueError("Los splits de datos son demasiado pequeños o resultan en conjuntos vacíos.")

    df_train = combined_df.iloc[:n_train]
    df_val = combined_df.iloc[n_train : n_train + n_val]
    df_test = combined_df.iloc[n_train + n_val :]

    logging.info(f"División Temporal Completada: {len(df_train)} (Train) | {len(df_val)} (Validation) | {len(df_test)} (Test)")

    # Preparar el dataset para el modelo (incluyendo el grafo y los tensores)
    dataset_info, splits = preparar_dataset_desde_splits(df_train, df_val, df_test)

    return dataset_info, splits