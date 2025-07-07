# training/trainer.py
import copy
import torch
import torch.nn.functional as F
from utils.augmentations import augment, corrupt
from utils.evaluation import evaluate
import logging

def update_target_network(online_net, target_net, ema_decay):
    for param_q, param_k in zip(online_net.parameters(), target_net.parameters()):
        param_k.data = param_k.data * ema_decay + param_q.data * (1.0 - ema_decay)

def pretrain_tbgrl(encoder, predictor, data, hparams):
    print("--- Iniciando Etapa 1: Pre-entrenamiento del Encoder (T-BGRL Fiel) ---")
    online_encoder = encoder
    target_encoder = copy.deepcopy(online_encoder)
    for param in target_encoder.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.Adam(list(online_encoder.parameters()) + list(predictor.parameters()), lr=hparams['lr_encoder'])
    
    def regression_loss(p, z):
        return 2 - 2 * (F.normalize(p, dim=1) * F.normalize(z, dim=1)).sum(dim=1).mean()

    for epoch in range(hparams['epochs_encoder']):
        online_encoder.train()
        predictor.train()

        x1, edge_index_1 = augment(data['features'], data['edge_index'], hparams['drop_feat_p'], hparams['drop_edge_p'])
        x2, edge_index_2 = augment(data['features'], data['edge_index'], hparams['drop_feat_p'], hparams['drop_edge_p'])
        xc, edge_index_c = corrupt(data['features'], data['edge_index'])

        p_online_1 = predictor(online_encoder(edge_index_1, data['tipo_ids'], data['mask_embed']))
        p_online_2 = predictor(online_encoder(edge_index_2, data['tipo_ids'], data['mask_embed']))

        with torch.no_grad():
            target_encoder.eval()
            h_target_1 = target_encoder(edge_index_1, data['tipo_ids'], data['mask_embed'])
            h_target_2 = target_encoder(edge_index_2, data['tipo_ids'], data['mask_embed'])
            h_target_c = target_encoder(edge_index_c, data['tipo_ids'], data['mask_embed'])
        
        pos_loss = regression_loss(p_online_1, h_target_2.detach()) + regression_loss(p_online_2, h_target_1.detach())
        neg_loss = regression_loss(p_online_1, h_target_c.detach())
        loss = (1 - hparams['lambda_t-bgrl']) * pos_loss + hparams['lambda_t-bgrl'] * neg_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_target_network(online_encoder, target_encoder, hparams['ema_decay'])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch Pre-train {epoch+1}, Loss: {loss.item():.4f}")

    print("--- Fin Etapa 1 ---")
    return online_encoder

def train_link_predictor(encoder, decoder, dataset, hparams):
    print("\n--- Iniciando Etapa 2: Entrenamiento del Decodificador ---")
    encoder.eval()
    for param in encoder.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=hparams['lr_decoder'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    best_val_metric = -1
    best_decoder_state = None

    for epoch in range(hparams['epochs_decoder']):
        decoder.train()
        
        with torch.no_grad():
            embeddings = encoder(dataset['edge_index'], dataset['tipo_ids'], dataset['mask_embed'])
        
        train_edges_t = torch.tensor(dataset['train_edges'], dtype=torch.long).t()
        neg_edges_t = torch.randint(0, dataset['num_nodes'], train_edges_t.size(), dtype=torch.long)
        
        pos_out = decoder(embeddings[train_edges_t[0]], embeddings[train_edges_t[1]])
        neg_out = decoder(embeddings[neg_edges_t[0]], embeddings[neg_edges_t[1]])
        
        out, labels = torch.cat([pos_out, neg_out]), torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        loss = loss_fn(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            #val_results = evaluate(decoder, embeddings, dataset['val_edges'], dataset['num_nodes'], k=hparams['k_hits'])
            #val_results, _ = evaluate(decoder, embeddings, dataset['val_edges'], dataset['num_nodes'], k=hparams['k_hits'], idx2id=dataset['idx2id'])
            val_results, _ = evaluate(
            decoder,
            embeddings,
            dataset['val_edges'],
            dataset['all_edges_set'],
            dataset['nodes_A_idx'],   # <-- Nuevo argumento
            dataset['nodes_B_idx'],   # <-- Nuevo argumento
            k=hparams['k_hits'],
            idx2id=dataset['idx2id']
        )
            val_metric = val_results[f'Hits@{hparams["k_hits"]}']
            #print(f"Decoder Epoch {epoch+1}, Loss: {loss.item():.4f}, Val {list(val_results.keys())[0]}: {val_metric:.4f}")
            logging.info(f"Decoder Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Hits@{hparams['k_hits']}: {val_metric:.4f}")
            if val_metric > best_val_metric:
                best_val_metric, best_decoder_state = val_metric, copy.deepcopy(decoder.state_dict())
                print(f"*** Nuevo mejor decodificador guardado (Val {list(val_results.keys())[0]}: {best_val_metric:.4f}) ***")
    
    print("--- Fin Etapa 2 ---")
    decoder.load_state_dict(best_decoder_state)
    return decoder