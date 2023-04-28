import os
import numpy as np
import torch
import sys
import pickle
from configparser import ConfigParser
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
import numpy as np

def get_prediction(model,
                   train: bool,
                   dataloader, 
                   loss_function, 
                   optimizer):
    l = 0.0
    
    rec = 0.0
    frob = 0.0
    energy = 0.0
    
    if train:
        model.train()
        is_train = True
    else: 
        model.eval()
        is_train = False
    alpha = 5e-2
    with torch.set_grad_enabled(is_train):
        for graph in iter(dataloader):
            if train:
                optimizer.zero_grad()
                
            yh, x, A = model(graph.to(model.device))
            
            ########### Computing the loss function ##############
            l1 = loss_function(input = yh, target = graph.y.float().to(model.device))
            l2 = alpha*torch.sum(torch. matmul(x, A))/x.shape[0]
            l3 = torch.norm(A)
            
            loss = l1 + l2 + l3
            
            if train:
                loss.backward()
                optimizer.step()
            
            l += loss.item()
            rec += l1.item()
            energy += l2.item()
            frob += l3.item()
    
    return (l, rec, energy, frob)


def training(model,
             dataloader_train: DataLoader,
             dataloader_validation: DataLoader,
             optimizer,
             scheduler, 
             config: ConfigParser):
    """"
    Funzione che permette di allenare il modello \n
    ogni 200 epoche faccio uno step allo scheduler
    """
    epoch = config.getint('optimizer', 'epochs')
    
    loss_train = {"loss" : [],
                  "rec"  : [],
                  "energy": [],
                  "frobenius": []}
    loss_validation = {"loss" : [],
                       "rec"  : [],
                       "energy": [],
                       "frobenius": []}
    
    loss_function = eval(f"{config['optimizer']['loss']}")
    
    be = np.Inf
    k = 1
    
    id_model = config['model']['id_model']
    saving_path = os.path.join(config['paths']['net_weights_train'],f"GNN_{id_model}.pt")
    
    # IN QUESTO MODO ELIMINO IL PROBLEMA DEL BUFFERING DATO CHE TQDM STAMPA OGNI RIGA
    with tqdm(total=epoch) as progress_bar: 
        for e in range(epoch):
            if e in [300, 400, 600, 800, 1000]:
                scheduler.step()
            ##TRAIN STEP
            l = 0
            loss_t = get_prediction(model = model, 
                                        train = True, 
                                        dataloader = dataloader_train, 
                                        loss_function = loss_function, 
                                        optimizer = optimizer)
            
            
            

            loss_v = get_prediction(model = model,
                                        train = False,
                                        dataloader = dataloader_validation, 
                                        loss_function = loss_function, 
                                        optimizer = optimizer)
            
            for i, key in enumerate(loss_train.keys()):
                loss_train[key].append(loss_t[i]/config.getint('dataset', 'n_obs_train'))
                loss_validation[key].append(loss_v[i]/config.getint('dataset', 'n_obs_validation'))

            if e%k==0:
                print(f" epoch {e+1} ".center(30, "#"))
                print("train loss == ", loss_train['loss'][-1])
                print("validation loss == ",loss_validation['loss'][-1])   
                if e == 0: 
                    progress_bar.update(1)
                else:
                    progress_bar.update(k)
                sys.stdout.flush()

            if e > 0:
                if loss_validation['loss'][-1] < be:
                    be = loss_validation['loss'][-1]
                    torch.save(model.state_dict(), saving_path)           
    
    losses = {
        'train':loss_train,
        'validation':loss_validation
    }
    with open(os.path.join(config['paths']['net_weights'], f'loss_{id_model}.pkl'), 'wb') as f:
        pickle.dump(losses, f) 