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

    if train:
        model.train()
        is_train = True
    else: 
        model.eval()
        is_train = False
    
    with torch.set_grad_enabled(is_train):
        for graph in iter(dataloader):
            if train:
                optimizer.zero_grad()
            yh = model(graph.to(model.device))
            loss = loss_function(input = yh, 
                                 target = graph.y.float().to(model.device))
            if train:
                loss.backward()
                optimizer.step()

            l += loss.item()
    return model, l


def training(model,
             dataloader_train: DataLoader,
             dataloader_validation: DataLoader,
             loss_function,
             optimizer,
             scheduler, 
             config: ConfigParser):

    epoch = config.getint('model', 'epochs')
    loss_train = []
    loss_validation = []
    be = np.Inf
    k = 3
    if config.getboolean('optimizer','custom_loss'):
        path_scaler = os.path.join(config['paths']['dataset'], 'scaler_min_max.pkl') 
        with open(path_scaler, 'rb') as f :
            scaler = pickle.load(f)
        m1 = scaler.data_max_[-1]-scaler.data_min_[-1]    
        
        def my_loss(input, target):
            crit = nn.L1Loss(reduction='sum')
            loss = crit(input = input, target = target)*m1
            return loss
        loss_function = my_loss
    
    id_model = config['model']['id_model']
    # IN QUESTO MODO ELIMINO IL PROBLEMA DEL BUFFERING DATO CHE TQDM STAMPA OGNI RIGA
    with tqdm(total=epoch) as progress_bar: 
        for e in range(epoch):
            if e in [200, 400, 600, 800, 1000]:
                scheduler.step()
            ##TRAIN STEP
            l = 0
            model, loss = get_prediction(model = model, 
                                        train = True, 
                                        dataloader = dataloader_train, 
                                        loss_function = loss_function, 
                                        optimizer = optimizer)

            loss_train.append(loss/config.getint('dataset', 'n_obs_train'))

            model, loss = get_prediction(model = model,
                                        train = False,
                                        dataloader = dataloader_validation, 
                                        loss_function = loss_function, 
                                        optimizer = optimizer)
            loss_validation.append(loss/config.getint('dataset', 'n_obs_validation'))

            if e%k==0:
                print(f" epoch {e+1} ".center(30, "#"))
                print("train loss == ", loss_train[-1])
                print("validation loss == ",loss_validation[-1])   
                if e == 0: 
                    progress_bar.update(1)
                else:
                    progress_bar.update(k)
                sys.stdout.flush()


            torch.save(model, os.path.join(config['paths']['net_weights_train'],f"GNN_{id_model}.pt"))
            """if loss_validation[-1] < be:
                be = loss_validation[-1]
                torch.save(model, os.path.join(config['paths']['net_weights_train'],f"GNN_{id_model}.pt"))"""            
    burn_in = int(len(loss_train)*0.05)
    return loss_train[burn_in:], loss_validation[burn_in:]