import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from configparser import ConfigParser

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

import argparse

from model import GAT

from train import training
from inference import get_plot
    
split = {'start_train': datetime(2017,11,15,0), 
        'end_train':datetime(2020,1,1,0),
        'start_validation':datetime(2020,1,1,0), 
        'end_validation':datetime(2020,12,1,0),
        'start_test': datetime(2021,3,5,0), 
        'end_test':datetime(2022,4,6,0)}

def train(config: ConfigParser) -> None:
    
    print(f'{" Upload the dataframe ":=^60s}')
    batch_size = config.getint('dataset', 'batch_size')
    A = np.triu(np.ones((265,265)))-np.diag(np.ones(265))

    path_dataset = os.path.join(config['paths']['dataset'], f"dataframes.pkl")    
    path_date = os.path.join(config['paths']['dataset'], f"dates.pkl")    
    path_emb = os.path.join(config['paths']['dataset'], f"embedding_setting.pkl")    

    with open(path_dataset, 'rb') as f :
        ds = pickle.load(f)
        
    with open(path_date, 'rb') as f :
        dates = pickle.load(f)

    with open(path_emb, 'rb') as f :
        emb = pickle.load(f)
        
    dl = {}    
    for key in ds.keys():
        dl[key] = DataLoader(ds[key], batch_size=batch_size, shuffle = True)
        config['dataset'][f'n_obs_{key}']=f"{len(ds[key])}"
    
    print(f'{" Upload complete ":=^60s}')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden = int(config['model']['hidden'])
    in_head = int(config['model']['in_head'])
    out_head = int(config['model']['out_head'])
    drop = config.getfloat('model', 'drop_out')
    in_feat = ds["train"][0].num_features
    id_model = config['model']['id_model']
    print(f'{f" Configuration (hidden,in_head,out_head) = ({id_model})":=^60s}')
    A = np.triu(np.ones((265,265)))-np.diag(np.ones(265))

    model = GAT(in_feat = in_feat, 
                out_feat = 1,
                hid = hidden,
                in_head = in_head,
                out_head = out_head, 
                drop_out = drop,  
                emb = emb,
                past = 200, 
                future = 65, 
                A = torch.tensor(A, device = device), 
                device = device)
    
    id_model = config['model']['id_model']
    model.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(),    
                             lr = config.getfloat('optimizer','lr'), 
                             weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                       gamma = config.getfloat('optimizer','gamma'))
    criterium = nn.L1Loss(reduction = 'sum')
    loss_train, loss_validation = training(model = model, 
                                            dataloader_train = dl['train'], 
                                            dataloader_validation = dl['validation'],
                                            loss_function = criterium, 
                                            optimizer = optimizer,
                                            scheduler = scheduler, 
                                            config = config)
    
    losses = {
        'train':loss_train,
        'validation':loss_validation
    }
    with open(os.path.join(config['paths']['net_weights'], f'loss_{id_model}.pkl'), 'wb') as f:
        pickle.dump(losses, f) 

    print(f'{" Creating the plot for the LAG ":=^60s}')
    model = torch.load(os.path.join(config['paths']['net_weights_train'], f'GNN_{id_model}.pt'))
    get_plot(model = model, 
            ds = ds, 
            dates=dates,
            losses = losses, 
            config = config,
            show = False)

def test(config: ConfigParser, name_model:str) -> None:
    print(f'{" Upload the dataframe ":=^60s}')
    path_dataset = os.path.join(config['paths']['dataset'], f"dataframes.pkl")    
    path_date = os.path.join(config['paths']['dataset'], f"dates.pkl")    

    with open(path_dataset, 'rb') as f :
        ds = pickle.load(f)
        
    with open(path_date, 'rb') as f :
        dates = pickle.load(f)

    for key in ds.keys():
        config['dataset'][f'n_obs_{key}']=f"{len(ds[key])}"

    
    model = torch.load(os.path.join(config['paths']['net_weights_train'], name_model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    plot = True
    
    config['model']['id_model'] = ""
    config['model']['hidden'] = f"{model.hid}"
    config['model']['in_head'] = f"{model.in_head}"
    config['model']['out_head'] = f"{model.out_head}"
    config['model']['drop_out'] = f"{model.drop}"

    for i, key in enumerate(["hidden", "in_head", "out_head", "drop_out"]):
        if i ==0:
            config['model']['id_model'] = f"{config['model'][key]}" 
        else:
            config['model']['id_model'] = f"{config['model']['id_model']}_{config['model'][key]}"

    id_model = f"{config['model']['id_model']}"
    print(f'{f" configuration {id_model} ":=^60s}')

    try:
        with open(os.path.join(config['paths']['net_weights'], f'loss_{id_model}.pkl'), 'rb') as f:
            losses = pickle.load(f) 
    except:
        print("unable to load the loss")
        plot = False
    if plot:
        get_plot(model = model, 
                ds = ds, 
                dates = dates,
                losses = losses, 
                config = config,
                show = False)

if __name__ == "__main__":
    # all standard stuffs here
    parser = argparse.ArgumentParser(description="Base model")
    parser.add_argument("-c", "--config", type=str, required = True, help="Config file")

    # parametri extra
    parser.add_argument("-hl", "--hidden", type=str, help="hidden layer")
    parser.add_argument("-ih", "--in_head", type=str, help="number of head in the input")
    parser.add_argument("-oh", "--out_head", type=str, help="number of head in the second conv")
    parser.add_argument("-dr", "--drop_out", type=str, help = "data manipulation")


    # non posso avere più di un'operazione alla volta
    parser.add_argument("-d", "--dataset", action="store_true", help = "data manipulation")
    parser.add_argument("-t", "--train", action="store_true", help="Train model flag")
    parser.add_argument("-i", "--inference", action="store_true", help="Inference model flag")
    parser.add_argument("-e", "--test", action="store_true", help="Inference model flag")

    parser.add_argument("-p", "--get_plot", action="store_true", help="Train model flag")
    parser.add_argument("-m", "--name_model", type=str, help="Train model flag")
    

    args = parser.parse_args()
    config_path = args.config
    config = ConfigParser()
    config.read(config_path)
    
    print(f'{" Cheking the paths ":=^60s}')
    if 'paths' in config.keys():
        for path in config['paths']:
            if os.path.exists(config['paths'][path]) != True:
                os.makedirs(config['paths'][path])

    print(f'{" Paths controlled ":=^60s}')
    
    print(f'{" Setting new possible parameters for the model ":=^60s}')
    config['model']['id_model'] = ""
    for i, key in enumerate(["hidden", "in_head", "out_head", "drop_out"]):
        if getattr(args, key) is not None:
            config['model'][key] = f"{getattr(args, key)}"
        if i ==0:
            config['model']['id_model'] = f"{config['model'][key]}" 
        else:
            config['model']['id_model'] = f"{config['model']['id_model']}_{config['model'][key]}" 
    
    
    print(f'{" Setting complete ":=^60s}')
    if args.get_plot:
        test(config = config, name_model = args.name_model)
    if args.dataset:
        os.system(f"python3 data_manipulation/data_generation.py --config {config_path}")
    if args.train:
        train(config)
