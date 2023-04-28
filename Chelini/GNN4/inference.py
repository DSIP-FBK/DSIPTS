
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from configparser import ConfigParser
import pdb

import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


def get_results(model, 
                dataloader,
                scaler, 
                hour, 
                config):
    model.eval()
    out_real = torch.tensor([], device="cpu")
    out_pred = torch.tensor([], device="cpu")
    m = scaler.scale_[-1] if config.getboolean('dataset','sc') else (scaler.data_max_[-1]-scaler.data_min_[-1])
    c = scaler.mean_[-1] if config.getboolean('dataset','sc') else scaler.data_min_[-1]
    with torch.no_grad():
        for graph in tqdm(iter(dataloader)):
            yh, _, _ = model(graph.to(model.device))
            yh = yh.cpu()*m + c

            y = graph.y.cpu()
            y = y*m + c
            out_real = torch.cat((out_real, y[:,hour]), 0)
            out_pred = torch.cat((out_pred, yh[:,hour]), 0)
            """import pdb
            pdb.set_trace()"""
    return out_real.cpu(), out_pred.cpu()

def get_lag(model, 
            dataloader,
            scaler, 
            config):
    y, yh = get_results(model = model, 
                        dataloader = dataloader, 
                        scaler = scaler, 
                        hour = range(65), 
                        config = config)
    err = torch.sqrt(torch.mean((y-yh)**2,0))
    return err.cpu().numpy()


def print_csv(model, 
              config: ConfigParser, 
              ds: list):
    
    with open(os.path.join(config['paths']['dataset'], f"scaler_min_max.pkl")    , 'rb') as f :
        scaler = pickle.load(f)
    dataloader = []
    for key in ds.keys():
        dataloader+= ds[key]
    dataloader = DataLoader(dataloader, 
                            batch_size = config.getint('dataset', 'batch_size'))
    
    real, pred = get_results(model = model, 
                            dataloader = dataloader,
                            scaler = scaler, 
                            hour = range(65), 
                            config = config)
    real = pd.DataFrame(real).astype("float")
    pred = pd.DataFrame(pred).astype("float")
    real.to_csv(os.path.join(config['paths']['prediction'], f"{config['model']['id_model']}_real.csv"))
    pred.to_csv(os.path.join(config['paths']['prediction'], f"{config['model']['id_model']}_pred.csv"))



def get_plot(model,
             config:ConfigParser,
             show = False):
    
    id_model = f"{config['model']['id_model']}"
    
    ################## Carico i file che mi servono ################
    path_losses = os.path.join(config['paths']['net_weights'], f'loss_{id_model}.pkl')
    path_dataset = os.path.join(config['paths']['dataset'], f"dataframes.pkl") 
    path_date = os.path.join(config['paths']['dataset'], f"dates.pkl")    

    with open(path_losses, 'rb') as f:
        losses = pickle.load(f) 
    
    with open(path_dataset, 'rb') as f :
        ds = pickle.load(f) 
    
    with open(path_date, 'rb') as f :
        dates = pickle.load(f)    
        
    batch_size = 16#config.getint('dataset', 'batch_size')
    lag = {}
    dl = {}
    for key in ds.keys():
        dl[key]=DataLoader(ds[key], 
                           batch_size = batch_size, 
                           shuffle = False)

    name_scaler = 'scaler_sc.pkl' if config.getboolean('dataset','sc') else 'scaler_min_max.pkl'
    with open(os.path.join(config['paths']['dataset'], name_scaler), 'rb') as f:
        scaler =  pickle.load(f)  
    
    fig, ax = plt.subplots(4, 2, figsize=(75, 55),  constrained_layout = True)
    fontsize_legend = 30
    fontsize_title = 40
    fontsize_label = 30
    

    for key in dl.keys():
        err = get_lag(model = model, 
                      dataloader = dl[key],
                      scaler = scaler, 
                      config = config)
        lag[key]= err

    
    ##################### train loss  #####################
    fig.suptitle(f'results of the model ({id_model})', fontsize=60)
    for key in losses.keys():
        for l in ["loss", "rec", "energy", "frobenius"]:
            ax[0,0].plot(range(1,len(losses[key][l])+1), losses[key][l], label = f"{key}-{l}")
    ax[0,0].set_ylabel('Loss', fontsize = fontsize_label)
    ax[0,0].set_xlabel('epoch', fontsize = fontsize_label)
    ax[0,0].set_title("Loss", fontsize=fontsize_title)
    ax[0,0].legend(fontsize = fontsize_legend)
    
    
    ##################### Lag plot  #####################
    for key in lag.keys():
        ax[0,1].plot(range(1, len(lag[key])+1), lag[key], label = key)
    
    ax[0,1].set_ylabel('RMSE', fontsize = fontsize_label)
    ax[0,1].set_xlabel('hour', fontsize = fontsize_label)
    ax[0,1].set_title("Hourly RMSE", fontsize = fontsize_title)
    ax[0,1].legend(fontsize = fontsize_legend)
    
    ##################### batch plot  #####################
    m = scaler.scale_[-1] if config.getboolean('dataset','sc') else (scaler.data_max_[-1]-scaler.data_min_[-1])
    c = scaler.mean_[-1] if config.getboolean('dataset','sc') else scaler.data_min_[-1]
    with torch.no_grad():
        for i,key in enumerate(['train', 'test']):
            batch = next(iter(dl[key]))
            yh, _, _ = model(batch.to(model.device))
            yh = yh.cpu()*m+c
            y = batch.y.cpu()*m+c

            ax[1,i].set_ylabel('Energy', fontsize = fontsize_label)
            ax[1,i].set_xlabel('hour', fontsize = fontsize_label)
            ax[1,i].plot(range(1, 66), yh.tolist()[0], label = "predicted")
            ax[1,i].plot(range(1, 66), y.tolist()[0], label = "real")

            ax[1,i].set_title(f"batch plot for {key}", fontsize = fontsize_title)
            ax[1,i].legend(fontsize = fontsize_legend)
           
    ##################### lag trend plot  #####################
    print(f'{" lag plot ":=^60s}')
    hour = [10,30]
    split = {'train': (np.datetime64('2019-06-01T00:00:00'), np.datetime64('2019-09-01T00:00:00')),
        'validation': (np.datetime64('2020-05-01T00:00:00'),np.datetime64('2020-08-01T00:00:00')),
        'test': (np.datetime64('2021-08-01T00:00:00'), np.datetime64('2021-11-01T00:00:00'))}
    
    for j, key in enumerate(['train', 'validation']):
        #### selezione le date che sono presente per tutte le ore di lag che devo controllare
        date_tmp = dates[key][(dates[key][hour[0]]>split[key][0])&(dates[key][hour[0]]<split[key][1])]
        d = set(date_tmp[hour[0]])
        for h in hour[1:]:
            d = d - set(date_tmp[h])
        d = list(set(date_tmp[hour[0]])-d)
        date_tmp = dates[key][dates[key][hour[0]].isin(d)]
        index_list = list(date_tmp[date_tmp[hour[0]].isin(d)].index)
        tmp = DataLoader([ds[key][i] for i in range(len(ds[key])) if i in index_list], 
                         batch_size = batch_size, 
                         shuffle = False)
        ######### devo creare i graphi ######

        y, yh = get_results(model = model, 
                            dataloader = tmp,
                            scaler = scaler, 
                            hour=hour, 
                            config = config)
        for i,h in enumerate(hour):
            ax[j+2,i].plot(date_tmp[h].values, yh[:,i].reshape(-1), label = "predicted")
            ax[j+2,i].plot(date_tmp[h].values, y[:,i].reshape(-1), label = "real")
            ax[j+2,i].legend(fontsize = fontsize_legend)
            ax[j+2,i].tick_params(axis = 'x', rotation=55)
            ax[j+2,i].set_title(f"lag plot in {key} for lag {h}", fontsize = fontsize_title)
            ax[j+2,i].set_ylabel('Energy', fontsize = fontsize_label)
            ax[j+2,i].set_xlabel('hour', fontsize = fontsize_label)
    
    ############ aggiusto la grandexxa dei valori sulla x #########
    for i in range(2):
        for j in range(4):
            ax[j, i].tick_params(axis='both', which='major', labelsize=30)
            
    plt.savefig(os.path.join(config['paths']['images'], f'{id_model}.png'))
    if show:
        plt.show()
    else:
        plt.close(fig)

def get_adj_density(x: torch.tensor, 
                    config: ConfigParser, 
                    show = False):

    id_model = config['model']['id_model']
    lines, cols = x.shape
    fontsize = 30
    #fig, ax = plt.subplots(1,2, figsize=(40, 20),  constrained_layout = True)
    fig = plt.figure(figsize=(70, 40))
    fig.suptitle("Adjacency matrix",fontsize = fontsize*2)

    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)
    density = fig.add_subplot(grid[:2, :2])
    matrix = fig.add_subplot(grid[:2, 2:])
    ts_density = fig.add_subplot(grid[2:, :])

    hight = (torch.sum(x,0)/torch.tensor(range(lines)))[1:]

    density.bar(np.arange(1,cols), height = hight.numpy())
    density.set_ylabel('density', fontsize = fontsize)
    density.set_xlabel('hour', fontsize = fontsize)
    density.tick_params(axis='both', which='major', labelsize=30)
    print(x.unique())

    matrix.matshow(x, interpolation='nearest')
    matrix.set_xlabel('hour', fontsize = fontsize)
    matrix.tick_params(axis='both', which='major', labelsize=30)
    
    
    path_adj = os.path.join(config['paths']['prediction'], f'adj_{id_model}.pkl') 
    with open(path_adj, 'rb') as f:
        density_ts = pickle.load(f)
    ts_density.plot(range(len(density_ts)), density_ts)
    ts_density.tick_params(axis='both', which='major', labelsize=30)
    ts_density.set_ylim(0,1)

    saving_path = os.path.join(config['paths']['images'], f"adj_{id_model}.png")
    plt.savefig(saving_path)
    if show:
        plt.show(show)
    else:
        plt.close()
