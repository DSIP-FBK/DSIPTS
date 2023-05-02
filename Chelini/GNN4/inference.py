
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from configparser import ConfigParser
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt


def get_results(model, 
                dataloader: DataLoader,
                scaler, 
                hour, 
                config):
    model.eval()
    out_real = torch.tensor([], device="cpu")
    out_pred = torch.tensor([], device="cpu")
    m = scaler.scale_[-1] if config.getboolean('dataset','sc') else (scaler.data_max_[-1]-scaler.data_min_[-1])
    c = scaler.mean_[-1] if config.getboolean('dataset','sc') else scaler.data_min_[-1]
    with torch.no_grad():
        for batch in tqdm(iter(dataloader)):
            yh, _, _ = model(batch[0].to(model.device))
            yh = yh.cpu()*m + c

            y = batch[1].cpu()
            y = y*m + c
            out_real = torch.cat((out_real, y[:,hour]), 0)
            out_pred = torch.cat((out_pred, yh[:,hour]), 0)
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
             ds: dict,
             show = False):
    
    id_model = f"{config['model']['id_model']}"
    
    ################## Carico i file che mi servono ################
    path_losses = os.path.join(config['paths']['net_weights'], f'loss_{id_model}.pkl')
    path_date = os.path.join(config['paths']['dataset'], f"dates.pkl")    
    name_scaler = 'scaler_sc.pkl' if config.getboolean('dataset','sc') else 'scaler_min_max.pkl'
    
    with open(os.path.join(config['paths']['dataset'], name_scaler), 'rb') as f:
        scaler =  pickle.load(f) 
    
    with open(path_losses, 'rb') as f:
        losses = pickle.load(f) 
    
    with open(path_date, 'rb') as f :
        dates = pickle.load(f)    
        
    batch_size = 16 #config.getint('dataset', 'batch_size')
    lag = {}
    dl = {}
    for key in ds.keys():
        dl[key]=DataLoader(ds[key], 
                           batch_size = batch_size, 
                           shuffle = False)
        
     
    
    fig, ax = plt.subplots(4, 2, figsize=(30, 20),  constrained_layout = True)
    fig.suptitle(f'results of the model ({id_model})', fontsize = 30)

    gs = {}
    for i, key in enumerate(['train', 'validation']):
        gs[key] = ax[2+i, 0].get_gridspec()
    # remove the underlying axes
    lag_plot = {}
    for i, key in enumerate(['train', 'validation']):
        for a in ax[2+i,:]:
            a.remove()
        lag_plot[key] = fig.add_subplot(gs[key][2+i, 0:])
    
    fontsize_legend = 15
    fontsize_title = 20
    fontsize_label = 15
    
    
    for key in dl.keys():
        err = get_lag(model = model, 
                      dataloader = dl[key],
                      scaler = scaler, 
                      config = config)
        lag[key]= err

    
    ##################### train loss  #####################
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
            yh, _, _ = model(batch[0].to(model.device))
            yh = yh.cpu()*m+c
            y = batch[1].cpu()*m+c

            ax[1,i].set_ylabel('Energy', fontsize = fontsize_label)
            ax[1,i].set_xlabel('hour', fontsize = fontsize_label)
            ax[1,i].plot(range(1, 66), yh[0].tolist(), label = "predicted")
            ax[1,i].plot(range(1, 66), y[0].tolist(), label = "real")

            ax[1,i].set_title(f"batch plot for {key}", fontsize = fontsize_title)
            ax[1,i].legend(fontsize = fontsize_legend)
           
    ##################### lag trend plot  #####################
    print(f'{" lag plot ":=^60s}')
    hour = [10,30]
    split = {'train': (np.datetime64('2019-06-01T00:00:00'), np.datetime64('2019-09-01T00:00:00')),
             'validation': (np.datetime64('2020-05-01T00:00:00'), np.datetime64('2020-08-01T00:00:00')),
             'test': (np.datetime64('2021-08-01T00:00:00'), np.datetime64('2021-11-01T00:00:00'))}

    for j, key in enumerate(['train', 'validation']):
    #### selezione le date che sono presente per tutte le ore di lag che devo controllare
        date_tmp = {}
        for h in hour:
            date_tmp[h] = dates[key][(dates[key][h] > split[key][0]) & 
                                     (dates[key][h] < split[key][1])]
            date_tmp[h] = set(date_tmp[h][h].values)

        intersection = date_tmp[hour[0]]
        for h in hour[1:]:  
            intersection = intersection.intersection(date_tmp[h])

        date_tmp = dates[key][dates[key][hour[0]].isin(list(intersection))]
        index_list = list(date_tmp.index)

        tmp = DataLoader([ds[key][i] for i in range(len(ds[key])) if i in index_list], 
                         batch_size = batch_size, 
                         shuffle = False)

        y, yh = get_results(model = model, 
                            dataloader = tmp,
                            scaler = scaler, 
                            hour = [x-1 for x in hour], 
                            config = config)
        pred = {}
        real = {}
        for i, h in enumerate(hour):
            pred[h]={}    
            for j, date in enumerate(date_tmp[h]):
                key_date = date + np.timedelta64(h, 'h') - np.timedelta64(hour[0], 'h')
                pred[h][key_date] = yh[j,i].item()

        for i, date in enumerate(date_tmp[hour[0]]):
            real[date] = y[i,0]

        df = pd.DataFrame(real.items())
        lag_plot[key].plot(df[0],df[1], label = 'real')
        lag_plot[key].set_ylim([df[1].max()+10,df[1].min()-10])
        for h in hour:
            df = pd.DataFrame(pred[h].items())
            lag_plot[key].plot(df[0],df[1], label = f"hour {h}")
        lag_plot[key].tick_params(axis = 'x', rotation = 55)
        lag_plot[key].set_title(f"lag plot in {key}", fontsize = fontsize_title)
        lag_plot[key].legend()

    ############ aggiusto la grandezza dei valori sulla x #########
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
