
from dsipts import RNN, LinearTS, Persistent, D3VAE, DilatedConv, TFT, Informer,VVA,VQVAEA,CrossFormer, beauty_string
import numpy as np
from sklearn.metrics import mean_squared_error
import os
def rmse(x:np.array,y:np.array)->float:
    """custom RMSE avoinding nan

    Args:
        x (np.array): predicted
        y (np.array): real

    Returns:
        float: RMSE
    """
    x = x.astype(float)
    y = y.astype(float)
    idx = list(np.where(~np.isnan(x*y))[0])
    return np.sqrt(mean_squared_error(x[idx],y[idx]))

def mse(x:np.array,y:np.array)->float:
    """custom MSE avoinding nan

    Args:
        x (np.array): predicted
        y (np.array): real

    Returns:
        float: MSE
    """
    x = x.astype(float)
    y = y.astype(float)
    idx = list(np.where(~np.isnan(x*y))[0])
    return mean_squared_error(x[idx],y[idx])

def mape(x:np.array,y:np.array)->float:
    """custom mape avoinding nan

    Args:
        x (np.array): predicted
        y (np.array): real

    Returns:
        float: mape
    """
    x = x.astype(float)
    y = y.astype(float)
    idx = list(np.where(~np.isnan(x*y))[0])
    res = 100*np.abs((x[idx]-y[idx])/y[idx])
    res = res[np.isfinite(res)]
    return np.nanmean(res)


def select_model(conf, model_conf,ts):
    
    if conf.model.type == 'linear':
        model =  LinearTS(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )
        
    elif conf.model.type == 'rnn':
        model =  RNN(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
        
    elif conf.model.type == 'dilated_conv':
        model =  DilatedConv(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )      
    
    elif conf.model.type == 'persistent':
        model_conf = {'future_steps':model_conf['future_steps'],
                      'past_steps':model_conf['past_steps']}
        model =  Persistent(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )
    elif conf.model.type == 'd3vae':

        model =  D3VAE(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config ) 
         
    elif conf.model.type == 'tft':
        model =  TFT(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
        
        
    elif conf.model.type == 'vva':
        model =  VVA(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
    elif conf.model.type == 'vqvae':
        model =  VQVAEA(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
    elif conf.model.type == 'crossformer':
        model =  CrossFormer(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )   
    elif conf.model.type == 'informer':
        ##gli servono, poi mette a 0 quelle che serve
        ts.future_variables +=ts.target_variables
        model_conf['future_channels']= len(ts.future_variables)
        model =  Informer(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
    else:
        model = None
        beauty_string(f"Not a valid model { conf.model.type}-{conf.ts.name}-{conf.ts.version}",'block')
        
    return model


def load_model(ts,conf):
    loaded = True
    if conf.model.type == 'linear':
        ts.load(LinearTS,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'rnn':
        ts.load(RNN,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'persistent':
        ts.load(Persistent,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'd3vae':
        ts.load(D3VAE,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'dilated_conv':
        ts.load(DilatedConv,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'tft':
        ts.load(TFT,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'informer':
        ts.load(Informer,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'vva':
        ts.load(VVA,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'vqvae':
        ts.load(VQVAEA,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
    elif conf.model.type == 'crossformer':
        ts.load(CrossFormer,os.path.join(conf.train_config.dirpath,'model'),load_last=conf.inference.load_last)
  
    else:
        beauty_string('NO VALID MODEL FOUND','block')
        loaded=False
    return loaded