

import pandas as pd
from dsipts import TimeSeries, RNN, Attention,read_public_dataset, LinearTS, Persistent
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
import os
import shutil
import numpy as np
import plotly.express as px



@hydra.main(version_base=None, config_path="config", config_name="config")
def train(conf: DictConfig) -> None:
    """Train a DL model

    Args:
        conf (DictConfig): dictionary whit all the parameters (split, normalization and training). Some of the parameters required will be filled looking to the timeserie definition. See the examples in the repo.
    """
    
    K = list(HydraConfig.get()['runtime']['choices'].keys())[0]
    print(OmegaConf.to_yaml(conf))  
    print(f"{''.join(['#']*100)}")
    print(f"{HydraConfig.get()['runtime']['choices'][K]:^100}")  
    print(f"{''.join(['#']*100)}")

    ##OCCHIO CHE tutti questi dataset hanno y come target! ###############################################
    data, columns = read_public_dataset(**conf.dataset)
    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data, enrich_cat= conf.ts.enrich,target_variables=['y'], past_variables=columns if conf.ts.use_covariates else [])
    ######################################################################################################
    
    model_conf = conf.model_configs
    model_conf['past_channels'] = len(ts.num_var)
    model_conf['future_channels'] = len(ts.future_variables)
    model_conf['embs'] = [ts.dataset[c].nunique() for c in ts.cat_var]
    model_conf['out_channels'] = len(ts.target_variables)

    if conf.model.type=='attention':
        if conf.split_params.shift==1:
            print('USING ATTENTION WITH y')
            ts.future_variables +=ts.target_variables
            model_conf['future_channels']= len(ts.future_variables)
        model =  Attention(**model_conf,
                           optim_config = conf.optim_config,
                           scheduler_config =conf.scheduler_config )
    elif conf.model.type == 'linear':
        model =  LinearTS(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )
        
    elif conf.model.type == 'rnn':
        model =  RNN(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
    elif conf.model.type == 'persistent':
        model_conf = {'future_steps':model_conf['future_steps'],
                      'past_steps':model_conf['past_steps']}
        model =  Persistent(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )
    else:
        print('use valid model')
    ##questa e' unica per ogni sequenza di dirpath type name version quindi dopo la RIMUOVO se mai ce n'e' una vecchia! 
    dirpath = os.path.join(conf.train_config.dirpath,'weights',conf.model.type,conf.ts.name, str(conf.ts.version))
    print(f'Model and weights will be placed and read from {dirpath}')
    
    
    ##clean folders
    if  os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    conf.train_config.dirpath = dirpath
    ts.set_model(model,config=dict(model_configs=model_conf,
                                    optim_config=conf.optim_config,
                                    scheduler_config =conf.scheduler_config ) )
    
    split_params = conf.split_params
    split_params['past_steps'] = model_conf['past_steps']
    split_params['future_steps'] = model_conf['future_steps']
    ts.train_model(split_params=split_params,**conf.train_config)
    ts.save(os.path.join(conf.train_config.dirpath,'model'))
    ##save the config for the comparison task
    if not os.path.exists(conf.used_config):
        os.mkdir(conf.used_config)
    with open(os.path.join(conf.used_config,HydraConfig.get()['runtime']['choices'][K]+'.yaml'),'w') as f:
        f.write(OmegaConf.to_yaml(conf))
        
        
if __name__ == '__main__': 
    
    #if not os.path.exists('config_used'):
    #    os.mkdir('config_used')
    train()
    if os.path.exists('multirun'):
        shutil.rmtree('multirun')
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
