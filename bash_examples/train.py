

import pandas as pd
from dsipts import TimeSeries, RNN, Attention,read_public_dataset, LinearTS, Persistent, D3VAE, MyModel, TFT, Informer
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
import os
import shutil
import numpy as np
import plotly.express as px
import logging
import sys
from load_data.load_data_edison import load_data_edison
from load_data.load_data_public import load_data_public
from load_data.load_data_incube import load_data_incube

#file_handler = logging.FileHandler(filename='tmp.log')
#stdout_handler = logging.StreamHandler(stream=sys.stdout)
#handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
 #   format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
 #   handlers=handlers
)




@hydra.main(version_base=None)
def train(conf: DictConfig) -> None:
    """Train a DL model

    Args:
        conf (DictConfig): dictionary whit all the parameters (split, normalization and training). Some of the parameters required will be filled looking to the timeserie definition. See the examples in the repo.
    """

    ##secondo me qui e' giusto mettere K = 'architecture'
    K = 'architecture'
    #K = list(HydraConfig.get()['runtime']['choices'].keys())[0]
    
    ##nel caso si faccia un multirun per    
    #import pdb
    #pdb.set_trace()
    tasks = HydraConfig.get()['overrides']['task']
    ##nel caso si faccia un multirun per cercare un parametro in particolare devo crearmi la versione giusta!
    version_modifier = ''
    for t in tasks:
        if 'model_configs' in t:
            version_modifier+=t.split('model_configs.')[1] ##cerco solo quelli che modifico

    version = str(conf.ts.version)
    if version_modifier!='':
        version = version+'_'+version_modifier
    conf.ts.version = version
    selection = HydraConfig.get()['runtime']['choices'][K]+'_'+version
    logging.info(f"{''.join(['#']*100)}")
    logging.info(f"{selection:^100}")  
    logging.info(f"{''.join(['#']*100)}")

    ##OCCHIO CHE tutti questi dataset hanno y come target! ###############################################
    
    
    if conf.dataset.dataset == 'edison':
        ts = load_data_edison(conf)
    elif conf.dataset.dataset == 'incube': 
        ts = load_data_incube(conf)
    else:
        ts = load_data_public(conf)
        

    ######################################################################################################
    
    
    model_conf = conf.model_configs
    if model_conf is None:
        model_conf = {}
    
    model_conf['past_channels'] = len(ts.num_var)
    model_conf['future_channels'] = len(ts.future_variables)
    model_conf['embs'] = [ts.dataset[c].nunique() for c in ts.cat_var]
    model_conf['out_channels'] = len(ts.target_variables)

    if conf.model.type=='attention':
        if conf.split_params.shift>0:
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
        
    elif conf.model.type == 'mymodel':
        model =  MyModel(**model_conf,
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
    elif conf.model.type == 'informer':
        ##gli servono, poi mette a 0 quelle che serve
        ts.future_variables +=ts.target_variables
        model_conf['future_channels']= len(ts.future_variables)
        model =  Informer(**model_conf,   optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )  
    else:
        logging.info(f"{''.join(['#']*300)}")
        logging.info(f"{''.join([' ']*300)}")
        logging.info(f'######use valid model { conf.model.type}-{conf.ts.name}-{conf.ts.version}########')
        logging.info(f"{''.join([' ']*300)}")
        logging.info(f"{''.join(['#']*300)}")
    ##questa e' unica per ogni sequenza di dirpath type name version quindi dopo la RIMUOVO se mai ce n'e' una vecchia! 
    dirpath = os.path.join(conf.train_config.dirpath,'weights',conf.model.type,conf.ts.name, version)
    logging.info(f'Model and weights will be placed and read from {dirpath}')
    
    retrain = True
    ##if there is a model file look if you want to retrain it
    if os.path.exists(os.path.join(dirpath,'model.pkl')):
        if conf.model.get('retrain',False):
            pass
        else:
            retrain = False
            
            
    

    if retrain==False:
        logging.info(f'##########MODEL{ conf.model.type}-{conf.ts.name}-{conf.ts.version}  ALREADY TRAINED#############')

        ## TODO if a model is altready trained with a config I should save the testloss somewhere
        return 0.0
    
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
    ## I save it here so i can use intermediate pth weights!
    ts.save(os.path.join(conf.train_config.dirpath,'model'))
    valid_loss = ts.train_model(split_params=split_params,**conf.train_config)
    ts.save(os.path.join(conf.train_config.dirpath,'model'))
    ##save the config for the comparison task
    
    path =  HydraConfig.get()['runtime']['config_sources'][1]['path']
    used_config = os.path.join(path,'config_used')
    if not os.path.exists(used_config):
        os.mkdir(used_config)
    with open(os.path.join(used_config,selection+'.yaml'),'w') as f:
        f.write(OmegaConf.to_yaml(conf))
    return valid_loss ##for optuna!    
        
if __name__ == '__main__': 
    
    #if not os.path.exists('config_used'):
    #    os.mkdir('config_used')
    val_loss = train()
    #if os.path.exists('multirun'):
    #    shutil.rmtree('multirun')
    
    if os.path.exists('outputs'):
        shutil.rmtree('outputs', ignore_errors=True)
    val_loss
