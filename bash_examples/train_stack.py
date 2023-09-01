

import pandas as pd
from dsipts import TimeSeries, RNN, read_public_dataset, LinearTS, Persistent, D3VAE, MyModel, TFT,TFT2, Informer,VVA,VQVAEA,CrossFormer
from omegaconf import DictConfig, OmegaConf,ListConfig
from hydra.core.hydra_config import HydraConfig
import hydra
import os
import shutil
import numpy as np
import plotly.express as px
import logging
import sys
from inference import inference
import inspect
from dsipts import extend_time_df
from datetime import timedelta
#file_handler = logging.FileHandler(filename='tmp.log')
#stdout_handler = logging.StreamHandler(stream=sys.stdout)
#handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
 #   format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
 #   handlers=handlers
)




@hydra.main(version_base=None)
def train_stack(conf: DictConfig) -> None:
    """Train a stacked DL model

    Args:
        conf (DictConfig): dictionary with the parameters for the stacked model, we can suppose that several stacked model can be trained
    """

    ##commented TODO FIX
    #K = 'architecture'
    
    ##nel caso si faccia un multirun per    
 
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
    ##commented TODO FIX
    #selection = HydraConfig.get()['runtime']['choices'][K]+'_'+conf.ts.name+'_'+version
    #logging.info(f"{''.join(['#']*100)}")
    #logging.info(f"{selection:^100}")  
   # logging.info(f"{''.join(['#']*100)}")
    selection = 'stacked_'+conf.ts.name+'_'+version



    ######################################################################################################
    
    
    if (type( conf.stack.models)==list) or (type( conf.stack.models)==ListConfig):
        files =  conf.stack.models
    
    elif os.path.isdir(conf.stack.models):
        ff = os.path.join(conf.stack.models,'config_used')
        files = [os.path.join(ff,f) for f in os.listdir(ff)]
    else:
        ## if we are here probably is becasue it falis to load the models from a list. There is a bug because hyrdra parse the string in an obscure way so we need to pass a string and parse it
        try:
            files = files.sep(',')
        except:
            logging.info('FAILED TO LOAD MODELS')
            
    predictions = None
    N_models = 0
    models_used = []
    input_columns = []
    i = 0
    ## Collect all the prediction in the selected set
    
    for conf_tmp in files:
     
        conf_tmp =  OmegaConf.load(conf_tmp) 
      
        if conf_tmp.ts.get('type','normal') == 'stacked':
            continue
        conf_tmp.inference.set = conf.stack.set

        conf_tmp.inference.rescaling = conf.stack.rescaling
        conf_tmp.inference.batch_size = conf.stack.get('batch_size',conf_tmp.inference.batch_size)

        logging.info(f"{''.join(['#']*200)}")
        logging.info(f"{''.join([' ']*200)}")
        logging.info(f'#####################PROCESSING {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version} ############## ')
        logging.info(f"{''.join([' ']*200)}")
        logging.info(f"{''.join(['#']*200)}")


        try:
            _,prediction, _ = inference(conf_tmp)
            
            ##this can be more informative but the names are too long
            #prediction['model'] = f'{conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version}'
            model_features = [c for c in prediction.columns if ('pred' in c or 'median' in c)]
            real_features = [c for c in prediction.columns if not any( k in c for k in ['median','pred','lag','low','high','time']  )]
            ##renaming columns
            prediction = prediction[real_features+model_features+['time','lag']]
            mapping = {}            
            for j,col in enumerate(model_features):
                mapping[col] = f'pred_model_{i}_target_{j}'
            prediction.rename(columns=mapping,inplace=True)
            input_columns+=list(mapping.values())
            
            
            
            if predictions is None:
                predictions = prediction[['time','lag']+list(mapping.values())+real_features]
                targets = real_features
                LAG = int(predictions.lag.max())


            else:
                assert(len(set(model_features).difference(set(model_features)))==0), print('Check models, seems with different targets')
                prediction = prediction[['time','lag']+list(mapping.values())]
                predictions = pd.merge(predictions, prediction)
            N_models+=1
            models_used.append(conf_tmp)
            i+=1
        except Exception as e:
            logging.info(f'#######can not load model {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version} {e} ######### ')
    
    
    
    logging.info(f'#######USING {N_models} models  ######### ')

    
    model_conf = conf.model_configs
    if model_conf is None:
        ##ste defaults
        model_conf = {}
        model_conf['quantiles'] = []
    
    ts = TimeSeries(conf.ts.name)  
    ts.models_used = models_used
    freq = prediction[prediction.lag==1].sort_values(by='time').time.diff()[1:].min()

    predictions['prediction_time'] = predictions.apply(lambda x: x.time-timedelta(seconds= x.lag*freq.seconds), axis=1)

    predictions = extend_time_df(predictions,freq,group='lag',global_minmax=True).merge(predictions,how='left')
    ts.load_signal(predictions, enrich_cat=conf.ts.enrich,target_variables=real_features, past_variables=input_columns, future_variables=[],check_holes_and_duplicates=False)
    ts.dataset.sort_values(by=['prediction_time','lag'],inplace=True)

    #these parameters depends on the model used not from the config file
    model_conf['past_steps'] = LAG
    model_conf['future_steps'] = LAG
    model_conf['past_channels'] = len(targets)*(N_models+1)
    model_conf['future_channels'] = 0
    model_conf['embs'] = [ts.dataset[c].nunique() for c in ts.cat_var]
    model_conf['out_channels'] = len(targets)

    if conf.model.type == 'linear':
        required = inspect.getfullargspec(LinearTS.__init__)
        model_conf = {k:model_conf[k] for k in required.args if k in model_conf.keys() }
        model =  LinearTS(**model_conf,
                          optim_config = conf.optim_config,
                          scheduler_config =conf.scheduler_config )
        
    elif conf.model.type == 'rnn':
        required = inspect.getfullargspec(RNN.__init__)
        model_conf = {k:model_conf[k] for k in required.args if k!='self'}

        model =  RNN(**model_conf,
                          optim_config = conf.optim_config,
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
    #/home/agobbi/Projexts/ExpTS/test/weights/linear/weather/1
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
    os.makedirs(dirpath)
    conf.train_config.dirpath = dirpath
    ts.set_model(model,config=dict(model_configs=model_conf,
                                    optim_config=conf.optim_config,
                                    scheduler_config =conf.scheduler_config ) )
    
    split_params = conf.split_params
    
    split_params['starting_point'] = {'lag':1}
    split_params['past_steps'] = LAG
    split_params['future_steps'] = LAG


    ts.dirpath = dirpath    
    ts.losses = None
    ts.checkpoint_file_last = os.path.join(dirpath,'checkpoint.ckpt')
    ts.save(os.path.join(conf.train_config.dirpath,'model'))

    ##save the config for the comparison task before training so we can get predictions during the training procedure
    path =  HydraConfig.get()['runtime']['config_sources'][1]['path']
    used_config = os.path.join(path,'config_used')
    if not os.path.exists(used_config):
        os.mkdir(used_config)
    with open(os.path.join(used_config,selection+'.yaml'),'w') as f:
        f.write(OmegaConf.to_yaml(conf))

    valid_loss = ts.train_model(split_params=split_params,**conf.train_config)
    ts.save(os.path.join(conf.train_config.dirpath,'model'))
    logging.info(f'##########FINISH TRAINING PROCEDURE with loss = {valid_loss}###############')
    
    
    
    
    
    
    return valid_loss ##for optuna!    
        
if __name__ == '__main__': 
    
    #if not os.path.exists('config_used'):
    #    os.mkdir('config_used')
    train_stack()

    #if os.path.exists('multirun'):
    #    shutil.rmtree('multirun')
    
    if os.path.exists('outputs'):
        shutil.rmtree('outputs', ignore_errors=True)

