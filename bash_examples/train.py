
from dsipts import  beauty_string
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
import os
import shutil
import traceback
from utils import select_model, check_split_parameters
import time

VERBOSE = True


@hydra.main(version_base=None)
def train(conf: DictConfig) -> None:
    """Train a DL model

    Args:
        conf (DictConfig): dictionary whit all the parameters (split, normalization and training). Some of the parameters required will be filled looking to the timeserie definition. See the examples in the repo.
    """

    K = 'architecture'
    tasks = HydraConfig.get()['overrides']['task']
    version_modifier = ''
    for t in tasks:
        if 'model_configs' in t:
            version_modifier+=t.split('model_configs.')[1] ##cerco solo quelli che modifico

    version = str(conf.ts.version)
    if version_modifier!='':
        version = version+'_'+version_modifier
    conf.ts.version = version
    selection = HydraConfig.get()['runtime']['choices'][K]+'_'+conf.ts.name+'_'+version
    beauty_string(selection,'block', VERBOSE)

    ##OCCHIO CHE tutti questi dataset hanno y come target! ###############################################
    
    if conf.dataset.dataset == 'incube': 
        from load_data.load_data_incube import load_data
    elif conf.dataset.dataset == 'pollen': 
        from load_data.load_data_pollen import load_data
    else:
        from load_data.load_data_public import load_data
    try:
        ts = load_data(conf)
    except Exception:
        beauty_string(f"LOADING {conf.dataset.dataset} ERROR {traceback.format_exc()}",'', True)

    ts.set_verbose(VERBOSE)
    ######################################################################################################
    ts
    check_split_parameters(conf)
    ######################################################################################################
    model_conf = conf.model_configs
    if model_conf is None:
        model_conf = {}
    
    model_conf['past_channels'] = len(ts.num_var)
    model_conf['future_channels'] = len(ts.future_variables)
    model_conf['embs'] = [ts.dataset[c].nunique() for c in ts.cat_var]
    model_conf['out_channels'] = len(ts.target_variables)


    model = select_model(conf,model_conf,ts)
    if model is None:
        return 1000

    dirpath = os.path.join(conf.train_config.dirpath,'weights',conf.model.type,conf.ts.name, version)
    beauty_string(f'Model and weights will be placed and read from {dirpath}','info', VERBOSE)
    
    retrain = True
    if os.path.exists(os.path.join(dirpath,'model.pkl')):
        if conf.model.get('retrain',False):
            pass
        else:
            retrain = False
            

    if retrain is False:
        beauty_string(f'MODEL{ conf.model.type}-{conf.ts.name}-{conf.ts.version}  ALREADY TRAINED if you want to overwrite set model.retrain=True in the config ','block', True)

        ## TODO if a model is altready trained with a config I should save the testloss somewhere
        return 1000
    
    ##clean folders
    if  os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
    conf.train_config.dirpath = dirpath
    ts.set_model(model,config=dict(model_configs=model_conf,
                                    optim_config=conf.optim_config,
                                    scheduler_config =conf.scheduler_config ) )
    
    split_params = conf.split_params
    split_params['past_steps'] = model_conf['past_steps']
    split_params['future_steps'] = model_conf['future_steps']
    ##save now so we can use it during the trainin step (or use intermediate pth files)
    ts.dirpath = dirpath    
    ts.losses = None
    ts.checkpoint_file_last = os.path.join(dirpath,'checkpoint.ckpt')
    ts.save(os.path.join(conf.train_config.dirpath,'model'))

    ##save the config for the comparison task before training so we can get predictions during the training procedure
    path =  HydraConfig.get()['runtime']['config_sources'][1]['path']
    used_config = os.path.join(path,'config_used')
    if not os.path.exists(used_config):
        os.mkdir(used_config)
    tot_seconds = time.time()
    try:    
        valid_loss = ts.train_model(split_params=split_params,**conf.train_config)
        ok = True
    except Exception as _:
        beauty_string(traceback.format_exc(),'', True)
        ok = False
        
    if ok:
        ts.save(os.path.join(conf.train_config.dirpath,'model'))
        with open(os.path.join(used_config,selection+'.yaml'),'w') as f:
            f.write(OmegaConf.to_yaml(conf))
        beauty_string(f'FINISH TRAINING PROCEDURE in {(time.time()-tot_seconds)/60} with loss = {valid_loss}','block', VERBOSE)
    
        
    return valid_loss ##for optuna!    
        
if __name__ == '__main__': 
    train()
    if os.path.exists('outputs'):
        shutil.rmtree('outputs', ignore_errors=True)

