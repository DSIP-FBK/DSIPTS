import re


import pandas as pd
from dsipts import TimeSeries, beauty_string,extend_time_df,read_public_dataset
from omegaconf import DictConfig, OmegaConf,ListConfig
from hydra.core.hydra_config import HydraConfig
import hydra
import os
import shutil
import logging
from inference import inference
from datetime import timedelta
from utils import select_model

VERBOSE = True

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
    
    
    if isinstance( conf.stack.models,list) or isinstance( conf.stack.models,ListConfig):
        files =  conf.stack.models
    
    elif os.path.isdir(conf.stack.models):
        ff = os.path.join(conf.stack.models,'config_used')
        files = [os.path.join(ff,f) for f in os.listdir(ff)]
    else:
        beauty_string('FAILED TO LOAD MODELS','block',True)

    predictions = None
    N_models = 0
    models_used = []
    input_columns = []
    i = 0
    ## Collect all the prediction in the selected set
    original_data = None
    predicted_columns = []
    for conf_tmp in files:
        
        conf_tmp =  OmegaConf.load(conf_tmp) 
        if conf_tmp.stack.dirpath is None:
            conf_tmp.inference.set = conf.stack.set
            conf_tmp.inference.rescaling = conf.stack.rescaling
            conf_tmp.inference.batch_size = conf.stack.get('batch_size',conf_tmp.inference.batch_size)
            beauty_string(f'PROCESSING {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version} ','block',VERBOSE)


            if original_data is None:
                if conf_tmp.dataset.dataset == 'custom': 
                    beauty_string('PLEASE WRITE CUSTOM PROCESSING FUNCTION','block',VERBOSE)
                    ##
                else:
                    original_data, columns = read_public_dataset(**conf_tmp.dataset)
                    targets = ['y']

            try:

                ##change skip otherwise a lot of non consecutive points
                conf_tmp.split_params.skip_step = 1 ##force this! and pass to the inference module
                _,prediction, _ = inference(conf_tmp,conf_tmp.split_params)
                
                ##this can be more informative but the names are too long
                #prediction['model'] = f'{conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version}'
                model_features = [c for c in prediction.columns if ('_pred' in c or c.endswith('_q0.5'))]
                real_features = [c for c in prediction.columns if not (any( k in c for k in ['pred','lag','time']) or re.search(r'_q[0-9.e-]+$',c))]
                ##renaming columns
                prediction = prediction[real_features+model_features+['time','lag']]
                mapping  ={}
                for j,col in enumerate(model_features):
                    mapping[col] = f'pred_model_{i}_target_{j}'
                prediction.rename(columns=mapping,inplace=True)
                input_columns+=list(mapping.values())
                predicted_columns+=list(mapping.values())

                
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
                import traceback
                beauty_string(traceback.format_exc(),'',True)
                beauty_string(f'Can not load model {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version} {e}','block', True)
                import pdb
                pdb.set_trace()
 

    beauty_string(f'USING {N_models} models','section',VERBOSE)

    
    model_conf = conf.model_configs
    if model_conf is None:
        model_conf = {}
        model_conf['quantiles'] = []
    
    ts = TimeSeries(conf.ts.name,stacked=True)  
    ts.models_used = models_used
    freq = prediction[prediction.lag==1].sort_values(by='time').time.diff()[1:].min()

    predictions['prediction_time'] = predictions.apply(lambda x: x.time-timedelta(seconds= x.lag*freq.seconds), axis=1)
    predictions = extend_time_df(predictions,freq,group='lag',global_minmax=True).merge(predictions,how='left')
    predictions['lag_m'] = predictions.lag.values
    silly_model = conf.ts.get('silly',False)

    '''
    df_long = predictions.drop(columns=['lag_m','prediction_time']).melt(id_vars=["time", "lag"]+targets, var_name="variable", value_name="value")
    df_long["variable"] = df_long["variable"] + "_lag" + df_long["lag"].astype(str)
    extra_columns = df_long["variable"].unique()
    df_wide = df_long.pivot(index=["time"]+targets, columns="variable", values="value").reset_index()
    df_wide = df_wide.merge(original_data.drop(columns=targets),how='left')
    

    ts.load_signal(df_wide, 
                   enrich_cat=conf.ts.enrich,
                   target_variables=real_features,
                   past_variables=columns if conf.ts.get('use_covariates',False) else [], 
                   future_variables=extra_columns,
                   check_holes_and_duplicates=True,
                   #cat_past_var=['lag_m'], ##dataset_dependent
                   #cat_fut_var=['lag_m'],##dataset_dependent
                   silly_model=silly_model)
    ts.dataset.sort_values(by=['time'],inplace=True)
    print(ts)

    import pdb
    pdb.set_trace()

    model_conf['past_steps'] = LAG
    model_conf['future_steps'] = LAG
    model_conf['past_channels'] = len(ts.past_variables)
    model_conf['future_channels'] = len(ts.future_variables)
    model_conf['embs_past'] = [ts.dataset[c].nunique() for c in ts.cat_past_var]
    model_conf['embs_fut'] = [ts.dataset[c].nunique() for c in ts.cat_fut_var]
    '''
    
    predictions = predictions.merge(original_data.drop(columns=targets),how='left')
    ts.load_signal(predictions, 
                   enrich_cat=conf.ts.enrich,
                   target_variables=real_features,
                   past_variables=columns if conf.ts.get('use_covariates',False) else [], 
                   future_variables=predicted_columns,
                   check_holes_and_duplicates=False,
                   cat_past_var=['lag_m'], ##dataset_dependent
                   cat_fut_var=['lag_m'],##dataset_dependent
                   silly_model=silly_model)
    ts.dataset.sort_values(by=['prediction_time','lag'],inplace=True)
    print(ts)


    model_conf['past_steps'] = LAG
    model_conf['future_steps'] = LAG
    model_conf['past_channels'] = len(ts.past_variables)
    model_conf['future_channels'] = len(ts.future_variables)
    model_conf['embs_past'] = [ts.dataset[c].nunique() for c in ts.cat_past_var]
    model_conf['embs_fut'] = [ts.dataset[c].nunique() for c in ts.cat_fut_var]    
    
    model_conf['out_channels'] = len(targets)


    model = select_model(conf,model_conf,ts)   
    dirpath = os.path.join(conf.train_config.dirpath,'weights',conf.model.type,conf.ts.name, version)
    beauty_string(f'Model and weights will be placed and read from {dirpath}','info',VERBOSE)
    retrain = True
    ##if there is a model file look if you want to retrain it
    if os.path.exists(os.path.join(dirpath,'model.pkl')):
        if conf.model.get('retrain',False):
            pass
        else:
            retrain = False
            
            
    

    if retrain is False:
        logging.info(f'##########MODEL{ conf.model.type}-{conf.ts.name}-{conf.ts.version}  ALREADY TRAINED#############',True)

        ## TODO if a model is altready trained with a config I should save the testloss somewhere
        return 0.0
    
    ##clean folders
    if  (os.path.exists(dirpath)) and (conf.model.get('restart',False) is False):
        shutil.rmtree(dirpath)
    if  os.path.exists(dirpath) is False:
        os.makedirs(dirpath)
    conf.train_config.dirpath = dirpath
    ts.set_verbose(VERBOSE)
    ts.set_model(model,config=dict(model_configs=model_conf,
                                    optim_config=conf.optim_config,
                                    scheduler_config =conf.scheduler_config ) )
    
    split_params = conf.split_params
    
    split_params['starting_point'] = {'lag':1} #THIS IS IMPORTANT!
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
    beauty_string(f'FINISH TRAINING PROCEDURE with loss = {valid_loss}','block',VERBOSE)
    
    return valid_loss 
        
if __name__ == '__main__': 
    train_stack()
    if os.path.exists('outputs'):
        shutil.rmtree('outputs', ignore_errors=True)

