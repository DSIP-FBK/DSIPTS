
import argparse
import pandas as pd
from omegaconf import DictConfig, OmegaConf,ListConfig
import os
import numpy as np
import plotly.express as px
from inference import inference
import hydra
from dsipts import beauty_string
VERBOSE = True



@hydra.main(version_base=None)
def compare(conf:DictConfig)-> None:
    """Compare all the models specified

    Args:
        conf (DictConfig): a config with a list of models to compare, the ouput folder and other parameters. See the examples in the repo
    """
    
    ##CREATE FOLDER IF NOT EXISTS
    if not os.path.exists(os.path.join(conf.dirpath,'plots')):
        os.makedirs(os.path.join(conf.dirpath,'plots'))
    if not os.path.exists(os.path.join(conf.dirpath,'csv')):
        os.makedirs(os.path.join(conf.dirpath,'csv'))
    
    res = []
    tot_losses = []
    tot_predictions = []
    
    if isinstance( conf.models,list) or isinstance( conf.models,ListConfig):
        files =  conf.models
    
    elif os.path.isdir(conf.models):
        ff = os.path.join(conf.models,'config_used')
        files = [os.path.join(ff,f) for f in os.listdir(ff)]
    else:
        import pdb
        pdb.set_trace()
        
    for conf_tmp in files:
        
        beauty_string(f'Processing file: {conf_tmp}','block',VERBOSE)
        conf_tmp =  OmegaConf.load(conf_tmp) 
        
        conf_tmp.inference.set = conf.set
        conf_tmp.inference.rescaling = conf.rescaling
        conf_tmp.inference.batch_size = conf.get('batch_size',conf_tmp.inference.batch_size)

        beauty_string(f'PROCESSING {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version}','section',VERBOSE)

        try:
            tmp,predictions, losses = inference(conf_tmp)
            tmp['model'] = f'{conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version}'
            predictions['model'] = f'{conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version}'
            if losses is not None:
                losses['epoch'] = list(range(losses.shape[0]))
                losses = losses.melt(id_vars='epoch')
                losses['model'] = f'{conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version}'
            else:
                beauty_string(f'Can not load losses {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version} maybe the train procedure is not completed','block',True)

            losses.value = np.log(losses.value)
            res.append(tmp )
            tot_losses.append(losses)
            tot_predictions.append(predictions)
        
        except Exception as e:
            beauty_string(f'Can not load model {conf_tmp.model.type}_{conf_tmp.ts.name}_{conf_tmp.ts.version} {e} ','',True)
            

    tot_losses = pd.concat(tot_losses,ignore_index=True)
    tot_predictions = pd.concat(tot_predictions,ignore_index=True)

    res = pd.concat(res,ignore_index=True)
    res.MAPE = np.round(res.MAPE/100,4)
    tot_losses.rename(columns = {'value':'loss','variable':'set'},inplace=True)

    '''
    fig_ass = px.line(res,x = 'lag',y='MSE',color = 'model',facet_row='variable')
    fig_rel = px.line(res,x = 'lag',y='MAPE',color = 'model',facet_row='variable')
    fig_losses = px.line(tot_losses,x = 'epoch',y='loss',color = 'set',facet_col='model')

        
        
    fig_ass.update_traces(mode="markers+lines", hovertemplate=None)
    fig_ass.update_layout(hovermode="x unified")

    fig_rel.update_traces(mode="markers+lines", hovertemplate=None)
    fig_rel.update_layout(hovermode="x unified")
    fig_rel.layout.yaxis.tickformat = ',.2%'



    fig_ass.update_layout(title = {'text':f'MSE {conf.set} set', 'x':0.5},
                          xaxis_title={'text':'Future step'},
                          yaxis_title={'text':'MSE'},
                        
                            )
    fig_ass.write_image(os.path.join(conf.dirpath,'plots',f'{conf.name}_{conf.set}_MSE.jpeg'),width=1000,scale=10)
    
    fig_rel.update_layout(title = {'text':f'MAPE {conf.set} set', 'x':0.5},
                          xaxis_title={'text':'Future step'},
                          yaxis_title={'text':'MAPE'},
                        
                            )
    fig_rel.write_image(os.path.join(conf.dirpath,'plots',f'{conf.name}_{conf.set}_MAPE.jpeg'),width=1000,scale=10)
    
    fig_losses.update_layout(title = {'text':'Losses', 'x':0.5},
                          xaxis_title={'text':'Epochs'},
                          yaxis_title={'text':'Value'},
                        
                            )
    
    ##TODO seems not working
    fig_losses.write_image(os.path.join(conf.dirpath,'plots',f'{conf.name}_{conf.set}_LOSSES.jpeg'),width=1000,scale=10)
    '''
    tot_losses.to_csv(os.path.join(conf.dirpath,'csv',f'{conf.name}_{conf.set}_LOSSES.csv'))
    res.to_csv(os.path.join(conf.dirpath,'csv',f'{conf.name}_{conf.set}_errors.csv'))
    tot_predictions.to_csv(os.path.join(conf.dirpath,'csv',f'{conf.name}_{conf.set}_tot_predictions.csv'))



    
    
if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser(description="Compare TS models")
    compare()

  