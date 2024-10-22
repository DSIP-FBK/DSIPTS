import os
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):
    
    
    
    '''
    data = np.load(os.path.join(conf.dataset.path,'temperature.npy'))
    data = pd.DataFrame({'time':range(len(data)),'y':data})
    data['cos_day'] = np.cos(2*np.pi*np.arange(0,data.shape[0],1)/364)
    data['sin_day'] = np.cos(2*np.pi*np.arange(0,data.shape[0],1)/364)
    '''
    data = pd.read_csv(os.path.join(conf.dataset.path,'press_temp_indexes_1940_2023.csv'))# '/home/agobbi/Scaricati/press_temp_indexes_1940_2023.csv')
    data['time'] = range( data.shape[0])
    y = ['temp_index_1']
    past = [x for x in data.columns if 'temp' in x] + [x for x in data.columns if 'press' in x]
    
    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data,past_variables = past ,future_variables = [],target_variables = y,enrich_cat= ['month'],silly_model=conf.ts.get('silly',False))
 

    return ts

 