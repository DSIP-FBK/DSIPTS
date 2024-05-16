import os
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):
    data = np.load(os.path.join(conf.dataset.path,'temperature.npy'))
    data = pd.DataFrame({'time':range(len(data)),'y':data})
    data['cos_day'] = np.cos(2*np.pi*list(range(len(data)))/364)
    data['sin_day'] = np.cos(2*np.pi*list(range(len(data)))/364)

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data,past_variables =['y','cos_day','sin_day'],future_variables = ['cos_day','sin_day'],target_variables =['y'],enrich_cat= [],silly_model=conf.ts.get('silly',False))

 

    return ts

 