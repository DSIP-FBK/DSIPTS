import os
import pickle
from dsipts import TimeSeries

def load_data(conf):
    with open(os.path.join(conf.dataset.path,'edison.pkl'),'rb') as f:
        res = pickle.load(f)
    data = res['data']
    data.rename(columns={'tempo':'time'},inplace=True)
    data[res['meteo']]=data[res['meteo']].interpolate()
    
    ts = TimeSeries(conf.ts.name)
    if conf.dataset.dataset == 'edison':
        if  conf.ts.use_covariates:
            ts.load_signal(data, cat_var= res['cat'],target_variables=['y'], past_variables=res['meteo'], future_variables=res['meteo'],silly_model=conf.ts.get('silly',False))
        else:
            ts.load_signal(data, cat_var= res['cat'],target_variables=['y'], past_variables=[], future_variables=[],silly_model=conf.ts.get('silly',False))
    return ts