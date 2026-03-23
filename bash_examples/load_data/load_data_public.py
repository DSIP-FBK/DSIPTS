from dsipts import TimeSeries, read_public_dataset
import numpy as np

def load_data(conf):
    data, columns = read_public_dataset(**conf.dataset)
    ts = TimeSeries(conf.ts.name)
    sampler_weights =  conf.ts.get('sampler_weights',None)
    if sampler_weights == 'fake_columns':
        data['fake_columns'] = np.random.rand(data.shape[0]) 
    ts.load_signal(data, enrich_cat= conf.ts.get('enrich',[]),
                   target_variables=['y'],
                   sampler_weights = conf.ts.get('sampler_weights',None),
                   past_variables=columns if conf.ts.get('use_covariates',False) else [],
                   future_variables=columns if conf.ts.get('use_future_covariates',False) else [],
                   silly_model=conf.ts.get('silly',False))
    print(ts)
    return ts