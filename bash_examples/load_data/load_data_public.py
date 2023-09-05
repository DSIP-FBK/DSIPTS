import os
import pickle
from dsipts import TimeSeries,read_public_dataset

def load_data(conf):
    data, columns = read_public_dataset(**conf.dataset)
    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data, enrich_cat= conf.ts.enrich,target_variables=['y'], past_variables=columns if conf.ts.use_covariates else [],silly_model=conf.ts.get('silly',False))
    return ts