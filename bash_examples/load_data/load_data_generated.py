from dsipts import TimeSeries
import pandas as pd
import os
def load_data(conf):

   data = pd.read_csv(os.path.join(conf.dataset.path,'generated.csv'))
   columns = [c for c in data.columns if c not in ['y','time']]
   
   ts = TimeSeries(conf.ts.name)
   
   ts.load_signal(data, 
                  enrich_cat= [],
               #    enrich_cat= conf.ts.get('enrich',[]),
                  target_variables=['y'],
                  sampler_weights = conf.ts.get('sampler_weights',None),
                  past_variables=columns if conf.ts.get('use_covariates',False) else [],
                  future_variables=[],
                  silly_model=conf.ts.get('silly',False))
   print(ts)
   return ts