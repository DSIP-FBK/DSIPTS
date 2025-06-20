import os
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):
   data = pd.read_csv(os.path.join(conf.dataset.path,'pollen.csv'))
   data.data = pd.to_datetime(data.data, utc=True)
   data.rename(columns={'data':'time'},inplace=True)
   data.totals[data.totals<0] = 0
   data.totals = np.log(data.totals+1)
   meteo = ['vsw_1', 'vsw_2', 'vsw_3', 'vsw_4', 'lai_H',
      'lai_L', 'st_1', 's_pres', 'u_comp', 'v_comp', '2mT', '2mTd', 'st_2',
      'st_3', 's_res_c', 'skin_T', 'st_4', 'fc_alb', 'net_sol', 'p_evap',
      't_evap', 'precip']
   group = 'region'
   target = ['totals']
   data.doy[data.doy>356] = 365
   quantile = data.groupby('region').apply(lambda x:np.quantile(x.totals,0.995)).reset_index().rename(columns={0:'quantiles'})
   data = pd.merge(data, quantile)
   data.totals = data.apply(lambda x: min(x.totals, x.quantiles),axis=1)
   ts = TimeSeries(conf.ts.name,stacked=False)
   ts.load_signal(data,past_variables = meteo + target, 
                  future_variables = meteo,
                  target_variables =target,
                  cat_var= [group],
                  enrich_cat= conf.ts.enrich,group=group,
                  silly_model=conf.ts.get('silly',False))
   return ts

 