import os
import pickle
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):
   data = pd.read_csv(os.path.join(conf.dataset.path,'pollen.csv'))
   data.data = pd.to_datetime(data.data, utc=True)
   data.rename(columns={'data':'time'},inplace=True)
   
   meteo = ['vsw_1', 'vsw_2', 'vsw_3', 'vsw_4', 'lai_H',
      'lai_L', 'st_1', 's_pres', 'u_comp', 'v_comp', '2mT', '2mTd', 'st_2',
      'st_3', 's_res_c', 'skin_T', 'st_4', 'fc_alb', 'net_sol', 'p_evap',
      't_evap', 'precip']
   group = 'region'
   cat_var = ['doy']
   target = ['totals']
   data.doy[data.doy>356] = 365

   ts = TimeSeries(conf.ts.name)
   ts.load_signal(data,past_variables = meteo + target, 
                  future_variables = meteo,
                  target_variables =target,
                  cat_var= cat_var+[group],
                  enrich_cat= conf.ts.enrich,group=group,
                  silly_model=conf.ts.get('silly',False))
   print(ts)
   return ts

 