import os
import pickle
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):

    data = pd.read_csv(os.path.join(conf.dataset.path,'data_consumption.csv'))
    data.Time = pd.to_datetime(data.Time)
    data.sort_values(by='Time',inplace=True)
    data_ex = data[data.PlantId==26913041]
    data_ex.rename(columns={'Time':'time'},inplace=True)
    data_ex.Value[data_ex.Value<0]=np.nan
    # data_ex.Value = np.log(data_ex.Value+1)
    data_ex = data_ex.groupby('time').mean().reset_index()
    #data_ex.index = data_ex.time
    # data_ex = data_ex.resample('1h').mean().reset_index()
    empty = pd.DataFrame({'time':pd.date_range(data_ex.time.min(),data_ex.time.max(),freq='900s')})
    data_ex = empty.merge(data_ex,how='left')
    data_ex.Value = data_ex.Value.interpolate(limit=1)
    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data_ex,past_variables =['Value'],future_variables = [],target_variables =['Value'],enrich_cat=['dow','hour','month', 'minute'])

 

    return ts

 