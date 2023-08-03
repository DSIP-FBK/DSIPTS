import os
import pickle
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):
    id = 27350178
    data = pd.read_csv(os.path.join(conf.dataset.path,'data_consumption.csv'))
    data.Time = pd.to_datetime(data.Time, utc=True)
    data.sort_values(by='Time',inplace=True)
    data_ex = data[data.PlantId==id]
    data_ex.rename(columns={'Time':'time'},inplace=True)
    data_ex.Value[data_ex.Value<0]=np.nan

    data_ex.drop_duplicates(inplace=True)
    data_ex.index = data_ex.time
    data_ex.drop(columns='time',inplace=True)
    data_ex = data_ex.resample('1h').mean().reset_index()

    meteo = pd.read_csv(os.path.join(conf.dataset.path,'meteo.csv'))
    meteo.Time = pd.to_datetime(meteo.Time, utc=True)
    meteo.rename(columns={'Time':'time'},inplace=True)
    data_ex = pd.merge(data_ex,meteo,how='left')
    data_ex.Value = data_ex.Value.interpolate(limit=1)  
    
    data_ex.Value = np.log(1+500*data_ex.Value)-2
    data_ex.rain = data_ex.rain.interpolate(limit=1)   
    data_ex.temp = data_ex.temp.interpolate(limit=1)

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data_ex,past_variables =['Value','rain','temp'],future_variables = ['rain','temp'],target_variables =['Value'],enrich_cat= conf.ts.enrich)

 

    return ts

 