import os
from dsipts import TimeSeries
import numpy as np
import pandas as pd


def load_data(conf):
    TIME_COL = "timestamp"
    TARGET_COl = "ElectricWConsumed"
    CAT_COLS = ['ora', 'weekend', 'giorno', 'festività']
    #EXOG_COLS = ['temperature','humidity']
    EXOG_COLS = []
    DEVICE = 'Z-WAVE_7E'

    data = pd.read_csv(os.path.join(conf.dataset.path,'target_withnan.csv'))
    data.rename(columns={TIME_COL: 'Time'}, inplace=True)
    data.Time = pd.to_datetime(data.Time, utc=True)
    data.sort_values(by='Time',inplace=True)
    data_ex = data[data.DEVICE==DEVICE].reset_index()
    data_ex.rename(columns={'Time':'time'},inplace=True)
    data_ex = data_ex[['time'] + [TARGET_COl] + CAT_COLS + EXOG_COLS]
    data_ex.loc[data_ex[TARGET_COl]<0,TARGET_COl]=np.nan

    data_ex.drop_duplicates(inplace=True)
    #data_ex.index = data_ex.time
    #data_ex.drop(columns='time',inplace=True)
    #data_ex = data_ex.resample('1h').mean().reset_index()

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data_ex,
                   past_variables = EXOG_COLS, 
                   cat_var = CAT_COLS,
                   future_variables = [TARGET_COl] + EXOG_COLS,
                   target_variables = [TARGET_COl])
    return ts

 