import os
from dsipts import TimeSeries
import numpy as np
import pandas as pd

 

def load_data(conf):
    TIME_COL = "timestamp"
    TARGET_COl = "ElectricWConsumed"
    CAT_COLS = ['ora', 'weekend', 'giorno', 'festività']
    EXOG_COLS = ['temperature','humidity']
    EXOG_COLS = []
    DEVICE = 'Z-WAVE_7E'

    def create_serie(start_date, end_date, freq='1h'):
        # Crea indice orario
        time_index = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Stagionalità giornaliera: una sinusoide su 24 ore
        hours = np.arange(len(time_index))
        daily_period = 24
        amplitude = 10
        baseline = 20
        noise = np.random.normal(0, 1, len(time_index))  # Rumore gaussiano

        # Funzione sinusoidale giornaliera con rumore
        values = baseline + amplitude * np.sin(2 * np.pi * (hours % daily_period) / daily_period) + noise

        # Serie temporale
        #serie_stagionale = pd.Series(values, index=time_index)
        return values

    data = pd.read_csv(os.path.join(conf.dataset.path,'target_withnan.csv'))
    data.rename(columns={TIME_COL: 'Time'}, inplace=True)
    data.Time = pd.to_datetime(data.Time, utc=True)
    data.sort_values(by='Time',inplace=True)
    data_ex = data[data.DEVICE==DEVICE].reset_index()
    data_ex.rename(columns={'Time':'time'},inplace=True)
    data_ex['t'] = create_serie(data_ex['time'].min(), data_ex['time'].max())
    data_ex = data_ex[['time'] + [TARGET_COl] + ['t'] + CAT_COLS + EXOG_COLS]
    data_ex.loc[data_ex[TARGET_COl]<0,TARGET_COl]=np.nan

    data_ex.drop_duplicates(inplace=True)
    #data_ex.index = data_ex.time
    #data_ex.drop(columns='time',inplace=True)
    #data_ex = data_ex.resample('1h').mean().reset_index()

    ts = TimeSeries(conf.ts.name)
    ts.load_signal(data_ex,
                   past_variables = EXOG_COLS, 
                   cat_var = CAT_COLS + EXOG_COLS,
                   future_variables = [TARGET_COl] + EXOG_COLS + ['t'],
                   target_variables = [TARGET_COl] + ['t'])
    return ts

 