##import modules
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

sys.path.append('/home/davide/Documents/WORKSPACE/timeseries/')  # Adjust the path as necessary to import from the parent directory

#from dsipts import TimeSeries
from dsipts.models.TTM import TTM
from dsipts.data_structure.data_structure import TimeSeries

print("> Libraries Imported.")

SEED = 42
set_seed(SEED)

#BEST_DEVICES = ['Z-WAVE_7E','Z-WAVE_8D', 'Z-WAVE_12F', 'Z-WAVE_8F', 'Z-WAVE_7D','Z-WAVE_7F', 'Z-WAVE_4C', 'Z-WAVE_5D']
TIME_COL = "timestamp"
CAT_COLS = ['ora', 'weekend', 'giorno', 'festività']
#CAT_COLS = []
EXOG_COLS = ['temperature','humidity']
#EXOG_COLS = []
TARGET_COl = ["ElectricWConsumed"] 
past_steps = 90
future_steps = 24

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

#PATH = '/home/davide/Documents/WORKSPACE/inCUBE/data/target.csv'
#PATH = '/home/davide/Documents/WORKSPACE/inCUBE/data/target_noscaled.csv'
PATH = '/home/davide/Documents/WORKSPACE/inCUBE/data/target_withnan.csv'
df = pd.read_csv(PATH, parse_dates=[TIME_COL])
df.rename(columns={TIME_COL: 'time'}, inplace=True)

print(f"> Read: {df.shape}")

df_device = df.loc[df['DEVICE'] == 'Z-WAVE_7E', 
                   ['time'] + TARGET_COl + CAT_COLS + EXOG_COLS].copy()

df_device['t'] =  create_serie(df_device['time'].min(), df_device['time'].max(), freq='1h')
print(f"> DF size: {df_device.shape}")

##initizate a timeseries object
ts = TimeSeries('prova')
ts.load_signal(df_device,
               past_variables=EXOG_COLS, 
               future_variables=TARGET_COl+EXOG_COLS+['t'], 
               target_variables = TARGET_COl+['t'],
               cat_var = CAT_COLS)


config = dict(model_configs =dict(
                                model_path="ibm-granite/granite-timeseries-ttm-r2",
                                num_input_channels=len(CAT_COLS) + len(EXOG_COLS) + len(TARGET_COl)+1,  # exog: number of input channels
                                decoder_mode="mix_channel",  # exog:  set to mix_channel for mixing channels in history IF REMOVED GIVES THE BEST RESULTS
                                #mode='mix_channel', # NOTE: ADDED THIS  IF REMOVED GIVES THE BEST RESULTS
                                prediction_channel_indices=[ts.dataset.columns.get_loc(c)-1 for c in ts.target_variables],
                                exogenous_channel_indices=[ts.dataset.columns.get_loc(c)-1 for c in ts.cat_var + EXOG_COLS],
                                past_steps=past_steps,
                                future_steps=future_steps,
                                freq_prefix_tuning=False,
                                freq='1h',
                                prefer_l1_loss=False,
                                prefer_longer_context=True,
                                fcm_context_length=1,  # exog: indicates lag length to use in the exog fusion. for Ex. if today sales can get affected by discount on +/- 2 days, mention 2
                                fcm_use_mixer=True,  # exog: Try true (1st option) or false
                                fcm_mix_layers=2,  # exog: Number of layers for exog mixing
                                enable_forecast_channel_mixing=True,  # exog: set true for exog mixing
                                fcm_prepend_past=True,  # exog: set true to include lag from history during exog infusion.
                                # Can also provide TTM Config args
                                embs = [ts.dataset[c].nunique() for c in ts.cat_var],
                                #quantiles=[0.1,0.5,0.9],
                                quantiles=[],
                                #persistence_weight= 0.010,
                                #loss_type= 'mse',
                                loss_type= 'mse',
                                remove_last= False,
                                optim= 'torch.optim.AdamW',
                                #activation= 'torch.nn.GELU', 
                                verbose = True,
                                out_channels = len(ts.target_variables)),
                scheduler_config = None,#dict(gamma=0.1,step_size=100),
                # 0.00478630092322638
                optim_config = dict(lr = 0.00478630092322638)) #,weight_decay=0.01 4.4306214575838814e-07
model_sum = TTM(**config['model_configs'],
                optim_config = config['optim_config'],
                scheduler_config =config['scheduler_config'] )
ts.set_model(model_sum,config=config )
print("> Model Set.")

ts.train_model(dirpath="/home/davide/Documents/WORKSPACE/timeseries/test/models",
               split_params=dict(perc_train=0.7, 
                                 perc_valid=0.1,
                                 past_steps = past_steps,
                                 future_steps=future_steps, 
                                 range_train=None, 
                                 range_validation=None, 
                                 range_test=None,
                                 shift = 0,
                                 starting_point=None,
                                 skip_step=1,
                                 #scaler='StandardScaler()'),
                                 scaler='MinMaxScaler()'),
                batch_size=64,
                num_workers=4,
                max_epochs=10,
                auto_lr_find=False,
                devices='auto')
print("> Model is trained.")

#loaded = ts.load(TTM,os.path.join('/home/davide/Documents/WORKSPACE/timeseries/test/models','model'),load_last=True)
#ts.checkpoint_file_last = '/home/davide/Documents/WORKSPACE/timeseries/test/models/last.ckpt'
ts.model = ts.model.load_from_checkpoint(ts.checkpoint_file_last)
print("> Model is loaded.")

#res = ts.inference_on_set(set='test',batch_size=64,num_workers=4)
#res.head()
results = []
for s in ['train','validation','test']:
    temp_res = ts.inference_on_set(set=s,batch_size=64,num_workers=4)
    temp_res['DATASET'] = s.upper()
    results.append(temp_res)
df_results = pd.concat(results,axis=0)
#df_results= df_results.rename(columns={'prediction_time':'timestamp_start', 'time':'timestamp'})
#df_results.loc[df_results['DATASET'] == 'VALIDATION','DATASET'] = 'VALID'
#df_results= df_results.rename(columns={'ElectricWConsumed':'actual', 'ElectricWConsumed_pred':'pred'})
#df_results['DEVICE'] = 'Z-WAVE_7E'
#df_results['lag'] = df_results['lag'] - 1
df_results.to_csv('preds_0.7.csv', index=False)
print('> Inference.')

