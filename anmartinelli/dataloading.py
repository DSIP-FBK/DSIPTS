import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from pandas import to_datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

def dataloading(batch_size, batch_size_test, seq_len, lag,
                hour_learning, hour_inference, train_bool,
                scaler_y, step, path='../data/edison/processed.pkl'):
    
    dictbound = {'start_train':'2017-11-15', #pd.to_datetime(dictbound['start_train'])
                'end_train':'2020-01-01',
                'start_val':'2020-01-01', 
                'end_val':'2020-12-01',
                'start_test':'2021-03-05', 
                'end_test':'2022-04-06'}

    with open(path,'rb') as f:
        df_el,col_el,df_term,col_term,le,max_id, meteo_columns = pickle.load(f)
        df_el.tempo = pd.to_datetime(df_el.tempo)
        f.close()

    df = df_el[['tempo','is_low','y']]
    df.y[df.y < 0] = np.nan
    
    train_df = df[df.tempo.between( to_datetime(dictbound['start_train']), to_datetime(dictbound['end_train']) )][:-1]
    val_df = df[df.tempo.between( to_datetime(dictbound['start_val']), to_datetime(dictbound['end_val']) )][:-1]
    test_df = df[df.tempo.between( to_datetime(dictbound['start_test']), to_datetime(dictbound['end_test']) )][:-1]

    # scaler fitted over training y
    train_y = torch.tensor(train_df['y'].values)
    scaler_y.fit(train_y.unsqueeze(1))
    
    print('-'*50+'\n')
    if train_bool:
        train_Ds = CustomDataset(train_df, scaler_y, seq_len, lag, step, hour=hour_learning)
        val_Ds = CustomDataset(val_df, scaler_y, seq_len, lag, step, hour=hour_learning)
        test_Ds = CustomDataset(test_df, scaler_y, seq_len, lag, step, hour=hour_inference)

        train_dl = DataLoader(train_Ds, batch_size = batch_size, shuffle = True)
        val_dl = DataLoader(val_Ds, batch_size = batch_size, shuffle = True)
        test_dl = DataLoader(test_Ds, batch_size = batch_size_test, shuffle = False)

        print('Learning & Testing')
        print(f'Hour Learning = {hour_learning}, Hour Testing = {hour_inference}')
        print(f'> len_train_dl = {len(train_dl)}')
        print(f'> len_val_dl   = {len(val_dl)}')
        print(f'> len_test_dl  = {len(test_dl)}')
        print(f'Train & Val Batch_size = {batch_size}')
        
    else:
        test_Ds = CustomDataset(test_df, scaler_y, seq_len, lag, step, hour=hour_inference)

        train_dl = None
        val_dl = None
        test_dl = DataLoader(test_Ds, batch_size = batch_size_test, shuffle = False)

        print('Only Testing')
        print(f'Hour Testing = {hour_inference}')
        print(f'len_test_dl = {len(test_dl)}')
        
    print(f'Test Batch_size = {batch_size_test}')
    return train_dl, val_dl, test_dl, scaler_y

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, seq_len, lag, step, hour):
        X = []
        Y = []

        x = df.tempo.apply(lambda x: [int(c) for c in x.strftime('%Y %m %d %H %w').split()]).values
        x = torch.tensor(np.array([np.array(c) for c in x]))
        y = torch.tensor(df.y.values)
        y = scaler.transform(y.unsqueeze(1)).squeeze()
        is_low = torch.tensor(df.is_low.values)

        # # for correct embedding?
        # max_month = max(x[:,1])
        # max_day = max(x[:,2])
        # max_hour = max(x[:,3])
        # max_dow = max(x[:,4])
        # max_low = max(is_low)

        # import pdb
        # pdb.set_trace()
        for i in range(seq_len,len(df),step):
            if not (np.isnan(df.y[i-seq_len:i]).any()):
                if not 0 in is_low[i-lag:i]:
                    if hour==24:
                        # ALL SEQUENCES
                        X.append(torch.cat((x[i-seq_len:i],is_low[i-seq_len:i].unsqueeze(1)), dim=1))
                        Y.append(y[i-seq_len:i])

                    else:
                        # SEQUENCES WITH REDICTION STARTING FROM 'hour'
                        if x[i-lag,-2] == hour: # x is [y,m,d,h,dow]
                            # x =  df.tempo[i-seq_len:i].apply(lambda x: x.strftime('%Y %m %d %H %w').split()).values
                            # y = torch.tensor(df.y[i-seq_len:i].values)
                            # is_low = torch.tensor(df.is_low[i-seq_len:i].values)
                            X.append(torch.cat((x[i-seq_len:i],is_low[i-seq_len:i].unsqueeze(1)), dim=1))
                            Y.append(y[i-seq_len:i])

        self.x = np.stack(X) # %Y %m %d %H %w is_low
        self.y = np.stack(Y)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
if __name__=='__main__':

    bs = 32
    bs_test = 8
    seq_len = 256
    lag = 60
    hour = 24
    hour_test = 24
    train = False
    path_data = '/home/andrea/timeseries/data/edison/processed.pkl' 
    step = 1
    scaler_type = StandardScaler()
    train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=bs, batch_size_test=bs_test, 
                                                        seq_len=seq_len, lag=lag,
                                                        hour_learning=hour, 
                                                        hour_inference=hour_test, 
                                                        train_bool=train,
                                                        step=step,
                                                        scaler_y=scaler_type,
                                                        path=path_data)
    
