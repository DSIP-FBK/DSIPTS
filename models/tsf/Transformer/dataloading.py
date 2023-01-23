import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from pandas import to_datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from time import time

def dataloading(path='./data/processed.pkl', batch_size:int = 20, batch_size_test:int = 20, seq_len:int = 256, lag:int = 61, step:int = 1, load_all_seq:bool=True, train_bool:bool = True):
    
    dictbound = {'start_train':'2017-11-15', #pd.to_datetime(dictbound['start_train'])
                'end_train':'2020-01-01',
                'start_val':'2020-01-01', 
                'end_val':'2020-12-01',
                'start_test':'2021-03-05', 
                'end_test':'2022-04-06'}

    with open(path,'rb') as f:
        df_el,col_el,df_term,col_term,le,max_id, meteo_columns = pickle.load(f)
        df_el.tempo = pd.to_datetime(df_el.tempo)

    df = df_el[['tempo','is_low','y']]
    df.y[df.y < 0] = np.nan
    
    train = df[df.tempo.between( to_datetime(dictbound['start_train']), to_datetime(dictbound['end_train']) )][:-1]
    val = df[df.tempo.between( to_datetime(dictbound['start_val']), to_datetime(dictbound['end_val']) )][:-1]
    test = df[df.tempo.between( to_datetime(dictbound['start_test']), to_datetime(dictbound['end_test']) )][:-1]

    # scaler fitted over training y
    scaler_y = StandardScaler()
    train_y = torch.tensor(train['y'].values)
    scaler_y.fit(train_y.unsqueeze(1))

    # train_data_y = scaler_y.transform(train_data_y.unsqueeze(1)).squeeze()
    # val_data_y = scaler_y.transform(val_data_y.unsqueeze(1)).squeeze()
    # test_data_y = scaler_y.transform(test_data_y.unsqueeze(1)).squeeze()
    L = len(train) + len(val) + len(test)
    print('-'*50)
    print(f'Train portion:      {len(train)/L*100:.4}%\t {len(train)}')
    print(f'Validation portion: {len(val)/L*100:.4}%\t {len(val)}')
    print(f'Test portion:       {len(test)/L*100:.4}%\t {len(test)}')
    print('-'*50)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, df, seq_len=256, lag=60, step=1, load_all_seq=load_all_seq):
            X = []
            Y = []
            Is_Low = []
            
            x = df.tempo.apply(lambda x: [int(c) for c in x.strftime('%Y %m %d %H %w').split()]).values
            x = torch.tensor(np.array([np.array(c) for c in x]))
            y = torch.tensor(df.y.values)
            y = scaler_y.transform(y.unsqueeze(1)).squeeze()
            is_low = torch.tensor(df.is_low.values)

            for i in range(seq_len,len(df),step):
                # if np.isfinite(df.y[i-seq_len:i].sum()):
                if not (np.isnan(df.y[i-seq_len:i]).any()):

                    if load_all_seq:
                        # ALL SEQUENCES
                        X.append(x[i-seq_len:i])
                        Y.append(y[i-seq_len:i])
                        Is_Low.append(is_low[i-seq_len:i])

                    else:
                        # SEQUENCES WITH REDICTION STARTING FROM 'hour'
                        hour = 12

                        if x[i-lag,-2] == hour: # x is [y,m,d,h,dow]
                            # x =  df.tempo[i-seq_len:i].apply(lambda x: x.strftime('%Y %m %d %H %w').split()).values
                            # y = torch.tensor(df.y[i-seq_len:i].values)
                            # is_low = torch.tensor(df.is_low[i-seq_len:i].values)
                            X.append(x[i-seq_len:i])
                            Y.append(y[i-seq_len:i])
                            Is_Low.append(is_low[i-seq_len:i])
            
            self.x = np.stack(X) # %Y %m %d %H %w
            self.y = np.stack(Y)
            self.is_low = np.stack(Is_Low)
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, index):
            return (self.x[index], self.y[index], self.is_low[index])

    if train_bool:
        train_Ds = CustomDataset(train, step=step, load_all_seq=load_all_seq)
        val_Ds = CustomDataset(val, step=step, load_all_seq=load_all_seq)
        test_Ds = CustomDataset(test, step=step, load_all_seq=load_all_seq)

        train_dl = DataLoader(train_Ds, batch_size = batch_size, shuffle = True)
        val_dl = DataLoader(val_Ds, batch_size = batch_size, shuffle = True)
        test_dl = DataLoader(test_Ds, batch_size = batch_size_test, shuffle = True)

        print(f'len_train_dl = {len(train_dl)}, len_val_dl = {len(val_dl)}, len_test_dl = {len(test_dl)}')
        print(f'Train & Val Batch_size = {batch_size}')
        
    else:
        train_dl = None
        val_dl = None
        test_Ds = CustomDataset(test, step=step, load_all_seq=load_all_seq)

        test_dl = DataLoader(test_Ds, batch_size = batch_size_test, shuffle = True)
        print(f'train_dl = None, val_dl = None, len_test_dl = {len(test_dl)}')
        
    print(f'Test Batch_size = {batch_size_test}')
    print('-'*50)
    return train_dl, val_dl, test_dl, scaler_y
            

if __name__=='__main__':
    t = time()
    train_dl, val_dl, test_dl, scaler_y = dataloading(batch_size=16, seq_len=256, lag=61, step=1, load_all_seq=True, train_bool=True)
    print(f'time elapsed: {time()-t}')
    # bs = 16, las = True -> len_train_dl = 1150, len_val_dl = 487, len_test_dl = 580
    # bs = 1, las = True -> len_train_dl = 18392, len_val_dl = 7784, len_test_dl = 9272

    # bs = 16, las = False -> len_train_dl = 48, len_val_dl = 21, len_test_dl = 25
    # bs = 1, las = False -> len_train_dl = 766, len_val_dl = 324, len_test_dl = 386

    it = iter(train_dl)
    data = next(it)
    import pdb 
    pdb.set_trace()
    print('bocia scemo')
