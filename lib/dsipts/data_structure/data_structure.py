import numpy as np
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
pd.options.mode.chained_assignment = None 
import pickle

def extend_df(x,freq):
    empty = pd.DataFrame({'time':pd.date_range(x.min(),x.max(),freq=freq)})
    return empty


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {'val_loss':[],'train_loss':[]}
        

    def on_validation_end(self, trainer, pl_module):

        for c in trainer.callback_metrics:
            self.metrics[c].append(trainer.callback_metrics[c].item())




class MyDataset(Dataset):
    def __init__(self, data,t=None):
        self.data = data
        self.t = t
    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, idxs):
        sample = {}
        for k in self.data:
            sample[k] = self.data[k][idxs]
        return sample

class ActionEnum(Enum):
    """
    action of categorical variable
    """
    multiplicative: str = 'multiplicative'
    additive: str = 'additive'

@dataclass
class Categorical():
    
    '''
    Create a categorical signal it can act as mutliplicative or additive. You can control the number of classesm the duration and the level 
    '''
    name: str
    frequency: int
    duration: List[int]
    classes: int
    action: ActionEnum
    level: List[float]
    
    def validate(self):
        if len(self.level) == self.classes:
            return True
        else:
            return False
        
    def __post_init__(self):
        if not self.validate():
            raise ValueError("Length must match")

    
    def generate_signal(self,length:int):
        if self.action  == 'multiplicative':
            signal = np.ones(length)
        elif self.action == 'additive':
            signal = np.zeros(length)
        classes = []    
        _class = 0
        _level = self.level[0]
        _duration = self.duration[0]

        count = 0
        count_freq = 0
        for i in range(length):
            if count_freq%self.frequency == 0:
                signal[i] = _level
                classes.append(_class)
                count+=1
                if count == _duration:
                    #change class
                    count = 0
                    _class+=1
                    count_freq+= _duration
                    _class = _class%self.classes
                    _level = self.level[_class]
                    _duration = self.duration[_class]

                   
            else:
                classes.append(-1)
                count_freq+=1
        
        self.classes_array = classes
        self.signal_array = signal

    def plot(self):
        tmp = pd.DataFrame({'time':range(len(self.classes_array)),'signal':self.signal,'class':self.classes_array})
        fig = px.scatter(tmp,x='time',y='signal',color='class',title=self.name)
        fig.show()


 
@dataclass
class TimeSeries():
    '''
        Class for generating time series object. If you don't have any time series you can build one fake timeseries using some
        helping classes (Categorical for instance).
    
            Parameters:
                    name (str): name of the series
    
    
    For example we can generate a toy timeseries:
    #- add a multiplicative categorical feature (weekly)
    settimana = Categorical('settimanale',1,[1,1,1,1,1,1,1],7,'multiplicative',[0.9,0.8,0.7,0.6,0.5,0.99,0.99])
    #- an additive montly feature (here a year is composed by 5 months)
    mese = Categorical('mensile',1,[31,28,20,10,33],5,'additive',[10,20,-10,20,0])
    #- a spotted categorical variable that happens every 100 days and lasts 1 day
    spot = Categorical('spot',100,[7],1,'additive',[10])
    ts = TimeSeries('prova')
    ts.generate_signal(length = 5000,categorical_variables = [settimana,mese,spot],noise_mean=1,type=0) ##we can add also noise
    
    '''
    name: str
    
    def _generate_base(self,length,type=0):
        if type==0:
            self.base_signal = 10*np.cos(np.arange(length)/(2*np.pi*length/100))
            self.out_vars = 1
        else:
            print('please implement your own method')
    def generate_signal(self,length=5000,categorical_variables=[],noise_mean=1,type=0):
        '''
        This will generate a syntetic signal with a selected length, a noise level and some categorical variables. The additive series are added at the end while the multiplicative series axt 
        '''
        dataset = pd.DataFrame({'time':range(length)})
        self._generate_base(length,type)
        signal = self.base_signal.copy()
        tot = None
        self.cat_var = []
        for c in categorical_variables:
            c.generate_signal(length)
            _signal = c.signal_array
            classes = c.classes_array
            dataset[c.name] = classes
            self.cat_var.append(c.name)
            if c.action=='multiplicative':
                signal*=_signal
            else:
                if tot is None:
                    additive = _signal
                else:
                    additive+=_signal
        signal+=additive
        dataset['signal'] = signal + noise_mean*np.random.randn(len(signal))
        self.dataset = dataset
        
        
        self.past_variables = ['signal']
        self.future_variables = []
        self.target_variables = ['signal']
        self.num_var = list(set(self.past_variables).union(set(self.future_variables)).union(set(self.target_variables)))
        
    def load_signal(self,data,enrich_cat = [],past_variables=[],future_variables=[],target_variables=[],cat_var=[],check_past=True):
        '''
        This is a crucial point in the data structure. We expect here to have a dataset with time as timestamp and signal as y
        The other columns can represent covariates, categorical features, know and unknow variables in the future
        The categorical variables will be processed with an label encoder in the function create_data_loader
        If enrich cat is provided it will added automatically some temporal categorical features
        '''
        
        ##some checks!
        
        dataset = data.copy()
        if dataset.time.diff()[1:].nunique()>1:
            print("There are holes in the dataset i will try to extend the dataframe inserting NAN")
            freq = pd.to_timedelta(np.diff(dataset.time).min())
            
            dataset = extend_df(dataset.time,freq).merge(dataset,how='left')
            

            
        assert len(target_variables)>0, f'Provide at least one column for target'
        assert 'time'  in dataset.columns, f'The temporal column must be called time'
        if set(target_variables).intersection(set(past_variables))!= set(target_variables): 
            if check_past:
                print('I will update past column adding all target columns, if you want to avoid this beahviour please use check_pass as false')
                past_variables = list(set(past_variables).union(set(target_variables)))
        
        self.cat_var = cat_var
        self.enrich_cat = enrich_cat
        for c in enrich_cat:
            self.cat_var = list(set(self.cat_var+[c]))
                
            if c in dataset.columns:
                print('categorical {c} already present, it will be added to categorical variable but not recomputed') 
            else:
                if c =='hour':
                    dataset[c] = dataset.time.dt.hour
                elif c=='dow':
                    dataset[c] = dataset.time.dt.weekday
                elif c=='month':
                    dataset[c] = dataset.time.dt.month
                elif c=='minute':
                    dataset[c] = dataset.time.dt.minute
                else:
                    print('I can not automatically add column {c} plase update this function accordlyng')
        self.dataset = dataset
        self.past_variables =past_variables
        self.future_variables = future_variables
        self.target_variables = target_variables
        self.out_vars = len(target_variables)
        self.num_var = list(set(self.past_variables).union(set(self.future_variables)).union(set(self.target_variables)))
        
    def plot(self):
        print('plotting only target variables')
        tmp = self.dataset[['time']+self.target_variables].melt(id_vars=['time'])
        fig = px.line(tmp,x='time',y='value',color='variable',title=self.name)
        fig.show()
        
    def create_data_loader(self,dataset,past_steps,future_steps,shift,starting_point=None):
        x_num_past_samples = []
        x_num_future_samples = []
        x_cat_past_samples = []
        x_cat_future_samples = []
        y_samples = []
        t_samples = []
        
        for c in self.cat_var:
            dataset[c] = self.scaler_cat[c].transform(dataset[c].values.ravel()).flatten()
        for c in self.num_var: 
            dataset[c] = self.scaler_num[c].transform(dataset[c].values.reshape(-1,1)).flatten()
         
        
        x_num_past = dataset[self.past_variables].values
        if len(self.future_variables)>0:
            x_num_future = dataset[self.future_variables].values
        if len(self.cat_var)>0:
            x_cat = dataset[self.cat_var].values
        y_target = dataset[self.target_variables].values
        t = dataset.time.values

        ##questo serve a forzare di iniziare i samples alla stessa ora per esempio (controllo sul primo indice della y)
        if starting_point is not None:
            check = dataset[list(starting_point.keys())[0]].values == starting_point[list(starting_point.keys())[0]]
        else:
            check = [True]*len(y_target)
        
        for i in range(past_steps,dataset.shape[0]-future_steps):
            if check[i]:

                if len(self.future_variables)>0:
                    xx = x_num_future[i-shift:i+future_steps-shift].mean()
                else:
                    xx = 0.0
                if np.isfinite(x_num_past[i-past_steps:i].min() + y_target[i:i+future_steps].min() + xx):
                    
                    x_num_past_samples.append(x_num_past[i-past_steps:i])
                    if len(self.future_variables)>0:
                        x_num_future_samples.append(x_num_future[i-shift:i+future_steps-shift])
                    if len(self.cat_var)>0:
                        x_cat_past_samples.append(x_cat[i-past_steps:i])
                        x_cat_future_samples.append(x_cat[i-shift:i+future_steps-shift])
                    y_samples.append(y_target[i:i+future_steps])
                    t_samples.append(t[i:i+future_steps])
            
        if len(self.future_variables)>0:
            x_num_future_samples = np.stack(x_num_future_samples)
        y_samples = np.stack(y_samples)

        t_samples = np.stack(t_samples)
        if len(self.cat_var)>0:
            x_cat_past_samples = np.stack(x_cat_past_samples)
            x_cat_future_samples = np.stack(x_cat_future_samples)
        x_num_past_samples = np.stack(x_num_past_samples)
        
        dd = {'y':y_samples.astype(np.float32),

              'x_num_past':x_num_past_samples.astype(np.float32)}
        if len(self.cat_var)>0:
            dd['x_cat_past'] = x_cat_past_samples
            dd['x_cat_future'] = x_cat_future_samples
        if len(self.future_variables)>0:
            dd['x_num_future'] = x_num_future_samples.astype(np.float32)
        
        return MyDataset(dd,t_samples)
    
    def split_for_train(self,perc_train=0.6, perc_valid=0.2, range_train=None, range_validation=None, range_test=None,past_steps = 100,future_steps=20,shift = 0,starting_point=None):
        try:
            l = self.dataset.shape[0]
        except:
            print('I will call generate signal because it is not initialized')
            self.generate_signal()
        

        if range_train is None:
            print(f'Split temporally using perc_train: {perc_train} and perc_valid:{perc_valid}')
        
            train = self.dataset.iloc[0:int(perc_train*l)]
            validation = self.dataset.iloc[int(perc_train*l):int(perc_train*l+perc_valid*l)]
            test = self.dataset.iloc[int(perc_train*l+perc_valid*l):]
        else:
            print('Split temporally using the time intervals provided')

            train = self.dataset[self.dataset.time.between(range_train[0],range_train[1])]
            validation =  self.dataset[self.dataset.time.between(range_validation[0],range_validation[1])]
            test =  self.dataset[self.dataset.time.between(range_test[0],range_test[1])]
                                      
        self.scaler_cat = {}
        self.scaler_num = {}
        for c in self.num_var:
            self.scaler_num[c] =  StandardScaler()
            self.scaler_num[c].fit(train[c].values.reshape(-1,1))
        for c in self.cat_var:                               
            self.scaler_cat [c] =  LabelEncoder()
            self.scaler_cat[c].fit(train[c].values.reshape(-1,1))  
                                      
    
        dl_train = self.create_data_loader(train,past_steps,future_steps,shift,starting_point)
        dl_validation = self.create_data_loader(validation,past_steps,future_steps,shift,starting_point)
        dl_test = self.create_data_loader(test,past_steps,future_steps,shift,starting_point)

        return dl_train,dl_validation,dl_test
            
    def set_model(self,model,quantile = False,config=None):
        self.model = model
        self.return_quantile = quantile
        if (self.out_vars>1 ) and quantile:
            print('care multioutoput with quantile not tested')
        self.config = config
    def train_model(self,dirpath,perc_train=0.6, perc_valid=0.2,past_steps = 100,future_steps=20,shift = 0,batch_size=100,num_workers=4,max_epochs=500,auto_lr_find=True,starting_point=None):
        self.split_params = {'perc_train':perc_train,
                             'perc_valid':perc_valid,
                             'past_steps':past_steps,
                             'future_steps':future_steps,
                             'shift':shift,
                             'starting_point':starting_point
                             }
        train,validation,test = self.split_for_train(perc_train=perc_train, perc_valid=perc_valid,past_steps = past_steps,future_steps=future_steps,shift = shift,starting_point=starting_point)
        print(f'train:{len(train)}, validation:{len(validation)}, test:{len(test)}')
        train_dl = DataLoader(train, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers)
        valid_dl = DataLoader(validation, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers)
        checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                     monitor='val_loss',
                                      save_last = True,
                                      every_n_epochs =1,
                                      verbose = False,
                                      save_top_k = 1,
                                     filename='checkpoint')
        
        
        logger = CSVLogger("logs", name=dirpath)
        trainer = pl.Trainer(logger = logger,max_epochs=max_epochs,callbacks=[checkpoint_callback,MetricsCallback()],auto_lr_find=auto_lr_find)

        if auto_lr_find:
            trainer.tune(self.model,train_dataloaders=train_dl,val_dataloaders = valid_dl)


        trainer.fit(self.model, train_dl,valid_dl)
        self.checkpoint_file_best = checkpoint_callback.best_model_path
        self.checkpoint_file_last =checkpoint_callback.last_model_path

        for c in trainer.callbacks:
            if 'metrics' in dir(c):
                self.losses = c.metrics
                
                ##non so perche' le prime due le chiama prima del train
                self.losses['val_loss'] = self.losses['val_loss'][2:]
                self.losses = pd.DataFrame(self.losses)
                break
        self.model = self.model.load_from_checkpoint(self.checkpoint_file_last)

    def inference_test(self,batch_size=100,num_workers=4,split_params=None):
        if split_params is None:
            print(f'splitting using train parameters {self.split_params}')
            _,_,test = self.split_for_train(**self.split_params)
        else:
            _,_,test = self.split_for_train(**split_params)
        train_dl = DataLoader(test, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)
        self.model.eval()
        res = []
        real = []
        for batch in train_dl:
            res.append(self.model.inference(batch).detach().numpy())
            real.append(batch['y'].detach().numpy())
        res = np.vstack(res)
        real = np.vstack(real)
        time = train_dl.dataset.t

        if self.return_quantile:
            ##qui sono sicuro che sia single output
            real = real[:,:,0]
            c = self.target_variables[0]
            time = pd.DataFrame(time,columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':'time'})
            real = pd.DataFrame(real,columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':c}).drop(columns=['variable'])
            median = pd.DataFrame(res[:,:,1],columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':'median'}).drop(columns=['variable'])
            low = pd.DataFrame(res[:,:,0],columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':'low'}).drop(columns=['variable'])
            high = pd.DataFrame(res[:,:,2],columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':'high'}).drop(columns=['variable'])
            
            res = pd.concat([time,real,median,low,high],axis=1)
            
        else:
            time = pd.DataFrame(time,columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':'time'})
            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(pd.DataFrame(real[:,:,i],columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':c}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i],columns=[f'lag_{i+1}' for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_pred'}).drop(columns=['variable']))
            res = pd.concat(tot,axis=1)

            
        return res
    def inference():
        pass
        
    def save(self, filename):
        print('Saving')
        with open(f'{filename}.pkl','wb') as f:
            params =  self.__dict__.copy()
            for k in ['model']:
                if k in params.keys():
                    _ = params.pop(k)
            pickle.dump(params,f)

    def load(self,model, filename,load_last=True):
        print('Loading')
        with open(filename+'.pkl','rb') as f:
            params = pickle.load(f)
            for p in params:
                setattr(self,p, params[p])    
        self.model = model(**self.config['model_configs'],optim_config = self.config['optim_config'],scheduler_config =self.config['scheduler_config'] )
        if load_last:
            self.model = self.model.load_from_checkpoint(self.checkpoint_file_last)
        else:
            self.model = self.model.load_from_checkpoint(self.checkpoint_file_best)