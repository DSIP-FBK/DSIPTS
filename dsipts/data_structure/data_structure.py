import numpy as np
import plotly.express as px
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from typing import Union
import os
import torch
import pickle
from .utils import extend_time_df,MetricsCallback, MyDataset, ActionEnum,beauty_string
from datetime import datetime
from ..models.base import Base
import logging 
from .modifiers import *
 


pd.options.mode.chained_assignment = None 
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())      
      
      
class Categorical():
    
    def __init__(self,name:str, frequency: int,duration: List[int], classes: int, action: ActionEnum,  level: List[float]):
        """Class for generating toy categorical data

        Args:
            name (str): name of the categorical signal
            frequency (int): frequency of the signal
            duration (List[int]): duration of each class
            classes (int): number of classes
            action (str): one between additive or multiplicative
            level (List[float]): intensity of each class

        """

        self.name = name 
        self.frequency = frequency
        self.duration = duration
        self.classes = classes
        self.action = action
        self.level = level
        self.validate()

    def validate(self):
        """Validate, maybe there will be other checks in the future
        
        :meta private:
        """
        if len(self.level) == self.classes:
            pass
        else:
            raise ValueError("Length must match")
    
    def generate_signal(self,length:int)->None:
        """Generate the resposne signal

        Args:
            length (int): length of the signal
        """
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

    def plot(self)->None:
        """Plot the series
        """
        tmp = pd.DataFrame({'time':range(len(self.classes_array)),'signal':self.signal,'class':self.classes_array})
        fig = px.scatter(tmp,x='time',y='signal',color='class',title=self.name)
        fig.show()

 
class TimeSeries():
    
    def __init__(self,name:str,stacked:bool=False):
        """Class for generating time series object. If you don't have any time series you can build one fake timeseries using some helping classes (Categorical for instance).


        Args:
            name (str): name of the series
            stacked (bool): if true it is a stacked model
                
        Usage:
            For example we can generate a toy timeseries:\n
            - add a multiplicative categorical feature (weekly)\n
            >>> settimana = Categorical('settimanale',1,[1,1,1,1,1,1,1],7,'multiplicative',[0.9,0.8,0.7,0.6,0.5,0.99,0.99])\n
            - an additive montly feature (here a year is composed by 5 months)\n
            >>> mese = Categorical('mensile',1,[31,28,20,10,33],5,'additive',[10,20,-10,20,0])\n
            - a spotted categorical variable that happens every 100 days and lasts 1 day\n
            >>> spot = Categorical('spot',100,[7],1,'additive',[10])\n
            >>> ts = TimeSeries('prova')\n
            >>> ts.generate_signal(length = 5000,categorical_variables = [settimana,mese,spot],noise_mean=1,type=0) ##we can add also noise\n
            >>> ts.plot()\n
        """
        self.is_trained = False
        self.name = name
        self.stacked = stacked
        self.verbose = True
    def __str__(self) -> str:
        return beauty_string(f"Timeseries named {self.name} of length {self.dataset.shape[0]}.\n Categorical variable: {self.cat_var},\n Future variables: {self.future_variables},\n Past variables: {self.past_variables},\n Target variables: {self.target_variables} \n With {'no group' if self.group is None else self.group+' as group' }",'',True)
    def __repr__(self) -> str:
        return f"Timeseries named {self.name} of length {self.dataset.shape[0]}.\n Categorical variable: {self.cat_var},\n Future variables: {self.future_variables},\n Past variables: {self.past_variables},\n Target variables: {self.target_variables}\n With {'no group' if self.group is None else self.group+' as group' }"
    
    def set_verbose(self,verbose:bool):
        self.verbose = verbose
    def _generate_base(self,length:int,type:int=0)-> None:
        """Generate a basic timeseries 

        Args:
            length (int): length
            type (int, optional): Type of the generated timeseries. Defaults to 0.
        """
        if type==0:
            self.base_signal = 10*np.cos(np.arange(length)/(2*np.pi*length/100))
            self.out_vars = 1
        else:
            beauty_string('Please implement your own method','block',True)
        """
        
        """    
    def generate_signal(self,length:int=5000,categorical_variables:List[Categorical]=[],noise_mean:int=1,type:int=0)->None:
        """This will generate a syntetic signal with a selected length, a noise level and some categorical variables. The additive series are added at the end while the multiplicative series acts on the original signal
        The TS structure will be populated

        Args:
            length (int, optional): length of the signal. Defaults to 5000.
            categorical_variables (List[Categorical], optional): list of Categorical variables. Defaults to [].
            noise_mean (int, optional): variance of the noise to add at the end. Defaults to 1.
            type (int, optional): type of the timeseries (only type=0 available right now). Defaults to 0.
        """
        
       
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
        
        
    def enrich(self,dataset,columns):
        if columns =='hour':
            dataset[columns] = dataset.time.dt.hour
        elif columns=='dow':
            dataset[columns] = dataset.time.dt.weekday
        elif columns=='month':
            dataset[columns] = dataset.time.dt.month
        elif columns=='minute':
            dataset[columns] = dataset.time.dt.minute
        else:
            if columns  not in dataset.columns:
                beauty_string(f'I can not automatically enrich column {columns}. Please contact the developers or add it manually to your dataset.','section',True)

    def load_signal(self,data:pd.DataFrame,
                    enrich_cat:List[str] = [],
                    past_variables:List[str]=[],
                    future_variables:List[str]=[],
                    target_variables:List[str]=[],
                    cat_var:List[str]=[],
                    check_past:bool=True,
                    group:Union[None,str]=None,
                    check_holes_and_duplicates:bool=True,
                    silly_model:bool=False)->None:
        """ This is a crucial point in the data structure. We expect here to have a dataset with time as timestamp.
            There are some checks:
                1- the duplicates will tbe removed taking the first instance
                2- the frequency will the inferred taking the minumum time distance between samples
                3- the dataset will be filled completing the missing timestamps

        Args:
            data (pd.DataFrame): input dataset the column indicating the time must be called `time`
            enrich_cat (List[str], optional): it is possible to let this function enrich the dataset for example adding the standard columns: hour, dow, month and minute. Defaults to [].
            past_variables (List[str], optional): list of column names of past variables not available for future times . Defaults to [].
            future_variables (List[str], optional): list of future variables available for tuture times. Defaults to [].
            target_variables (List[str], optional): list of the target variables. They will added to past_variables by default unless `check_past` is false. Defaults to [].
            cat_var (List[str], optional): list of the categortial variables (same for past and future). Defaults to [].
            check_past (bool, optional): see `target_variables`. Defaults to True.
            group (str or None, optional): if not None the time serie dataset is considered composed by omogeneus timeseries coming from different realization (for example point of sales, cities, locations)
            and the relative series are not splitted during the sample generation. Defaults to None
            check_holes_and_duplicates (bool, optional): if False duplicates or holes will not checked, the dataloader can not correctly work, disable at your own risk. Defaults True
            silly_model (bool, optional): if True, target variables will be added to the pool of the future variables. This can be useful to see if information passes throught the decoder part of your model (if any)
        """
        
        
        
        dataset = data.copy()
        dataset.sort_values(by='time',inplace=True)
        
        if check_holes_and_duplicates:
            beauty_string('I will drop duplicates, I dont like them','section',self.verbose)
            dataset.drop_duplicates(subset=['time'] if group is None else [group,'time'],  keep='first', inplace=True, ignore_index=True)
            
            if group is None:
                differences = dataset.time.diff()[1:]
            else:
                differences = dataset[dataset[group]==dataset[group].unique()[0]].time.diff()[1:]
                
            if differences.nunique()>1:
                beauty_string("There are holes in the dataset i will try to extend the dataframe inserting NAN",'info',self.verbose)
                freq = pd.to_timedelta(differences.min())
                beauty_string(f'Detected minumum frequency: {freq}','section',self.verbose)
                dataset = extend_time_df(dataset,freq,group).merge(dataset,how='left')
            

            
        assert len(target_variables)>0, 'Provide at least one column for target'
        assert 'time'  in dataset.columns, 'The temporal column must be called time'
        if set(target_variables).intersection(set(past_variables))!= set(target_variables): 
            if check_past:
                beauty_string('I will update past column adding all target columns, if you want to avoid this beahviour please use check_pass as false','info',self.verbose)
                past_variables = list(set(past_variables).union(set(target_variables)))
        
        self.cat_var = cat_var
        self.group = group 
        if group is not None:
            if group not in cat_var:
                beauty_string(f'I will add {group} to the categorical variables','info',self.verbose)
                self.cat_var.append(group)
                
                
                
        self.enrich_cat = enrich_cat
        for c in enrich_cat:
            self.cat_var = list(set(self.cat_var+[c]))
                
            if c in dataset.columns:
                beauty_string('Categorical {c} already present, it will be added to categorical variable but not call the enriching function','info',self.verbose) 
            else:
                self.enrich(dataset,c)
        self.dataset = dataset
        self.past_variables =past_variables
        self.future_variables = future_variables
        self.target_variables = target_variables
        self.out_vars = len(target_variables)
        self.num_var = list(set(self.past_variables).union(set(self.future_variables)).union(set(self.target_variables)))
        if silly_model:
            beauty_string('YOU ARE TRAINING A SILLY MODEL WITH THE TARGETS IN THE INPUTS','section',self.verbose) 
            self.future_variables+=self.target_variables
            
    def plot(self):
        """  
        Easy way to control the loaded data
        Returns:
            plotly.graph_objects._figure.Figure: figure of the target variables
        """
      
        beauty_string('Plotting only target variables','block',self.verbose)
        if self.group is None:
            tmp = self.dataset[['time']+self.target_variables].melt(id_vars=['time'])
            fig = px.line(tmp,x='time',y='value',color='variable',title=self.name)
            fig.show()
        else:
            tmp = self.dataset[['time',self.group]+self.target_variables].melt(id_vars=['time',self.group])
            fig = px.line(tmp,x='time',y='value',color='variable',title=self.name,facet_row=self.group)
            fig.show()
        return fig
    
        
    def create_data_loader(self,data:pd.DataFrame,
                           past_steps:int,
                           future_steps:int,
                           shift:int=0,
                           keep_entire_seq_while_shifting:bool=False,
                           starting_point:Union[None,dict]=None,
                           skip_step:int=1
                         
                           )->MyDataset:
        """ Create the dataset for the training/inference step

        Args:
            data (pd.DataFrame): input dataset, usually a subset of self.data
            past_steps (int): past context length
            future_steps (int): future lags to predict
            shift (int, optional): if >0 the future input variables will be shifted (categorical and numerical). For example for attention model it is better to start with a know value of y and use it during the process. Defaults to 0.
            keep_entire_seq_while_shifting (bool, optional): if the dataset is shifted, you may want the future data be of length future_step+shift (like informer), default false
            starting_point (Union[None,dict], optional): a dictionary indicating if a sample must be considered. It is checked for the first lag in the future (useful in the case your model has to predict only starting from hour 12). Defaults to None.
            skip_step (int, optional): list of the categortial variables (same for past and future). Usual there is a skip of one between two saples but for debugging  or training time purposes you can skip some samples. Defaults to 1.
        Returns:
            MyDataset: class thath extends torch.utils.data.Dataset (see utils)
                keys of a batch:
                y : the target variable(s)
                x_num_past: the numerical past variables
                x_num_future: the numerical future variables
                x_cat_past: the categorical past variables
                x_cat_future: the categorical future variables
                idx_target: index of target features in the past array
        """
        beauty_string('Creating data loader','block',self.verbose)
        
        x_num_past_samples = []
        x_num_future_samples = []
        x_cat_past_samples = []
        x_cat_future_samples = []
        y_samples = []
        t_samples = []
        g_samples = []
        
        if starting_point is not None:
            kk = list(starting_point.keys())[0]
            assert kk not in self.cat_var, beauty_string('CAN NOT USE FEATURE {kk} as starting point it may have a different value due to the normalization step, please add a second column with a suitable name','info',True)
        
        ##overwrite categorical columns
        for c in self.cat_var:
            self.enrich(data,c)

        if self.group is None:
            data['_GROUP_'] = '1'
        else:
            data['_GROUP_'] = data[self.group].values
            
            
        if self.normalize_per_group:
            tot = []
            groups = data[self.group].unique()
            
            data[self.group] = self.scaler_cat[self.group].transform(data[self.group].values.ravel()).flatten()
            
            for group in groups:
                tmp = data[data['_GROUP_']==group].copy()
                
                for c in self.num_var:
                    tmp[c] = self.scaler_num[f'{c}_{group}'].transform(tmp[c].values.reshape(-1,1)).flatten()
                for c in self.cat_var:      
                    if c!=self.group:                         
                        tmp[c] = self.scaler_cat[f'{c}_{group}'].transform(tmp[c].values.ravel()).flatten()
                tot.append(tmp)
            data = pd.concat(tot,ignore_index=True)
        else:
            for c in self.cat_var:
                data[c] = self.scaler_cat[c].transform(data[c].values.ravel()).flatten()
            for c in self.num_var: 
                data[c] = self.scaler_num[c].transform(data[c].values.reshape(-1,1)).flatten()

        idx_target = []
        for c in self.target_variables:
            idx_target.append(self.past_variables.index(c))
            
        idx_target_future = []
        
        for c in self.target_variables:
            if c in self.future_variables:
                idx_target_future.append(self.future_variables.index(c))    
        if len(idx_target_future)==0:
            idx_target_future = None
        

        if self.stacked:
            skip_stacked = future_steps*future_steps-future_steps
        else:
            skip_stacked = 0
        for group in data['_GROUP_'].unique():
            tmp = data[data['_GROUP_']==group]
            groups = tmp['_GROUP_'].values  
            t = tmp.time.values 
            x_num_past = tmp[self.past_variables].values
            if len(self.future_variables)>0:
                x_num_future = tmp[self.future_variables].values
            if len(self.cat_var)>0:
                x_cat = tmp[self.cat_var].values
            y_target = tmp[self.target_variables].values

        
            if starting_point is not None:
                check = tmp[list(starting_point.keys())[0]].values == starting_point[list(starting_point.keys())[0]]
            else:
                check = [True]*len(y_target)
            
            for i in range(past_steps,tmp.shape[0]-future_steps-skip_stacked,skip_step):
                if check[i]:

                    if len(self.future_variables)>0:
                        if keep_entire_seq_while_shifting:
                            xx = x_num_future[i-shift+skip_stacked:i+future_steps+skip_stacked].mean()
                        else:
                            xx = x_num_future[i-shift+skip_stacked:i+future_steps-shift+skip_stacked].mean()
                    else:
                        xx = 0.0
                    if np.isfinite(x_num_past[i-past_steps:i].min() + y_target[i+skip_stacked:i+future_steps+skip_stacked].min() + xx):
                        
                        x_num_past_samples.append(x_num_past[i-past_steps:i])
                        if len(self.future_variables)>0:
                            if keep_entire_seq_while_shifting:
                                x_num_future_samples.append(x_num_future[i-shift+skip_stacked:i+future_steps+skip_stacked])
                            else:
                                x_num_future_samples.append(x_num_future[i-shift+skip_stacked:i+future_steps-shift+skip_stacked])
                        if len(self.cat_var)>0:
                            x_cat_past_samples.append(x_cat[i-past_steps:i])
                            if keep_entire_seq_while_shifting:
                                x_cat_future_samples.append(x_cat[i-shift+skip_stacked:i+future_steps+skip_stacked])
                            else:
                                x_cat_future_samples.append(x_cat[i-shift+skip_stacked:i+future_steps-shift+skip_stacked])
                        y_samples.append(y_target[i+skip_stacked:i+future_steps+skip_stacked])
                        t_samples.append(t[i+skip_stacked:i+future_steps+skip_stacked])
                        g_samples.append(groups[i])

    
  
        if len(self.future_variables)>0:
            try:
                x_num_future_samples = np.stack(x_num_future_samples)
            except Exception as e:
                beauty_string('WARNING x_num_future_samples is empty and it should not','info',True)
        y_samples = np.stack(y_samples)
        t_samples = np.stack(t_samples)   
        g_samples = np.stack(g_samples)

        if len(self.cat_var)>0:
            x_cat_past_samples = np.stack(x_cat_past_samples)
            x_cat_future_samples = np.stack(x_cat_future_samples)
        x_num_past_samples = np.stack(x_num_past_samples)
        if self.stacked:
            mod = 0
        else:
            mod = 1.0
        dd = {'y':y_samples.astype(np.float32),

              'x_num_past':(x_num_past_samples*mod).astype(np.float32)}
        if len(self.cat_var)>0:
            dd['x_cat_past'] = x_cat_past_samples
            dd['x_cat_future'] = x_cat_future_samples
        if len(self.future_variables)>0:
            dd['x_num_future'] = x_num_future_samples.astype(np.float32)
        
        return MyDataset(dd,t_samples,g_samples,idx_target,idx_target_future)
    
          
    
    def split_for_train(self,
                        perc_train:Union[float,None]=0.6,
                        perc_valid:Union[float,None]=0.2,
                        range_train:Union[List[Union[datetime, str]],None]=None,
                        range_validation:Union[List[Union[datetime, str]],None]=None,
                        range_test:Union[List[Union[datetime, str]],None]=None,
                        past_steps:int = 100,
                        future_steps:int=20,
                        shift:int = 0,
                        keep_entire_seq_while_shifting:bool=False,
                        starting_point:Union[None, dict]=None,
                        skip_step:int=1,
                        normalize_per_group: bool=False
                        )->List[DataLoader]:
        """Split the data and create the datasets.

        Args:
            perc_train (Union[float,None], optional): fraction of the training set. Defaults to 0.6.
            perc_valid (Union[float,None], optional): fraction of the test set. Defaults to 0.2.
            range_train (Union[List[Union[datetime, str]],None], optional): a list of two elements indicating the starting point and end point of the training set (string date style or datetime). Defaults to None.
            range_validation (Union[List[Union[datetime, str]],None], optional):a list of two elements indicating the starting point and end point of the validation set (string date style or datetime). Defaults to None.
            range_test (Union[List[Union[datetime, str]],None], optional): a list of two elements indicating the starting point and end point of the test set (string date style or datetime). Defaults to None.
            past_steps (int, optional): past step to consider for making the prediction. Defaults to 100.
            future_steps (int, optional): future step to predict. Defaults to 20.
            shift (int, optional): see `create_data_loader`. Defaults to 0.
            keep_entire_seq_while_shifting (bool, optional): if the dataset is shifted, you may want the future data be of length future_step+shift (like informer), default false

            starting_point (Union[None, dict], optional):  see `create_data_loader`. Defaults to None.
            skip_step (int, optional):  see `create_data_loader`. Defaults to 1.
            normalize_per_group (boolean, optional): if true and self.group is not None, the variables are scaled respect to the groups
        Returns:
            List[DataLoader,DataLoader,DataLoadtrainer]: three dataloader used for training or inference
        """

        beauty_string('Splitting for train','block',self.verbose)

        
        try:
            ls = self.dataset.shape[0]
        except Exception as _:
            beauty_string('Empty dataset','info', True)
            return None, None, None
        
        if range_train is None:
            if self.group is None:
                beauty_string(f'Split temporally using perc_train: {perc_train} and perc_valid:{perc_valid}','section',self.verbose)
                train = self.dataset.iloc[0:int(perc_train*ls)]
                validation = self.dataset.iloc[int(perc_train*ls):int(perc_train*ls+perc_valid*ls)]
                test = self.dataset.iloc[int(perc_train*ls+perc_valid*ls):]
            else:
                beauty_string(f'Split temporally using perc_train: {perc_train} and perc_valid:{perc_valid} for each group!','info',self.verbose)
                train = []
                validation =[]
                test = []
                ls = self.dataset.groupby(self.group).time.count().reset_index()
                for group in self.dataset[self.group].unique():
                    tmp = self.dataset[self.dataset[self.group]==group]
                    lt = ls[ls[self.group]==group].time.values[0]
                    train.append(tmp[0:int(perc_train*lt)])
                    validation.append(tmp[int(perc_train*lt):int(perc_train*lt+perc_valid*lt)])
                    test.append(tmp[int(perc_train*lt+perc_valid*lt):])

                train = pd.concat(train,ignore_index=True)
                validation = pd.concat(validation,ignore_index=True)
                test = pd.concat(test,ignore_index=True)


        else:

            beauty_string('Split temporally using the time intervals provided','section',self.verbose)
            train = self.dataset[self.dataset.time.between(range_train[0],range_train[1])]
            validation =  self.dataset[self.dataset.time.between(range_validation[0],range_validation[1])]
            test =  self.dataset[self.dataset.time.between(range_test[0],range_test[1])]


        beauty_string('Train categorical and numerical scalers','block',self.verbose)

        if self.is_trained:
            pass
        else:
            self.scaler_cat = {}
            self.scaler_num = {}
        
            if self.group is None or normalize_per_group is False:
                self.normalize_per_group = False
                for c in self.num_var:
                    self.scaler_num[c] =  StandardScaler()
                    self.scaler_num[c].fit(train[c].values.reshape(-1,1))
                for c in self.cat_var:                               
                    self.scaler_cat[c] =  LabelEncoder()
                    self.scaler_cat[c].fit(train[c].values.ravel())  
            else:
                self.normalize_per_group = True
                self.scaler_cat[self.group] =  LabelEncoder()
                self.scaler_cat[self.group].fit(train[self.group].values.ravel())  
                for group in train[self.group].unique():
                    tmp = train[train[self.group]==group]

                    for c in self.num_var:
                        self.scaler_num[f'{c}_{group}'] =  StandardScaler()
                        self.scaler_num[f'{c}_{group}'].fit(tmp[c].values.reshape(-1,1))
                    for c in self.cat_var:
                        if c!=self.group:                               
                            self.scaler_cat[f'{c}_{group}'] =  LabelEncoder()
                            self.scaler_cat[f'{c}_{group}'].fit(tmp[c].values.ravel())  
        
        dl_train = self.create_data_loader(train,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        dl_validation = self.create_data_loader(validation,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        if test.shape[0]>0:
            dl_test = self.create_data_loader(test,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        else:
            dl_test = None
        return dl_train,dl_validation,dl_test
            
    def set_model(self,model:Base,config:dict=None):
        """Set the model to train

        Args:
            model (Base): see `models`
            config (dict, optional): usually the configuration used by the model. Defaults to None.
        """
        self.model = model
        #self.model.apply(weight_init)
        self.config = config
        
        beauty_string('Setting the model','block',self.verbose)
        beauty_string(model,'',self.verbose)
              
    def train_model(self,dirpath:str,
                    split_params:dict,
                    batch_size:int=100,
                    num_workers:int=4,
                    max_epochs:int=500,
                    auto_lr_find:bool=True,
                    gradient_clip_val:Union[float,None]=None,
                    gradient_clip_algorithm:str="value",
                    devices:Union[str,List[int]]='auto',
                    precision:Union[str,int]=32,
                    modifier:Union[None,str]=None,
                    modifier_params:Union[None,dict]=None,
                    seed:int=42
                    )-> float:
        """Train the model

        Args:
            dirpath (str): path where to put all the useful things
            split_params (dict): see `split_for_train`
            batch_size (int, optional): batch size. Defaults to 100.
            num_workers (int, optional): num_workers for the dataloader. Defaults to 4.
            max_epochs (int, optional): maximum epochs to perform. Defaults to 500.
            auto_lr_find (bool, optional): find initial learning rate, see  `pytorch-lightening`. Defaults to True.
            gradient_clip_val (Union[float,None], optional): gradient_clip_val. Defaults to None. See https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
            gradient_clip_algorithm (str, optional): gradient_clip_algorithm. Defaults to 'norm '. See https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
            devices (Union[str,List[int]], optional): devices to use. Use auto if cpu or the list of gpu to use otherwise. Defaults to 'auto'.
            precision  (Union[str,int], optional): precision to use. Usually 32 bit is fine but for larger model you should try 'bf16'. If 'auto' it will use bf16 for GPU and 32 for cpu
            modifier (Union[str,int], optional): if not None a modifier is applyed to the dataloader. Sometimes lightening has very restrictive rules on the dataloader, or we want to use a ML model before or after the DL model (See readme for more information)
            modifier_params (Union[dict,int], optional): parameters of the modifier
            seed (int, optional): seed for reproducibility
        """

        beauty_string('Training the model','block',self.verbose)

        self.split_params = split_params
        self.check_custom = False
        train,validation,test = self.split_for_train(**self.split_params)
        accelerator = 'gpu' if torch.cuda.is_available() else "cpu"
        strategy = "auto"
        if accelerator == 'gpu':
            strategy = "auto" ##TODO in future investigate on this
            if precision=='auto':
                precision = 'bf16'
            #"bf16" ##in futuro magari inserirlo nei config, potrebbe essere che per alcuni modelli possa non andare bfloat32
            torch.set_float32_matmul_precision('medium')
            beauty_string('Setting multiplication precision to medium','info',self.verbose)
        else:
            devices = 'auto'
            if precision=='auto':
                precision  = 32
        beauty_string(f'train:{len(train)}, validation:{len(validation)}, test:{len(test) if test is not None else 0}','section',self.verbose)
        if (accelerator=='gpu') and (num_workers>0):
            persistent_workers = True
        else:
            persistent_workers = False
            
            
        if modifier is not None:
            modifier = eval(modifier)
            modifier = modifier(**modifier_params)
            train, validation = modifier.fit_transform(train=train,val=validation)
            self.modifier = modifier
        else:
            self.modifier = None
        train_dl = DataLoader(train, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers,persistent_workers=persistent_workers)
        valid_dl = DataLoader(validation, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers,persistent_workers=persistent_workers)
   
        train_dl = DataLoader(train, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers,persistent_workers=persistent_workers)
        valid_dl = DataLoader(validation, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers,persistent_workers=persistent_workers)
        checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                     monitor='val_loss',
                                      save_last = True,
                                      every_n_epochs =1,
                                      verbose = self.verbose,
                                      save_top_k = 1,
                                     filename='checkpoint')
        
        
        logger = CSVLogger("logs", name=dirpath)

        mc = MetricsCallback(dirpath)
        ## TODO se ci sono 2 o piu gpu MetricsCallback non funziona (secondo me fa una istanza per ogni dataparallel che lancia e poi non riesce a recuperare info)
        pl.seed_everything(seed, workers=True)
        trainer = pl.Trainer(default_root_dir=dirpath,
                             logger = logger,
                             max_epochs=max_epochs,
                             callbacks=[checkpoint_callback,mc],
                             auto_lr_find=auto_lr_find, 
                             accelerator=accelerator,
                             devices=devices,
                             strategy=strategy,
                             enable_progress_bar=False,
                             precision=precision,
                             gradient_clip_val=gradient_clip_val,
                             gradient_clip_algorithm=gradient_clip_algorithm)#,devices=1)

        if auto_lr_find:
            trainer.tune(self.model,train_dataloaders=train_dl,val_dataloaders = valid_dl)
            files = os.listdir(dirpath)
            for f in files:
                if '.lr_find' in f:
                    os.remove(os.path.join(dirpath,f))
 
        trainer.fit(self.model, train_dl,valid_dl)
        self.checkpoint_file_best = checkpoint_callback.best_model_path
        self.checkpoint_file_last = checkpoint_callback.last_model_path 
        if self.checkpoint_file_last=='':
            beauty_string('There is a bug on saving last model I will try to fix it','info',self.verbose)
            self.checkpoint_file_last = checkpoint_callback.best_model_path.replace('checkpoint','last')

        self.dirpath = dirpath
        
        self.losses = mc.metrics

        files = os.listdir(dirpath)
        ##accrocchio per multi gpu
        for f in files:
            if '__losses__.csv' in f:
                if len(self.losses['val_loss'])>0:
                    self.losses = pd.DataFrame(self.losses)
                else:
                    self.losses = pd.read_csv(os.path.join(os.path.join(dirpath,f)))
                os.remove(os.path.join(os.path.join(dirpath,f)))
        if isinstance(self.losses,dict):
            self.losses = pd.DataFrame()
        try:
            self.model = self.model.load_from_checkpoint(self.checkpoint_file_last)
        except Exception as _:
            beauty_string(f'There is a problem loading the weights on file {self.checkpoint_file_last}','section',self.verbose)

        try:
            val_loss = self.losses.val_loss.values[-1]
        except Exception as _:
            beauty_string('Can not extract the validation loss, maybe it is a persistent model','info',self.verbose)
            val_loss = 100
        self.is_trained = True
        
        beauty_string('END of the training process','block',self.verbose)
        return val_loss 
    
    def inference_on_set(self,batch_size:int=100,
                         num_workers:int=4,
                         split_params:Union[None,dict]=None,set:str='test',
                         rescaling:bool=True,
                         data:Union[None,torch.utils.data.Dataset]=None)->pd.DataFrame:
        """This function allows to get the prediction on a particular set (train, test or validation). 

        Args:
            batch_size (int, optional): barch sise. Defaults to 100.
            num_workers (int, optional): num workers. Defaults to 4.
            split_params (Union[None,dict], optional): if not None  the spliting procedure will use the given data otherwise it will use the same configuration used in train. Defaults to None.
            set (str, optional): trai, validation or test. Defaults to 'test'.
            rescaling (bool, optional):  If rescaling is true the output will be rescaled to the initial values. . Defaults to True.
            data (None or pd.DataFrame, optional). If not None the inference is performed on the given data. In the case of custom data please call inference because it will normalize the data for you!
        Returns:
            pd.DataFrame: the predicted values in a pandas format
        """
        
        beauty_string('Inference on a set (train, validation o test)','block',self.verbose)
     
        if data is None:
            if split_params is None:
                beauty_string(f'splitting using train parameters {self.split_params}','section',self.verbose)
                train,validation,test = self.split_for_train(**self.split_params)
            else:
                train,validation,test = self.split_for_train(**split_params)

        if set=='test':
            if self.modifier is not None:
                test = self.modifier.transform(test)
            dl = DataLoader(test, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)
        elif set=='validation':
            if self.modifier is not None:
                validation = self.modifier.transform(validation)
            dl = DataLoader(validation, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)
        elif set=='train':
            if self.modifier is not None:
                train = self.modifier.transform(train)
            dl = DataLoader(train, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)    
        elif set=='custom':
            if self.check_custom:
                pass
            else:
                beauty_string('If you are here something went wrong, please report it','section',self.verbose)
            if self.modifier is not None:
                data = self.modifier.transform(data)
            dl = DataLoader(data, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)    
  
        else:
            beauty_string('Select one of train, test, or validation set','section',self.verbose)
        self.model.eval()
        
        res = []
        real = []
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        beauty_string(f'Device used: {self.model.device}','info',self.verbose)

        for batch in dl:
            res.append(self.model.inference(batch).cpu().detach().numpy())
            real.append(batch['y'].cpu().detach().numpy())
       
        res = np.vstack(res)
 
        real = np.vstack(real)
        time = dl.dataset.t
        groups = dl.dataset.groups
        #import pdb
        #pdb.set_trace()
        if self.modifier is not None:
            res,real = self.modifier.inverse_transform(res,real)

        ## BxLxCx3
        if rescaling:
            beauty_string('Scaling back','info',self.verbose)
            if self.normalize_per_group is False:
                for i, c in enumerate(self.target_variables):
                    real[:,:,i] = self.scaler_num[c].inverse_transform(real[:,:,i].reshape(-1,1)).reshape(-1,real.shape[1])
                    for j in range(res.shape[3]):
                        res[:,:,i,j] = self.scaler_num[c].inverse_transform(res[:,:,i,j].reshape(-1,1)).reshape(-1,res.shape[1])
            else:
                for group in np.unique(groups):
                    idx = np.where(groups==group)[0]
                    for i, c in enumerate(self.target_variables):
                        real[idx,:,i] = self.scaler_num[f'{c}_{group}'].inverse_transform(real[idx,:,i].reshape(-1,1)).reshape(-1,real.shape[1])
                        for j in range(res.shape[3]):
                            res[idx,:,i,j] = self.scaler_num[f'{c}_{group}'].inverse_transform(res[idx,:,i,j].reshape(-1,1)).reshape(-1,res.shape[1])

        if self.model.use_quantiles:
            time = pd.DataFrame(time,columns=[i+1 for i in range(res.shape[1])])
            
            if self.group is not None:
                time[self.group] = groups
                time = time.melt(id_vars=['region'])
            else:
                time = time.melt()
            time.rename(columns={'value':'time','variable':'lag'},inplace=True)

                
            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(pd.DataFrame(real[:,:,i],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,0],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_low'}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,1],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_median'}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,2],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_high'}).drop(columns=['variable']))

            res = pd.concat(tot,axis=1)
        
            
        ## BxLxCx1
        else:
            time = pd.DataFrame(time,columns=[i+1 for i in range(res.shape[1])])#.melt()

            if self.group is not None:
                time[self.group] = groups
                time = time.melt(id_vars=['region'])
            else:
                time = time.melt()
            time.rename(columns={'value':'time','variable':'lag'},inplace=True)
                 

            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(pd.DataFrame(real[:,:,i],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,0],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_pred'}).drop(columns=['variable']))
            res = pd.concat(tot,axis=1)

            
        return res
    def inference(self,batch_size:int=100,
                  num_workers:int=4,
                  split_params:Union[None,dict]=None,
                  rescaling:bool=True,
                  data:pd.DataFrame=None,
                  steps_in_future:int=0,
                  check_holes_and_duplicates:bool=True)->pd.DataFrame:
        
        """similar to `inference_on_set`
        only change is split_params that must contain this keys but using the default can be sufficient:
        'past_steps','future_steps','shift','keep_entire_seq_while_shifting','starting_point'
        
        skip_step is set to 1 for convenience (generally you want all the predictions)
        You can set split_params to None and use the standard parameters (at your own risck)
   

        Args:
            batch_size (int, optional): see inference_on_set. Defaults to 100.
            num_workers (int, optional): inference_on_set. Defaults to 4.
            split_params (Union[None,dict], optional): inference_on_set. Defaults to None.
            rescaling (bool, optional): inference_on_set. Defaults to True.
            data (pd.DataFrame, optional): startin dataset. Defaults to None.
            steps_in_future (int, optional): if>0 the dataset is extendend in order to make predictions in the future. Defaults to 0.
            check_holes_and_duplicates (bool, optional): if False the routine does not check for holes or for duplicates, set to False for stacked model. Defaults to True.

        Returns:
            pd.DataFrame: predicted values
        """
        beauty_string('Inference on a custom dataset','block',self.verbose)
        
        self.check_custom = True ##this is a check for the dataset loading
        ## enlarge the dataset in order to have all the rows needed
        if check_holes_and_duplicates:
            if self.group is None:
                freq = pd.to_timedelta(np.diff(data.time).min())
                beauty_string(f'Detected minumum frequency: {freq}','section',self.verbose)
                empty = pd.DataFrame({'time':pd.date_range(data.time.min(),data.time.max()+freq*(steps_in_future+self.split_params['past_steps']+self.split_params['future_steps']),freq=freq)})

            else:
                freq = pd.to_timedelta(np.diff(data[data[self.group==data[self.group].unique()[0]]].time).min())
                beauty_string(f'Detected minumum frequency: {freq} supposing constant frequence inside the groups','section',self.verbose)
                _min = data.groupby(self.group).time.min()
                _max = data.groupby(self.group).time.max()
                empty = []
                for c in data[self.group].unique():
                    empty.append(pd.DataFrame({self.group:c,'time':pd.date_range(_min.time[_min[self.group]==c].values[0],_max.time[_max[self.group]==c].values[0]+freq*(steps_in_future+self.split_params['past_steps']+self.split_params['future_steps']),freq=freq)}))
                empty = pd.concat(empty,ignore_index=True)
            dataset = empty.merge(data,how='left')
        else:
            dataset = data.copy()
        
        
        if split_params is None:
            split_params = {}
            for c in self.split_params.keys():
                if c in ['past_steps','future_steps','shift','keep_entire_seq_while_shifting','starting_point']:
                    split_params[c] = self.split_params[c]
            split_params['skip_step']=1
            data = self.create_data_loader(dataset,**split_params)
        else:
            data = self.create_data_loader(data,**split_params)

        res = self.inference_on_set(batch_size=batch_size,num_workers=num_workers,split_params=None,set='custom',rescaling=rescaling,data=data)
        self.check_custom = False
        return res
        
    def save(self, filename:str)->None:
        """save the timeseries object

        Args:
            filename (str): name of the file
        """
        beauty_string('Saving','block',self.verbose)
        with open(f'{filename}.pkl','wb') as f:
            params =  self.__dict__.copy()
            for k in ['model']:
                if k in params.keys():
                    _ = params.pop(k)
            pickle.dump(params,f)


    def load(self,model:Base, filename:str,load_last:bool=True,dirpath:Union[str,None]=None,weight_path:Union[str, None]=None)->None:
        """ Load a saved model

        Args:
            model (Base): class of the model to load (it will be initiated by pytorch-lightening)
            filename (str): filename of the saved model
            load_last (bool, optional): if true the last checkpoint will be loaded otherwise the best (in the validation set). Defaults to True.
            dirpath (Union[str,None], optional): if None we asssume that the model is loaded from the same pc where it has been trained, otherwise we can pass the dirpath where all the stuff has been saved . Defaults to None.
            weight_path (Union[str, None], optional): if None the standard path will be used. Defaults to None.
        """

            
        
        beauty_string('Loading','block',self.verbose)
        self.modifier = None
        self.check_custom = False
        self.is_trained = True
        with open(filename+'.pkl','rb') as f:
            params = pickle.load(f)
            for p in params:
                setattr(self,p, params[p])    
        self.model = model(**self.config['model_configs'],optim_config = self.config['optim_config'],scheduler_config =self.config['scheduler_config'],verbose=self.verbose )
        
        
        if weight_path is not None:
            tmp_path = weight_path
        else:
            if self.dirpath is not None:
                directory = self.dirpath
            else:
                directory = dirpath
        
        
            if load_last:
                
                try:
                    tmp_path = os.path.join(directory,self.checkpoint_file_last.split('/')[-1])
                except Exception as _:
                    beauty_string('checkpoint_file_last not defined try to load best','section',self.verbose)
                    tmp_path = os.path.join(directory,self.checkpoint_file_best.split('/')[-1])
            else:
                try:
                    tmp_path = os.path.join(directory,self.checkpoint_file_best.split('/')[-1])
                except Exception as _:
                    beauty_string('checkpoint_file_best not defined try to load best','section',self.verbose)
                    tmp_path = os.path.join(directory,self.checkpoint_file_last.split('/')[-1])
        try:
            self.model = self.model.load_from_checkpoint(tmp_path,verbose=self.verbose)
        except Exception as e:
            beauty_string(f'There is a problem loading the weights on file {tmp_path} {e}',True)
