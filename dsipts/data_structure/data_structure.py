import numpy as np
from dataclasses import dataclass
import plotly.express as px
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from typing import Optional, Union
import os
import torch
pd.options.mode.chained_assignment = None 
import pickle
from .utils import extend_df,MetricsCallback, MyDataset, ActionEnum
from datetime import datetime
from ..models.base import Base
from ..models.utils import weight_init
import logging 
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
    
    def __init__(self,name:str):
        """Class for generating time series object. If you don't have any time series you can build one fake timeseries using some helping classes (Categorical for instance).


        Args:
            name (str): name of the series
                
                
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
        
        self.name = name

    def __str__(self) -> str:
        return f"Timeseries named {self.name} of length {self.dataset.shape[0]}.\n Categorical variable: {self.cat_var},\n Future variables: {self.future_variables},\n Past variables: {self.past_variables},\n Target variables: {self.target_variables}"
    def __repr__(self) -> str:
        return f"Timeseries named {self.name} of length {self.dataset.shape[0]}.\n Categorical variable: {self.cat_var},\n Future variables: {self.future_variables},\n Past variables: {self.past_variables},\n Target variables: {self.target_variables}"
    
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
            logging.error('Please implement your own method')
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
        
        
        
    def load_signal(self,data:pd.DataFrame,enrich_cat:List[str] = [],past_variables:List[str]=[],future_variables:List[str]=[],target_variables:List[str]=[],cat_var:List[str]=[],check_past:bool=True)->None:
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
        """
        
        
        
        dataset = data.copy()
        logging.info('################I will drop duplicates, I dont like them###################')
        dataset.drop_duplicates(subset=['time'],  keep='first', inplace=True, ignore_index=True)

        if dataset.time.diff()[1:].nunique()>1:
            logging.info("#########There are holes in the dataset i will try to extend the dataframe inserting NAN#############3")
            freq = pd.to_timedelta(np.diff(dataset.time).min())
            logging.info(f'#############Detected minumum frequency: {freq}#############')
            dataset = extend_df(dataset.time,freq).merge(dataset,how='left')
            

            
        assert len(target_variables)>0, f'Provide at least one column for target'
        assert 'time'  in dataset.columns, f'The temporal column must be called time'
        if set(target_variables).intersection(set(past_variables))!= set(target_variables): 
            if check_past:
                logging.info('##########I will update past column adding all target columns, if you want to avoid this beahviour please use check_pass as false############')
                past_variables = list(set(past_variables).union(set(target_variables)))
        
        self.cat_var = cat_var
        self.enrich_cat = enrich_cat
        for c in enrich_cat:
            self.cat_var = list(set(self.cat_var+[c]))
                
            if c in dataset.columns:
                logging.info('#########Categorical {c} already present, it will be added to categorical variable but not recomputed#########') 
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
                    logging.info('I can not automatically add column {c} plase update this function accordlyng')
        self.dataset = dataset
        self.past_variables =past_variables
        self.future_variables = future_variables
        self.target_variables = target_variables
        self.out_vars = len(target_variables)
        self.num_var = list(set(self.past_variables).union(set(self.future_variables)).union(set(self.target_variables)))
        
    def plot(self):
        """  
        Easy way to control the loaded data
        Returns:
            plotly.graph_objects._figure.Figure: figure of the target variables
        """
      
        logging.info('plotting only target variables')
        tmp = self.dataset[['time']+self.target_variables].melt(id_vars=['time'])
        fig = px.line(tmp,x='time',y='value',color='variable',title=self.name)
        fig.show()
        return fig
    
        
    def create_data_loader(self,data:pd.DataFrame,past_steps:int,future_steps:int,shift:int=0,keep_entire_seq_while_shifting:bool=False,starting_point:Union[None,dict]=None,skip_step:int=1)->MyDataset:
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

        
        x_num_past_samples = []
        x_num_future_samples = []
        x_cat_past_samples = []
        x_cat_future_samples = []
        y_samples = []
        t_samples = []
        
        for c in self.cat_var:
            data[c] = self.scaler_cat[c].transform(data[c].values.ravel()).flatten()
        for c in self.num_var: 
            data[c] = self.scaler_num[c].transform(data[c].values.reshape(-1,1)).flatten()
        
        idx_target = []
        for c in self.target_variables:
            idx_target.append(self.past_variables.index(c))
         
        
        x_num_past = data[self.past_variables].values
        if len(self.future_variables)>0:
            x_num_future = data[self.future_variables].values
        if len(self.cat_var)>0:
            x_cat = data[self.cat_var].values
        y_target = data[self.target_variables].values
        t = data.time.values

        ##questo serve a forzare di iniziare i samples alla stessa ora per esempio (controllo sul primo indice della y)
        if starting_point is not None:
            check = data[list(starting_point.keys())[0]].values == starting_point[list(starting_point.keys())[0]]
        else:
            check = [True]*len(y_target)
        
        for i in range(past_steps,data.shape[0]-future_steps,skip_step):
            if check[i]:

                if len(self.future_variables)>0:
                    if keep_entire_seq_while_shifting:
                        xx = x_num_future[i-shift:i+future_steps].mean()
                    else:
                        xx = x_num_future[i-shift:i+future_steps-shift].mean()
                else:
                    xx = 0.0
                if np.isfinite(x_num_past[i-past_steps:i].min() + y_target[i:i+future_steps].min() + xx):
                    
                    x_num_past_samples.append(x_num_past[i-past_steps:i])
                    if len(self.future_variables)>0:
                        if keep_entire_seq_while_shifting:
                            x_num_future_samples.append(x_num_future[i-shift:i+future_steps])
                        else:
                            x_num_future_samples.append(x_num_future[i-shift:i+future_steps-shift])
                    if len(self.cat_var)>0:
                        x_cat_past_samples.append(x_cat[i-past_steps:i])
                        if keep_entire_seq_while_shifting:
                            x_cat_future_samples.append(x_cat[i-shift:i+future_steps])
                        else:
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
        
        return MyDataset(dd,t_samples,idx_target)
    
          
    
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
                        skip_step:int=1
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

        Returns:
            List[DataLoader,DataLoader,DataLoader]: three dataloader used for training or inference
        """

        
        
        try:
            l = self.dataset.shape[0]
        except:
            logging.error('Empty dataset')
            return None, None, None
        

        if range_train is None:
            logging.info(f'Split temporally using perc_train: {perc_train} and perc_valid:{perc_valid}')
        
            train = self.dataset.iloc[0:int(perc_train*l)]
            validation = self.dataset.iloc[int(perc_train*l):int(perc_train*l+perc_valid*l)]
            test = self.dataset.iloc[int(perc_train*l+perc_valid*l):]
        else:
            logging.info('Split temporally using the time intervals provided')

            train = self.dataset[self.dataset.time.between(range_train[0],range_train[1])]
            validation =  self.dataset[self.dataset.time.between(range_validation[0],range_validation[1])]
            test =  self.dataset[self.dataset.time.between(range_test[0],range_test[1])]
                                      
        self.scaler_cat = {}
        self.scaler_num = {}
        logging.info('######################################################################################################')
        logging.info('######Scaling numerical (standard scaler) and categorical (label encorer) on the training data! ######')
        logging.info('######################################################################################################')

        for c in self.num_var:
            self.scaler_num[c] =  StandardScaler()
            self.scaler_num[c].fit(train[c].values.reshape(-1,1))
        for c in self.cat_var:                               
            self.scaler_cat [c] =  LabelEncoder()
            self.scaler_cat[c].fit(train[c].values.reshape(-1,1))  
                                      
    
        dl_train = self.create_data_loader(train,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        dl_validation = self.create_data_loader(validation,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)
        dl_test = self.create_data_loader(test,past_steps,future_steps,shift,keep_entire_seq_while_shifting,starting_point,skip_step)

        return dl_train,dl_validation,dl_test
            
    def set_model(self,model:Base,config:dict=None):
        """Set the model to train

        Args:
            model (Base): see `models`
            config (dict, optional): usually the configuration used by the model. Defaults to None.
        """
        self.model = model
        logging.info('######################################################################################################')
        logging.info('###########LOOK THE INIT PROCEDURE IF YOU NEED A CUSTOM INITIALIZATION of the weighjts################')
        logging.info('######################################################################################################')
        self.model.apply(weight_init)
        self.config = config
        
        logging.info('######################################################################################################')
        logging.info('######################################################MODEL###########################################')
        logging.info('######################################################################################################')
        logging.info(model)
              
    def train_model(self,dirpath:str,
                    split_params:dict,
                    batch_size:int=100,
                    num_workers:int=4,
                    max_epochs:int=500,
                    auto_lr_find:bool=True,
                    gradient_clip_val:Union[float,None]=None,
                    gradient_clip_algorithm:str="value",
                    devices:Union[str,List[int]]='auto',
                    precision:Union[str,int]=32)-> float:
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
        """

        logging.info('###############################################################################')
        logging.info('############################TRAINING###########################################')
        logging.info('###############################################################################')
        self.split_params = split_params
        train,validation,test = self.split_for_train(**self.split_params)
        accelerator = 'gpu' if torch.cuda.is_available() else "cpu"
        strategy = "auto"
        if accelerator == 'gpu':
            strategy = "auto" ##TODO in future investigate on this
            if precision=='auto':
                precision = 'bf16'
            #"bf16" ##in futuro magari inserirlo nei config, potrebbe essere che per alcuni modelli possa non andare bfloat32
            torch.set_float32_matmul_precision('medium')
            logging.info('setting multiplication precision to medium')
        else:
            devices = 'auto'
            if precision=='auto':
                precision  = 32
        logging.info(f'train:{len(train)}, validation:{len(validation)}, test:{len(test)}')
        if (accelerator=='gpu') and (num_workers>0):
            persistent_workers = True
        else:
            persistent_workers = False
            
        train_dl = DataLoader(train, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers,persistent_workers=persistent_workers)
        valid_dl = DataLoader(validation, batch_size = batch_size , shuffle=True,drop_last=True,num_workers=num_workers,persistent_workers=persistent_workers)
        checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                     monitor='val_loss',
                                      save_last = True,
                                      every_n_epochs =1,
                                      verbose = False,
                                      save_top_k = 1,
                                     filename='checkpoint')
        
        
        logger = CSVLogger("logs", name=dirpath)

        mc = MetricsCallback(dirpath)
        ## TODO se ci sono 2 o piu gpu MetricsCallback non funziona (secondo me fa una istanza per ogni dataparallel che lancia e poi non riesce a recuperare info)
        trainer = pl.Trainer(logger = logger,max_epochs=max_epochs,callbacks=[checkpoint_callback,mc],
                             auto_lr_find=auto_lr_find, accelerator=accelerator,devices=devices,strategy=strategy,
                             precision=precision,gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)#,devices=1)

        if auto_lr_find:
            trainer.tune(self.model,train_dataloaders=train_dl,val_dataloaders = valid_dl)
            
        ##clean lr finder
        files = os.listdir()
        for f in files:
            if '.lr_find' in f:
                os.remove(f)
 
        trainer.fit(self.model, train_dl,valid_dl)
        self.checkpoint_file_best = checkpoint_callback.best_model_path
        self.checkpoint_file_last = checkpoint_callback.last_model_path 
        if self.checkpoint_file_last=='':
            logging.info('There is a bug on saving last model I will try to fix it')
            self.checkpoint_file_last = checkpoint_callback.best_model_path.replace('checkpoint','last')

        self.dirpath = dirpath
        
        self.losses = mc.metrics

        files = os.listdir()
        ##accrocchio per multi gpu
        for f in files:
            if '__losses__.csv' in f:
                if len(self.losses['val_loss'])>0:
                    self.losses = pd.DataFrame(self.losses)
                else:
                    self.losses = pd.read_csv(f)
                os.remove(f)
        if type(self.losses)==dict:
            self.losses = pd.DataFrame()
        try:
            self.model = self.model.load_from_checkpoint(self.checkpoint_file_last)
        except:
            logging.info(f'There is a problem loading the weights on file {self.checkpoint_file_last}')


        
        return self.losses.val_loss.values[-1]
    
    def inference_on_set(self,batch_size:int=100,num_workers:int=4,split_params:Union[None,dict]=None,set:str='test',rescaling:bool=True)->pd.DataFrame:
        """This function allows to get the prediction on a particular set (train, test or validation). TODO add inference on a custom dataset

        Args:
            batch_size (int, optional): barch sise. Defaults to 100.
            num_workers (int, optional): num workers. Defaults to 4.
            split_params (Union[None,dict], optional): if not None  the spliting procedure will use the given data otherwise it will use the same configuration used in train. Defaults to None.
            set (str, optional): trai, validation or test. Defaults to 'test'.
            rescaling (bool, optional):  If rescaling is true the output will be rescaled to the initial values. . Defaults to True.

        Returns:
            pd.DataFrame: the predicted values in a pandas format
        """
     
        
        if split_params is None:
            logging.info(f'splitting using train parameters {self.split_params}')
            train,validation,test = self.split_for_train(**self.split_params)
        else:
            train,validation,test = self.split_for_train(**split_params)

        if set=='test':
            dl = DataLoader(test, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)
        elif set=='validation':
            dl = DataLoader(validation, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)
        elif set=='train':
            dl = DataLoader(train, batch_size = batch_size , shuffle=False,drop_last=False,num_workers=num_workers)    
        else:
            logging.error('select one of train, test, or validation set')
        self.model.eval()
        res = []
        real = []
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        logging.info(f'Device used: {self.model.device}')
        for batch in dl:
            res.append(self.model.inference(batch).cpu().detach().numpy())
            real.append(batch['y'].cpu().detach().numpy())
        res = np.vstack(res)
        real = np.vstack(real)
        time = dl.dataset.t

        ## BxLxCx3
        if rescaling:
            logging.info('Scaling back')
            for i, c in enumerate(self.target_variables):
                real[:,:,i] = self.scaler_num[c].inverse_transform(real[:,:,i].reshape(-1,1)).reshape(-1,real.shape[1])
                for j in range(res.shape[3]):
                    res[:,:,i,j] = self.scaler_num[c].inverse_transform(res[:,:,i,j].reshape(-1,1)).reshape(-1,res.shape[1])

        if self.model.use_quantiles:
            ##i+1 
            time = pd.DataFrame(time,columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':'time','variable':'lag'})
            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(pd.DataFrame(real[:,:,i],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,0],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_low'}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,1],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_median'}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,2],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_high'}).drop(columns=['variable']))

            res = pd.concat(tot,axis=1)
        
            
        ## BxLxCx1
        else:
            time = pd.DataFrame(time,columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':'time','variable':'lag'})
            tot = [time]
            for i, c in enumerate(self.target_variables):
                tot.append(pd.DataFrame(real[:,:,i],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c}).drop(columns=['variable']))
                tot.append(pd.DataFrame(res[:,:,i,0],columns=[i+1 for i in range(res.shape[1])]).melt().rename(columns={'value':c+'_pred'}).drop(columns=['variable']))
            res = pd.concat(tot,axis=1)

            
        return res
    def inference():
        ##TODO implement!
        pass
        
    def save(self, filename:str)->None:
        """save the timeseries object

        Args:
            filename (str): name of the file
        """
        logging.info('Saving')
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

            
        
        logging.info('################Loading#################################')
        with open(filename+'.pkl','rb') as f:
            params = pickle.load(f)
            for p in params:
                setattr(self,p, params[p])    
        self.model = model(**self.config['model_configs'],optim_config = self.config['optim_config'],scheduler_config =self.config['scheduler_config'] )
        
        if weight_path is not None:
            tmp_path = weight_path
        else:
            if self.dirpath is not None:
                if load_last:
                    tmp_path = os.path.join(self.dirpath,self.checkpoint_file_last.split('/')[-1])
                else:
                    tmp_path = os.path.join(self.dirpath,self.checkpoint_file_best.split('/')[-1])
            else:
                if load_last:
                    tmp_path = os.path.join(dirpath,self.checkpoint_file_last.split('/')[-1])
                else:
                    tmp_path = os.path.join(dirpath,self.checkpoint_file_best.split('/')[-1])
            
        try:
            self.model = self.model.load_from_checkpoint(tmp_path)
        except Exception as e:
            logging.info(f'There is a problem loading the weights on file {tmp_path} {e}')
