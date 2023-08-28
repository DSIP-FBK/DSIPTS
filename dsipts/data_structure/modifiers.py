
from abc import  abstractmethod,ABC
from sklearn.cluster import BisectingKMeans
from scipy.stats import bootstrap
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import logging
from .utils import MyDataset

    
    
class VVADataset(Dataset):


    def __init__(self,x,y,y_orig,t,length_in,length_out, num_digits):
        self.length_in = length_in
        self.length_out = length_out
        self.num_digits = num_digits
        self.x_emb = torch.tensor(x).long()
        self.y_emb = torch.tensor(y).long()
        self.y = torch.tensor(y_orig)
        self.t = t

    def __len__(self):
        """
        
        :meta private:
        """
        return len(self.x_emb) # ...
    
    def get_vocab_size(self):
        """
        
        :meta private:
        """
        return self.num_digits
    
    def get_block_size(self):
        """
        
        :meta private:
        """
        return self.length * 2 - 1

    def __getitem__(self, idx):
        """
        :meta private:
        """

        inp = self.x_emb[idx]
        sol = self.y_emb[idx]
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length_out-1] = -1
        return {'x_emb':x, 'y_emb':y, 'y':self.y[idx]}


class Modifier(ABC):
    def __init__(self,**kwargs):
        """In the constructor you can store some parameters of the modifier. It will be saved when the timeseries is saved.
        """
        super(Modifier, self).__init__()
        self.__dict__.update(kwargs)
        
    @abstractmethod
    def fit_transform(self,train:MyDataset,val:MyDataset)->[Dataset,Dataset]:
        """This funtion is called before the training procedure and it should tasnform the standard Dataset into the new Dataset

        Args:
            train (MyDataset): initial train `Dataset`
            val (MyDataset): initial validation `Dataset`

        Returns:
            Dataset, Dataset: transformed train and validation `Datasets`
        """
        return train,val
    
    @abstractmethod
    def transform(self,test:MyDataset)->Dataset:
        """Similar to `fit_transform` but only transformation task will be performed, it is used in the inference function before calling the inference method
        Args:
            test (MyDataset): initial test `Dataset`

        Returns:
            Dataset: transformed test `Dataset`
        """
        return test
    
    @abstractmethod
    def inverse_transform(self,res:np.array,real:np.array)->[np.array,np.array]:
        """The results must be reverted respect to the prediction task

        Args:
            res (np.array): raw prediction
            real (np.array): raw real data

        Returns:
            [np.array, np.array] : inverse transfrmation of the predictions and the real data
        """
        return res


class ModifierVVA(Modifier):
    """This modifiers is used for the custom model VVA. The initial data are divided in smaller segments and then tokenized using a clustering procedure (fit_trasform).
    The centroids of the clusters are stored. A GPT model is then trained on the tokens an the predictions are reverted using the centroid information.
    """
   

    def fit_transform(self,train:MyDataset,val:MyDataset)->[Dataset,Dataset]:
        """BisectingKMeans is used on segments of length `token_split`

        Args:
            train (MyDataset): initial train `Dataset`
            val (MyDataset): initial validation `Dataset`

        Returns:
            Dataset, Dataset: transformed train and validation `Datasets`
        """
        idx_target =  train.idx_target
        assert len(idx_target)==1, print('This works only with single channel prediction')
        
        samples,length,_ = train.data['y'].shape
        tmp = train.data['x_num_past'][:,:,idx_target[0]].reshape(samples,-1,self.token_split)
        _,length_in, _ = tmp.shape
        length_out = length//self.token_split
        tmp = tmp.reshape(-1,self.token_split)
        cl = BisectingKMeans(n_clusters=self.max_voc_size)
        clusters = cl.fit_predict(tmp)
        self.cl = cl
        self.centroids = []
        cls, counts = np.unique(clusters,return_counts=True)
        logging.info(counts)
        
        for i in cls:
            res = []
            data = tmp[np.where(clusters==i)[0]]
            if len(data)>1:
                for j in range(data.shape[1]):
                    bootstrap_ci = bootstrap((data[:,j],), np.median,n_resamples=50, confidence_level=0.9,random_state=1, method='percentile')
                    res.append([bootstrap_ci.confidence_interval.low,np.median(data[:,j]),bootstrap_ci.confidence_interval.high])
                self.centroids.append(np.array(res))
            else:
                self.centroids.append(np.repeat(data.T,3,axis=1))
        
        self.centroids = np.array(self.centroids) ##clusters x length x 3 

        x_train = clusters.reshape(-1,length_in)
        samples = train.data['y'].shape[0]
        y_train = cl.predict(train.data['y'].squeeze().reshape(samples,-1,self.token_split).reshape(-1,self.token_split)).reshape(-1,length_out)
        samples = val.data['y'].shape[0]
        y_validation = cl.predict(val.data['y'].squeeze().reshape(samples,-1,self.token_split).reshape(-1,self.token_split)).reshape(-1,length_out)
        x_validation = cl.predict(val.data['x_num_past'][:,:,idx_target[0]].reshape(samples,-1,self.token_split).reshape(-1,self.token_split)).reshape(-1,length_in)
        train_dataset = VVADataset(x_train,y_train,train.data['y'].squeeze(),train.t,length_in,length_out,self.max_voc_size)
        validation_dataset = VVADataset(x_validation,y_validation,val.data['y'].squeeze(),val.t,length_in,length_out,self.max_voc_size)
        return train_dataset,validation_dataset
    
    
    
    def transform(self,test:MyDataset)->Dataset:
        """Similar to `fit_transform` but only transformation task will be performed
        Args:
            test (MyDataset): test val `Dataset`

        Returns:
            Dataset: transformed test `Dataset`
        """
    
        idx_target =  test.idx_target

        samples,length,_ = test.data['y'].shape
        tmp = test.data['x_num_past'][:,:,idx_target[0]].reshape(samples,-1,self.token_split)
        _,length_in, _ = tmp.shape
        length_out = length//self.token_split
        
        tmp = tmp.reshape(-1,self.token_split)
        clusters = self.cl.predict(tmp)
        x = clusters.reshape(-1,length_in)
        y = self.cl.predict(test.data['y'].squeeze().reshape(samples,-1,self.token_split).reshape(-1,self.token_split)).reshape(-1,length_out)
      
        return VVADataset(x,y,test.data['y'].squeeze(),test.t,length_in,length_out,self.max_voc_size)
    
    def inverse_transform(self,res:np.array,real:np.array)->[np.array,np.array]:
        """The results must be reverted respect to the prediction task

        Args:
            res (np.array): raw prediction

        Returns:
            np.array: inverse transofrmation of the predictions
        """
        tot = []
        for sample in res:
            tmp_sample = []
            for index in sample:
                tmp = []
                for i in index:
                    tmp.append(self.centroids[i])
                tmp = np.array(tmp)
                if tmp.shape[0]==1:
                    tmp2 = tmp[0,:,:]
                else:
                    tmp2 = tmp.mean(axis=0)
                    tmp2[:,0] -= 1.96*tmp.std(axis=0)[:,0]  #using confidence interval
                    tmp2[:,2] += 1.96*tmp.std(axis=0)[:,2]
                tmp_sample.append(tmp2)
            tot.append(np.vstack(tmp_sample))

        return np.expand_dims(np.stack(tot),2),np.expand_dims(real,2)
