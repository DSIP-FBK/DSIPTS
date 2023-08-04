from dsipts import TimeSeries,read_public_dataset
from sklearn.cluster import BisectingKMeans
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from minigpt import GPT
from trainer import Trainer
import logging
logging.basicConfig(level=logging.INFO)
import argparse
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
import pickle
import numpy as np
class SortDataset(Dataset):
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self,x,y,length, num_digits):
        self.length = length
        self.num_digits = num_digits
        self.x = torch.tensor(x).long()
        self.y = torch.tensor(y).long()
    
    def __len__(self):
        return len(self.x) # ...
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        return self.length * 2 - 1

    def __getitem__(self, idx):
        
        inp = self.x[idx]
        sol = self.y[idx]
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y
    
@hydra.main(version_base=None,config_path=".")
def run(conf: DictConfig) -> None:
    data, columns = read_public_dataset(  dataset= 'weather', path= '/home/agobbi/Projects/ExpTS/data')
    ts = TimeSeries('prova')
    use_covariates = False
    ts.load_signal(data, enrich_cat=  ['hour'],target_variables=['y'], past_variables=columns if use_covariates else [])
    train,validation,test= ts.split_for_train(  perc_train= 0.8,  perc_valid= 0.1,  shift= 0,  skip_step=1,past_steps=16,future_steps=16)
    token_split = 4
    max_voc_size = 256
    samples,length,_ = train.data['y'].shape
    tmp = train.data['x_num_past'].squeeze().reshape(samples,-1,token_split)
    _,sentence_length, _ = tmp.shape
    tmp = tmp.reshape(-1,token_split)
    cl = BisectingKMeans(n_clusters=max_voc_size)
    clusters = cl.fit_predict(tmp)
    with open('cluster_model.pkl','wb') as f:
        pickle.dump(cl,f)
    x_train = clusters.reshape(-1,sentence_length)
    samples = train.data['y'].shape[0]
    y_train = cl.predict(train.data['y'].squeeze().reshape(samples,-1,token_split).reshape(-1,token_split)).reshape(-1,sentence_length)
    samples = validation.data['y'].shape[0]
    y_validation = cl.predict(validation.data['y'].squeeze().reshape(samples,-1,token_split).reshape(-1,token_split)).reshape(-1,sentence_length)
    x_validation = cl.predict(validation.data['x_num_past'].squeeze().reshape(samples,-1,token_split).reshape(-1,token_split)).reshape(-1,sentence_length)

    samples = test.data['y'].shape[0]
    y_test = cl.predict(test.data['y'].squeeze().reshape(samples,-1,token_split).reshape(-1,token_split)).reshape(-1,sentence_length)
    x_test = cl.predict(test.data['x_num_past'].squeeze().reshape(samples,-1,token_split).reshape(-1,token_split)).reshape(-1,sentence_length)


    train_dataset = SortDataset(x_train,y_train,sentence_length,max_voc_size)
    test_dataset = SortDataset(x_test,y_test,sentence_length,max_voc_size)
    validation_dataset = SortDataset(x_validation,y_validation,sentence_length,max_voc_size)

    x, y = test_dataset[0]
    for a, b in zip(x,y):
        logging.info(int(a),int(b))



    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = max_voc_size
    model_config.block_size = x_train.shape[1] +  y_train.shape[1] -1
    trans = GPT(model_config)
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 200000
    #optimizer = trans.configure_optimizers(train_config)

    trainer = Trainer(train_config, trans, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            logging.info(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

    trans.eval();

    def eval_split(trainer, split, max_batches):
        dataset = {'train':train_dataset, 'test':test_dataset,'validation':validation_dataset}[split]
        n = train_dataset.length # naugy direct access shrug
        results = []
        real = []
        predicted = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            y = y.to(trainer.device)

            # isolate the input pattern alone
            inp = x[:, :n]
            sol = y[:, -n:]
            # let the model sample the rest of the sequence
            cat = trans.generate(inp, n, do_sample=False) # using greedy argmax, not sampling

            sol_candidate = cat[:, n:] # isolate the filled in sequence
            # compare the predicted sequence to the true sequence
            correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 10: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    logging.info("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
            if max_batches is not None and b+1 >= max_batches:
                break
            real.append(sol.cpu())
            predicted.append(sol_candidate.cpu())
        rt = torch.tensor(results, dtype=torch.float)
        logging.info("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum(),np.vstack(real),np.vstack(predicted)

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        train_score,_,_ = eval_split(trainer, 'train', max_batches=50)
        test_score,_,_  = eval_split(trainer, 'validation',  max_batches=50)
        test_score,real,predicted  = eval_split(trainer, 'test',  max_batches=None)


    with open('res.pkl','wb') as f:
        pickle.dump([real,predicted],f)
        
if __name__ == '__main__': 
    try:
        run()
    except Exception as e:
        logging.info(e)