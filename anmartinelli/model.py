import torch
import torch.nn as nn
from embedding_nn import *
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np
import sys

class Model(nn.Module):
    def __init__(self, 
                 use_target_past: bool, 
                 use_yprec: bool,
                 n_cat_var,
                 n_target_var,
                 seq_len,
                 lag,
                 d_model,
                 n_enc,
                 n_dec,
                 head_size,
                 num_heads,
                 fw_exp,
                 dropout,
                 num_lstm_layers,
                 device,
                 quantiles = None,
                 path_model_save = None,
                 model_name = None) -> None:
        super().__init__()
        self.path_model_save = path_model_save
        self.model_name = model_name
        self.device = device
        self.use_target_past = use_target_past
        self.use_yprec = use_yprec
        self.lag = lag
        self.quantiles = quantiles
        # Start Embedding
        emb_dims = [12+1, 31+1, 24, 7, 3]
        self.emb_cat_var = embedding_cat_variables(seq_len, lag, d_model, emb_dims, device)
        self.emb_y_var = embedding_target(d_model)
        # Encoder (past)
        self.EncVariableSelection = Encoder_Var_Selection(use_target_past, n_cat_var+3, n_target_var, d_model, dropout, device)
        self.EncLSTM = Encoder_LSTM(num_lstm_layers, d_model, dropout, device)
        self.EncGRN = GRN(d_model, dropout)
        self.Encoder = Encoder(n_enc, d_model, num_heads, head_size, fw_exp, dropout)
        # Decoder (future)
        self.DecVariableSelection = Decoder_Var_Selection(use_yprec, n_cat_var+3, n_target_var, d_model, dropout, device)
        self.DecLSTM = Decoder_LSTM(num_lstm_layers, d_model, dropout)
        self.DecGRN = GRN(d_model, dropout)
        self.Decoder = Decoder(n_dec, d_model, num_heads, head_size, fw_exp, lag, dropout)
        # PostTransformer (future)
        self.postTransformer = postTransformer(d_model, dropout)
        # last linear wrt quantile
        if quantiles is not None:
            self.outLinear = nn.Linear(d_model, len(quantiles))
        else:
            self.outLinear = nn.Linear(d_model, n_target_var)
        
    def forward(self, categorical, y):
        embed_categorical = self.emb_cat_var(categorical)
        embed_categorical_past = embed_categorical[:,:-self.lag,:,:]
        embed_categorical_fut = embed_categorical[:,-self.lag:,:,:]
        y_past = y[:,:-self.lag]
        embed_y_past = self.emb_y_var(y_past.unsqueeze(dim=2))

        if self.use_target_past:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past, embed_y_past.unsqueeze(2))
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(encoding, encoding, encoding)
        else:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(embed_y_past, encoding, encoding)
        
        if self.use_yprec:
            ### Iterative Computation
            # Decoder_out stores y_0 and all predicted fututre values of y overwritten during for  
            decoder_out = y[:,-self.lag-1:-1].unsqueeze(2)
            embed_y = self.emb_y_var(decoder_out)
            variable_selection_fut = self.DecVariableSelection(embed_categorical_fut, embed_y.unsqueeze(2))
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            pred_decoding = self.DecGRN(fut_LSTM)
            pred_decoded = self.Decoder(pred_decoding, encoded, encoded)
            out = self.postTransformer(pred_decoded, pred_decoding, fut_LSTM)
            out = self.outLinear(out)
        else:
            ### Direct Computation
            variable_selection_fut = self.DecVariableSelection(embed_categorical_fut)
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            decoding = self.DecGRN(fut_LSTM)
            decoded = self.Decoder(decoding, encoded, encoded)
            out = self.postTransformer(decoded, decoding, fut_LSTM)
            out = self.outLinear(out)
        return out

    def iter_forward(self, categorical, y):
        embed_categorical = self.emb_cat_var(categorical)
        embed_categorical_past = embed_categorical[:,:-self.lag,:,:]
        embed_categorical_fut = embed_categorical[:,-self.lag:,:,:]
        y_past = y[:,:-self.lag]
        embed_y_past = self.emb_y_var(y_past.unsqueeze(dim=2))

        if self.use_target_past:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past, embed_y_past.unsqueeze(2))
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(encoding, encoding, encoding)
        else:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(embed_y_past, encoding, encoding)
        
        ### Iterative Computation 
        decoder_out = y[:,-self.lag-1].unsqueeze(1).unsqueeze(2)
        for tau in range(1,self.lag+1):
            embed_tau_y = self.emb_y_var(decoder_out)
            variable_selection_fut = self.DecVariableSelection(embed_categorical_fut[:,:tau,:,:], embed_tau_y.unsqueeze(2))
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            pred_decoding = self.DecGRN(fut_LSTM)
            pred_decoded = self.Decoder(pred_decoding, encoded, encoded)
            out = self.postTransformer(pred_decoded, pred_decoding, fut_LSTM)
            out = self.outLinear(out)
            decoder_out = torch.cat((decoder_out, out[:,-1,:].unsqueeze(1)), dim = 1)
        # extract only predicted y without y_0
        out = decoder_out[:,1:,:]
        return out

    def learning(self, epochs, dl_training, dl_validation, cost_func, optimizer, scheduler):
        
        res = {'train':[],
               'val': []} # store train and validation_loss
        val_loss = float('inf') # initialized validation loss

        for iter in range(epochs): # starting epochs
            sys.stdout.flush()
            if (iter)%10==0: # validation update each 10 epochs
                #* EVALUATION 
                curr_val_loss = self.validation_step(iter, dl_validation, cost_func)
                # update best model if the model reaches better performances
                if curr_val_loss < val_loss:
                    val_loss = curr_val_loss
                    torch.save(self.state_dict(), self.path_model_save + '_best.pt')
                    print('  - IMPROVED')
            #* TRAINING
            if (iter%5)==3: 
                # every 5 epochs an iterative training step
                curr_train_loss = self.training_step(iter, dl_training, cost_func, optimizer, scheduler, if_iterative=True)
            else:
                curr_train_loss = self.training_step(iter, dl_training, cost_func, optimizer, scheduler, if_iterative=False)

            res['train'].append(curr_train_loss)
            res['val'].append(curr_val_loss)

            with open(self.path_model_save + '.pkl', 'rb') as f:
                dict, _ = pkl.load(f)
                f.close()
            with open(self.path_model_save + '.pkl', 'wb') as f:
                pkl.dump([dict, res], f)
                f.close()
            
            scheduler.step()
            #always update last_model
            torch.save(self.state_dict(), self.path_model_save + '_last.pt') 
        curr_val_loss = self.validation_step(iter, dl_validation, cost_func)
        print(f'Last Val_Loss: {curr_val_loss}')
        res['val'][-1] = curr_val_loss
        with open(self.path_model_save + '.pkl', 'wb') as f:
                pkl.dump([dict, res], f)
                f.close()

    def training_step(self, ep, dl, cost_function, optimizer, scheduler, if_iterative):
        self.train()
        train_cumulative_loss = 0.

        for i, (ds, y) in enumerate(tqdm(dl, desc = f"> {ep} Train Step")):            
            y = y.to(self.device)
            ds = ds.to(self.device)
            if if_iterative:
                output = self.iter_forward(ds[:,:,1:], y)
            else:
                output = self(ds[:,:,1:], y)
            output = output.squeeze()
            y = y[:,-self.lag:]

            loss = cost_function(output,y)
            train_cumulative_loss += loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_cumulative_loss = train_cumulative_loss/(i+1)
        
        scheduler.step()
        print(f' iter {ep} - train_loss: {train_cumulative_loss:4f}')
        self.eval()
        return train_cumulative_loss
    
    def validation_step(self, ep, dl, cost_function):
        self.eval()
        val_cumulative_loss = 0.

        with torch.no_grad():
            for i, (ds, y) in enumerate(tqdm(dl, desc = "Validation Step")):
                y = y.to(self.device)
                ds = ds.to(self.device)
                output = self.iter_forward(ds[:,:,1:], y)
                output = output.squeeze()
                y = y[:,-self.lag:].float()
                loss = cost_function(output,y)
                val_cumulative_loss += loss.item()

        val_cumulative_loss = val_cumulative_loss/(i+1)
        print(f' iter {ep} - val_loss: {val_cumulative_loss:4f}')
        return val_cumulative_loss
    
    def inference(self, dl, scaler):
        
        self.eval()
        sns.set(rc={'figure.facecolor':'lightgray'}) # 'axes.facecolor':'black',
        fig, axs = plt.subplots(3, 2, figsize=(18, 18))
        fig.suptitle(self.model_name, fontsize=25, fontweight='bold')

        # assign figure's slot
        plots = axs.flat

        loss_plot = plots[0]
        dict_loss = self.get_losses(self.path_model_save+'.pkl')
        total_epochs = dict_loss['total_epochs']
        x = torch.arange(1, dict_loss['total_epochs'])

        loss_plot.cla()
        loss_plot.set_yscale('log')
        loss_plot.plot(x, dict_loss['train'][0], label = 'train_loss')
        loss_plot.plot(x, dict_loss['val'][0], label = 'val_loss')
        loss_plot.grid(True)
        loss_plot.set_title(f'Epochs: {total_epochs},\n'\
                            f'TRAIN: -last: {dict_loss["train"][0][-1]:.03f} -argmin: {dict_loss["train"][1]:.4f} -min: {dict_loss["train"][2]:.4f}\n'\
                            f'VAL:   -last: {dict_loss["val"][0][-1]:.03f} -argmin: {dict_loss["val"][1]:.4f} -min: {dict_loss["val"][2]:.4f}')
        loss_plot.legend()

        #* RMSE AND TEST BATCH
        net_best = self
        net_best.eval()  
        net_last = copy.deepcopy(self)
        net_last.eval()

        path_best = self.path_model_save + '_best.pt'
        net_best.load_state_dict(torch.load(path_best, map_location=self.device))
        net_best.eval()

        path_last = self.path_model_save + '_last.pt'
        net_last.load_state_dict(torch.load(path_last, map_location=self.device))
        net_last.eval()
    
        y_data = []
        pred_best = []
        pred_last = []

        for i,(ds,y) in enumerate(tqdm(dl, desc = " > On Test DataLoader - ")):
            ds = ds.to(self.device)
            y = y.to(self.device)
            y_clone_best = y.detach().clone().to(self.device)
            y_clone_last = y.detach().clone().to(self.device)
            
            output_best = net_best.iter_forward(ds[:,:,1:], y_clone_best).detach()
            output_last = net_last.iter_forward(ds[:,:,1:], y_clone_last).detach()

            # RESCALE THE OUTPUTS TO STORE Y_DATA,PRED_BEST, PRED_LAST
            y = scaler.inverse_transform(y[:,-self.lag:].cpu()).flatten()
            prediction_best = scaler.inverse_transform(output_best[:,-self.lag:].squeeze(2).cpu()).flatten()
            prediction_last = scaler.inverse_transform(output_last[:,-self.lag:].squeeze(2).cpu()).flatten()

            y_data.append(y.reshape(-1,self.lag))
            pred_best.append(prediction_best.reshape(-1,self.lag))
            pred_last.append(prediction_last.reshape(-1,self.lag))
            
            del y,y_clone_best,y_clone_last,ds
            torch.cuda.empty_cache()
        
        assert len(y_data)==len(pred_best)
        assert len(y_data)==len(pred_last)
        
        y_data = np.vstack(y_data)
        pred_best = np.vstack(pred_best)
        pred_last = np.vstack(pred_last)
        #* -- RMSE --
        rmse_best = np.sqrt(np.mean((y_data-pred_best)**2, axis = 0))
        rmse_last = np.sqrt(np.mean((y_data-pred_last)**2, axis = 0))
        
        mean_rmse_best = np.mean(rmse_best)
        mean_rmse_last = np.mean(rmse_last)
        min_rmse_best = rmse_best.min()
        min_rmse_last = rmse_last.min()
        max_rmse_best = rmse_best.max()
        max_rmse_last = rmse_last.max()
        x=np.arange(1, self.lag+1)

        rmse_plot = plots[1]
        rmse_plot.cla()
        rmse_plot.plot(x, rmse_best, label = 'rmse_best')
        rmse_plot.plot(x, rmse_last, label = 'rmse_last')
        rmse_plot.grid(True)
        rmse_plot.set_title(f' BEST: min {min_rmse_best:.2f}, mean {mean_rmse_best:.2f}, max {max_rmse_best:.2f}\n \
                            LAST: min {min_rmse_last:.2f}, mean {mean_rmse_last:.2f}, max {max_rmse_last:.2f}')
        rmse_plot.legend()
        
        batch_plot = plots[2:4]
        shift_plot = plots[4:]
        
        for k, ax in enumerate(batch_plot):
            ax.cla()
            y_lim = [0, 6000]
            ax.plot(x, pred_best[k*-1], 'p' ,label = 'yhat_best')
            ax.plot(x, pred_last[k*-1], 'p', label = 'yhat_last')
            ax.plot(x, y_data[k*-1], label = ' y')
            ax.set_ylim(y_lim)
            ax.grid(True)
            ax.legend()

        shift_10, shift_30 = shift_plot

        shift_10.cla()
        y_lim = [0, 6000]
        shift_10.plot(pred_best[:,9], label = 'yhat_best_10')
        shift_10.plot(pred_last[:,9], label = 'yhat_last_10')
        shift_10.plot(y_data[:,9], label = ' y_10')
        shift_10.set_ylim(y_lim)
        shift_10.set_title('SHIFT 10 - ALL VALUES')
        shift_10.grid(True)
        shift_10.legend()

        shift_30.cla()
        y_lim = [0, 6000]
        shift_30.plot(pred_best[:,29], label = 'yhat_best_30')
        shift_30.plot(pred_last[:,29], label = 'yhat_last_30')
        shift_30.plot(y_data[:,29], label = ' y_30')
        shift_30.set_ylim(y_lim)
        shift_30.set_title('SHIFT 30 - ALL VALUES')
        shift_30.grid(True)
        shift_30.legend()

        fig.savefig(self.path_model_save + f'_plot_{total_epochs}.png') #! PATH

    def get_losses(self, path):
        with open(path, 'rb') as f:
            _, losses = pkl.load(f)
            f.close()
        # import pdb
        # pdb.set_trace()
        train_loss = losses['train']
        val_loss = losses['val']
        total_epochs = len(train_loss)+1

        dict_loss = {
            # [ list, argmin, min ]
            'train': [train_loss, train_loss.index(min(train_loss))+1, min(train_loss)],
            'val': [val_loss, val_loss.index(min(val_loss))+1, min(val_loss)],
            'total_epochs': total_epochs
        }
        return dict_loss
    
    def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__=='__main__':
    from dataloading import dataloading
    from sklearn.preprocessing import StandardScaler

    bs = 8
    bs_test = 4
    seq_len = 265
    lag = 65
    hour = 24
    hour_test = 24
    train = True
    step = 1
    scaler_type = StandardScaler()
    path_data = '/home/andrea/timeseries/data/edison/processed.pkl' 
    train_dl, _, _, _ = dataloading(batch_size=bs, batch_size_test=bs_test, 
                                                        seq_len=seq_len, lag=lag,
                                                        hour_learning=hour, 
                                                        hour_inference=hour_test, 
                                                        train_bool=train,
                                                        step = step,
                                                        scaler_y = scaler_type,
                                                        path=path_data)
    
    d_model = 4
    n_enc = 2
    n_dec = 2
    head_size = 2
    num_heads = 2
    fw_exp = 3
    device = 'cpu'
    dropout = 0.1
    num_lstm_layers = 3
    use_target_past = True
    use_yprec = True
    n_cat_var = 8 # 6 from dataloader, but delete 'year' and add 'pos_seq','pos_fut','is_fut' (-1 +3)
    n_target_var = 1

    model = Model(use_target_past,
                  use_target_past,
                  n_cat_var,
                  n_target_var,
                  seq_len,
                  lag,
                  d_model,
                  n_enc,
                  n_dec,
                  head_size,
                  num_heads,
                  fw_exp,
                  dropout,
                  num_lstm_layers,
                  device)

    x, y = next(iter(train_dl))
    # x.shape = [8, 256, 6]
    # y.shape = [8, 256]
    categorical = x[:,:,1:]
    out = model(categorical, y)
    # import pdb
    # pdb.set_trace()