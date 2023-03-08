import torch
import torch.nn as nn
from embed import Embed
from embed import Embed_tft
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np

class Full_Model(nn.Module):
    def __init__(self, mix, prec, tft, seq_len, lag, n_enc, n_dec, n_embd, num_heads, head_size, fw_exp, dropout, device) :
        super().__init__()
        self.mix = mix
        self.prec = prec
        self.tft = tft
        self.lag = lag
        self.save_str = self.get_save_str()
        self.device = device
        if tft:
            pass # Embed_tft(...)
        else:
            self.embed = Embed(mix, prec, tft, seq_len, lag, n_embd, device)
        self.encoder = Encoder(n_enc, n_embd, num_heads, head_size, fw_exp, dropout)
        self.decoder = Decoder(n_dec, n_embd, num_heads, head_size, fw_exp, lag, dropout)
        self.out_linear = nn.Linear(n_embd, 1)

    def forward(self, x, y):

        # if self.tft:
        #     pass # other way to compute emb_past and emb_future, depend also on self.mix
        # import pdb
        # pdb.set_trace()
        emb_x, emb_y_past, emb_y_fut, y_lin = self.embed(x, y)
        # emb_y_past => y[:,:-self.lag,:]
        # emb_y_fut => y[:,-self.lag-1:-1,:] SHIFT!
        emb_x_past = emb_x[:,:-self.lag,:,:]
        emb_x_fut = emb_x[:,-self.lag:,:,:]

        if self.mix:
            # MIX & NO Y_PREC (SelfAtt in Enc)
            emb_past = torch.sum(emb_x_past, dim=2, keepdim=False) + emb_y_past
            emb_future = torch.sum(emb_x_fut, dim=2, keepdim=False)
            if self.prec:
                emb_future = emb_future + emb_y_fut

            encoding = self.encoder(emb_past, emb_past, emb_past) # mixing past x and y
            decoding = self.decoder(emb_future, encoding, encoding)
            out = self.out_linear(decoding)
        else:
            # NO MIX & NO Y_PREC
            emb_past = torch.sum(emb_x_past, dim=2, keepdim=False)
            emb_future = torch.sum(emb_x_fut, dim=2, keepdim=False)
            if self.prec:
                emb_future = emb_future + emb_y_fut
            
            encoding = self.encoder(emb_y_past, emb_past, emb_past) # not mixing past x and y
            decoding = self.decoder(emb_future, encoding, encoding)
            out = self.out_linear(decoding)

        return out
    
    def learning(self, path_model, epochs, train_dl, val_dl, cost_function, optimizer, sched_step):

        res = {'train':[],
               'val': []} # store train and vali
        val_loss = float('inf') # initialized validation loss
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=0.3)

        for iter in range(epochs): # starting epochs

            #* EVALUATION 
            if (iter)%10==0: # validation update each 10 epochs

                self.eval()  
                val_cumulative_loss = 0.
                with torch.no_grad():
                    for i, (ds, y) in enumerate(tqdm(val_dl, desc = "val step")):
                        # import pdb
                        # pdb.set_trace()
                        y = y.to(self.device)
                        ds = ds.to(self.device)
                        output = self(ds, y)
                        
                        output = output.squeeze()
                        y = y[:,-self.lag:].float()

                        loss = cost_function(output,y)
                        val_cumulative_loss += loss.item()

                val_cumulative_loss = val_cumulative_loss/(i+1)
                print(f' iter {iter}: val_loss: {val_cumulative_loss:4f}')

                if val_cumulative_loss < val_loss:
                    val_loss = val_cumulative_loss
                    best_model = torch.save(self.state_dict(), path_model + self.save_str + '_best.pt') #! PATH
                    print('  - IMPROVED')

            #* TRAINING
            self.train()
            train_cumulative_loss = 0.
            for i, (ds, y) in enumerate(tqdm(train_dl, desc = "train step")):
                
                y = y.to(self.device)
                ds = ds.to(self.device)
                output = self(ds, y)
                
                output = output.squeeze()
                y = y[:,-self.lag:]

                loss = cost_function(output,y)
                train_cumulative_loss += loss.item()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_cumulative_loss = train_cumulative_loss/(i+1)
            print(f' iter {iter+1}: train_loss: {train_cumulative_loss:4f}')

            res['train'].append(train_cumulative_loss)
            res['val'].append(val_cumulative_loss)

            with open(path_model + self.save_str + '.pkl', 'wb') as f: #! PATH
                pkl.dump(res, f)
                f.close()
            
            scheduler.step()
            last_model = torch.save(self.state_dict(), path_model + self.save_str + '_last.pt') #! PATH

    def iter_forward(self, x, y):
        
        # if self.tft:
        #     pass # other way to compute emb_past and emb_future, depend also on self.mix

        emb_x, emb_y_past, emb_y_fut, y_lin = self.embed(x, y)
        # emb_y_past => y[:,:-self.lag,:]
        # emb_y_fut => y[:,-self.lag-1:-1,:] SHIFT!
        emb_x_past = emb_x[:,:-self.lag,:,:]
        emb_x_future = emb_x[:,-self.lag:,:,:]
        predictions = y.unsqueeze(2)[:,-self.lag-1:,:]

        emb_past = torch.sum(emb_x_past, dim=2, keepdim=False)
        if self.mix:
            emb_past = emb_past + emb_y_past
            for tau in range(1,self.lag+1):
                # import pdb
                # pdb.set_trace()
                future = torch.sum(emb_x_future[:,:tau,:,:], dim=2, keepdim=False) + y_lin(predictions[:,:tau,:].float())
                encoding = self.encoder(emb_past, emb_past, emb_past)
                decoding = self.decoder(future, encoding, encoding)
                out = self.out_linear(decoding)
                # import pdb
                # pdb.set_trace()
                predictions[:,tau,:] = out[:,-1,:]
        else:
            for tau in range(1,self.lag+1):
                future =  torch.sum(emb_x_future[:,:tau,:,:], dim=2, keepdim=False) + y_lin(predictions[:,:tau,:].float())
                encoding = self.encoder(emb_y_past, emb_past, emb_past)
                decoding = self.decoder(future, encoding, encoding)
                out = self.out_linear(decoding)
                predictions[:,tau,:] = out[:,-1,:]

        return predictions[:,1:]

    def inference(self, path_model, dl, scaler):

        sns.set(rc={'figure.facecolor':'lightgray'}) # 'axes.facecolor':'black',
        fig,axs = plt.subplots(3, 2, figsize=(18, 18))
        title = path_model.split('/')[-1]
        fig.suptitle(title, fontsize=25, fontweight='bold')

        # assign figure's slot
        plots = axs.flat
        loss_plot = plots[0]
        rmse_plot = plots[1]
        batch_plot = plots[2:]

        #* LOSS
        path_pkl = path_model + self.save_str + '.pkl' #! PATH
        with open(path_pkl, 'rb') as f:
            losses = pkl.load(f)
        
            train_loss = losses['train']
            val_loss = losses['val']
            # import pdb
            # pdb.set_trace()
            min_val_loss = [val_loss.index(min(val_loss))+1, min(val_loss)]
            actual_epochs = len(train_loss)+1
            x = np.arange(1, len(train_loss)+1)

            loss_plot.cla()
            loss_plot.set_yscale('log')
            loss_plot.plot(x, train_loss, label = 'train_loss')
            loss_plot.plot(x, val_loss, label = 'val_loss')
            loss_plot.grid(True)
            loss_plot.set_title(f'Epochs: {actual_epochs}, last: {val_loss[-1]}\n argmin: {min_val_loss[0]}, min: {min_val_loss[1]}')
            loss_plot.legend()
            f.close()

        #* RMSE AND TEST BATCH
        net_best = self
        net_best.eval()  
        net_last = copy.deepcopy(self)
        net_last.eval()

        path_best = path_model + self.save_str + '_best.pt' #! PATH
        net_best.load_state_dict(torch.load(path_best, map_location=self.device))
        net_best.eval()

        path_last = path_model + self.save_str + '_last.pt' #! PATH
        net_last.load_state_dict(torch.load(path_last, map_location=self.device))
        net_last.eval()
    
        y_data = []
        pred_best = []
        pred_last = []

        for i,(ds,y) in enumerate(tqdm(dl, desc = f" > On Test DataLoader - ")):
            ds = ds.to(self.device)
            y = y.to(self.device)
            y_clone_best = y.detach().clone().to(self.device)
            y_clone_last = y.detach().clone().to(self.device)
            
            if self.prec:
                output_best = net_best.iter_forward(ds,y_clone_best).detach().cpu().numpy()
                output_last = net_last.iter_forward(ds,y_clone_last).detach().cpu().numpy()
            else:
                output_best = net_best.forward(ds,y_clone_best).detach().cpu().numpy()
                output_last = net_last.forward(ds,y_clone_last).detach().cpu().numpy()

            # RESCALE THE OUTPUTS TO STORE Y_DATA,PRED_BEST, PRED_LAST
            y = scaler.inverse_transform(y[:,-self.lag:].cpu()).flatten()
            prediction_best = scaler.inverse_transform(output_best[:,-self.lag:].squeeze(2)).flatten()
            prediction_last = scaler.inverse_transform(output_last[:,-self.lag:].squeeze(2)).flatten()

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

        rmse_plot.cla()
        rmse_plot.plot(x, rmse_best, label = 'rmse_best')
        rmse_plot.plot(x, rmse_last, label = 'rmse_last')
        rmse_plot.grid(True)
        rmse_plot.set_title(f' BEST: min {min_rmse_best:.2f}, mean {mean_rmse_best:.2f}, max {max_rmse_best:.2f}\n LAST: min {min_rmse_last:.2f}, mean {mean_rmse_last:.2f}, max {max_rmse_last:.2f}')
        rmse_plot.legend()
            
        for k, ax in enumerate(batch_plot):
            ax.cla()
            y_lim = [0, 6000]
            ax.plot(x, pred_best[k], 'p' ,label = 'yhat_best')
            ax.plot(x, pred_last[k], 'p', label = 'yhat_last')
            ax.plot(x, y_data[k], label = ' y')
            ax.set_ylim(y_lim)
            ax.grid(True)
            ax.legend()

        fig.savefig(path_model + self.save_str + f'_plot_{actual_epochs}.png') #! PATH

    def get_save_str(self):
        
        if self.mix:
            if self.prec:
                if self.tft:
                    path = '_Mix_Prec_Tft'
                else:
                    path = '_Mix_Prec_NoTft'
            else:
                if self.tft:
                    path = '_Mix_NoPrec_Tft'
                else:
                    path = '_Mix_NoPrec_NoTft'
        else:
            if self.prec:
                if self.tft:
                    path = '_NoMix_Prec_Tft'
                else:
                    path = '_NoMix_Prec_NoTft'
            else:
                if self.tft:
                    path = '_NoMix_NoPrec_Tft'
                else:
                    path = '_NoMix_NoPrec_NoTft'

        return path
            
        