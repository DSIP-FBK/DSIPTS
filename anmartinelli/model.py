import torch
import torch.nn as nn
from embedding_nn import *
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
import pickle as pkl

class Model(nn.Module):
    def __init__(self, 
                 mix: bool, 
                 prec: bool,
                 n_cat_var,
                 n_num_var,
                 seq_len,
                 lag,
                 n_embd,
                 n_enc,
                 n_dec,
                 head_size,
                 num_heads,
                 fw_exp,
                 dropout,
                 num_lstm_layers,
                 device,
                 quantiles = None) -> None:
        super().__init__()
        self.path = ''
        self.mix = mix
        self.prec = prec
        self.lag = lag
        self.quantiles = quantiles
        # Start Embedding
        self.emb_cat_var = embedding_cat_variables(seq_len, lag, n_embd, device)
        self.emb_y_var = embedding_target(n_embd)
        # Encoder (past)
        self.EncVariableSelection = Encoder_Var_Selection(mix, seq_len, lag, n_cat_var, n_num_var, n_embd, dropout, device)
        self.EncLSTM = Encoder_LSTM(num_lstm_layers, n_embd, dropout, device)
        self.EncGRN = GRN(n_embd=n_embd, dropout=dropout)
        self.Encoder = Encoder(n_enc, n_embd, num_heads, head_size, fw_exp, dropout)
        # Decoder (future)
        self.DecVariableSelection = Decoder_Var_Selection(prec, seq_len, lag, n_cat_var, n_num_var, n_embd, dropout, device)
        self.DecLSTM = Decoder_LSTM(num_lstm_layers, n_embd, dropout, device)
        self.DecGRN = GRN(n_embd=n_embd, dropout=dropout)
        self.Decoder = Decoder(n_dec, n_embd, num_heads, head_size, fw_exp, lag, dropout)
        # PostTransformer (future)
        self.postTransformer = postTransformer(n_embd, dropout)
        # last linear wrt quantile
        if quantiles is not None:
            self.outLinear = nn.Linear(n_embd, len(quantiles))
        else:
            self.outLinear = nn.Linear(n_embd, 1)
        
    def forward(self, categorical, y):
        '''
        Input: 
        - categorical: torch.Size([bs, seq_len, 5]) # 5 number of cat_variables available from data
        - y(target): torch.Size([bs, seq_len])
        Categorical contains past and future, not separated
        Y (taget) both past and future. Future used only with prec=True as y[:,-self.lag-1:-1]
        No numerical variables, only the target one

        Output:
        - torch.Size([bs, seq_len, 1]) if quantiles=None, else len(self.quantiles)
        '''
        # categorical.shape = [bs, seq_len, 5]
        # y.shape = [bs, seq_len]
        #   - y will be unsqueezed the first time to embed each timestep in 'n_embd' dimension
        #   - The second one will be to cat y with other cat_variables on dim=2
        import pdb
        pdb.set_trace()

        #Emb cat var and then split for past and fut
        embed_categorical = self.emb_cat_var(categorical)
        embed_categorical_past = embed_categorical[:,:-lag,:,:]
        embed_categorical_fut = embed_categorical[:,-lag:,:,:]
        
        # Emb only y_past, y_fut will be emb possibly (if prec=True) before decoder
        y_past = y[:,:-self.lag]
        embed_y_past = self.emb_y_var(y_past.unsqueeze(dim=2))

        # mix==True:
        #   - Both categorical and y_past (embedded) go through VariableSelection
        #   - The 'mixture' will be Q, K and V of the Encoder
        # mix==False
        #   - Only the categorical variables will be passed in the VariableSelection
        #   - Embed y_past will be the Q of Encoder, while the selected categorical the K and V
        if self.mix:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past, embed_y_past.unsqueeze(2))
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(encoding, encoding, encoding)
        else:
            variable_selection_past = self.EncVariableSelection(embed_categorical_past)
            past_LSTM, hn, cn = self.EncLSTM(variable_selection_past)
            encoding = self.EncGRN(past_LSTM)
            encoded = self.Encoder(embed_y_past, encoding, encoding)
        
        # prec==True
        #   - Y_{t-1} is passed for future steps, but it need iterative forward
        #   - For each future step, we compute the output y reusing the last one computed
        # prec==False
        #   - No need of the iterative forward
        #   - Info about Y in future steps are not used
        if self.prec:
            ### Iterative Computation
            # Decoder_out stores y_0 and all predicted fututre values of y overwritten during for  
            decoder_out = y[:,-self.lag-1:].unsqueeze(2)
            for tau in range(1,self.lag+1):
                embed_tau_y = self.emb_y_var(decoder_out[:,:tau,:])
                variable_selection_fut = self.DecVariableSelection(embed_categorical_fut[:,:tau,:,:], embed_tau_y.unsqueeze(2))
                fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
                pred_decoding = self.DecGRN(fut_LSTM)
                pred_decoded = self.Decoder(pred_decoding, encoded, encoded)
                out = self.postTransformer(pred_decoded, fut_LSTM)
                out = self.outLinear(out)
                decoder_out[:,tau,:] = out[:,-1,:]
            # extract only predicted y without y_0
            out = decoder_out[:,1:,:]
        else:
            ### Direct Computation
            variable_selection_fut = self.DecVariableSelection(embed_categorical_fut)
            fut_LSTM = self.DecLSTM(variable_selection_fut, hn, cn)
            decoding = self.DecGRN(fut_LSTM)
            decoded = self.Decoder(decoding, encoded, encoded)
            out = self.postTransformer(decoded, fut_LSTM)
            out = self.outLinear(out)
        return out

    def learning(self, epochs, dl_training, dl_validation, cost_func, optimizer, scheduler):
        res = {'train':[],
               'val': []} # store train and validation_loss
        val_loss = float('inf') # initialized validation loss

        for iter in range(epochs): # starting epochs

            if (iter)%10==0: # validation update each 10 epochs
                #* EVALUATION 
                curr_val_loss = self.validation_step(dl_validation, cost_func)
                # update best model if the model reaches better performances
                if curr_val_loss < val_loss:
                    val_loss = curr_val_loss
                    best_model = torch.save(self.state_dict(), self.path + '_best.pt') #! PATH for BEST
                    print('  - IMPROVED')
                        
            #* TRAINING
            curr_train_loss = self.training_step(iter, dl_training, cost_func, optimizer, scheduler)

            res['train'].append(curr_train_loss)
            res['val'].append(curr_val_loss)

            with open(self.path + '.pkl', 'wb') as f: #! PATH for PKL
                pkl.dump(res, f)
                f.close()
            
            scheduler.step()
            #always update last_model
            last_model = torch.save(self.state_dict(), self.path + '_last.pt') #! PATH for LAST
        curr_val_loss = self.validation_step(dl_validation, cost_func)

    def training_step(self, ep, dl, cost_function, optimizer, scheduler):
        self.train()
        train_cumulative_loss = 0.

        for i, (ds, y) in enumerate(tqdm(dl, desc = f"> {ep} Train Step")):            
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
        
        scheduler.step()
        print(f' iter {iter+1}: train_loss: {train_cumulative_loss:4f}')
        self.eval()
        return train_cumulative_loss
    
    def validation_step(self, dl, cost_function):
        self.eval()
        val_cumulative_loss = 0.

        with torch.no_grad():
            for i, (ds, y) in enumerate(tqdm(dl, desc = "Validation Step")):
                y = y.to(self.device)
                ds = ds.to(self.device)
                output = self(ds, y)
                output = output.squeeze()
                y = y[:,-self.lag:].float()

                loss = cost_function(output,y)
                val_cumulative_loss += loss.item()

        val_cumulative_loss = val_cumulative_loss/(i+1)
        print(f' iter {iter}: val_loss: {val_cumulative_loss:4f}')
        return val_cumulative_loss
    
    def get_losses(self, path):
        with open(path, 'rb') as f:
            losses, _ = pkl.load(f)
        
            train_loss = losses['train']
            val_loss = losses['val']

            # both [argmin, min]
            min_val_loss = [val_loss.index(min(val_loss))+1, min(val_loss)]
            min_train_loss = [train_loss.index(min(train_loss))+1, min(train_loss)]

            total_epochs = len(train_loss)+1

        dict_loss = {
            'train': {
            train_loss,
            min_train_loss
            },
            'val': {
            val_loss,
            min_val_loss
            },
            'total_epochs': total_epochs
        }
        return dict_loss
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
    
    
    n_embd = 4
    n_enc = 2
    n_dec = 2
    head_size = 2
    num_heads = 2
    fw_exp = 3
    device = 'cpu'
    dropout = 0.1
    num_lstm_layers = 3
    mix = True
    prec = True
    n_cat_var = 8 # 6 from dataloader, but delete 'year' and add 'pos_seq','pos_fut','is_fut' (-1 +3)
    n_num_var = 1

    model = Model(mix,prec,n_cat_var,n_num_var,seq_len,lag,n_embd,n_enc,n_dec,head_size,num_heads,fw_exp,dropout,num_lstm_layers,device,quantiles = None)

    x, y = next(iter(train_dl))
    # x.shape = [8, 256, 6]
    # y.shape = [8, 256]
    categorical = x[:,:,1:]
    out = model(categorical, y)