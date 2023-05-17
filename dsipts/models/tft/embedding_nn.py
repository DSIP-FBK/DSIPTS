import torch.nn as nn
import torch

class embedding_cat_variables(nn.Module):
    # at the moment cat_past and cat_fut together
    def __init__(self, seq_len: int, future_steps: int, d_model: int, emb_dims: list):
        """Class for embedding categorical variables, adding 3 positional variables during forward.
        
        Args:
            seq_len (int): length of the sequence (sum of past and future steps)
            future_steps (int): number of future step to be predicted
            d_model (int): dimension of all variables after they are embedded
            emb_dims (list): size of the dictionary for embedding. One dimension for each categorical variable
        """
        super().__init__()
        self.seq_len = seq_len
        self.future_steps = future_steps # past_steps = seq_len-future_steps
        self.updated_cat_embed_dims = emb_dims + [seq_len, future_steps+1, 2] # add embedding dimensions for variables added during forward
        self.embed_layers_to_d_model = nn.ModuleList([
            nn.Embedding(emb_dim, d_model) for emb_dim in self.updated_cat_embed_dims # list of Embedding layer for each variable
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Must be applied to both past and future variables: a concatenation is needed!
        To x's components, 3 new variables are added; in the order:
        - pos_seq: assign at each step its time-position
        - pos_fut: assign at each step its future position. 0 if it is a past step, 1-self.seq_len for future.
        - is_fut: explicit for each step if it is a future(1) or past(0)

        Args:
            x (torch.Tensor): [bs, seq_len, num_vars]

        Returns:
            torch.Tensor: [bs, seq_len, num_vars+3, d_model] 
        """
        # fetch device and batch size from x
        device = x.device.type
        B = x.shape[0]

        # create the 3 new variables
        pos_seq = self.get_pos_seq(bs=B).to(device)
        pos_fut = self.get_pos_fut(bs=B).to(device)
        is_fut = self.get_is_fut(bs=B).to(device)

        # concat everything
        cat_vars = torch.cat((x, pos_seq, pos_fut, is_fut), dim=2)

        # embed each variable to the d_model dimension
        cat_n_embd = self.get_cat_n_embd(cat_vars)
        return cat_n_embd

    def get_pos_seq(self, bs: int) -> torch.Tensor:
        # [0, 1,....,seq_len-1] repeated bs times
        pos_seq = torch.arange(0, self.seq_len)
        pos_seq = pos_seq.repeat(bs,1).unsqueeze(2)
        return pos_seq
    
    def get_pos_fut(self, bs: int) -> torch.Tensor:
        # [0,0,..,0,1,...lag], where 0 is repeated past_steps-times
        pos_fut = torch.cat((torch.zeros((self.seq_len-self.future_steps), dtype=torch.long),torch.arange(1,self.future_steps+1)))
        pos_fut = pos_fut.repeat(bs,1).unsqueeze(2)
        return pos_fut
    
    def get_is_fut(self, bs: int) -> torch.Tensor:
        # [0,0,...,0,1,...,1] with #past_steps 0s and #future_steps 1s
        is_fut = torch.cat((torch.zeros((self.seq_len-self.future_steps), dtype=torch.long),torch.ones((self.future_steps), dtype=torch.long)))
        is_fut = is_fut.repeat(bs,1).unsqueeze(2)
        return is_fut
    
    def get_cat_n_embd(self, cat_vars: torch.Tensor) -> torch.Tensor:
        device = cat_vars.device.type

        cat_n_embd = torch.Tensor().to(device)
        for index, layer in enumerate(self.embed_layers_to_d_model):
            # embed each variable to d_model dimension and concat 
            emb = layer(cat_vars[:, :, index])
            cat_n_embd = torch.cat((cat_n_embd, emb.unsqueeze(2)),dim=2)
        return cat_n_embd
    
class embedding_num_past_variables(nn.Module):
    def __init__(self, channels:int, d_model: int):
        """Class to embed past numerical variables.
        Only past, do not concat with future 

        Args:
            channels (int): number of numerical past variables to take in consideration
            d_model (int): model dimension
        """
        super().__init__()
        self.past_num_linears = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(channels) # list of linear layers
        ])

    def forward(self, num_past_tensor: torch.Tensor) -> torch.Tensor:
        """Class to embed past numerical variables.

        Args:
            num_past_tensor (torch.Tensor): numerical past variables just fetchd from batch

        Returns:
            torch.Tensor: embedded numerical past variables
        """
        device = num_past_tensor.device.type
        # use embed_vars to store numerical variables sent to d_model dimension
        embed_vars = torch.Tensor().to(device)
        for index, layer in enumerate(self.past_num_linears):
            emb = layer(num_past_tensor[:, :, index].unsqueeze(2))
            embed_vars = torch.cat((embed_vars, emb.unsqueeze(2)),dim=2)
        return embed_vars

class embedding_num_future_variables(nn.Module):
    def __init__(self, max_steps: int, channels:int, d_model: int):
        """ Embedding the target varible. (Only one)
        'channels' now not used, but ready to extend the model to predict more variables

        Args:
            max_steps (int): total number of future_steps
            channels (int): number of target variables we want to predict
            d_model (int): model dimension
        """
        super().__init__()
        # 'channels' now not used, but ready to extend the model to predict more variables
        self.max_steps = max_steps
        self.fut_num_linears = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(max_steps)
        ])
        self.emb_pos_layer = nn.Embedding(max_steps, d_model)

    def forward(self, num_fut_tensor: torch.Tensor) -> torch.Tensor:
        """Embedding the target varible. (Only one)

        Args:
            num_fut_tensor (torch.Tensor): future steps of scaled target variable

        Returns:
            torch.Tensor: embedded numerical future variables
        """
        # fetch device, batch_size and actual length of the size from num_fut_tensor
        device = num_fut_tensor.device.type
        B, L = num_fut_tensor.shape[0], num_fut_tensor.shape[1]

        # create and embed the tensor of time positions  
        pos_seq = self.get_pos_seq(B, L).to(device)
        emb_pos_seq = self.emb_pos_layer(pos_seq)

        # embed the future variables
        embedded_num_past_vars = self.get_num_fut_embedded(num_fut_tensor).to(device)

        # concat with time positions and return
        embedded_num_past_vars = torch.cat((embedded_num_past_vars, emb_pos_seq), dim=2)
        return embedded_num_past_vars
    
    def get_pos_seq(self, bs: int, length: int) -> torch.Tensor:
        pos_seq = torch.arange(0, length)
        pos_seq = pos_seq.repeat(bs,1).unsqueeze(2)
        return pos_seq

    def get_num_fut_embedded(self, num_fut_vars: torch.Tensor) -> torch.Tensor:
        device = num_fut_vars.device.type
        embed_vars = torch.Tensor().to(device)
        # at each iteration use the first L 
        L = num_fut_vars.shape[1] # get_the number of steps, number of channels will be always the same
        for index in range(L):
            emb = self.fut_num_linears[index](num_fut_vars[:,index,:]).unsqueeze(1)
            embed_vars = torch.cat((embed_vars, emb.unsqueeze(2)), dim=1)
        return embed_vars
    
class GLU(nn.Module):
    def __init__(self, d_model: int):
        """Gated Linear Unit, 'Gate' block in TFT paper 
        Sub net of GRN: linear(x) * sigmoid(linear(x))
        No dimension changes

        Args:
            d_model (int): model dimension
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated Linear Unit
        Sub net of GRN: linear(x) * sigmoid(linear(x))
        No dimension changes: [bs, seq_len, d_model]

        Args:
            x (torch.Tensor)

        Returns:
            torch.Tensor
        """
        x1 = self.sigmoid(self.linear1(x))
        x2 = self.linear2(x)
        out = x1*x2 #element-wise multiplication
        return out
    
class GRN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        """Gated Residual Network
        Used alone or as subnet of VariableSelection
        Norm(x + GLU(dropout( linear(ELU(linear)) )) )

        Args:
            d_model (int): model dimension
            dropout (float)
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model) 
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated Residual Network
        Used alone or as subnet of VariableSelection
        Norm(x + GLU(dropout( linear(ELU(linear)) )) )
        No dimension changes: [bs, seq_len, d_model]

        Args:
            x (torch.Tensor)

        Returns:
            torch.Tensor
        """
        eta1 = self.elu(self.linear1(x))
        eta2 = self.dropout(self.linear2(eta1))
        out = self.norm(x + self.glu(eta2))
        return out

class flatten_GRN(nn.Module):
    def __init__(self, d_model: int, num_var: int, dropout: float):
        """Modified GRN for flattened variables
        We start from the starting dimension (emb_dims[0]) and gradually switch to mid dimension (emb_dims[1]).
        Ending with end dimension (emb_dims[2]), which will be the total number of variables to be selected.
        The aim of those different dimensions is to avoid a bottleneck effect passing from 'd_model' to 'tot_num_var'.
        The first one can be also 100 times larger than the latter one, so we introduce an intermidiate dimension (usually mean of the two).
        Norm(x + GLU(dropout( linear(ELU(linear)) )) )

        Args:
            d_model (int): model dimension
            dropout (float): -
        """
        super().__init__()

        self.res_conn = nn.Linear(d_model, 1, bias = False)
        self.dropout_res_conn = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_model//2, bias = False) 
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(d_model//2, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(1)
        self.norm = nn.LayerNorm(num_var)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modified GRN for flattened variables to get Variable Selection Weights
        Norm(x + GLU(dropout( linear(ELU(linear)) )) )

        Args:
            x (torch.Tensor): [bs, seq_len, emb_dims[0]]

        Returns:
            torch.Tensor: [bs, seq_len, emb_dims[-1]]
        """
        import pdb 
        pdb.set_trace()
        res_conn = self.dropout_res_conn(self.res_conn(x).squeeze(3))
        eta1 = self.elu(self.linear1(x))
        eta2 = self.dropout(self.linear2(eta1))
        res_conn += self.glu(eta2).squeeze(3)
        out = self.norm(res_conn)
        out = self.softmax(out)
        return out

class Encoder_Var_Selection(nn.Module): # input already embedded
    def __init__(self, use_target_past: bool, n_past_cat_var: int, n_past_num_var: int, d_model: int, dropout: float):
        """Variable Selection Network in Encoder(past)
        Apply GRN to each variable, compute selection weights
        Element-wise multiplication to have a Tensor [bs, seq_len, d_model] encoding the single variable.

        Args:
            use_target_past (bool): True if we want to use the past target variable mixing it to past variables, False to use only past vars
            n_past_cat_var (int): number of categorical variables for past steps
            n_past_num_var (int): number of target variables for past steps. If use_target_past==False it is ignored
            d_model (int): model dimension
            dropout (float): -
        """
        super().__init__()
        self.use_target_past = use_target_past

        #categorical
        self.n_grn_cat = n_past_cat_var
        self.GRNs_cat = nn.ModuleList([
            GRN(d_model, dropout) for _ in range(self.n_grn_cat)
        ])
        tot_var = n_past_cat_var

        # numerical
        # if use_target_past==True, we apply selection on more varibles and we have to enlarge the number of GRNs
        if use_target_past:
            self.n_grn_num = n_past_num_var
            self.GRNs_num = nn.ModuleList([
                GRN(d_model, dropout) for _ in range(self.n_grn_num)
            ])
            tot_var = tot_var + n_past_num_var

        #flatten
        # flat_emb_dims = [d_model*tot_var, int(((d_model+1)*tot_var)/2), tot_var]
        
        self.flatten_GRN = flatten_GRN(d_model, tot_var, dropout)

    def forward(self, categorical: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        """Variable Selection Network in Encoder(past)
        If y is not None, we want to apply selection also to numerical(y) variables

        Args:
            categorical (torch.Tensor): [bs, past_steps, n_cat_var, d_model] past_cat_variables to be selected
            y (torch.Tensor, optional): [bs, past_steps, n_num_var, d_model]. Defaults to None. past_num_variables to be selected

        Returns:
            torch.Tensor: [bs, past_steps, d_model]
        """

        # categorical GRNs
        var_sel = self.get_cat_GRN(categorical)
        to_be_flat = categorical
        
        # numerical GRNs
        # computed and concatenated to categorical ones
        if y is not None:
            num_var_sel = self.get_num_GRN(y)
            # concat over second dimension parallelizing everything
            var_sel = torch.cat((var_sel, num_var_sel), dim = 2)
            to_be_flat = torch.cat((to_be_flat, y), dim=2)

        # GRN for flattened variables
        var_sel_wei = self.get_flat_GRN(to_be_flat)

        # element-wise multiplication
        out = var_sel*var_sel_wei.unsqueeze(3)
        # obtaining [bs, past_steps, d_model] by mean over the second dimension
        out = torch.sum(out, dim = 2)/out.shape[2]
        return out

    def get_cat_GRN(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device.type
        # cat_after_GRN to store variables in parallel over the second dimension
        cat_after_GRN = torch.Tensor().to(device)
        for index, layer in enumerate(self.GRNs_cat):
            grn = layer(x[:,:,index,:])
            cat_after_GRN = torch.cat((cat_after_GRN, grn.unsqueeze(2)), dim=2)
        return cat_after_GRN
    
    def get_num_GRN(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device.type
        # num_after_GRN to store variables in parallel over the second dimension
        num_after_GRN = torch.Tensor().to(device)
        for index, layer in enumerate(self.GRNs_num):
            grn = layer(x[:,:,index,:])
            num_after_GRN = torch.cat((num_after_GRN, grn.unsqueeze(2)), dim=2)
        return num_after_GRN
    
    def get_flat_GRN(self, to_be_flat: torch.Tensor) -> torch.Tensor:
        var_sel_wei = self.flatten_GRN(to_be_flat)
        return var_sel_wei

class Encoder_LSTM(nn.Module):
    def __init__(self, n_layers_LSTM: int, d_model: int, dropout: float):
        """LSTM Encoder with GLU, Add and Norm
        norm( x + GLU(dropout( LSTM(x) )) )

        Args:
            n_layers_EncLSTM (int): number of layers involved by LSTM 
            d_model (int): model dimension
            dropout (float): -
        """
        super().__init__()
        self.n_layers_EncLSTM = n_layers_LSTM
        self.hidden_size = d_model
        self.LSTM = nn.LSTM(input_size=d_model, hidden_size=self.hidden_size, num_layers=self.n_layers_EncLSTM, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.LSTM_enc_GLU = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> list:
        """LSTM Encoder with GLU, Add and Norm

        Args:
            x (torch.Tensor): [bs, past_steps, d_model]

        Returns:
            list of tensors: [output_enc, hn, cn] where hn and cn must be used for Decoder_LSTM. 
        """
        # init and move to device h0 and c0 of RNN 
        device = x.device.type
        h0 = torch.zeros(self.n_layers_EncLSTM, x.size(0), x.size(2)).to(device)
        c0 = torch.zeros(self.n_layers_EncLSTM, x.size(0), x.size(2)).to(device)

        # computations
        lstm_enc, (hn, cn) = self.LSTM(x, (h0,c0))
        lstm_enc = self.dropout(lstm_enc)
        output_enc = self.norm(self.LSTM_enc_GLU(lstm_enc) + x)
        return [output_enc, hn, cn]
    
class Decoder_Var_Selection(nn.Module): # input already embedded
    def __init__(self, use_yprec: bool, n_fut_cat_var: int, n_fut_num_var: int, d_model: int, dropout: float):
        """Variable Selection Network in Decoder(future)
        Apply GRN to each variable, compute selection weights
        Element-wise multiplication to have a Tensor [bs, seq_len, d_model] encoding the single variable.

        Args:
            use_yprec (bool): True if we want to use the last predicted values of target variable(s)
            n_fut_cat_var (int): number of categorical variables for future steps
            n_fut_tar_var (int): number of target variables for future steps. If use_yprec==False it is ignored
            d_model (int): model dimension
            dropout (float): -
        """
        super().__init__()
        self.use_yprec = use_yprec

        #categorical
        self.n_grn_cat = n_fut_cat_var
        self.GRNs_cat = nn.ModuleList([
            GRN(d_model, dropout) for _ in range(self.n_grn_cat)
        ])
        tot_var = n_fut_cat_var

        #numerical
        # if use_yprec==True, we apply selection on more varibles and we have to enlarge the number of GRNs
        if use_yprec:
            self.n_grn_num = n_fut_num_var
            self.GRNs_num = nn.ModuleList([
                GRN(d_model, dropout) for _ in range(self.n_grn_num)
            ])
            tot_var = tot_var + n_fut_num_var

        #flatten
        flat_emb_dims = [d_model*tot_var, int(((d_model+1)*tot_var)/2), tot_var]
        self.flatten_GRN = flatten_GRN(flat_emb_dims, dropout)

    def forward(self, categorical: torch.Tensor, y: torch.Tensor=None) -> torch.Tensor:
        """Variable Selection Network in Decoder(future)

        Args:
            categorical (torch.Tensor): [bs, fut_steps, n_cat_var, d_model] fut_cat_var to be selected
            y (torch.Tensor, optional): [bs, fut_steps, n_num_var, d_model] fut_num_var to be selected if the process is iterative. Defaults to None.

        Returns:
            torch.Tensor: [bs, fut_steps, d_model]
        """
        # categorical GRNs
        var_sel = self.get_cat_GRN(categorical)
        to_be_flat = categorical

        # numerical GRNs
        # computed and concatenated to categorical ones
        if y is not None:
            num_after_GRN = self.get_num_GRN(y)
            # concat over second dimension parallelizing everything
            var_sel = torch.cat((var_sel, num_after_GRN), dim = 2)
            to_be_flat = torch.cat((to_be_flat, y), dim=2)

        # GRN for flattened variables
        var_sel_wei = self.get_flat_GRN(to_be_flat)
        
        # element-wise multiplication
        out = var_sel*var_sel_wei.unsqueeze(3)
        # obtaining [bs, fut_steps, d_model] by mean over the second dimension
        out = torch.sum(out, 2)/out.size(2)
        return out

    def get_cat_GRN(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device.type
        # cat_after_GRN to store variables in parallel over the second dimension
        cat_after_GRN = torch.Tensor().to(device)
        for index, layer in enumerate(self.GRNs_cat):
            grn = layer(x[:,:,index,:])
            cat_after_GRN = torch.cat((cat_after_GRN, grn.unsqueeze(2)), dim=2)
        return cat_after_GRN
    
    def get_num_GRN(self, x: torch.Tensor) -> torch.Tensor:
        # num_after_GRN to store variables in parallel over the second dimension
        device = x.device.type
        num_after_GRN = torch.Tensor().to(device)
        for index, layer in enumerate(self.GRNs_num):
            grn = layer(x[:,:,index,:])
            num_after_GRN = torch.cat((num_after_GRN, grn.unsqueeze(2)), dim=2)
        return num_after_GRN
    
    def get_flat_GRN(self, to_be_flat: torch.Tensor) -> torch.Tensor:
        emb = torch.flatten(to_be_flat, start_dim=2)
        var_sel_wei = self.flatten_GRN(emb)
        return var_sel_wei
    
class Decoder_LSTM(nn.Module):
    def __init__(self, n_layers_LSTM: int, d_model: int, dropout: float):
        """LSTM Decoder with GLU, Add and Norm
        norm( x + GLU(dropout( LSTM(x) )) )

        Args:
            n_layers_LSTM (int): number of layers involved by LSTM 
            d_model (int): model dimension
            dropout (float): -
        """
        super().__init__()
        self.n_layers_DecLSTM = n_layers_LSTM
        self.hidden_size = d_model
        self.LSTM = nn.LSTM(input_size=d_model, hidden_size=self.hidden_size, num_layers=self.n_layers_DecLSTM, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.LSTM_enc_GLU = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, cn: torch.Tensor) -> torch.Tensor:
        """LSTM Decoder with GLU, Add and Norm

        Args:
            x (torch.Tensor): [bs, past_steps, d_model] main Tensor
            hn (torch.Tensor): [n_layers_DecLSTM, bs, d_model] Tensor of hidden states from Encoder
            cn (torch.Tensor): [n_layers_DecLSTM, bs, d_model] Tensor of initial cell states from Encoder

        Returns:
            torch.Tensor: [bs, past_steps, d_model]
        """
        lstm_dec, _ = self.LSTM(x, (hn,cn)) # we ignore the (hc,cn) coming from LSTM, no needed for future computations
        lstm_dec = self.dropout(lstm_dec)
        output_dec = self.norm(self.LSTM_enc_GLU(lstm_dec) + x)
        return output_dec

class postTransformer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        """Last part of TFT after decoder and before last linear
        norm( res_conn_postLSTM + ( GLU(GRN( norm(res_conn_postGRN + GLU(dropout(x))) )) ) )

        Args:
            d_model (int): model dimension
            dropout (float): -
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.GLU1 = GLU(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.GRN = GRN(d_model, dropout)
        self.GLU2 = GLU(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, res_conn_dec: torch.Tensor, res_conn_grn: torch.Tensor) -> torch.Tensor:
        """Last part of TFT after decoder and before last linear

        Args:
            x (torch.Tensor): [bs, past_steps, d_model] main Tensor
            res_conn_dec (torch.Tensor): [bs, past_steps, d_model] residual connection pre decoder
            res_conn_grn (torch.Tensor): [bs, past_steps, d_model] residual connection pre GRN-Static Enrichment

        Returns:
            torch.Tensor: [bs, past_steps, d_model]
        """
        x = self.dropout(x)
        # first res_conn
        x = self.norm1(res_conn_dec + self.GLU1(x))
        x = self.GRN(x)
        #second res_conn
        out = self.norm2(res_conn_grn + self.GLU2(x))
        return out
    
