## Copyright Contributors to the s4casting project, MPL-2.0
## https://github.com/alliander-opensource/s4casting/blob/main/src/s4casting/model/transformer.py
## Adapted to the DSIPTS batch structure; the layers are in grouped_transformer/layers.py

from typing import List, Union

import torch
from torch import nn

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    if pl.__version__>='2.0.0':
        from .base_v2 import Base
        OLD_PL = False
    else:
        from .base import Base
        OLD_PL = True
from ..data_structure.utils import beauty_string
from .utils import get_scope
from .utils import Embedding_cat_variables
from .grouped_transformer.layers import SeriesPatchEncoder, PatchDecoder, TransformerLayer, masked_norm


class GroupedTransformer(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    description += '\n Grouped time-series transformer (s4casting): time attention + variable (group) attention on patches'

    def __init__(self,
                 latent_dim:int=64,
                 n_heads:int=4,
                 n_layers:int=2,
                 patch_size:int=8,
                 dropout_rate:float=0.0,
                 attn_bias:bool=True,
                 mlp_layers:int=2,
                 out_emb:Union[int,None]=None,
                 use_norm:bool=True,
                 norm_clamp:Union[float,None]=10.0,
                 norm_eps:float=1e-5,
                 group_gate_init:float=-3.0,
                 causal:bool=False,
                 future_num_in_past:Union[List[int],None]=None,
                 **kwargs)->None:
        """Grouped time-series transformer from s4casting (Alliander).

        Every variable (targets, past covariates, known-future covariates) is a series over the whole window
        past+future, with a mask that marks where it is observed (targets and past covariates: past only, future
        covariates: future, plus the past when they are also past variables). Each series is cut in patches
        (values + mask -> latent_dim), then each layer does time attention (RoPE) over the patches of a series and
        a gated group attention over the series of a patch. The target series are decoded back to time steps and
        the future part is returned.

        Args:
            latent_dim (int): embedding size. latent_dim // n_heads must be even (RoPE).
            n_heads (int): attention heads.
            n_layers (int): transformer layers.
            patch_size (int): patch length. past_steps is left-padded and future_steps right-padded to a multiple
                of it (padding is marked missing), so the past/future boundary falls on a patch boundary.
            dropout_rate (float): dropout in attention and MLP.
            attn_bias (bool): bias of the qkv/out projections.
            mlp_layers (int): layers of the feed-forward block (>=2).
            out_emb (int, optional): size of the decoded per-step embedding before the output head, default latent_dim.
            use_norm (bool): per-window instance norm of every series with masked statistics (RevIN-like, as in
                s4casting); the targets are de-normalized with the statistics of their past.
            norm_clamp (float, optional): clamp of the normalized inputs, None = off.
            norm_eps (float): added to the instance std.
            group_gate_init (float): initial logit of the group-attention gate (sigmoid(-3)~0.05: starts almost off).
            causal (bool): causal time attention (a patch sees only previous patches). Group attention is never causal.
            future_num_in_past (List[int], optional): for each future numerical variable, its index among the past
                variables (-1 if absent). A shared variable then becomes ONE series with past and future values;
                without it, it appears twice (filled by train.py).

        Any DSIPTS loss works (l1/mse, quantiles, crps, persistence losses, classification) and also
        loss_type='gmm_nll' (s4casting's GMMHead + NLL, n_gaussians components, now in Base): the de-normalization
        goes through Base.rescale_output, which treats the mixture parameters correctly.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.patch_size = patch_size
        self.use_norm = use_norm
        self.norm_clamp = norm_clamp
        self.norm_eps = norm_eps
        self.future_num_in_past = list(future_num_in_past) if future_num_in_past is not None else None
        if self.future_channels>0 and self.future_num_in_past is None:
            beauty_string('GroupedTransformer: future_num_in_past not given, future variables also present in the past will appear twice','info',self.verbose)

        P = patch_size
        self.Lp = -(-self.past_steps//P)*P     ## padded past
        self.Hp = -(-self.future_steps//P)*P   ## padded future
        self.W = self.Lp+self.Hp
        self.n_patches = self.W//P
        if out_emb is None:
            out_emb = latent_dim

        ## categorical variables: one embedding per time step, averaged per patch and added to every series token
        ## (it replaces the unix-time/lat-lon embedding of s4casting). Zero-initialized projection as there.
        self.use_cat_past = len(self.embs_past)>0
        self.use_cat_fut = len(self.embs_fut)>0
        if self.use_cat_past:
            self.emb_past = Embedding_cat_variables(self.past_steps,self.emb_dim,self.embs_past, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
            self.proj_cat_past = nn.Linear(self.emb_past.output_channels, latent_dim)
            nn.init.zeros_(self.proj_cat_past.weight)
        if self.use_cat_fut:
            self.emb_fut = Embedding_cat_variables(self.future_steps,self.emb_dim,self.embs_fut, reduction_mode=self.reduction_mode,use_classical_positional_encoder=self.use_classical_positional_encoder,device = self.device)
            self.proj_cat_fut = nn.Linear(self.emb_fut.output_channels, latent_dim)
            nn.init.zeros_(self.proj_cat_fut.weight)

        self.patch_encoder = SeriesPatchEncoder(latent_dim, P)
        self.layers = nn.ModuleList([TransformerLayer(latent_dim, n_heads, self.n_patches, dropout=dropout_rate,
                                                      attn_bias=attn_bias, mlp_layers=mlp_layers, norm_eps=norm_eps,
                                                      group_gate_init=group_gate_init, causal=causal)
                                     for _ in range(n_layers)])
        self.patch_decoder = PatchDecoder(latent_dim, out_emb, P)
        self.head = nn.Linear(out_emb, self.mul)

    def can_be_compiled(self):
        ## index lists are read from the batch (idx_target) with .tolist()
        return False

    def _series(self, batch):
        """Build the (B, W, F) values and mask of all the series, targets first."""
        x = batch['x_num_past'].float()
        B, L, Cp = x.shape
        H = self.future_steps
        dev = x.device
        s = self.Lp-L   ## start of the past inside the window

        idx_target = batch.get('idx_target', None)
        if idx_target is not None and idx_target.numel()>0:
            tgt_idx = [int(i) for i in idx_target[0].tolist()]
        else:
            tgt_idx = []   ## targets not among the past variables: fully missing target series
        nt = self.out_channels

        skip_fut = []
        if batch.get('idx_target_future', None) is not None:
            skip_fut = [int(i) for i in batch['idx_target_future'][0].tolist()]
        x_fut = batch['x_num_future'].float() if 'x_num_future' in batch.keys() else None

        ## future variables: mapped onto a past one (shared series) or future-only
        mapped = {}       ## past idx -> future idx
        fut_only = []
        if x_fut is not None:
            for f in range(x_fut.shape[2]):
                if f in skip_fut:
                    continue
                j = self.future_num_in_past[f] if (self.future_num_in_past is not None and f<len(self.future_num_in_past)) else -1
                if j is not None and 0<=j<Cp and j not in tgt_idx:
                    mapped[j] = f
                else:
                    fut_only.append(f)
        past_only = [j for j in range(Cp) if j not in tgt_idx and j not in mapped]
        shared = list(mapped.keys())

        F = nt+len(past_only)+len(shared)+len(fut_only)
        v = torch.zeros((B, self.W, F), device=dev)
        m = torch.zeros((B, self.W, F), device=dev)
        k = 0
        if len(tgt_idx)>0:
            v[:, s:self.Lp, :nt] = x[:, :, tgt_idx]
            m[:, s:self.Lp, :nt] = 1
        k = nt
        cols = past_only+shared
        if len(cols)>0:
            v[:, s:self.Lp, k:k+len(cols)] = x[:, :, cols]
            m[:, s:self.Lp, k:k+len(cols)] = 1
        k += len(past_only)
        fcols = [mapped[j] for j in shared]+fut_only
        if len(fcols)>0:
            Hf = x_fut.shape[1]
            v[:, self.Lp:self.Lp+Hf, k:k+len(fcols)] = x_fut[:, :, fcols]
            m[:, self.Lp:self.Lp+Hf, k:k+len(fcols)] = 1

        ok = torch.isfinite(v)
        m = m*ok
        v = torch.where(ok, v, torch.zeros_like(v))
        return v, m, nt

    def _cat_embedding(self, batch, B, dev):
        """(B, n_patches, latent_dim) or None"""
        out = None
        if self.use_cat_past and 'x_cat_past' in batch.keys():
            e = self.proj_cat_past(self.emb_past(B, batch['x_cat_past'].to(dev)))
            out = torch.zeros((B, self.W, e.shape[-1]), device=dev, dtype=e.dtype)
            out[:, self.Lp-e.shape[1]:self.Lp] = e
        if self.use_cat_fut and 'x_cat_future' in batch.keys():
            e = self.proj_cat_fut(self.emb_fut(B, batch['x_cat_future'].to(dev)))
            if out is None:
                out = torch.zeros((B, self.W, e.shape[-1]), device=dev, dtype=e.dtype)
            out[:, self.Lp:self.Lp+e.shape[1]] = e
        if out is None:
            return None
        return out.reshape(B, self.n_patches, self.patch_size, -1).mean(2)

    def forward(self, batch:dict)->torch.Tensor:
        v, m, nt = self._series(batch)
        B, W, F = v.shape
        H = self.future_steps

        if self.use_norm:
            v, mean, std = masked_norm(v, m, eps=self.norm_eps, clamp=self.norm_clamp)
        else:
            v = v*m

        tok = self.patch_encoder(v, m)                       ## B, F, N, D
        cat = self._cat_embedding(batch, B, v.device)
        if cat is not None:
            tok = tok+cat[:, None]

        ## a series is a key of the group attention in a patch only if it is observed there; targets always
        ## (otherwise a patch with no observed series has no key -> NaN)
        key_present = m.reshape(B, self.n_patches, self.patch_size, F).bool().any(dim=2)
        key_present[..., :nt] = True
        key_present = key_present.flatten(0, 1)

        for layer in self.layers:
            tok = layer(tok, key_present)

        z = tok[:, :nt].reshape(B*nt, self.n_patches, -1)
        z = self.head(self.patch_decoder(z))                 ## B*nt, W, mul
        z = z[:, self.Lp:self.Lp+H].reshape(B, nt, H, self.mul).permute(0, 2, 1, 3)   ## B, H, nt, mul

        if self.use_norm and not self.is_classification:
            z = self.rescale_output(z, loc=mean[:, :, :nt].unsqueeze(-1), scale=std[:, :, :nt].unsqueeze(-1))
        return z
