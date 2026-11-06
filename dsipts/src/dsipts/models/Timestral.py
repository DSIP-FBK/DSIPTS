## Wrapper around Timestral (Mistral AI, https://github.com/mistralai/timestral, code Apache-2.0).
## The network is vendored in ./timestral (imports made relative, nothing else changed); the
## pretrained weights are downloaded from the HF Hub (mistralai/Timestral-{5M,28M,123M,209M},
## Modified MIT license) the first time the model is built and then read from the HF cache.

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

from .timestral.loading import load_model
from .timestral.forecaster import TimestralForecaster
from ..data_structure.utils import beauty_string
from .utils import get_scope


class Timestral(Base):
    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True
    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    description += '\n PRETRAINED FOUNDATION MODEL: categorical variables are passed as raw codes (numeric rows), quantiles are mapped to the closest of the 21 pretrained levels'

    def __init__(self,
                 model_path:str = 'mistralai/Timestral-5M',
                 freeze:str = 'none',
                 use_categorical:bool = True,
                 symmetric_averaging:bool = False,
                 future_num_in_past:Union[List[int],None] = None,
                 future_cat_in_past:Union[List[int],None] = None,
                 local_files_only:bool = False,
                 **kwargs)->None:
        """Timestral (decoder-only patch transformer, 21 monotone quantiles) wrapped as a dsipts model.

        Every variable becomes one row of the model (channel-as-token):
          - targets: x_num_past[:, :, idx_target]  (role TARGET)
          - past numerical / categorical variables that are not future variables  (role PAST_COV)
          - future numerical / categorical variables (role KNOWN_COV): their past values are taken from
            x_num_past / x_cat_past when the same variable is also a past variable (see future_*_in_past),
            otherwise the past part is marked as missing.
        Future columns that are targets (idx_target_future) are always dropped: they would leak y.

        Output: (BS, future_steps, out_channels, mul). mul is chosen by Base from quantiles/loss_type and
        each requested level is mapped to the CLOSEST pretrained level (0.01, 0.05, 0.10, ..., 0.95, 0.99):
          - quantiles=[]            -> the median (any point loss: l1, mse, the persistence losses, ...)
          - quantiles=[q1,q2,q3]    -> the three closest levels (QuantileLossMO)
          - loss_type='crps' (M)    -> member i -> level closest to (i+0.5)/M (the convention Base.inference uses)
        Classification is not supported.

        Args:
            model_path (str): HF Hub repo id or local directory with config.json + model.safetensors.
            freeze (str): 'none' = fine-tune everything, 'backbone' = train only the quantile head.
                For a pure zero-shot run keep freeze='none' and set optim_config.lr=0 (and max_epochs=1):
                Lightning still needs a loss with a gradient, the weights do not move.
            use_categorical (bool): feed x_cat_past / x_cat_future as numeric rows (raw codes, as the official
                code does for multivariate targets). False drops them.
            symmetric_averaging (bool): test-time augmentation of the official forecaster, average the forecast
                with the reflected forecast of the negated input. Doubles the cost, also during training.
            future_num_in_past (List[int], optional): for each future numerical variable, its index in the past
                numerical variables or -1. Filled by bash_examples/train.py. None = no future variable is
                assumed to be in the past (works, but shared variables appear twice).
            future_cat_in_past (List[int], optional): same for the categorical variables.
            local_files_only (bool): do not contact the HF Hub (use the cache only), e.g. on offline nodes.
                HF_HUB_OFFLINE=1 does the same.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        assert self.n_classes==0, 'Timestral: classification is not supported'
        assert freeze in ['none','backbone'], f"Timestral: freeze must be 'none' or 'backbone', got {freeze}"

        self.timestral = load_model(model_path, local_files_only=local_files_only)
        self.timestral.train() ##no dropout/BN inside, but keep lightning's mode coherent
        cfg = self.timestral.config
        ##plain helper object (not a nn.Module) holding a reference to self.timestral: it moves with .to()
        self.forecaster = TimestralForecaster(self.timestral)
        self.use_categorical = use_categorical
        self.symmetric_averaging = symmetric_averaging
        self.future_num_in_past = list(future_num_in_past) if future_num_in_past is not None else None
        self.future_cat_in_past = list(future_cat_in_past) if future_cat_in_past is not None else None
        self.supports_known = cfg.supports_known_covariates
        if not self.supports_known:
            beauty_string('Timestral: this checkpoint can not read known-future covariates, they are used as past covariates only','info',True)
        if self.future_channels>0 and self.future_num_in_past is None:
            beauty_string('Timestral: future_num_in_past not given, future variables also present in the past will appear twice','info',self.verbose)

        ## requested levels -> index of the closest pretrained level
        knots = list(cfg.quantile_knots)
        if self.loss_type=='crps':
            M = len(self.quantiles)
            levels = [(i+0.5)/M for i in range(M)]
        elif self.use_quantiles:
            levels = list(self.quantiles)
        else:
            levels = [0.5]
        idx = [min(range(len(knots)), key=lambda k: abs(knots[k]-l)) for l in levels]
        for l,i in zip(levels,idx):
            if abs(knots[i]-l)>1e-6:
                beauty_string(f'Timestral: level {l:.4f} mapped to the closest pretrained level {knots[i]}','info',self.verbose)
        if len(set(idx))<len(idx):
            beauty_string(f'Timestral: some levels map to the same pretrained level ({idx}), those outputs are identical at init','info',True)
        self.register_buffer('q_idx', torch.tensor(idx, dtype=torch.long), persistent=False)

        if freeze=='backbone':
            for name, p in self.timestral.named_parameters():
                p.requires_grad = name.startswith('head.')
        n_tot = sum(p.numel() for p in self.timestral.parameters())
        n_train = sum(p.numel() for p in self.timestral.parameters() if p.requires_grad)
        beauty_string(f'Timestral {model_path}: {n_tot} parameters, {n_train} trainable, patch {cfg.patch_size}, levels {[knots[i] for i in idx]}','info',self.verbose)

    def can_be_compiled(self):
        ##data dependent branches (role_ids.all(), torch.unique) inside the vendored model
        return False

    @staticmethod
    def _fit_length(x:torch.Tensor, length:int)->torch.Tensor:
        """(B, T, C) -> (B, length, C): truncate or right-pad with NaN (= missing)"""
        if x.shape[1]>=length:
            return x[:, :length]
        pad = torch.full((x.shape[0], length-x.shape[1], x.shape[2]), float('nan'), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def _rows(self, past:torch.Tensor, fut:Union[torch.Tensor,None], fut_in_past:Union[List[int],None], skip_past:List[int], skip_fut:List[int]):
        """Split one family of variables (numerical or categorical) into past-only rows (B, C, L) and known
        rows (B, C, L+H). past (B, L, Cp), fut (B, H, Cf) or None."""
        B, L, Cp = past.shape
        H = self.future_steps
        past_only, known = [], []
        mapped = {}
        if fut is not None:
            fut = self._fit_length(fut, H)
            for f in range(fut.shape[2]):
                if f in skip_fut:
                    continue
                j = fut_in_past[f] if (fut_in_past is not None and f<len(fut_in_past)) else -1
                if j is not None and 0<=j<Cp:
                    mapped[j] = f
                    if self.supports_known:
                        known.append(torch.cat([past[:, :, j], fut[:, :, f]], dim=1))
                    else:
                        past_only.append(past[:, :, j])
                elif self.supports_known:
                    nan_past = torch.full((B, L), float('nan'), device=past.device, dtype=past.dtype)
                    known.append(torch.cat([nan_past, fut[:, :, f]], dim=1))
        for j in range(Cp):
            if j in skip_past or j in mapped:
                continue
            past_only.append(past[:, :, j])
        past_only = torch.stack(past_only, dim=1) if len(past_only)>0 else None
        known = torch.stack(known, dim=1) if len(known)>0 else None
        return past_only, known

    def _decode(self, target, mask, past_cov, known_cov):
        """(B, n_tgt, L) context -> (B, n_tgt, H, 21) quantiles in the input (already scaled) space."""
        H = self.future_steps
        rows, rmask, role_ids, kf, kfm, n_tgt = self.forecaster._assemble_rows(target, mask, H, past_cov, known_cov)
        q = self.forecaster._block_ar_decode(rows, rmask, H, role_ids, kf, kfm)
        return q[:, :n_tgt]

    def forward(self, batch:dict)->torch.Tensor:
        x = batch['x_num_past'].float()
        B, L, Cp = x.shape
        H = self.future_steps

        idx_target = batch.get('idx_target', None)
        if idx_target is not None and idx_target.numel()>0:
            tgt_idx = [int(i) for i in idx_target[0].tolist()]
            target = x[:, :, tgt_idx].permute(0, 2, 1)
        else:
            ##targets not among the past variables: target rows with a fully missing context
            tgt_idx = []
            target = torch.full((B, self.out_channels, L), float('nan'), device=x.device)
        obs = torch.isfinite(target)
        target = torch.where(obs, target, torch.zeros_like(target))

        skip_fut = []
        if batch.get('idx_target_future', None) is not None:
            skip_fut = [int(i) for i in batch['idx_target_future'][0].tolist()]

        x_fut = batch['x_num_future'].float() if 'x_num_future' in batch.keys() else None
        past_cov, known_cov = self._rows(x, x_fut, self.future_num_in_past, tgt_idx, skip_fut)

        if self.use_categorical and 'x_cat_past' in batch.keys():
            c_fut = batch['x_cat_future'].float() if 'x_cat_future' in batch.keys() else None
            p_cat, k_cat = self._rows(batch['x_cat_past'].float(), c_fut, self.future_cat_in_past, [], [])
            past_cov = torch.cat([t for t in [past_cov, p_cat] if t is not None], dim=1) if (past_cov is not None or p_cat is not None) else None
            known_cov = torch.cat([t for t in [known_cov, k_cat] if t is not None], dim=1) if (known_cov is not None or k_cat is not None) else None
        elif self.use_categorical and 'x_cat_future' in batch.keys():
            _, k_cat = self._rows(torch.zeros((B, L, 0), device=x.device), batch['x_cat_future'].float(), None, [], [])
            if k_cat is not None:
                known_cov = k_cat if known_cov is None else torch.cat([known_cov, k_cat], dim=1)

        q = self._decode(target, obs, past_cov, known_cov)
        if self.symmetric_averaging:
            neg = lambda t: -t if t is not None else None
            q_m = self._decode(-target, obs, neg(past_cov), neg(known_cov))
            q = 0.5*(q + (-q_m).flip(-1))

        q = q.index_select(-1, self.q_idx)          # (B, n_tgt, H, mul)
        return q.permute(0, 2, 1, 3).reshape(B, H, -1, self.mul)
