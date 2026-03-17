
from dataclasses import dataclass
from torch import nn
from einops import rearrange, repeat
from typing import cast
import torch
import copy
from .chronos2.layers import ResidualBlock, Chronos2LayerNorm, MLP, MHA, Patch, InstanceNorm, Chronos2Encoder, Chronos2EncoderOutput
from .chronos2.config import Chronos2CoreConfig, Chronos2ForecastingConfig
from transformers.utils.generic import ModelOutput
from .utils import get_scope

try:
    import lightning.pytorch as pl
    from .base_v2 import Base
    OLD_PL = False
except:
    import pytorch_lightning as pl
    OLD_PL = True
    from .base import Base
    

@dataclass
class Chronos2Output(ModelOutput):
    loss: torch.Tensor | None = None
    quantile_preds: torch.Tensor | None = None
    enc_time_self_attn_weights: tuple[torch.Tensor, ...] | None = None
    enc_group_self_attn_weights: tuple[torch.Tensor, ...] | None = None


class Chronos2(Base): # type: ignore

    handle_multivariate = True
    handle_future_covariates = True
    handle_categorical_variables = True
    handle_quantile_loss = True

    description = get_scope(handle_multivariate,handle_future_covariates,handle_categorical_variables,handle_quantile_loss)
    
    config_class = Chronos2CoreConfig  # type: ignore[assignment]
    _supports_long_horizon: bool = True
    _supports_future_covariates: bool = True
    _supports_sdpa: bool = True

    def __init__(self,
                context_length,
                input_patch_size,
                input_patch_stride,
                max_output_patches,
                output_patch_size,
                time_encoding_scale,
                use_arcsinh,
                use_reg_token,

                d_ff,
                d_kv,
                d_model,
                dense_act_fn,
                dropout_rate,
                feed_forward_proj,
                initializer_factor,
                layer_norm_epsilon,
                num_heads,
                num_layers,
                pad_token_id,
                reg_token_id,
                rope_theta,
                vocab_size,
                attn_implementation,
                **kwargs)->None:

        super().__init__(**kwargs)
        
        # self.config = config
        self.model_dim = d_model
        self.initializer_factor = initializer_factor
        self.d_ff = d_ff
        self.d_kv = d_kv
        self.num_heads = num_heads
        self.reg_token_id = reg_token_id

        # if past variables has to contain future variables
        self.need_shared_past_fut_variables = True
        # if needed data ordering
        self.need_ordered_variables = True
        self.n_past_only_num_vars = 0
        self.n_past_only_cat_vars = 0

        chronos_config = {'context_length':context_length, 
                        'output_patch_size':output_patch_size, 
                        'input_patch_size':input_patch_size, 
                        'input_patch_stride':input_patch_stride,
                        'quantiles':kwargs['quantiles'],
                        'use_reg_token':use_reg_token,
                        'use_arcsinh':use_arcsinh,
                        'max_output_patches':max_output_patches,
                        'time_encoding_scale':time_encoding_scale,
                        }
        self.chronos_config = Chronos2ForecastingConfig(**chronos_config)

        self.shared = nn.Embedding(vocab_size, d_model)

        self.input_patch_embedding = ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim = self.chronos_config.input_patch_size * 3,
            h_dim = d_ff,
            out_dim = d_model,
            act_fn_name = dense_act_fn,
            dropout_p = dropout_rate,
        )
        self.patch = Patch(
            patch_size = self.chronos_config.input_patch_size, 
            patch_stride = self.chronos_config.input_patch_stride
        )
        
        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm(use_arcsinh = self.chronos_config.use_arcsinh)

        encoder_config = Chronos2CoreConfig(
                d_model = d_model,
                d_kv = d_kv,
                d_ff = d_ff,
                num_layers = num_layers,
                num_heads = num_heads,
                dropout_rate = dropout_rate,
                layer_norm_epsilon = layer_norm_epsilon,
                initializer_factor = initializer_factor,
                feed_forward_proj = feed_forward_proj,
                vocab_size = vocab_size,
                pad_token_id = pad_token_id,
                rope_theta = rope_theta,
                attn_implementation = attn_implementation,
                **kwargs,)
        # encoder_config['is_decoder'] = False
        self.encoder = Chronos2Encoder(encoder_config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles_tensor = torch.tensor(self.chronos_config.quantiles, dtype=self.dtype)
        self.chronos_quantiles: torch.Tensor
        self.register_buffer("chronos_quantiles", quantiles_tensor, persistent=False)

        self.output_patch_embedding = ResidualBlock(
            in_dim = d_model,
            h_dim = d_ff,
            out_dim = self.num_quantiles * self.chronos_config.output_patch_size,
            act_fn_name = dense_act_fn,
            dropout_p = dropout_rate,
        )

    def can_be_compiled(self):
        return False
    
    def _init_weights(self, module):
        super()._init_weights(module)
        """Initialize the weights"""
        factor = self.initializer_factor
        if isinstance(module, Chronos2LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, MLP):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MHA):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.d_model
            kv_proj_dim = self.d_kv
            n_heads = self.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * kv_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * kv_proj_dim) ** -0.5))
        elif isinstance(module, (Chronos2)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, ResidualBlock):
            module.hidden_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.hidden_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.hidden_layer, "bias") and module.hidden_layer.bias is not None:
                module.hidden_layer.bias.data.zero_()

            module.residual_layer.weight.data.normal_(
                mean=0.0,
                std=factor * (module.residual_layer.weight.size(-1) ** -0.5),
            )
            if hasattr(module.residual_layer, "bias") and module.residual_layer.bias is not None:
                module.residual_layer.bias.data.zero_()

            module.output_layer.weight.data.normal_(
                mean=0.0, std=factor * (module.output_layer.weight.size(-1) ** -0.5)
            )
            if hasattr(module.output_layer, "bias") and module.output_layer.bias is not None:
                module.output_layer.bias.data.zero_()

    def _validate_input(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        group_ids: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        num_output_patches: int,
        future_target: torch.Tensor | None,
        future_target_mask: torch.Tensor | None,
    ):
        output_patch_size = self.chronos_config.output_patch_size
        if context.ndim != 2:
            raise ValueError(f"context must have shape (batch_size, context_length), found: {tuple(context.shape)}")
        if context_mask is not None and context_mask.shape != context.shape:
            raise ValueError(f"mask must have shape {tuple(context.shape)}, found: {tuple(context_mask.shape)}")
        if future_covariates is not None:
            if future_covariates.shape[0] != context.shape[0] or future_covariates.ndim != 2:
                raise ValueError(
                    f"future_covariates must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_covariates.shape)}"
                )
            if future_covariates.shape[-1] > num_output_patches * output_patch_size:
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_covariates, "
                    f"found: {future_covariates.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )
            if future_target is not None and future_target.shape != future_covariates.shape:
                raise ValueError(
                    f"future_target must have the same shape as future_covariates, found: {tuple(future_target.shape)} and {tuple(future_covariates.shape)}"
                )
        if future_covariates_mask is not None:
            if future_covariates is None:
                raise ValueError("future_covariates must be provided if future_covariates_mask is provided")
            if future_covariates_mask.shape != future_covariates.shape:
                raise ValueError(
                    f"future_covariates_mask must have the same shape as future_covariates, "
                    f"found: {tuple(future_covariates_mask.shape)} and {tuple(future_covariates.shape)}"
                )
        if group_ids is not None and group_ids.shape != (context.shape[0],):
            raise ValueError(f"group_ids must have shape (batch_size,), found: {tuple(group_ids.shape)}")
        if future_target is not None:
            if future_target.shape[0] != context.shape[0] or future_target.ndim != 2:
                raise ValueError(
                    f"future_target must have shape (batch_size={context.shape[0]}, future_length), found: {tuple(future_target.shape)}"
                )
            if future_target.shape[-1] > output_patch_size * num_output_patches:
                raise ValueError(
                    f"{num_output_patches=} must be large enough to accommodate the length of future_target, "
                    f"found: {future_target.shape[-1]} > {num_output_patches} * {output_patch_size}"
                )
        if future_target_mask is not None:
            if future_target is None:
                raise ValueError("future_target must be provided if future_target_mask is provided")
            if future_target_mask.shape != future_target.shape:
                raise ValueError(
                    f"future_target_mask must have the same shape as future_target, found: {tuple(future_target_mask.shape)} and {tuple(future_target.shape)}"
                )

    def _prepare_patched_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, context_length = context.shape
        # truncate context if it's longer than model's context length
        if context_length > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            context_mask = context_mask[..., -self.chronos_config.context_length :]

        # scaling
        context, loc_scale = self.instance_norm(context)

        # scaling is done in 32-bit precision, then the context is moved to model's dtype
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, num_patches)
        num_context_patches = attention_mask.shape[-1]

        # context time encoding: every observation is assigned a sequential time index,
        # scaled by model's context length = [-C, -(C-1), ..., -1] / context_length
        final_context_length = num_context_patches * self.chronos_config.input_patch_size
        context_time_enc = torch.arange(start=-final_context_length, end=0, device=self.device, dtype=torch.float32)
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.chronos_config.input_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        # concat time encoding, context and mask along the last (feature) dim
        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)

        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(
        self,
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_patch_size = self.chronos_config.output_patch_size
        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = cast(torch.Tensor, future_covariates)
            future_covariates = future_covariates.to(self.dtype)

            if future_covariates_mask is None:
                future_covariates_mask = torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)

            future_covariates = torch.where(future_covariates_mask > 0.0, future_covariates, 0.0)

            if torch.isnan(future_covariates).any():
                raise ValueError(
                    "future_covariates contains NaN values at indices not masked by future_covariates_mask. "
                    "Input the correct future_covariates_mask or omit it to automatically infer the mask based on NaN values."
                )

            # add padding if the length of future_covariates is not an integer multiple of output_patch_size
            if num_output_patches * output_patch_size > future_covariates.shape[-1]:
                padding_shape = (
                    *future_covariates.shape[:-1],
                    num_output_patches * output_patch_size - future_covariates.shape[-1],
                )
                future_covariates = torch.cat(
                    [future_covariates, torch.zeros(padding_shape).to(future_covariates)], dim=-1
                )
                future_covariates_mask = torch.cat(
                    [future_covariates_mask, torch.zeros(padding_shape).to(future_covariates_mask)], dim=-1
                )

            patched_future_covariates = rearrange(
                future_covariates, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
            patched_future_covariates_mask = rearrange(
                future_covariates_mask, "b (n p) -> b n p", n=num_output_patches, p=output_patch_size
            )
        else:
            patched_future_covariates = torch.zeros(
                batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype
            )
            patched_future_covariates_mask = torch.zeros(
                batch_size, num_output_patches, output_patch_size, device=self.device, dtype=self.dtype
            )

        # future time encoding: every future timestep is assigned a sequential time index,
        # scaled by model's context length = [0, 1, ..., h-1] / context_length
        final_future_length = num_output_patches * output_patch_size
        future_time_enc = torch.arange(start=0, end=final_future_length, device=self.device, dtype=torch.float32)
        future_time_enc = (
            repeat(
                future_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_output_patches,
                p=output_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
        )

        patched_future = torch.cat(
            [future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1
        )

        return patched_future, patched_future_covariates_mask

    def _compute_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size
        assert quantile_preds.shape[0] == batch_size and quantile_preds.shape[-1] >= future_target.shape[-1]

        # normalize target and mask
        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1)
        future_target = future_target.to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        # pad target and target_mask if they are shorter than model's prediction
        if quantile_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (*future_target.shape[:-1], quantile_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat([future_target, torch.zeros(padding_shape).to(future_target)], dim=-1)
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(padding_shape).to(future_target_mask)], dim=-1
            )

        quantiles = rearrange(self.chronos_quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds) * ((future_target <= quantile_preds).float() - quantiles)
        )
        inv_future_covariate_mask = 1 - rearrange(
            patched_future_covariates_mask,
            "b n p -> b 1 (n p)",
            b=batch_size,
            n=num_output_patches,
            p=output_patch_size,
        )
        # the first components masks any missing targets and the second component masks known future values
        loss_mask = future_target_mask.float() * inv_future_covariate_mask
        loss = quantile_loss * loss_mask
        # mean over prediction horizon, sum over quantile levels and mean over batch
        loss = loss.mean(dim=-1).sum(dim=-1).mean()

        return loss

    def encode(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        self._validate_input(
            context=context,
            context_mask=context_mask,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            group_ids=group_ids,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
        )

        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask
        )
        num_context_patches = attention_mask.shape[-1]

        # get input embeddings of shape (batch, num_context_patches, d_model)
        input_embeds: torch.Tensor = self.input_patch_embedding(patched_context)
        # append [REG] special token embedding, if needed
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)], dim=-1
            )

        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)

        # get future embeddings of shape (batch, num_output_patches, d_model)
        future_embeds: torch.Tensor = self.input_patch_embedding(patched_future)

        # concatenate context and future embeddings and masks
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        if group_ids is None:
            # by default, each time series is treated independently, i.e., no mixing across the batch
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

        encoder_outputs: Chronos2EncoderOutput = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            output_attentions=output_attentions,
        )
        return encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches

    def forward(
        self,
        batch
    ):
        # ADAPT from BATCH to STANDARD CHRONOS2 INPUTS
        batch_size, horizon, n_target_vars = batch['y'].shape
        num_past_with_y = batch['x_num_past']
        _, context_length, number_past_no_cat_vars = num_past_with_y.shape

        ### context
        target_idx_expanded = batch['idx_target'].unsqueeze(1).expand(-1, context_length, -1)
        past_target = torch.gather(num_past_with_y, dim=2, index=target_idx_expanded)

        # remove target from num_past
        n_past_num_vars = number_past_no_cat_vars - n_target_vars
        mask = torch.ones((batch_size, number_past_no_cat_vars), device=num_past_with_y.device, dtype=torch.bool)
        mask.scatter_(1, target_idx_expanded[:, 0, :], False)
        mask_3d = mask.unsqueeze(1).expand(-1, context_length, -1)
        num_past = num_past_with_y[mask_3d].view(batch_size, context_length, n_past_num_vars)

        # context: with new workflow they are sorted
        if 'x_cat_past' in batch.keys():
            context = torch.cat([
                past_target, # Past Targets
                num_past[:, :, :self.n_past_only_num_vars], # Past-only Numerical
                batch['x_cat_past'][:, :, :self.n_past_only_cat_vars], # Past-only Categorical
                num_past[:, :, self.n_past_only_num_vars:], # Other Numerical
                batch['x_cat_past'][:, :, self.n_past_only_cat_vars:]  # Other Categorical
            ], dim=-1)
        else:
            context = torch.cat([past_target, num_past], dim=-1)
        tot_vars = context.shape[-1]

        ### group id
        group_ids = torch.arange(batch_size)
        group_ids = torch.repeat_interleave(group_ids, repeats = tot_vars, dim = 0)

        ### future_covariates
        x_num_future = batch.get('x_num_future', torch.empty((batch_size, horizon, 0), device=context.device))
        x_cat_future = batch.get('x_cat_future', torch.empty((batch_size, horizon, 0), device=context.device))
        future_covariates = torch.cat([x_num_future, x_cat_future.float()], dim=-1)

        fut_cov_pad_size = tot_vars - future_covariates.shape[-1]
        if fut_cov_pad_size != tot_vars:
            if fut_cov_pad_size > 0:
                future_covariates = nn.functional.pad(
                    future_covariates, 
                    # Padding tuple for 3D: 
                    # (Dim2_L, Dim2_R, Dim1_T, Dim1_B, Dim0_F, Dim0_B)
                    (fut_cov_pad_size, 0, 0, 0, 0, 0), 
                    value=float('nan')
                )
            future_covariates = rearrange(future_covariates, "b t f -> (b f) t")
        else:
            future_covariates = None

        ### future_target
        future_target = batch['y']
        fut_y_pad_size = tot_vars - future_target.shape[-1]
        future_target = nn.functional.pad(future_target, (0, fut_y_pad_size, 0, 0, 0, 0), value=float('nan'))

        # aux
        num_output_patches = horizon // self.chronos_config.output_patch_size +1
        output_attentions = False
        context_mask = None
        future_covariates_mask = None
        future_target_mask = None

        # reshape all
        context = rearrange(context, "b t f -> (b f) t")
        future_target = rearrange(future_target, "b t f -> (b f) t")

        """Forward pass of the Chronos2 model.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
        group_ids : torch.Tensor | None, optional
            Group IDs of shape (batch_size,) indicating which times series in the batch form a group.
            A group indicates a task, for example, for a batch of size 6:
            - if groups_ids = [0, 1, 2, 3, 4, 5], each time series is treated independently.
            - if groups_ids = [0, 0, 1, 1, 1, 2], information is mixed across the first two time series (id=0),
                the next three time series (id=1) and the last time series is treated separately. Information is
                NOT shared among time series from different groups.
            The ordering and specific values of group_ids are not important, all time series with the same group
            ID form a group.
        future_covariates
            Tensor of shape (batch_size, future_length) containing future covariates. Note that the size of
            tensor along the first axis is equal to the batch_size. This means that future values (which may be NaNs)
            must be provided for each time series in the batch. For any time series that need to be forecasted, the
            future_covariates can be set to NaNs, if ``future_covariates_mask`` is omitted or to an arbitrary dummy
            value when ``future_covariates_mask`` is provided. ``future_covariates`` can be used with ``group_ids``
            to construct heterogenous forecasting tasks in a single batch. For example:
            - future_covariates = [[nan, ...], [nan, ...], [v1, ...], [v2, ...], [nan, ...], [nan, ...]]
            - groups_ids = [0, 0, 1, 1, 1, 2]
            - future_covariates_mask = None
            contains 3 types of forecasting tasks:
            - [0, 0]: The first task, both future_covariates are missing, which implies that the two time series need to
                be forecasted jointly, i.e., multivariate forecasting.
            - [1, 1, 1]: In the next task, the first two future_covariates are available and the last one is missing
                ([v1, ...], [v2, ...], [nan, ...]), where [v1, ...] and [v1, ...] denote an arbitrary sequence of values.
                This indicates that the first two time series are known covariates and the third one needs to be forecasted
                by the model.
            - [2]: The last task has a single time series in the group which needs to be forecasted independently.
            There is no theoretical limit on the number of time series in a group, i.e., the number of targets and known
            covariates in a task. The above setup subsumes tasks with past-only covariates as the model's prediction for
            those time series can simply be ignored downstream.
        future_covariates_mask
            Binary mask tensor of same shape as future_covariates indicating which future values are known
            If omitted, future_covariates_mask is automatically constructed based on future_covariates with
            all non-NaN values treated as known future values.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
            When ``future_covariates`` and/or ``future_target`` are provided, num_output_patches should be large enough to accommodate
            their lengths, i.e., num_output_patches * output_patch_size >= future_length
        future_target
            Target tensor of shape (batch_size, future_length) used during training. If ``future_covariates`` are provided, both
            target and future_covariates must have the same shape.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in `future_target`.
        output_attentions
            Whether to return attention weights, by default False

        Returns
        -------
        Chronos2Output containing:
        - loss: Training loss, if `future_target` is provided
        - quantile_preds: Quantile predictions of shape (batch_size, num_quantiles, num_output_patches * output_patch_size).
            quantile_preds will contain an entry for every time series in the context batch regardless of whether it was a
            known future covariate.
        - enc_time_self_attn_weights: Time self attention weights, if output_attentions=True
        - enc_group_self_attn_weights: Group self attention weights, if output_attentions=True
        """

        encoder_outputs, loc_scale, patched_future_covariates_mask, num_context_patches = self.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            future_target=future_target,
            future_target_mask=future_target_mask,
            output_attentions=output_attentions,
        )
        hidden_states: torch.Tensor = encoder_outputs[0]
        assert hidden_states.shape == (batch_size*tot_vars, num_context_patches + 1 + num_output_patches, self.model_dim)

        # slice the last num_output_patches hidden states to be input into the output_patch_embedding
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )
        ## skipping loss computation

        # Unscale predictions
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size*tot_vars,
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )

        ## customizing output
        quantile_preds = quantile_preds[:batch_size*n_target_vars,:, :horizon] # check
        # breakpoint()
        quantile_preds = rearrange(quantile_preds,
                                   "(b c) q l -> b l c q",
                                   b = batch_size,
                                   c = n_target_vars,
                                #    q = self.num_quantiles, 
                                #    l = horizon 
                                   )

        return quantile_preds
        # return Chronos2Output(
        #     loss=loss,
        #     quantile_preds=quantile_preds,
        #     enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
        #     enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        # )
