# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0
#
# Adapted from https://github.com/alliander-opensource/s4casting (src/s4casting/model/{transformer,_blocks,
# _encoders,_norm}.py). Changes for DSIPTS:
#   - no pydantic, no module instances as default arguments (they were built once at import and shared)
#   - every target is decoded (s4casting decodes only feature 0)
#   - masked instance norm returns safe statistics for a fully masked series (no past target)
#   - the RoPE cache is sized to the actual number of patches
#   - unix-time / lat-lon embeddings, GMM head, NLL loss and sample-interval logic are dropped

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def masked_norm(x: torch.Tensor, xm: torch.Tensor, eps: float = 1e-5, clamp: float | None = 10.0):
    """Per-sample, per-feature instance norm over time using only the observed values (xm=1).

    Args:
        x: (B, T, F)
        xm: (B, T, F) 1 where observed
        eps: added to the std
        clamp: clamp of the normalized values (None = no clamp)

    Returns:
        x normalized (masked entries set to 0), mean (B, 1, F), std (B, 1, F).
        A feature without observed values gets mean 0, std 1.
    """
    m = xm.to(x.dtype)
    n = m.sum(dim=1, keepdim=True)
    mean = (x * m).sum(dim=1, keepdim=True) / n.clamp(min=1)
    var = (((x - mean) * m) ** 2).sum(dim=1, keepdim=True) / (n - 1).clamp(min=1)
    std = torch.sqrt(var + 1e-12) + eps
    empty = n < 1
    mean = torch.where(empty, torch.zeros_like(mean), mean)
    std = torch.where(empty, torch.ones_like(std), std)
    x = (x - mean) / std
    if clamp is not None:
        x = x.clamp(min=-clamp, max=clamp)
    return x * m, mean, std


class MLP(nn.Module):
    """Multi-layer perceptron (transformer feed-forward)."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None, n_layers: int = 2, bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert n_layers >= 2, 'mlp_layers must be >= 2'
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        self.first_layer = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.mid_layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=bias), nn.GELU()) for _ in range(n_layers - 2)
        ])
        self.final_layer = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.first_layer(x))
        x = self.mid_layers(x)
        x = self.final_layer(x)
        return self.dropout_layer(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE), Su et al. (2021)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, 'RoPE head dimension (latent_dim // n_heads) must be even'
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = torch.outer(positions, frequencies)
        self.register_buffer('cache', torch.stack((angles.cos(), angles.sin())), persistent=False)

    def forward(self, x):
        ## x: (B, L, H, hd)
        cos, sin = self.cache[:, : x.shape[1], None].to(x)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE and key masking."""

    def __init__(self, latent_dim: int, n_heads: int, bias: bool = True, dropout: float = 0.0,
                 is_causal: bool = False, rope: nn.Module | None = None):
        super().__init__()
        assert latent_dim % n_heads == 0, 'latent_dim must be divisible by n_heads'
        self.n_heads = n_heads
        self.is_causal = is_causal
        self.dropout = dropout
        self.rope = rope
        self.qkv = nn.Linear(latent_dim, 3 * latent_dim, bias=bias)
        self.out = nn.Linear(latent_dim, latent_dim, bias=bias)

    def forward(self, x, key_present: torch.Tensor | None = None):
        b, p, _ = x.shape
        query, key, value = self.qkv(x).view(b, p, 3, self.n_heads, -1).unbind(2)
        if self.rope is not None:
            query, key = self.rope(query), self.rope(key)
        query, key, value = (t.transpose(1, 2) for t in (query, key, value))
        attention_mask = key_present[:, None, None, :] if key_present is not None else None
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=self.is_causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x.transpose(1, 2).flatten(2)
        return F.dropout(self.out(x), self.dropout, self.training)


class SeriesPatchEncoder(nn.Module):
    """Each series is cut in patches; a patch (values + mask) is projected to latent_dim."""

    def __init__(self, latent_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(2 * patch_size, latent_dim)

    def forward(self, x, xm):
        """(B, T, F), (B, T, F) -> (B, F, N, D)"""
        x_patches = rearrange(x, 'b (n p) f -> b f n p', p=self.patch_size)
        xm_patches = rearrange(xm, 'b (n p) f -> b f n p', p=self.patch_size).to(x.dtype)
        return self.proj(torch.cat([x_patches, xm_patches], dim=-1))


class PatchDecoder(nn.Module):
    """Each patch embedding, concatenated with the next one, is projected back to patch_size time steps."""

    def __init__(self, latent_dim: int, out_emb: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Sequential(nn.LayerNorm(2 * latent_dim), nn.Linear(2 * latent_dim, out_emb * patch_size),
                                     nn.GELU())

    def forward(self, x):
        """(B, N, D) -> (B, N*patch_size, out_emb)"""
        x = F.pad(x, pad=(0, 0, 0, 1, 0, 0), mode='constant', value=0)
        x = torch.cat([x[:, :-1, :], x[:, 1:, :]], dim=-1)
        x = self.project(x)
        return rearrange(x, 'b t (p e) -> b (t p) e', p=self.patch_size)


class TransformerLayer(nn.Module):
    """Time attention (per series, over patches), group attention (per patch, over series), MLP."""

    def __init__(self, latent_dim: int, n_heads: int, n_patches: int, dropout: float = 0.0, attn_bias: bool = True,
                 mlp_layers: int = 2, norm_eps: float = 1e-5, group_gate_init: float = -3.0, causal: bool = False):
        super().__init__()
        self.time_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.time_attention = SelfAttention(latent_dim, n_heads, bias=attn_bias, dropout=dropout, is_causal=causal,
                                            rope=RotaryPositionalEmbedding(dim=latent_dim // n_heads,
                                                                           max_seq_len=n_patches))
        self.group_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        ## never causal: feature order carries no causal structure
        self.group_attention = SelfAttention(latent_dim, n_heads, bias=attn_bias, dropout=dropout, is_causal=False)
        ## sigmoid(-3) ~ 0.05: group mixing starts (almost) off
        self.group_gate = nn.Parameter(torch.tensor(float(group_gate_init)))
        self.mlp_norm = nn.LayerNorm(latent_dim, eps=norm_eps)
        self.mlp = MLP(embed_dim=latent_dim, n_layers=mlp_layers, dropout=dropout)

    def forward(self, x, key_present):
        """x: (B, F, N, D); key_present: (B*N, F) bool, True where the series has an observed value in the patch."""
        b, f, n, _ = x.shape
        t = rearrange(x, 'b f n e -> (b f) n e')
        t = t + self.time_attention(self.time_norm(t))
        g = rearrange(t, '(b f) n e -> (b n) f e', b=b, f=f)
        g = g + torch.sigmoid(self.group_gate) * self.group_attention(self.group_norm(g), key_present=key_present)
        x = rearrange(g, '(b n) f e -> b f n e', b=b, n=n)
        return x + self.mlp(self.mlp_norm(x))
