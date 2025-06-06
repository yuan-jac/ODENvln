"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x, attn_mask=None):
        attn_mask_ = None
        if attn_mask is not None:
            attn_mask_ = torch.zeros(attn_mask.shape[0], attn_mask.shape[-1]).to(attn_mask.device).to(x.dtype)
            attn_mask_[attn_mask == False] = float('-inf')
            attn_mask_ = attn_mask_.unsqueeze(1).unsqueeze(-2).repeat(1, 12, attn_mask.shape[-1], 1).view(-1,
                                                                                                          attn_mask.shape[
                                                                                                              -1],
                                                                                                          attn_mask.shape[
                                                                                                              -1])

        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x):
        x, attn_mask = x
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x, attn_mask=None):
        return self.resblocks((x, attn_mask))[0]

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()
        self.input_resolution = input_resolution

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

    def forward(self, x, attn_mask=None):
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

class CLIP(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int):
        super().__init__()

        self.visual = VisionTransformer(input_resolution, patch_size, width, layers, heads)

    def forward(self, x, attn_mask=None):
        return self.visual(x, attn_mask)
