import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x



class PreBlock(torch.nn.Module):
    """
    Preprocessing module. It is designed to replace filtering and baseline correction.

    Args:
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
    """
    def __init__(self, sampling_point):
        super().__init__()
        self.pool1 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.pool2 = torch.nn.AvgPool1d(kernel_size=13, stride=1, padding=6)
        self.pool3 = torch.nn.AvgPool1d(kernel_size=7, stride=1, padding=3)
        self.ln_0 = torch.nn.LayerNorm(sampling_point)
        self.ln_1 = torch.nn.LayerNorm(sampling_point)

    def forward(self, x):
        x0 = x[:, 0, :, :]
        x1 = x[:, 1, :, :]

        x0 = x0.squeeze()
        x0 = self.pool1(x0)
        x0 = self.pool2(x0)
        x0 = self.pool3(x0)
        x0 = self.ln_0(x0)
        x0 = x0.unsqueeze(dim=1)

        x1 = x1.squeeze()
        x1 = self.pool1(x1)
        x1 = self.pool2(x1)
        x1 = self.pool3(x1)
        x1 = self.ln_1(x1)
        x1 = x1.unsqueeze(dim=1)

        x = torch.cat((x0, x1), 1)

        return x


import torch
import torch.nn as nn
from einops import rearrange, repeat
import math


class fNIRS_T(nn.Module):
    """
    fNIRS-T model (支持输入形状 [B, channels, samples])

    输入约定：
        前 20 个通道为 HbO
        后 20 个通道为 HbR
    """

    def __init__(self, channels, sampling_point, n_class, dim=64, depth=6, heads=8, mlp_dim=64,
                 pool='cls', dim_head=64, dropout=0.5, emb_dropout=0.):
        super().__init__()

        # 验证通道数
        assert channels % 2 == 0, "channels 必须是偶数（前一半为HbO，后一半为HbR）"
        self.hbo_channels = channels // 2
        self.hbr_channels = channels // 2
        self.sampling_point = sampling_point
        self.pool = pool

        # Patch embedding 分支（时间维卷积）
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(5, 30), stride=(1, 4)),
            Rearrange('b c h w -> b h (c w)'),
            nn.Linear((math.floor((sampling_point - 30) / 4) + 1) * 8, dim),
            nn.LayerNorm(dim)
        )

        # Channel embedding 分支（仅通道维）
        self.to_channel_embedding = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 30), stride=(1, 4)),
            Rearrange('b c h w -> b h (c w)'),
            nn.Linear((math.floor((sampling_point - 30) / 4) + 1) * 8, dim),
            nn.LayerNorm(dim)
        )

        # Transformer 块（假设你已有 Transformer 类定义）
        num_patches = self.hbo_channels  # 每个通道一个patch
        num_channels = self.hbo_channels

        self.pos_embedding_patch = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token_patch = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_patch = nn.Dropout(emb_dropout)
        self.transformer_patch = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pos_embedding_channel = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_channel = nn.Dropout(emb_dropout)
        self.transformer_channel = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 输出层
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, n_class)
        )

    def forward(self, x, mask=None):
        """
        x: [B, channels, samples]
        """
        # === 自动拆分 HbO / HbR ===
        hbo = x[:, :self.hbo_channels, :]  # 前20通道
        hbr = x[:, self.hbo_channels:, :]  # 后20通道

        # 拼成 [B, 2, channels/2, samples]
        x_in = torch.stack([hbo, hbr], dim=1)

        # Patch-level 特征提取
        x_patch = self.to_patch_embedding(x_in)
        b, n, _ = x_patch.shape
        cls_tokens = repeat(self.cls_token_patch, '() n d -> b n d', b=b)
        x_patch = torch.cat((cls_tokens, x_patch), dim=1)
        x_patch += self.pos_embedding_patch[:, :(n + 1)]
        x_patch = self.dropout_patch(x_patch)
        x_patch = self.transformer_patch(x_patch, mask)

        # Channel-level 特征提取
        x_channel = self.to_channel_embedding(x_in)
        b, n, _ = x_channel.shape
        cls_tokens = repeat(self.cls_token_channel, '() n d -> b n d', b=b)
        x_channel = torch.cat((cls_tokens, x_channel), dim=1)
        x_channel += self.pos_embedding_channel[:, :(n + 1)]
        x_channel = self.dropout_channel(x_channel)
        x_channel = self.transformer_channel(x_channel, mask)

        # 池化
        x_patch = x_patch.mean(dim=1) if self.pool == 'mean' else x_patch[:, 0]
        x_channel = x_channel.mean(dim=1) if self.pool == 'mean' else x_channel[:, 0]

        # 拼接输出
        x_out = torch.cat((x_patch, x_channel), dim=1)
        return self.mlp_head(x_out)


class fNIRS_PreT(nn.Module):
    """
    fNIRS-PreT model

    Args:
        n_class: number of classes.
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
        dim: last dimension of output tensor after linear transformation.
        depth: number of Transformer blocks.
        heads: number of the multi-head self-attention.
        mlp_dim: dimension of the MLP layer.
        pool: MLP layer classification mode, 'cls' is [CLS] token pooling, 'mean' is  average pooling, default='cls'.
        dim_head: dimension of the multi-head self-attention, default=64.
        dropout: dropout rate, default=0.
        emb_dropout: dropout for patch embeddings, default=0.
    """
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.pre = PreBlock(sampling_point)
        self.fNIRS_T = fNIRS_T(n_class, sampling_point, dim, depth, heads, mlp_dim, pool, dim_head, dropout, emb_dropout)

    def forward(self, img):
        img = self.pre(img)
        x = self.fNIRS_T(img)
        return x
