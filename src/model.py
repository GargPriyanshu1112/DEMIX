import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass, field

from utils import MoEStats, Outputs
from moe import MoEConfig, IdentityMoELayer, MoELayer

@dataclass
class DiffusionUNetConfig:
    name: str = "ddpm"
    in_c: int = 3
    out_c: int = 3
    init_c: int = 128
    chls_mult_factor: List[int] = field(default_factory=lambda: [1, 2, 2, 4])
    has_attn: List[bool] = field(default_factory=lambda: [False, False, False, True])
    num_res_blocks: int = 2
    t_emb_dim: int = 512
    dropout: float = 0.1
    num_attn_heads: int = 4
    n_classes: int = 0 # > 0 will use classifer free guidance


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings (from Transformer paper).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, t_emb_dim, dropout=0.1, use_moe=False, moe_cfg=None):
        super().__init__()
        self.t_emb_proj = nn.Linear(t_emb_dim, out_c)

        self.g_norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

        self.g_norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.moe_layer = MoELayer(out_c, moe_cfg) if use_moe else IdentityMoELayer()

        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        # Time embedding
        t = F.silu(t_emb)
        t = self.t_emb_proj(t)[:, :, None, None] # [B, out_c, 1, 1]

        h = self.g_norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + t
        h, moe_stats = self.moe_layer(h)
        h = self.g_norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = h + self.shortcut(x)
        return h, moe_stats


# For global 'spatial-mixing' across feature-map positions.
class AttentionBlock(nn.Module):
    def __init__(self, num_c, num_heads=4, attn_dropout=0.0):
        super().__init__()
        self.g_norm = nn.GroupNorm(8, num_c)
        self.mha = nn.MultiheadAttention(
            embed_dim=num_c, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.shortcut = nn.Conv2d(num_c, num_c, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.g_norm(x)
        out = out.view(b, c, h*w).permute(0, 2, 1) # [B, HW, C]
        out, _ = self.mha(query=out, key=out, value=out) # [B, HW, C]
        out = out.permute(0, 2, 1).view(b, c, h, w) # [B, C, H, W]
        out = x + self.shortcut(out) # [B, C, H, W]
        return out, MoEStats.empty_like(x)


class Downsample(nn.Module):
    def __init__(self, num_c):
        super().__init__()
        self.conv = nn.Conv2d(
            num_c, num_c, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x), MoEStats.empty_like(x)


class Upsample(nn.Module):
    def __init__(self, num_c):
        super().__init__()
        self.conv = nn.Conv2d(
            num_c, num_c, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x), MoEStats.empty_like(x)


class DiffusionUNet(nn.Module):
    def __init__(self, img_size: int, cfg: DiffusionUNetConfig, moe_cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.moe_cfg = moe_cfg
        self.t_embed = nn.Sequential(
            nn.Linear(cfg.init_c, cfg.t_emb_dim),
            nn.SiLU(),
            nn.Linear(cfg.t_emb_dim, cfg.t_emb_dim)
        )
        self.conv_in = nn.Conv2d(cfg.in_c, cfg.init_c, kernel_size=3, padding=1)

        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.middle_block = nn.ModuleList()

        if cfg.n_classes:
            self.label_emb = nn.Embedding(cfg.n_classes, cfg.t_emb_dim)

        curr_res, curr_c = img_size, cfg.init_c
        in_chls = [curr_c]

        # Downsampling layers
        for i, mult_factor in enumerate(cfg.chls_mult_factor):
            out_c = cfg.init_c * mult_factor
            for block_idx in range(cfg.num_res_blocks):
                use_moe = self.use_moe("downblock", curr_res, block_idx)
                block = [ResidualBlock(curr_c, out_c, cfg.t_emb_dim, cfg.dropout, use_moe, moe_cfg)]
                if cfg.has_attn[i]:
                    block.append(AttentionBlock(out_c, cfg.num_attn_heads))
                self.down_blocks.append(nn.ModuleList(block))
                curr_c = out_c
                in_chls.append(curr_c)

            if i != len(cfg.chls_mult_factor) - 1: # don't downsample last layer
                self.down_blocks.append(nn.ModuleList([Downsample(curr_c)]))
                in_chls.append(curr_c)
                curr_res //= 2

        # Middle block
        self.middle_block.extend([
            ResidualBlock(curr_c, curr_c, cfg.t_emb_dim, cfg.dropout),
            AttentionBlock(curr_c, cfg.num_attn_heads),
            ResidualBlock(curr_c, curr_c, cfg.t_emb_dim, cfg.dropout)
        ])

        # Upsampling layers
        for i, mult_factor in reversed(list(enumerate(cfg.chls_mult_factor))):
            out_c = cfg.init_c * mult_factor
            for block_idx in range(cfg.num_res_blocks + 1):
                skip_c = in_chls.pop()
                use_moe = self.use_moe("upblock", curr_res, block_idx)
                block = [ResidualBlock(curr_c + skip_c, out_c, cfg.t_emb_dim, cfg.dropout, use_moe, moe_cfg)]
                if cfg.has_attn[i]:
                    block.append(AttentionBlock(out_c))
                if i and block_idx == cfg.num_res_blocks:
                    block.append(Upsample(out_c))
                    curr_res *= 2
                self.up_blocks.append(nn.ModuleList(block))
                curr_c = out_c

        # Final layers
        self.out_norm = nn.GroupNorm(8, curr_c)
        self.conv_out = nn.Conv2d(curr_c, cfg.out_c, 3, padding=1)

    def forward(self, x, t, y=None):
        moe_routing_info = []
        aux_loss, z_loss, scale_reg = x.new_zeros(()), x.new_zeros(()), x.new_zeros(())

        t_emb = self.t_embed(
            get_timestep_embedding(t, self.cfg.init_c)
        )
        if y is not None:
            t_emb += self.label_emb(y)

        h = self.conv_in(x)
        skips = [h]

        for block in self.down_blocks:
            for layer in block:
                h, moe_stats = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)
                aux_loss += moe_stats.aux_loss
                z_loss += moe_stats.z_loss
                scale_reg += moe_stats.scale_reg
                if moe_stats.routing is not None:
                    moe_routing_info.append(moe_stats.routing)
            skips.append(h)

        for layer in self.middle_block:
            h, moe_stats = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)
            aux_loss += moe_stats.aux_loss
            z_loss += moe_stats.z_loss
            scale_reg += moe_stats.scale_reg
            if moe_stats.routing is not None:
                moe_routing_info.append(moe_stats.routing)

        for block in self.up_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for layer in block:
                h, moe_stats = layer(h, t_emb) if isinstance(layer, ResidualBlock) else layer(h)
                aux_loss += moe_stats.aux_loss
                z_loss += moe_stats.z_loss
                scale_reg += moe_stats.scale_reg
                if moe_stats.routing is not None:
                    moe_routing_info.append(moe_stats.routing)

        output = self.conv_out(F.silu(self.out_norm(h)))
        return Outputs(
            pred_noise=output, aux_loss=aux_loss, z_loss=z_loss, scale_reg=scale_reg, moe_routing_info=moe_routing_info
        )

    def use_moe(self, block_type, resolution, block_idx):
        if self.moe_cfg.enable:
            block_cfg = getattr(self.moe_cfg.placement, block_type)
            if (resolution in block_cfg.resolutions) and (block_idx in block_cfg.blocks):
                return True
        return False
