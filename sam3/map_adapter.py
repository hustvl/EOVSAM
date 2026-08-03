import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
import torch.utils.checkpoint as cp
from .convnext import ConvNextBlock
from einops import rearrange

@SEM_SEG_HEADS_REGISTRY.register()
class MAP_ADAPTER(nn.Module):

    @configurable
    def __init__(
        self,
        feat_dim: int,
        mask_in_chans: int,
        num_channels: int,
        use_checkpoint: bool,
        num_output_maps: int,
        num_attn_maps: int = 1,
        chunk_size: int = 0,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.chunk_size = chunk_size
        self.num_output_maps = num_output_maps

        self.fuse = nn.Conv2d(feat_dim, num_channels, 1)

        self.cnext1 = ConvNextBlock(num_channels)
        self.cnext2 = ConvNextBlock(num_channels)
        self.cnext3 = ConvNextBlock(num_channels)

        self.norm = nn.LayerNorm(num_channels)
        self.final = nn.Conv2d(num_channels, num_output_maps, 1)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(num_attn_maps, mask_in_chans // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, feat_dim, kernel_size=1),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "feat_dim": cfg.MODEL.MAP_ADAPTER.FEAT_DIM,
            "num_attn_maps": cfg.MODEL.MAP_ADAPTER.NUM_ATTN_MAPS,
            "mask_in_chans": cfg.MODEL.MAP_ADAPTER.MASK_IN_CHANNELS,
            "num_channels": cfg.MODEL.MAP_ADAPTER.NUM_CHANNELS,
            "use_checkpoint": cfg.MODEL.MAP_ADAPTER.USE_CHECKPOINT,
            "num_output_maps": cfg.MODEL.MAP_ADAPTER.NUM_OUTPUT_MAPS,
            "chunk_size": getattr(cfg.MODEL.MAP_ADAPTER, "CHUNK_SIZE", 0),
        }

    def forward(self, img_feature: torch.Tensor, masks: List[torch.Tensor]) -> List[torch.Tensor]:

        B = len(masks)
        H_feat, W_feat = img_feature.shape[-2:]
        batch_outputs = []

        for b in range(B):
            mask_b = masks[b] # (N_i, K, H_m, W_m)
            N_i = mask_b.size(0)


            if N_i == 0:

                empty_out = torch.empty(
                    (1, 0, H_feat, W_feat),
                    dtype=img_feature.dtype,
                    device=img_feature.device
                )
                batch_outputs.append(empty_out)
                continue


            img_feat_b = img_feature[b:b+1]

            chunk_size = self.chunk_size if (self.chunk_size > 0 and self.chunk_size < N_i) else N_i
            final_outputs_b = []


            for i in range(0, N_i, chunk_size):
                m_chunk = mask_b[i:i+chunk_size] # (N_c, K, H_m, W_m)

                # --- Checkpoint 1 ---
                def _pre_forward(m_c, i_f):

                    m = F.interpolate(m_c.float(), size=(H_feat*4, W_feat*4), mode='bilinear', align_corners=False)


                    m = self.mask_downscaling(m)

                    return self.fuse(m + i_f)

                if self.use_checkpoint and self.training:
                    pre_out = cp.checkpoint(_pre_forward, m_chunk, img_feat_b, use_reentrant=False)
                else:
                    pre_out = _pre_forward(m_chunk, img_feat_b)

                # --- Checkpoint 2 ---
                def _inner_forward(x):
                    x = self.cnext1(x)
                    x = self.cnext2(x)
                    x = self.cnext3(x)

                    x = x.permute(0, 2, 3, 1)
                    x = self.norm(x.contiguous())
                    x = x.permute(0, 3, 1, 2)

                    x = self.final(x.contiguous()) # (N_c, num_output_maps, H_feat, W_feat)


                    x = rearrange(x, 'N_c C H W -> 1 (N_c C) H W')
                    return x

                if self.use_checkpoint and self.training:
                    chunk_out = cp.checkpoint(_inner_forward, pre_out, use_reentrant=False)
                else:
                    chunk_out = _inner_forward(pre_out)

                final_outputs_b.append(chunk_out)


            out_b = torch.cat(final_outputs_b, dim=1)
            batch_outputs.append(out_b)

        return batch_outputs


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def build_map_adapter(cfg, name):
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg)
