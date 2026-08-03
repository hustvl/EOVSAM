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
from einops import rearrange,repeat

def load_mask_adapter_standalone(
    weight_path,
    clip_model_name='_large',
    mask_in_chans=8,
    num_channels=256,
    num_output_maps=1,
    use_checkpoint=False
):


    adapter = MASKAdapterHead(
        clip_model_name=clip_model_name,
        mask_in_chans=mask_in_chans,
        num_channels=num_channels,
        use_checkpoint=use_checkpoint,
        num_output_maps=num_output_maps
    )


    print(f"Loading weights from {weight_path}...")
    checkpoint = torch.load(weight_path, map_location="cpu")


    if "model" in checkpoint:
        full_state_dict = checkpoint["model"]
    else:
        full_state_dict = checkpoint


    adapter_state_dict = {}
    prefix = "mask_adapter."
    for k, v in full_state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            adapter_state_dict[new_key] = v


    if not adapter_state_dict:

        print("Warning: No 'mask_adapter.' prefix found, trying to load direct state_dict...")
        adapter_state_dict = full_state_dict

    msg = adapter.load_state_dict(adapter_state_dict, strict=True)
    print(f"Successfully loaded MaskAdapter with result: {msg}")

    return adapter


@SEM_SEG_HEADS_REGISTRY.register()
class MASKAdapterHead(nn.Module):

    @configurable
    def __init__(
        self,
        feat_dim: int,
        mask_in_chans: int,
        num_channels: int,
        use_checkpoint: bool,
        num_output_maps: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint

        clip_dim = feat_dim

        self.fuse = nn.Conv2d(clip_dim, num_channels, 1)

        self.cnext1 = ConvNextBlock(num_channels)

        self.cnext2 = ConvNextBlock(num_channels)

        self.cnext3 = ConvNextBlock(num_channels)

        self.norm = nn.LayerNorm(num_channels)
        self.final = nn.Conv2d(num_channels, num_output_maps, 1)

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, clip_dim, kernel_size=1),
        )


    @classmethod
    def from_config(cls, cfg):

        return {
            "feat_dim": cfg.MODEL.MASK_ADAPTER.FEAT_DIM,
            "mask_in_chans": cfg.MODEL.MASK_ADAPTER.MASK_IN_CHANNELS,
            "num_channels": cfg.MODEL.MASK_ADAPTER.NUM_CHANNELS,
            "use_checkpoint": cfg.MODEL.MASK_ADAPTER.USE_CHECKPOINT,
            "num_output_maps": cfg.MODEL.MASK_ADAPTER.NUM_OUTPUT_MAPS,
        }

    def forward(self, clip_feature, masks):


        N = masks.size(1)
        masks = rearrange(masks, 'B N H W -> (B N) H W').unsqueeze(dim=1)

        clip_feature = repeat(clip_feature, "B C H W -> (B N) C H W", N=N)

        H,W = clip_feature.shape[-2:]
        masks = F.interpolate(masks.float(), size=(H*4,W*4),
                                                mode='bilinear', align_corners=False)
        masks = self.mask_downscaling(masks)

        outputs = clip_feature + masks

        def _inner_forward(outputs):
            outputs = self.fuse(outputs)

            outputs = self.cnext1(outputs)

            outputs = self.cnext2(outputs)

            outputs = self.cnext3(outputs)

            outputs = outputs.permute(0, 2, 3, 1)
            outputs = self.norm(outputs.contiguous())
            outputs = outputs.permute(0, 3, 1, 2)

            outputs = self.final(outputs.contiguous())

            outputs = rearrange(outputs, '(B N) C H W -> B (N C) H W',N=N)

            return outputs

        if self.use_checkpoint and self.training:
            outputs = cp.checkpoint(_inner_forward, outputs,use_reentrant=False)
        else:
            outputs = _inner_forward(outputs)
        return outputs

def build_mask_adapter(cfg,name):
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg)

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
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