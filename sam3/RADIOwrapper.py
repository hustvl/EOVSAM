"""
Utility functions for replacing SAM3's vision encoder with RADIO.

This module provides functionality to:
1. Load RADIO models from torch.hub
2. Replace SAM3's vision encoder with RADIO
3. Create appropriate Sam3Processor with correct image sizes for RADIO
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from timm.layers.helpers import to_2tuple

from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_RADIO_RESOLUTION = 1008 * 16 // 14  # RADIO uses 16x16 patches vs SAM3's 14x14


class RADIO_Adaptor(nn.Module):
    """Replaces SAM3's ViT trunk with student features."""

    def __init__(self,
                 student: nn.Module,
                 input_size: int,
                 output_channels: int,
    ):
        super().__init__()

        self.student = student
        self.input_size = to_2tuple(input_size)

        # SAM3's ViT has a channel_list attribute that the neck uses
        self.channel_list = [output_channels]

        self.sig2_adaptor = self.student.adaptors['siglip2-g']

    @torch.no_grad()
    def get_text_classifier(self, text_list, device):
        text_input = self.sig2_adaptor.tokenizer(text_list).to(device)
        text_tokens = self.sig2_adaptor.encode_text(text_input, normalize=True)
        return text_tokens

    # @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        Forward pass that mimics ViT trunk output.

        Args:
            images: Input tensor [B, C, H, W]

        Returns:
            List of feature tensors in [B, C, H, W] format (ViT output format)
        """
        # Normalize inputs to [0, 1]
        images = (images + 1) / 2

        # if images.shape[-2:] != self.input_size:
        #     images = F.interpolate(images, self.input_size, mode='bilinear', align_corners=False)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            student_output = self.student(images)
            if isinstance(student_output, dict):
                features = student_output['sam3'][1]
                dino_features = student_output['dino_v3_7b'][1]
            _, backbone_features = student_output['backbone']
            sig2_vis_features = self.sig2_adaptor.head_mlp(backbone_features)
            del student_output
            del backbone_features


        patch_size = int(round(math.sqrt(images.shape[-2] * images.shape[-1] / features.shape[1])))

        rows = images.shape[-2] // patch_size
        cols = images.shape[-1] // patch_size

        # Reshape from [B, N, C] to [B, C, H, W] to match ViT output format
        features = rearrange(features, 'b (r c) d -> b d r c', r=rows, c=cols)

        sig2_vis_features = rearrange(sig2_vis_features, 'b (r c) d -> b d r c', r=rows, c=cols)

        dino_features = rearrange(dino_features, 'b (r c) d -> b d r c', r=rows, c=cols)

        other_output = {
            "siglip2-g": {
                "features": sig2_vis_features.float(),
            },
            "dino_v3_7b":{
                "features": dino_features.float(),
            }
        }
        # Return as a list (ViT returns list of outputs from global attention blocks)
        return [other_output, features.float()]


def load_radio_model(
    model_version: str,
    device: str = 'cuda',
    vitdet: Optional[int] = None,
    num_prompts_to_insert: int = 200,
    insert_start_layer: int = 1,
    insert_end_layer: int = -1,
):
    """
    Load RADIO model from torch.hub.

    Args:
        checkpoint_path: Path to the RADIO model checkpoint
        device: Device to load model on

    Returns:
        RADIO model with SAM3 adaptor
    """
    extra = dict()
    if vitdet:
        extra['vitdet_window_size'] = vitdet
    if num_prompts_to_insert is not None:
        extra['num_prompts_to_insert'] = int(num_prompts_to_insert)
    if insert_start_layer is not None:
        extra['insert_start_layer'] = int(insert_start_layer)
    if insert_end_layer is not None:
        extra['insert_end_layer'] = int(insert_end_layer)

    print(f"Loading RADIO model from {model_version}...")
    model: torch.nn.Module = torch.hub.load(
        'NVlabs/RADIO',
        'radio_model',
        model_version,
        adaptor_names=['sam3','siglip2-g','dino_v3_7b'],
        source='local',
        **extra,
    )
    model = model.to(device)
    model.eval()
    print("RADIO model loaded successfully!")
    return model


def set_radio_grad_checkpointing(radio_model: torch.nn.Module, enable: bool = True):
    inner_model = getattr(radio_model, "model", radio_model)
    set_fn = getattr(inner_model, "set_grad_checkpointing", None)
    if callable(set_fn):
        set_fn(enable)
    elif hasattr(inner_model, "grad_checkpointing"):
        inner_model.grad_checkpointing = enable
    else:
        print("Warning: RADIO model does not expose grad checkpointing control.")

    blocks = getattr(inner_model, "blocks", None)
    if hasattr(blocks, "grad_checkpointing"):
        blocks.grad_checkpointing = enable


def replace_sam3_encoder(sam3_model, radio_model, device: str = 'cuda'):
    """
    Replace SAM3's vision encoder with RADIO model.

    Args:
        sam3_model: The SAM3 image model
        radio_model: The RADIO model with SAM3 adaptor
        device: Device for computations

    Returns:
        Modified SAM3 model with RADIO encoder
    """
    print("Replacing SAM3 vision encoder with RADIO...")

    # Get the original SAM3 vision encoder to extract configuration
    original_encoder = sam3_model.backbone.vision_backbone

    sam3_dim = original_encoder.trunk.patch_embed.proj.out_channels

    # Create the adaptor
    input_size = 1152  # SAM3 standard input size
    adaptor = RADIO_Adaptor(
        student=radio_model,
        input_size=input_size,
        output_channels=sam3_dim,
    )

    # Replace the trunk in SAM3's vision encoder
    sam3_model.backbone.vision_backbone.trunk = adaptor

    print("Vision encoder replaced successfully!")
    return sam3_model, adaptor


def create_sam3_radio_processor(sam3_model,
                                confidence_threshold: float = 0.5,
                                resolution: Optional[int] = None) -> Sam3Processor:
    """
    Create a Sam3Processor configured for use with RADIO encoder.

    RADIO uses 16x16 patches while SAM3's default ViT uses 14x14 patches,
    so we need a different resolution for optimal performance.

    Args:
        sam3_model: The SAM3 model (potentially with RADIO encoder)
        confidence_threshold: Confidence threshold for predictions
        resolution: Image resolution to use. If None, uses DEFAULT_RADIO_RESOLUTION (1152)
                   for RADIO-based models. Set to 1008 for original SAM3 encoder.

    Returns:
        Configured Sam3Processor instance
    """
    if resolution is None:
        resolution = DEFAULT_RADIO_RESOLUTION

    print(f"Creating Sam3Processor with resolution={resolution}")
    processor = Sam3Processor(
        sam3_model,
        resolution=resolution,
        confidence_threshold=confidence_threshold
    )

    return processor