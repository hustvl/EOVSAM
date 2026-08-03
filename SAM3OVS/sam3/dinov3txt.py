import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils import comm

import dataclasses
import math
import warnings
from typing import Callable
import os


import numpy as np

import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
import tqdm
from omegaconf import OmegaConf
from torch import Tensor, nn

DINOv3_REPO_DIR = "/data/hmp/dinov3" # Please add here the path to your DINOv3 repository
import sys
sys.path.append(DINOv3_REPO_DIR)
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

@BACKBONE_REGISTRY.register()
class DINOv3TXT(Backbone):
    def __init__(self, ):
        super().__init__()
        model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l()
        model.to("cuda", non_blocking=True)
        model.eval()
        tokenizer = tokenizer.tokenize

        self.model = model
        self.tokenizer = tokenizer
        self.mode = "whole"

        patch_size = model.visual_model.backbone.patch_size
        output_dim = model.visual_model.backbone.embed_dim

        
        self._out_features = ["clip_vis_dense"]
        self._out_feature_strides = {"clip_vis_dense": patch_size}
        self._out_feature_channels = {"clip_vis_dense": output_dim}
        
        self.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def dim_latent(self):
        return self.model.model_config.embed_dim
    
    @property
    def clip_model(self):
        return self.model

    def tokenize_text(self, text_list):
        tokens = self.tokenizer(text_list).to("cuda", non_blocking=True)
        return tokens

    def get_text_classifier(self, text_list, device):
        ""
        tokens = self.tokenize_text(text_list).to(device)
        feats = self.model.encode_text(tokens)
        feats = feats[:, feats.shape[1] // 2 :]  # The 1st half of the features corresponds to the CLS token, drop it
        feats = F.normalize(feats, p=2, dim=-1)  # Normalize each text embedding
        
        # feats = feats.mean(dim=0)  # Average over all prompt embeddings per class
        # feats = F.normalize(feats, p=2, dim=-1)  # Normalize again
        
        return feats
    
    def forward(self, x):
        return self.extract_features(x)

    def extract_features(self, img):
        if self.mode == "whole":
            _, blocks_feats = self.encode_image(img) 
            blocks_feats = blocks_feats.permute(0, 3, 1, 2).contiguous()  # [B, D, h, w]
            return {"clip_vis_dense": blocks_feats}

        elif self.mode == "slide":
            pass
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        
    
    def encode_image(self,img: Tensor):
        """Extract image features from the backbone and the additional blocks."""
        B, _, H, W = img.shape
        model = self.model
        P = model.visual_model.backbone.patch_size # In the case of our DINOv3
        new_H = math.ceil(H / P) * P
        new_W = math.ceil(W / P) * P

        # Stretch image to a multiple of patch size
        if (H, W) != (new_H, new_W):
            img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)  # [B, 3, H', W']

        B, _, h_i, w_i = img.shape

        backbone_patches = None
        cls_tokens, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)
        blocks_patches = (
            patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
        ) # [1, h, w, D]

        return backbone_patches, blocks_patches
