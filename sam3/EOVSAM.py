from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import shutil
import cv2

import os
import numpy as np
import torchvision
import logging
import copy

from sam3.model_builder import (
    _create_vision_backbone,
    _create_text_encoder,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
)
from sam3.model.data_misc import FindStage, interpolate

from sam3.model.model_misc import (
    gen_sineembed_for_position,
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    MLP,
)

from maft.utils.text_templetes import VILD_PROMPT


from .loss.matcher import HungarianMatcher
from .loss.criterion import SetCriterion
from .loss.fcclip_criterion import FcclipSetCriterion
from .loss.fcclip_map_criterion import MapAdapterFcclipSetCriterion
from .loss.fcclip_matcher import FcclipHungarianMatcher
from sam3.model.content_dependent_transfer import ContentDependentTransfer
from sam3.model.box_ops import masks_to_boxes, box_xyxy_to_cxcywh


from maft.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling

import random

import math

from .mask_adapter_head import build_mask_adapter
from .map_adapter import build_map_adapter
from .convnext import ConvNextBlock


from.RADIOwrapper import replace_sam3_encoder, load_radio_model, set_radio_grad_checkpointing

@META_ARCH_REGISTRY.register()
class EOVSAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device_type = cfg.MODEL.DEVICE
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)


        compile_mode = "default" if cfg.MODEL.SAM3.COMPILE else None

        vision_encoder = _create_vision_backbone(
            compile_mode=compile_mode,
            enable_inst_interactivity=cfg.MODEL.SAM3.ENABLE_INST_INTERACTIVITY
        )
        text_encoder = _create_text_encoder(cfg.MODEL.SAM3.BPE_PATH)
        backbone = _create_vl_backbone(vision_encoder, text_encoder)
        transformer = _create_sam3_transformer(
            use_gate = cfg.MODEL.SAM3.USE_GATE,
        )
        dot_prod_scoring = _create_dot_product_scoring()

        segmentation_head = (
            _create_segmentation_head(compile_mode=compile_mode)
            if cfg.MODEL.SAM3.ENABLE_SEGMENTATION
            else None
        )

        input_geometry_encoder = _create_geometry_encoder()

        enable_inst_interactivity = False
        if enable_inst_interactivity:
            sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
            inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
        else:
            inst_predictor = None

        self.detector = _create_sam3_model(
            backbone,
            transformer,
            input_geometry_encoder,
            segmentation_head,
            dot_prod_scoring,
            inst_predictor,
            cfg.eval_only,
        )
        if cfg.eval_only:
            self.detector.eval()
        print("SAM3 created successfully!")

        radio_model = load_radio_model(
            "c-radio_v4-h",
            device=self.pixel_mean.device,
            vitdet=None,
            num_prompts_to_insert=cfg.MODEL.VPT.NUM_PROMPTS_TO_INSERT,
            insert_start_layer=cfg.MODEL.VPT.START_LAYER,
            insert_end_layer=cfg.MODEL.VPT.END_LAYER,
        )
        self.detector, self.radio_adaptor = replace_sam3_encoder(self.detector, radio_model, device=self.pixel_mean.device)


        self.use_pe_text = cfg.MODEL.SAM3.USE_PE_TEXT

        self.use_cos_sim = getattr(cfg.MODEL.SAM3, "COS_SIM", False)

        self.out_vocab_logit_scale = None


        self.use_query_proj = cfg.MODEL.SAM3.USE_QUERY_PROJ
        if self.use_query_proj:
            self.query_proj = MLP(256, 256, 1024, 3)
        else:
            self.query_proj = None

        self.use_pixel_proj = cfg.MODEL.SAM3.USE_PIXEL_PROJ
        if self.use_pixel_proj:
            self.pixel_proj = nn.Linear(256, 256)
        else:
            self.pixel_proj = None

        self.num_decoder_layers = len(self.detector.transformer.decoder.layers)
        self.num_decoder_cross_attn_heads = (
            self.detector.transformer.decoder.layers[0].cross_attn.num_heads
        )
        self.cross_attn_head_pool_logits = nn.Parameter(
            torch.zeros(
                self.num_decoder_layers,
                self.num_decoder_cross_attn_heads,
            )
        )
        self.num_cdt = cfg.MODEL.SAM3.NUM_CDT
        self.use_cdt = False if cfg.MODEL.SAM3.NUM_CDT == 0 else True
        if self.use_cdt:
            if self.use_pe_text:
                print("The CDT module currently supports only 256-dimensional text features. Please disable PE text features.")
            text_classifier_dim = 1536
            self.cdt = ContentDependentTransfer(d_model=text_classifier_dim, nhead=8, panoptic_on=False)

            self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.cdt = None


        self.DynamicQuery = cfg.MODEL.SAM3.DYNAMIC_QUERY
        if self.DynamicQuery:
            self.encoder_box_head = MLP(256, 256, 4, 3)
            nn.init.constant_(self.encoder_box_head.layers[-1].weight.data, 0)
            nn.init.constant_(self.encoder_box_head.layers[-1].bias.data, 0)

        self.mask_pooling = MaskPooling()

        self.use_MaskAdapter = cfg.MODEL.USE_MASKADAPTER
        if self.use_MaskAdapter:
            from .mask_adapter_head import load_mask_adapter_standalone
            self.mask_adapter = build_mask_adapter(cfg, cfg.MODEL.MASK_ADAPTER.NAME)
            self.num_output_maps = cfg.MODEL.MASK_ADAPTER.NUM_OUTPUT_MAPS
            self.iou_threshold = cfg.MODEL.MASK_ADAPTER.IOU_THRESHOLD
            self.mask_threshold = cfg.MODEL.MASK_ADAPTER.MASK_THRESHOLD
            self.num_gt_masks = cfg.MODEL.MASK_ADAPTER.NUM_GT_MASKS
            self.num_pred_masks = cfg.MODEL.MASK_ADAPTER.NUM_PRED_MASKS

            self.mask_adapter_weight_dict = {"loss_ce": cfg.MODEL.MASK_ADAPTER.CLASS_WEIGHT, "loss_cosine": cfg.MODEL.MASK_ADAPTER.COS_WEIGHT}
            self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.use_map_adapter = cfg.MODEL.MAP_ADAPTER.ENABLE
        self.map_adapter_train_only = getattr(cfg.MODEL.MAP_ADAPTER, "TRAIN_ONLY", False)
        if self.use_map_adapter:
            self.map_adapter = build_map_adapter(cfg, cfg.MODEL.MAP_ADAPTER.NAME)
            self.map_num_output_maps = cfg.MODEL.MAP_ADAPTER.NUM_OUTPUT_MAPS
            self.map_mask_threshold = cfg.MODEL.MASK_ADAPTER.MASK_THRESHOLD
        else:
            self.map_adapter = None
            self.map_num_output_maps = 0
            self.map_mask_threshold = cfg.MODEL.MASK_ADAPTER.MASK_THRESHOLD


        self.use_attnpool = cfg.MODEL.ATTNPOOL.ENABLE
        if self.use_attnpool:
            self.attnpool_weight_dict = {
                "loss_cls": cfg.MODEL.ATTNPOOL.CLASS_WEIGHT,
                'loss_bbox':cfg.SOLVER.BBOX_WEIGHT,
                'loss_giou':cfg.SOLVER.GIOU_WEIGHT,
            }
            self.num_gt_masks = cfg.MODEL.ATTNPOOL.NUM_GT_MASKS
            self.iou_threshold = cfg.MODEL.ATTNPOOL.IOU_THRESHOLD
            self.num_pred_masks = cfg.MODEL.ATTNPOOL.NUM_PRED_MASKS
            self.aux_attn_pool = cfg.MODEL.ATTNPOOL.USE_AUX

            if self.aux_attn_pool:
                attnpool_weight_dict_aux = {}
                for i in range (5):
                    for k in self.attnpool_weight_dict.keys():
                        attnpool_weight_dict_aux[f"{k}_{i}"] = self.attnpool_weight_dict[k]
                self.attnpool_weight_dict.update(attnpool_weight_dict_aux)
            self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.attn_pool_logit_scale = nn.Parameter(torch.ones([]) * np.log(1))


        self.new_score_head = cfg.MODEL.SAM3.NEW_SCORE_HEAD

        self.use_dot_prod_head = cfg.MODEL.SAM3.USE_DOT_PROD_HEAD

        self.vpt_enable = cfg.MODEL.VPT.ENABLE
        self.vpt_grad_checkpoint = getattr(cfg.MODEL.VPT, "GRAD_CHECKPOINT", False)
        radio_blocks = getattr(getattr(self.radio_adaptor, "student", None), "model", None)
        radio_blocks = getattr(radio_blocks, "blocks", None)
        if hasattr(radio_blocks, "is_active"):
            radio_blocks.is_active = self.vpt_enable
        if self.vpt_enable and self.vpt_grad_checkpoint:
            set_radio_grad_checkpointing(self.radio_adaptor.student, True)
        elif hasattr(radio_blocks, "grad_checkpointing"):
            radio_blocks.grad_checkpointing = False


        self.PROMPT = VILD_PROMPT
        self.SAM_PROMPT = ['{}']

        self.add_pixelfeat = cfg.MODEL.SAM3.ADD_PIXELFEAT
        self.alpha = cfg.MODEL.SAM3.ALPHA
        self.beta = cfg.MODEL.SAM3.BETA
        self.OracleSelect_on = cfg.MODEL.SAM3.ORACLE_SELECT


        self.train_dataname = None
        self.test_dataname = None
        self.SAM_train_dataname = None
        self.SAM_test_dataname = None
        self.text_encoder_cache = {}
        self.SAM_text_encoder_cache = {}


        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata_dict = {name: MetadataCatalog.get(name) for name in cfg.DATASETS.TRAIN}


        if len(cfg.DATASETS.TRAIN) > 0:
            self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        else:
            self.train_metadata = None

        self.train_num_templates = None
        self.train_class_names = None

        self.use_aux = cfg.SOLVER.USE_AUX
        self.only_instance = cfg.DATASETS.ONLY_INSTANCE
        if self.only_instance:
            raise NotImplementedError("only_instance has not been adapted for prepare_targets_for_maskadapter")

        self.train_mask = cfg.SOLVER.TRAIN_MASK
        if self.map_adapter_train_only:
            if not self.use_map_adapter:
                raise ValueError("MODEL.MAP_ADAPTER.TRAIN_ONLY=True requires MODEL.MAP_ADAPTER.ENABLE=True")
            if not cfg.SOLVER.TRAIN_OUT_VOCAB:
                raise ValueError("MODEL.MAP_ADAPTER.TRAIN_ONLY=True requires SOLVER.TRAIN_OUT_VOCAB=True")
            self.train_mask = False


        losses = ["labels", "masks", "boxes"]


        class_weight = cfg.SOLVER.CLASS_WEIGHT
        dice_weight = cfg.SOLVER.DICE_WEIGHT
        mask_weight = cfg.SOLVER.MASK_WEIGHT
        bbox_weight = cfg.SOLVER.BBOX_WEIGHT
        giou_weight = cfg.SOLVER.GIOU_WEIGHT

        objectness_weight = cfg.SOLVER.OBJECT_WEIGHT

        self.use_softmax = cfg.MODEL.SAM3.USE_SOFTMAX
        self.void_embedding = None
        if self.use_softmax:
            self.void_embedding = nn.Embedding(1, 256)


        self.logit_bias = None
        if not self.use_softmax:
            prior_prob = 0.01
            bias_value = -np.log((1 - prior_prob) / prior_prob)
            self.logit_bias = nn.Parameter(torch.ones([]) * bias_value)

        if self.use_cos_sim:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        weight_dict = {}
        criterion_weight_dict = {
            "loss_cls": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            'loss_bbox':bbox_weight,
            'loss_giou':giou_weight,

        }
        if self.new_score_head:
            criterion_weight_dict["loss_objectness"] = objectness_weight
        weight_dict.update(criterion_weight_dict)

        if self.use_aux:
            for i in range (5):
                for k in criterion_weight_dict.keys():
                    weight_dict[f"{k}_{i}"] = criterion_weight_dict[k]


        if self.use_softmax:
            no_object_weight = 0.1

            matcher = FcclipHungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                cost_bbox=bbox_weight,
                cost_giou=giou_weight,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            )
            self.criterion = FcclipSetCriterion(
                num_classes = 133,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
            )

        else:
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            )


            self.criterion = SetCriterion(
                matcher=matcher,
                weight_dict=weight_dict,
                losses=losses,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
                tau=cfg.SOLVER.CONTRAST_TEMPERATURE,
            )

        self.train_out_vocab= cfg.SOLVER.TRAIN_OUT_VOCAB
        if self.train_out_vocab:
            self.iou_threshold = cfg.MODEL.MASK_ADAPTER.IOU_THRESHOLD
            self.mask_threshold = 0.50
            self.num_gt_masks = cfg.MODEL.MASK_ADAPTER.NUM_GT_MASKS
            self.num_pred_masks = cfg.MODEL.MASK_ADAPTER.NUM_PRED_MASKS

            self.out_vocab_weight_dict = {

                "loss_attn_cls": cfg.MODEL.ATTNPOOL.CLASS_WEIGHT,
            }
            if self.use_map_adapter and self.use_attnpool:
                self.out_vocab_weight_dict["loss_attn_pool_cls"] = cfg.MODEL.ATTNPOOL.CLASS_WEIGHT
            tp_out_vocab_weight_dict = {}
            for k in self.out_vocab_weight_dict.keys():
                for i in range(5):
                    tp_out_vocab_weight_dict[f"{k}_{i}"] = self.out_vocab_weight_dict[k]
            self.out_vocab_weight_dict.update(tp_out_vocab_weight_dict)

            self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            no_object_weight = 0.0

            matcher = FcclipHungarianMatcher(
                cost_class=0,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                cost_bbox=bbox_weight,
                cost_giou=giou_weight,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            )

            losses = ["labels", ]
            out_vocab_criterion_cls = MapAdapterFcclipSetCriterion if self.use_map_adapter else FcclipSetCriterion
            self.out_vocab_criterion = out_vocab_criterion_cls(
                num_classes=133,
                matcher=matcher,
                weight_dict=self.out_vocab_weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
            )


        self.semantic_on = cfg.TEST.SEMANTIC_ON
        self.instance_on = cfg.TEST.INSTANCE_ON
        self.panoptic_on = cfg.TEST.PANOPTIC_ON


        self.object_mask_threshold = 0.01
        self.overlap_threshold = 0.8
        self.test_topk_per_image = 100

        self._freeze()

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "map_adapter_train_only", False):
            for name, module in self.named_children():
                if name == "map_adapter":
                    module.train(True)
                else:
                    module.eval()
        return self

    def _freeze(self):

        logger = logging.getLogger("detectron2")

        if getattr(self, "map_adapter_train_only", False):
            trainable_count = 0
            for name, param in self.named_parameters():
                param.requires_grad = name.startswith("map_adapter.")
                if param.requires_grad:
                    logger.info(f"TRAINABLE: {name}")
                    trainable_count += 1
                else:
                    logger.info(f"FROZEN: {name}")
            logger.info(f"MapAdapter train-only mode enabled. Total trainable parameter groups: {trainable_count}")
            logger.info("="*40)
            return


        freeze_keywords = [
            'geometry_encoder',
            'language_backbone',
            'text_model',
            'sig2_adaptor',
            "student"
        ]


        train_keywords = [
            "tuner",
        ]


        for name, param in self.named_parameters():

            match_freeze = any(key in name for key in freeze_keywords)
            match_train = any(key in name for key in train_keywords)

            if match_freeze and match_train:

                logger.warning(f"OVERRIDE: '{name}' matches both freeze and train keywords. Force setting to TRAINABLE.")
                param.requires_grad = True
            elif match_freeze:

                param.requires_grad = False
            else:

                param.requires_grad = True


        trainable_count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:

                logger.info(f"TRAINABLE: {name}")
                trainable_count += 1
            else:
                logger.info(f"FROZEN: {name}")

        logger.info(f"Total trainable parameter groups: {trainable_count}")
        logger.info('='*40)


    def prepare_targets(self, targets, images, batched_inputs):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []


        for i, targets_per_image in enumerate(targets):

            dataname = get_dataname(batched_inputs[i])


            gt_classes = targets_per_image.gt_classes
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor


            keep_indices = torch.ones(len(gt_classes), dtype=torch.bool, device=gt_classes.device)


            if self.only_instance:

                is_coco_stuff = "openvocab_coco_2017_train_stuff_sem_seg" in dataname

                if is_coco_stuff:

                    keep_indices = gt_classes >= 80


                    gt_classes = gt_classes[keep_indices]
                    gt_masks = gt_masks[keep_indices]


                    gt_classes = gt_classes - 80


                else:

                    gt_classes = gt_classes[keep_indices]
                    gt_masks = gt_masks[keep_indices]


            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)

            if gt_masks.shape[0] > 0:
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks


            if padded_masks.shape[0] > 0:
                gt_boxes_xyxy = masks_to_boxes(padded_masks)


                scale = torch.tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float32, device=gt_boxes_xyxy.device)
                gt_boxes_norm = gt_boxes_xyxy / scale


                gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes_norm)
            else:

                gt_boxes_cxcywh = torch.zeros((0, 4), device=padded_masks.device)

            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": padded_masks,
                    "boxes": gt_boxes_cxcywh,
                }
            )
        return new_targets

    def prepare_targets_for_maskadapter(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        masks_list = []
        labels_list = []

        num_masks = self.num_gt_masks

        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor
            valid_mask_indices = [i for i, mask in enumerate(gt_masks) if mask.sum() > 0]

            if len(valid_mask_indices) > 0:
                valid_gt_masks = gt_masks[valid_mask_indices]
                valid_gt_classes = targets_per_image.gt_classes[valid_mask_indices]

                padded_masks = torch.zeros((valid_gt_masks.shape[0], h_pad, w_pad), dtype=valid_gt_masks.dtype, device=valid_gt_masks.device)
                padded_masks[:, :valid_gt_masks.shape[1], :valid_gt_masks.shape[2]] = valid_gt_masks
                new_targets.append(
                    {
                        "labels": valid_gt_classes,
                        "masks": padded_masks,
                    }
                )

                total_masks = torch.zeros((num_masks, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                selected_labels = torch.full((num_masks,), -1, dtype=valid_gt_classes.dtype, device=gt_masks.device)

                if valid_gt_masks.shape[0] > num_masks:
                    selected_indices = torch.randperm(valid_gt_masks.shape[0])[:num_masks]
                    for idx, mask_idx in enumerate(selected_indices):
                        total_masks[idx, :valid_gt_masks[mask_idx].shape[0], :valid_gt_masks[mask_idx].shape[1]] = valid_gt_masks[mask_idx]
                        selected_labels[idx] = valid_gt_classes[mask_idx]
                else:
                    for idx in range(valid_gt_masks.shape[0]):
                        total_masks[idx, :valid_gt_masks[idx].shape[0], :valid_gt_masks[idx].shape[1]] = valid_gt_masks[idx]
                        selected_labels[idx] = valid_gt_classes[idx]

                    for idx in range(valid_gt_masks.shape[0], num_masks):
                        total_masks[idx] = torch.zeros((h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                        selected_labels[idx] = -1
            else:
                total_masks = torch.zeros((num_masks, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                selected_labels = torch.full((num_masks,), -1, dtype=torch.long, device=gt_masks.device)

                padded_masks = torch.zeros((0, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                valid_gt_classes = torch.zeros((0), device=gt_masks.device)
                new_targets.append(
                    {
                        "labels": valid_gt_classes,
                        "masks": padded_masks,
                    }
                )

            masks_list.append(total_masks)
            labels_list.append(selected_labels)

        masks = torch.stack(masks_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        labels = labels.long()

        return new_targets, masks, labels

    def _build_query_map_inputs(self, attn_weights, pred_masks, query_indices_per_image, feat_hw):
        if attn_weights is None:
            raise ValueError("attn_weights is required for map-adapter pooling")
        if attn_weights.dim() != 4:
            raise ValueError(
                f"Expected per-layer multi-head attention maps with shape [B, H, Q, L], got {tuple(attn_weights.shape)}"
            )

        bs, num_heads, num_queries, num_tokens = attn_weights.shape
        feat_h, feat_w = feat_hw
        if num_tokens != feat_h * feat_w:
            raise ValueError(
                f"Attention token count {num_tokens} does not match feature size {feat_h}x{feat_w}"
            )

        mask_h, mask_w = pred_masks.shape[-2:]
        attn_maps = attn_weights.view(bs, num_heads, num_queries, feat_h, feat_w)
        attn_maps = attn_maps.permute(0, 2, 1, 3, 4).reshape(bs * num_queries, num_heads, feat_h, feat_w)
        attn_maps = F.interpolate(attn_maps.float(), size=(mask_h, mask_w), mode="bilinear", align_corners=False)
        attn_maps = attn_maps.view(bs, num_queries, num_heads, mask_h, mask_w)

        binary_masks = (pred_masks.sigmoid() > self.map_mask_threshold).unsqueeze(2).float()
        query_map_inputs = torch.cat([attn_maps, binary_masks], dim=2)

        return [query_map_inputs[b, query_indices] for b, query_indices in enumerate(query_indices_per_image)]

    def _pool_map_adapter_features(self, map_src_feature, pool_feature, query_map_inputs):
        map_adapter_outputs = self.map_adapter(map_src_feature, query_map_inputs)
        pooled_query_features = []
        for b, maps_for_pooling in enumerate(map_adapter_outputs):
            num_queries = query_map_inputs[b].shape[0]
            if num_queries == 0:
                pooled_query_features.append(pool_feature.new_empty((0, pool_feature.shape[1])))
                continue

            maps_for_pooling = F.interpolate(
                maps_for_pooling,
                size=pool_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            num_maps = maps_for_pooling.size(1)
            weights = F.softmax(F.logsigmoid(maps_for_pooling).view(1, num_maps, -1), dim=-1)
            pooled = torch.bmm(weights, pool_feature[b:b+1].flatten(2).transpose(1, 2))
            pooled = pooled.view(1, num_queries, self.map_num_output_maps, -1).mean(dim=2).squeeze(0).contiguous()
            pooled_query_features.append(pooled)
        return pooled_query_features

    def _compute_map_adapter_cls_logits(self, attn_weights, pred_masks, map_src_feature, pool_feature, text_classifier, num_templates, query_indices_per_image=None):
        bs, _, num_queries, _ = attn_weights.shape
        if query_indices_per_image is None:
            query_indices_per_image = [
                torch.arange(num_queries, device=pred_masks.device, dtype=torch.long)
                for _ in range(bs)
            ]

        query_map_inputs = self._build_query_map_inputs(
            attn_weights,
            pred_masks,
            query_indices_per_image,
            map_src_feature.shape[-2:],
        )
        pooled_query_features = self._pool_map_adapter_features(map_src_feature, pool_feature, query_map_inputs)

        num_classes = len(num_templates)
        pooled_logits = pred_masks.new_zeros((bs, num_queries, num_classes))
        for b, query_indices in enumerate(query_indices_per_image):
            if len(query_indices) == 0:
                continue
            cur_text_classifier = text_classifier[b:b + 1] if text_classifier.dim() == 3 else text_classifier
            cur_logits = get_classification_logits(
                pooled_query_features[b].unsqueeze(0),
                cur_text_classifier,
                self.out_vocab_logit_scale,
                num_templates,
            ).squeeze(0)
            pooled_logits[b, query_indices] = cur_logits

        return pooled_logits

    def _get_cross_attn_head_pool_weights(self, num_layers=None, device=None, dtype=None):
        head_pool_weights = F.softmax(self.cross_attn_head_pool_logits, dim=-1)
        if num_layers is not None:
            head_pool_weights = head_pool_weights[:num_layers]
        if device is not None or dtype is not None:
            head_pool_weights = head_pool_weights.to(
                device=device if device is not None else head_pool_weights.device,
                dtype=dtype if dtype is not None else head_pool_weights.dtype,
            )
        return head_pool_weights

    def _aggregate_cross_attn_weights(self, cross_attn_weights):
        if cross_attn_weights is None:
            raise ValueError("cross_attn_weights is required for attention pooling")
        if cross_attn_weights.dim() == 5:
            head_pool_weights = self._get_cross_attn_head_pool_weights(
                num_layers=cross_attn_weights.shape[0],
                device=cross_attn_weights.device,
                dtype=cross_attn_weights.dtype,
            )
            return torch.einsum("kh,kbhnl->kbnl", head_pool_weights, cross_attn_weights)
        if cross_attn_weights.dim() == 4:
            head_pool_weights = self._get_cross_attn_head_pool_weights(
                device=cross_attn_weights.device,
                dtype=cross_attn_weights.dtype,
            )[-1]
            return torch.einsum("h,bhnl->bnl", head_pool_weights, cross_attn_weights)
        raise ValueError(
            f"Expected cross-attention weights with shape [K, B, H, Q, L] or [B, H, Q, L], got {tuple(cross_attn_weights.shape)}"
        )

    def _compute_attn_pool_cls_logits(self, cross_attn_weights, pool_feature, text_classifier, num_templates):
        pooled_cross_attn_weights = self._aggregate_cross_attn_weights(cross_attn_weights)
        if pooled_cross_attn_weights.dim() == 3:
            pooled_cross_attn_weights = pooled_cross_attn_weights.unsqueeze(0)

        bs = pool_feature.shape[0]
        num_queries = pooled_cross_attn_weights.shape[2]
        radio_img_feat = pool_feature.view(bs, pool_feature.shape[1], -1).permute(0, 2, 1)

        attn_pool_logit_scale = torch.clamp(self.attn_pool_logit_scale.exp(), max=100)
        pooled_cross_attn_weights = (
            torch.log(pooled_cross_attn_weights + 1e-6) * attn_pool_logit_scale
        ).softmax(-1)
        pooled_img_feat = torch.einsum("bld,kbnl->kbnd", radio_img_feat, pooled_cross_attn_weights)

        if text_classifier.dim() == 3:
            attn_cls_results = []
            for b in range(bs):
                attn_cls_result = get_classification_logits(
                    pooled_img_feat[:, b, :, :],
                    text_classifier[b],
                    self.out_vocab_logit_scale,
                    num_templates,
                )
                attn_cls_results.append(attn_cls_result)
            return torch.stack(attn_cls_results, dim=1)

        attn_cls_results = get_classification_logits(
            pooled_img_feat.reshape(-1, num_queries, pooled_img_feat.shape[-1]),
            text_classifier,
            self.out_vocab_logit_scale,
            num_templates,
        )
        return attn_cls_results.view(
            pooled_img_feat.shape[0],
            bs,
            num_queries,
            attn_cls_results.shape[-1],
        )

    def _compute_matched_attn_pool_loss(self, matcher_outputs, attn_cls_logits, targets):
        indices = self.out_vocab_criterion.matcher(matcher_outputs, targets)
        src_logits = matcher_outputs["pred_logits"].float()
        current_num_classes = src_logits.shape[-1] - 1
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if target_classes_o.numel() == 0:
            return src_logits.sum() * 0.0

        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        matched_logits = attn_cls_logits[batch_idx, src_idx]
        empty_weight = torch.ones(current_num_classes, device=src_logits.device)
        return F.cross_entropy(matched_logits.float(), target_classes_o, empty_weight)

    def _compute_attn_pool_supervision_losses(self, criterion_pred, targets, attn_cls_results):
        if attn_cls_results is None:
            return {}

        losses = {}
        outputs_without_aux = {k: v for k, v in criterion_pred.items() if k != "aux_outputs"}
        losses["loss_attn_pool_cls"] = self._compute_matched_attn_pool_loss(
            outputs_without_aux,
            attn_cls_results[-1],
            targets,
        )

        aux_outputs = criterion_pred.get("aux_outputs") or []
        for i, aux_output in enumerate(aux_outputs):
            losses[f"loss_attn_pool_cls_{i}"] = self._compute_matched_attn_pool_loss(
                aux_output,
                attn_cls_results[i],
                targets,
            )
        return losses

    def prepare_class_names_from_metadata(self, metadata, train_metadata, prompt_list):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',')
                res.append(x_)
            return res


        try:
            class_names = split_labels(metadata.stuff_classes)
        except AttributeError:
            class_names = split_labels(metadata.thing_classes)


        try:
            train_class_names = split_labels(train_metadata.stuff_classes)
        except AttributeError:
            train_class_names = split_labels(train_metadata.thing_classes)


        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        self.vis_class_names = class_names
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in prompt_list:
                    res.append(template.format(x))
            return res, len(res) // len(prompt_list)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names


    def _compress_language_features(self, language_features, language_mask):
        valid_mask = ~language_mask
        valid_lengths = valid_mask.sum(dim=1)
        max_valid_len = int(valid_lengths.max().item()) if valid_lengths.numel() > 0 else 0

        compressed_features = language_features[:, :max_valid_len, :].new_zeros(
            language_features.shape[0],
            max_valid_len,
            language_features.shape[-1],
        )
        compressed_mask = torch.ones(
            language_mask.shape[0],
            max_valid_len,
            dtype=torch.bool,
            device=language_mask.device,
        )

        for idx in range(language_features.shape[0]):
            curr_len = int(valid_lengths[idx].item())
            if curr_len == 0:
                continue
            compressed_features[idx, :curr_len] = language_features[idx, valid_mask[idx]]
            compressed_mask[idx, :curr_len] = False

        return compressed_features, compressed_mask

    def get_text_classifier(self, dataname):
        if self.training:
            if getattr(self, 'test_text_classifier', None) is not None:
                self.test_text_classifier = None
                torch.cuda.empty_cache()

            if self.train_dataname != dataname or getattr(self, 'train_text_classifier', None) is None:
                if dataname in self.text_encoder_cache:
                    cache = self.text_encoder_cache[dataname]
                    self.train_text_classifier = cache["text_classifier"].to(self.device)
                    self.train_num_templates = cache["num_templates"]
                    self.train_class_names = cache["class_names"]
                else:
                    if dataname in self.train_metadata_dict:
                        current_metadata = self.train_metadata_dict[dataname]
                    else:
                        current_metadata = MetadataCatalog.get(dataname)

                    _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(
                        current_metadata, current_metadata, self.PROMPT
                    )

                    is_coco_stuff = "openvocab_coco_2017_train_stuff_sem_seg" in dataname
                    if self.only_instance and is_coco_stuff:
                        num_things_classes = 80
                        if len(self.train_num_templates) > num_things_classes:
                            print(f"[{dataname}] Filtering out {num_things_classes} Thing classes for Stuff training.")
                            num_synonyms_to_skip = sum(self.train_num_templates[:num_things_classes])
                            offset_text_idx = num_synonyms_to_skip * len(self.PROMPT)
                            self.train_class_names = self.train_class_names[offset_text_idx:]
                            self.train_num_templates = self.train_num_templates[num_things_classes:]

                    text_classifier = []
                    bs = 128
                    print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                    for idx in range(0, len(self.train_class_names), bs):
                        batch_text_feat = self.radio_adaptor.get_text_classifier(
                            self.train_class_names[idx:idx+bs],
                            device=self.device,
                        )
                        text_classifier.append(batch_text_feat)

                    text_classifier = torch.cat(text_classifier, dim=0)
                    text_classifier = text_classifier.reshape(
                        text_classifier.shape[0] // len(self.PROMPT),
                        len(self.PROMPT),
                        -1,
                        text_classifier.shape[-1],
                    )
                    print("text_classifier:", text_classifier.shape)
                    text_classifier = text_classifier.mean(-2)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                    self.train_text_classifier = text_classifier.detach()
                    self.text_encoder_cache[dataname] = {
                        "text_classifier": self.train_text_classifier.cpu(),
                        "num_templates": self.train_num_templates,
                        "class_names": self.train_class_names,
                    }

                self.train_dataname = dataname
            return self.train_text_classifier, self.train_num_templates

        else:
            if getattr(self, 'train_text_classifier', None) is not None:
                self.train_text_classifier = None
                torch.cuda.empty_cache()

            if self.test_dataname != dataname or getattr(self, 'test_text_classifier', None) is None:
                if dataname in self.text_encoder_cache:
                    cache = self.text_encoder_cache[dataname]
                    self.test_text_classifier = cache["text_classifier"].to(self.device)
                    self.test_num_templates = cache["num_templates"]
                    self.test_class_names = cache["class_names"]
                    self.category_overlapping_mask = cache["category_overlapping_mask"]
                else:
                    self.category_overlapping_mask, self.test_num_templates, self.test_class_names = (
                        self.prepare_class_names_from_metadata(
                            self.test_metadata[dataname], self.train_metadata, self.PROMPT
                        )
                    )
                    text_classifier = []
                    bs = 128
                    print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                    for idx in range(0, len(self.test_class_names), bs):
                        batch_text_feat = self.radio_adaptor.get_text_classifier(
                            self.test_class_names[idx:idx+bs],
                            device=self.device,
                        )
                        text_classifier.append(batch_text_feat)

                    text_classifier = torch.cat(text_classifier, dim=0)
                    text_classifier = text_classifier.reshape(
                        text_classifier.shape[0] // len(self.PROMPT),
                        len(self.PROMPT),
                        -1,
                        text_classifier.shape[-1],
                    )
                    text_classifier = text_classifier.mean(-2)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                    self.test_text_classifier = text_classifier.detach()
                    self.text_encoder_cache[dataname] = {
                        "text_classifier": self.test_text_classifier.cpu(),
                        "num_templates": self.test_num_templates,
                        "class_names": self.test_class_names,
                        "category_overlapping_mask": self.category_overlapping_mask,
                    }

                self.test_dataname = dataname
            return self.test_text_classifier, self.test_num_templates


    def get_SAM_text_classifier(self, dataname):
        if self.training:
            if getattr(self, 'SAM_test_text_classifier', None) is not None:
                self.SAM_test_text_classifier = None
                torch.cuda.empty_cache()

            if self.SAM_train_dataname != dataname or getattr(self, 'SAM_train_text_classifier', None) is None:
                if (
                    dataname in self.SAM_text_encoder_cache
                    and "language_features" in self.SAM_text_encoder_cache[dataname]
                    and "language_mask" in self.SAM_text_encoder_cache[dataname]
                ):
                    cache = self.SAM_text_encoder_cache[dataname]
                    self.SAM_language_features, self.SAM_language_mask = self._compress_language_features(
                        cache["language_features"].to(self.device),
                        cache["language_mask"].to(self.device),
                    )
                    self.SAM_train_text_classifier = cache["text_classifier"].to(self.device)
                    self.SAM_train_num_templates = cache["num_templates"]
                    self.SAM_train_class_names = cache["class_names"]
                else:
                    if dataname in self.train_metadata_dict:
                        current_metadata = self.train_metadata_dict[dataname]
                    else:
                        current_metadata = MetadataCatalog.get(dataname)

                    _, self.SAM_train_num_templates, self.SAM_train_class_names = (
                        self.prepare_class_names_from_metadata(
                            current_metadata, current_metadata, self.SAM_PROMPT
                        )
                    )

                    is_coco_stuff = "openvocab_coco_2017_train_stuff_sem_seg" in dataname
                    if self.only_instance and is_coco_stuff:
                        num_things_classes = 80
                        if len(self.SAM_train_num_templates) > num_things_classes:
                            print(f"[{dataname}] Filtering out {num_things_classes} Thing classes for Stuff training.")
                            num_synonyms_to_skip = sum(self.SAM_train_num_templates[:num_things_classes])
                            offset_text_idx = num_synonyms_to_skip * len(self.SAM_PROMPT)
                            self.SAM_train_class_names = self.SAM_train_class_names[offset_text_idx:]
                            self.SAM_train_num_templates = self.SAM_train_num_templates[num_things_classes:]

                    text_classifier = []
                    text_feat = []
                    language_mask = []
                    bs = 128
                    print("Generating text classifier for", dataname, "with", len(self.SAM_train_class_names), "classes.")
                    for idx in range(0, len(self.SAM_train_class_names), bs):
                        state_text = self.detector.backbone.forward_text(
                            self.SAM_train_class_names[idx:idx+bs],
                            device=self.device,
                        )

                        batch_text_feat = state_text["language_features"].detach()
                        mask = state_text["language_mask"]
                        batch_text_feat = batch_text_feat.permute(1, 0, 2)
                        if self.use_pe_text:
                            text_classifier.append(state_text["pe_text_out"]["pe_textfeat"])
                        else:
                            text_classifier.append(batch_text_feat)
                        text_feat.append(batch_text_feat)
                        language_mask.append(mask)

                    text_classifier = torch.cat(text_classifier, dim=0)
                    text_feat = torch.cat(text_feat, dim=0)
                    language_mask = torch.cat(language_mask, dim=0)

                    text_feat = text_feat.reshape(
                        text_feat.shape[0] // len(self.SAM_PROMPT),
                        len(self.SAM_PROMPT),
                        text_feat.shape[-2],
                        text_feat.shape[-1],
                    )
                    text_feat /= (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    text_feat[
                        language_mask.view(
                            text_feat.shape[0],
                            text_feat.shape[1],
                            text_feat.shape[2],
                        )
                    ] = 0.0
                    language_features = text_feat.mean(1)

                    text_classifier = text_classifier.reshape(
                        text_classifier.shape[0] // len(self.SAM_PROMPT),
                        len(self.SAM_PROMPT),
                        text_classifier.shape[-2],
                        text_classifier.shape[-1],
                    )
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier[
                        language_mask.view(
                            text_classifier.shape[0],
                            text_classifier.shape[1],
                            text_classifier.shape[2],
                        )
                    ] = 0.0
                    text_classifier = text_classifier.mean(-2)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                    sam_language_mask = torch.min(
                        language_mask.view(
                            language_features.shape[0],
                            len(self.SAM_PROMPT),
                            language_features.shape[1],
                        ),
                        dim=1,
                    ).values
                    self.SAM_language_features, self.SAM_language_mask = self._compress_language_features(
                        language_features.detach(),
                        sam_language_mask,
                    )
                    self.SAM_train_text_classifier = text_classifier.detach()
                    self.SAM_text_encoder_cache[dataname] = {
                        "language_features": self.SAM_language_features.cpu(),
                        "language_mask": self.SAM_language_mask.cpu(),
                        "text_classifier": self.SAM_train_text_classifier.cpu(),
                        "num_templates": self.SAM_train_num_templates,
                        "class_names": self.SAM_train_class_names,
                    }

                self.SAM_train_dataname = dataname
            return self.SAM_train_text_classifier, self.SAM_train_num_templates

        else:
            if getattr(self, 'SAM_train_text_classifier', None) is not None:
                self.SAM_train_text_classifier = None
                torch.cuda.empty_cache()

            if self.SAM_test_dataname != dataname or getattr(self, 'SAM_test_text_classifier', None) is None:
                if (
                    dataname in self.SAM_text_encoder_cache
                    and "language_features" in self.SAM_text_encoder_cache[dataname]
                    and "language_mask" in self.SAM_text_encoder_cache[dataname]
                ):
                    cache = self.SAM_text_encoder_cache[dataname]
                    self.SAM_language_features, self.SAM_language_mask = self._compress_language_features(
                        cache["language_features"].to(self.device),
                        cache["language_mask"].to(self.device),
                    )
                    self.SAM_test_text_classifier = cache["text_classifier"].to(self.device)
                    self.SAM_test_num_templates = cache["num_templates"]
                    self.SAM_test_class_names = cache["class_names"]
                    if "category_overlapping_mask" in cache:
                        self.SAM_category_overlapping_mask = cache["category_overlapping_mask"]
                    else:
                        self.SAM_category_overlapping_mask, _, _ = self.prepare_class_names_from_metadata(
                            self.test_metadata[dataname], self.train_metadata, self.SAM_PROMPT
                        )
                else:
                    self.SAM_category_overlapping_mask, self.SAM_test_num_templates, self.SAM_test_class_names = (
                        self.prepare_class_names_from_metadata(
                            self.test_metadata[dataname], self.train_metadata, self.SAM_PROMPT
                        )
                    )
                    text_classifier = []
                    text_feat = []
                    language_mask = []
                    bs = 128
                    print("Generating text classifier for", dataname, "with", len(self.SAM_test_class_names), "classes.")
                    for idx in range(0, len(self.SAM_test_class_names), bs):
                        state_text = self.detector.backbone.forward_text(
                            self.SAM_test_class_names[idx:idx+bs],
                            device=self.device,
                        )

                        batch_text_feat = state_text["language_features"].detach()
                        mask = state_text["language_mask"]
                        batch_text_feat = batch_text_feat.permute(1, 0, 2)
                        if self.use_pe_text:
                            text_classifier.append(state_text["pe_text_out"]["pe_textfeat"])
                        else:
                            text_classifier.append(batch_text_feat)
                        text_feat.append(batch_text_feat)
                        language_mask.append(mask)

                    text_classifier = torch.cat(text_classifier, dim=0)
                    text_feat = torch.cat(text_feat, dim=0)
                    language_mask = torch.cat(language_mask, dim=0)

                    text_feat = text_feat.reshape(
                        text_feat.shape[0] // len(self.SAM_PROMPT),
                        len(self.SAM_PROMPT),
                        text_feat.shape[-2],
                        text_feat.shape[-1],
                    )
                    text_feat /= (text_feat.norm(dim=-1, keepdim=True) + 1e-6)
                    text_feat[
                        language_mask.view(
                            text_feat.shape[0],
                            text_feat.shape[1],
                            text_feat.shape[2],
                        )
                    ] = 0.0
                    language_features = text_feat.mean(1)

                    text_classifier = text_classifier.reshape(
                        text_classifier.shape[0] // len(self.SAM_PROMPT),
                        len(self.SAM_PROMPT),
                        text_classifier.shape[-2],
                        text_classifier.shape[-1],
                    )
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier[
                        language_mask.view(
                            text_classifier.shape[0],
                            text_classifier.shape[1],
                            text_classifier.shape[2],
                        )
                    ] = 0.0
                    text_classifier = text_classifier.mean(-2)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                    text_classifier = text_classifier.mean(1)
                    text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                    sam_language_mask = torch.min(
                        language_mask.view(
                            language_features.shape[0],
                            len(self.SAM_PROMPT),
                            language_features.shape[1],
                        ),
                        dim=1,
                    ).values
                    self.SAM_language_features, self.SAM_language_mask = self._compress_language_features(
                        language_features.detach(),
                        sam_language_mask,
                    )
                    self.SAM_test_text_classifier = text_classifier.detach()
                    self.SAM_text_encoder_cache[dataname] = {
                        "language_features": self.SAM_language_features.cpu(),
                        "language_mask": self.SAM_language_mask.cpu(),
                        "text_classifier": self.SAM_test_text_classifier.cpu(),
                        "num_templates": self.SAM_test_num_templates,
                        "class_names": self.SAM_test_class_names,
                        "category_overlapping_mask": self.SAM_category_overlapping_mask,
                    }

                self.SAM_test_dataname = dataname
            return self.SAM_test_text_classifier, self.SAM_test_num_templates


    def clear_inference_cache(self):
        attrs_to_clear = [
            "train_text_classifier",
            "test_text_classifier",
            "SAM_train_text_classifier",
            "SAM_test_text_classifier",
            "SAM_language_features",
            "SAM_language_mask",
            "SAM_category_overlapping_mask",
            "category_overlapping_mask",
            "find_stage",
        ]
        for attr_name in attrs_to_clear:
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):

        images = [x["image"].to(self.device) for x in batched_inputs]

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 16)

        img_h, img_w = images.tensor.shape[-2:]

        bs = images.tensor.shape[0]

        self.find_stage = FindStage(
            img_ids=torch.arange(bs, device=self.device, dtype=torch.long),
            text_ids=torch.arange(bs, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        with torch.no_grad():

            file_names = [x["file_name"] for x in batched_inputs]
            file_names = [x.split('/')[-1].split('.')[0] for x in file_names]

            if 'meta' in batched_inputs[0]:
                meta = batched_inputs[0]["meta"]
            else:
                meta = batched_inputs[0]


            dataname = get_dataname(batched_inputs[0])

            SAM_text_classifier, SAM_num_templates = self.get_SAM_text_classifier(dataname)
            if self.void_embedding is not None:
                SAM_text_classifier = torch.cat([SAM_text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)


            geometric_prompt = self.detector._get_dummy_prompt(bs)

        if self.vpt_enable and self.training:
            images.tensor.requires_grad_(True)
            if self.vpt_grad_checkpoint:
                backbone_out_vision = checkpoint(
                    self.detector.backbone.forward_image,
                    images.tensor,
                    use_reentrant=False,
                )
            else:
                backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
        else:
            with torch.no_grad():
                backbone_out_vision = self.detector.backbone.forward_image(images.tensor)

        with torch.no_grad():
            img_feat = backbone_out_vision["vision_features"].detach()
            backbone_fpn = backbone_out_vision["backbone_fpn"]
            for k in range(len(backbone_fpn)):
                backbone_fpn[k] = backbone_fpn[k].detach()

        flat_language_features = self.SAM_language_features.reshape(
            -1, self.SAM_language_features.shape[-1]
        )
        flat_language_mask = self.SAM_language_mask.reshape(-1)
        flat_language_features = flat_language_features[~flat_language_mask]
        language_features_input = flat_language_features.unsqueeze(0).expand(bs, -1, -1)
        language_mask_input = torch.zeros(
            bs,
            language_features_input.shape[1],
            dtype=torch.bool,
            device=self.device,
        )

        language_features_input = language_features_input.reshape(
            bs, -1, language_features_input.shape[-1]
        )
        language_mask_input = language_mask_input.reshape(bs, -1)

        backbone_out = {
            "img_batch_all_stages": img_feat,
            "vision_pos_enc": backbone_out_vision["vision_pos_enc"],
            "backbone_fpn": backbone_fpn,
            "language_features": language_features_input.permute(1, 0, 2),
            "language_mask": language_mask_input,
        }

        enable_mask_grad = self.training and self.train_mask

        with torch.set_grad_enabled(enable_mask_grad):

            find_input = self.find_stage

            with torch.profiler.record_function("SAM3Image._encode_prompt"):
                prompt, prompt_mask, backbone_out = self.detector._encode_prompt(
                    backbone_out, find_input, geometric_prompt
                )

            with torch.profiler.record_function("SAM3Image._run_encoder"):
                backbone_out, encoder_out, _ = self.detector._run_encoder(
                    backbone_out, find_input, prompt, prompt_mask
                )

            fusion_feat = encoder_out["encoder_hidden_states"]
            fusion_feat = fusion_feat.permute(1,0,2)
            if self.use_cos_sim:
                fusion_feat = F.normalize(fusion_feat, dim=-1)


            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": backbone_out,
                },
            }


            with torch.profiler.record_function("SAM3Image._run_decoder"):


                query_embed = self.detector.transformer.decoder.query_embed.weight
                query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

                hs, reference_boxes, dec_presence_out, dec_presence_feats, cross_attn_weights = (
                    self.detector.transformer.decoder(
                        tgt=query_embed,
                        memory=out["encoder_hidden_states"],
                        memory_key_padding_mask=encoder_out["padding_mask"],
                        pos=encoder_out["pos_embed"],
                        reference_boxes=None,
                        level_start_index=encoder_out["level_start_index"],
                        spatial_shapes=encoder_out["spatial_shapes"],
                        valid_ratios=encoder_out["valid_ratios"],
                        tgt_mask=None,
                        memory_text=prompt,
                        text_attention_mask=prompt_mask,
                        apply_dac=False,

                        use_presence_token = False,
                        return_attn_weights=self.training or self.use_attnpool,
                        return_all_attn_weights=self.training,
                    )
                )

                hs = hs.transpose(1, 2)
                reference_boxes = reference_boxes.transpose(1, 2)
                if dec_presence_out is not None:

                    dec_presence_out = dec_presence_out.transpose(1, 2)

                out["presence_feats"] = dec_presence_feats
                self.detector._update_scores_and_boxes(
                    out,
                    hs,
                    reference_boxes,
                    prompt,
                    prompt_mask,
                    dec_presence_out=dec_presence_out,
                )

            if self.use_dot_prod_head:
                dot_prod = self.detector.compute_open_vocab_classification_scores(
                    hs,
                    SAM_text_classifier,
                )


            with torch.profiler.record_function("SAM3Image._run_segmentation_heads"):
                self.detector._run_segmentation_heads(
                    out=out,
                    backbone_out=backbone_out,
                    img_ids=find_input.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                    aux_masks=self.training,
                )


            outputs = out


            if self.training:

                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images, batched_inputs)
                else:
                    targets = None

            N = outputs["pred_masks"].shape[1]


            pred_boxes = outputs['pred_boxes']
            pred_boxes_xyxy = outputs['pred_boxes_xyxy']

            C_ = SAM_text_classifier.shape[0]

            queries = outputs["obj_queries"]


            use_aux = self.use_aux and self.training
            aux_outputs = []

            obj_logits = None
            if self.new_score_head:
                obj_logits = self.score_head(queries).squeeze(-1)

            attn_cls_results = None
            if self.use_attnpool and (not self.map_adapter_train_only) and (self.training or not self.use_map_adapter):
                radio_img_feat = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]
                pool_cross_attn_weights = cross_attn_weights if self.training else cross_attn_weights[-1]

                text_classifier, num_templates = self.get_text_classifier(dataname)
                if self.cdt is not None:
                    text_classifier = self.cdt(
                        backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"].detach(),
                        text_classifier
                    )

                attn_cls_results = self._compute_attn_pool_cls_logits(
                    pool_cross_attn_weights,
                    radio_img_feat,
                    text_classifier,
                    num_templates,
                )

            for i in range(self.num_decoder_layers):
                assert queries.shape[0] == self.num_decoder_layers
                assert queries.shape[2] == N
                if use_aux or i == 5 :
                    if self.use_dot_prod_head:
                        query_names_results = dot_prod[i,:,:,:]
                    else:
                        tp_queries = queries[i]

                        if self.add_pixelfeat:
                            pixel_embed = outputs["pixel_embed"]

                            pooled_pixel_embed = self.mask_pooling(pixel_embed, outputs["pred_masks"] if i ==5 else outputs['aux_outputs'][i]["pred_masks"])

                            if self.use_pixel_proj:
                                pooled_pixel_embed = self.pixel_proj(pooled_pixel_embed)

                            tp_queries = tp_queries + pooled_pixel_embed


                        if self.use_query_proj:
                            tp_queries = self.query_proj(tp_queries)


                        if self.use_cos_sim:
                            tp_queries = F.normalize(tp_queries, dim=-1, p=2)

                        query_names_results = torch.einsum("bnd,cd->bnc", tp_queries, SAM_text_classifier)

                        if self.use_cos_sim:
                            cur_logit_scale = self.logit_scale.exp()
                            cur_logit_scale = torch.clamp(cur_logit_scale, max=100.0)
                            query_names_results = cur_logit_scale * query_names_results
                            if self.logit_bias is not None:
                                cur_logit_bias = self.logit_bias
                                query_names_results = query_names_results + cur_logit_bias
                        else:
                            if self.logit_bias is not None:
                                cur_logit_bias = self.logit_bias
                                query_names_results = query_names_results + self.logit_bias

                    cur_obj_logits = obj_logits[i] if obj_logits is not None else None

                    query_cls_results= []
                    cur_idx = 0

                    tp_num_templates = SAM_num_templates

                    for num_t in tp_num_templates:
                        query_cls_results.append(query_names_results[:,:, cur_idx: cur_idx + num_t].max(-1).values)
                        cur_idx += num_t


                    if self.void_embedding is not None:
                        query_cls_results.append(query_names_results[:,:, -1])
                        query_cls_results = torch.stack(query_cls_results, dim=-1)

                    else:
                        query_cls_results = torch.stack(query_cls_results, dim=-1)

                    if i<5:
                        aux_out = {
                            'pred_logits': query_cls_results,
                            'pred_masks': outputs['aux_outputs'][i]["pred_masks"],
                            'pred_boxes': outputs['aux_outputs'][i]['pred_boxes'],
                            'pred_boxes_xyxy': outputs['aux_outputs'][i]["pred_boxes_xyxy"],
                        }
                        if attn_cls_results is not None and not self.use_map_adapter:
                            aux_out["attn_cls_logits"] = attn_cls_results[i]
                        if self.use_map_adapter:
                            aux_out["map_attn_weights"] = cross_attn_weights[i].detach()
                        if cur_obj_logits is not None:
                            aux_out['pred_objectness_logits'] = cur_obj_logits

                        aux_outputs.append(aux_out)
                    else:
                        query_cls_results_final = query_cls_results
                        obj_logits_final = cur_obj_logits


        if self.training:
            losses = {}

            if self.train_mask and not self.map_adapter_train_only:
                criterion_pred = {
                    'pred_logits': query_cls_results_final,
                    'pred_masks': outputs["pred_masks"],
                    'pred_boxes': outputs['pred_boxes'],
                    'pred_boxes_xyxy': outputs["pred_boxes_xyxy"],

                    'aux_outputs': aux_outputs if use_aux is True else None,
                }
                if obj_logits_final is not None:
                    criterion_pred['pred_objectness_logits'] = obj_logits_final

                fcclip_losses = self.criterion(criterion_pred, targets)


                for k in list(fcclip_losses.keys()):

                    if k in self.criterion.weight_dict:
                        fcclip_losses[k] *= self.criterion.weight_dict[k]
                    else:

                        fcclip_losses.pop(k)

                losses.update(fcclip_losses)


            if self.train_out_vocab:
                criterion_pred = {
                    'pred_logits': query_cls_results_final,
                    'pred_masks': outputs["pred_masks"],
                    'pred_boxes': outputs['pred_boxes'],
                    'pred_boxes_xyxy': outputs["pred_boxes_xyxy"],
                    'aux_outputs': aux_outputs if use_aux is True else None,
                }
                out_vocab_extra_context = None
                if self.use_map_adapter:
                    if self.map_adapter_train_only:
                        with torch.no_grad():
                            text_classifier, num_templates = self.get_text_classifier(dataname)
                            if self.cdt is not None:
                                text_classifier = self.cdt(
                                    backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"],
                                    text_classifier,
                                )
                    else:
                        text_classifier, num_templates = self.get_text_classifier(dataname)
                        if self.cdt is not None:
                            text_classifier = self.cdt(
                                backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"],
                                text_classifier,
                            )
                    criterion_pred["map_attn_weights"] = cross_attn_weights[-1].detach()
                    out_vocab_extra_context = {
                        "use_map_adapter": True,
                        "map_adapter": self.map_adapter,
                        "map_src_feature": backbone_out_vision['vit_feature'][0]["dino_v3_7b"]["features"].detach(),
                        "pool_feature": backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"].detach() if self.map_adapter_train_only else backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"],
                        "text_classifier": text_classifier.detach() if self.map_adapter_train_only else text_classifier,
                        "num_templates": num_templates,
                        "out_vocab_logit_scale": self.out_vocab_logit_scale,
                        "mask_threshold": self.map_mask_threshold,
                        "num_output_maps": self.map_num_output_maps,
                    }
                elif attn_cls_results is not None:
                    criterion_pred["attn_cls_logits"] = attn_cls_results[-1]

                if self.use_map_adapter:
                    out_vocab_losses = self.out_vocab_criterion(criterion_pred, targets, out_vocab_extra_context)
                else:
                    out_vocab_losses = self.out_vocab_criterion(criterion_pred, targets)

                for k in list(out_vocab_losses.keys()):
                    if k in self.out_vocab_weight_dict:
                        out_vocab_losses[k] *= self.out_vocab_weight_dict[k]
                    else:
                        out_vocab_losses.pop(k)

                losses.update(out_vocab_losses)

                if self.use_map_adapter and attn_cls_results is not None and not self.map_adapter_train_only:
                    attn_pool_losses = self._compute_attn_pool_supervision_losses(
                        criterion_pred,
                        targets,
                        attn_cls_results,
                    )
                    for k in list(attn_pool_losses.keys()):
                        if k in self.out_vocab_weight_dict:
                            attn_pool_losses[k] *= self.out_vocab_weight_dict[k]
                        else:
                            attn_pool_losses.pop(k)

                    losses.update(attn_pool_losses)

            all_keys = list(losses.keys())
            aux_suffixes = [f"_{i}" for i in range(5)]
            main_keys = sorted([k for k in all_keys if not any(k.endswith(s) for s in aux_suffixes)])
            aux_keys = sorted([k for k in all_keys if any(k.endswith(s) for s in aux_suffixes)])

            ordered_losses = {}
            for k in main_keys:
                ordered_losses[k] = losses[k]
            for k in aux_keys:
                ordered_losses[k] = losses[k]

            return ordered_losses

        else:

            if self.out_vocab_logit_scale is None:
                self.out_vocab_logit_scale = nn.Parameter(torch.ones([]) * np.log(100))

            if self.use_attnpool:
                if self.use_map_adapter:
                    text_classifier, num_templates = self.get_text_classifier(dataname)
                    if self.cdt is not None:
                        text_classifier = self.cdt(
                            backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"],
                            text_classifier,
                        )
                    pool_cls_logits = self._compute_map_adapter_cls_logits(
                        cross_attn_weights[-1],
                        outputs["pred_masks"],
                        backbone_out_vision['vit_feature'][0]["dino_v3_7b"]["features"],
                        backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"],
                        text_classifier,
                        num_templates,
                    )
                else:
                    pool_cls_logits = attn_cls_results[-1]

            elif self.use_MaskAdapter:
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

                img_feat_for_pool = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]

                binary_masks = mask_pred_results.sigmoid() > self.mask_threshold


                adapter_batchsize = 32
                maps_for_pooling_list = []
                for i in range(0, mask_pred_results.shape[1], adapter_batchsize):
                    batch_binary_masks = binary_masks[:, i:i+adapter_batchsize, :, :]
                    maps_for_pooling_batch = self.mask_adapter(img_feat_for_pool, batch_binary_masks)
                    maps_for_pooling_list.append(maps_for_pooling_batch)
                maps_for_pooling = torch.cat(maps_for_pooling_list, dim=1)

                maps_for_pooling = F.interpolate(maps_for_pooling, size=img_feat_for_pool.shape[-2:],
                                            mode='bilinear', align_corners=False)
                N_maps = maps_for_pooling.size(1)
                num_instances = N_maps // self.num_output_maps
                maps_for_pooling = F.softmax(F.logsigmoid(maps_for_pooling).view(bs, N_maps,-1), dim=-1)
                pooled_img_feature = torch.bmm(maps_for_pooling, img_feat_for_pool.view(bs, img_feat_for_pool.size(1), -1).permute(0, 2, 1))
                pooled_img_feature = (pooled_img_feature.reshape(bs,num_instances, self.num_output_maps, -1).mean(dim=-2).contiguous())
                pooled_img_feat = pooled_img_feature
                pooled_img_feat = F.normalize(pooled_img_feat, dim=-1, p=2)

            else:
                img_feat_for_pool = backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"]
                mask_for_pool = F.interpolate(outputs["pred_masks"], size=img_feat_for_pool.shape[-2:],
                                                    mode='bilinear', align_corners=False)
                pooled_img_feat = self.mask_pooling(img_feat_for_pool, mask_for_pool)
                pooled_img_feat = F.normalize(pooled_img_feat, dim=-1, p=2)

                text_classifier, num_templates = self.get_text_classifier(dataname)
                if self.cdt is not None:
                    text_classifier = self.cdt(
                        backbone_out_vision['vit_feature'][0]["siglip2-g"]["features"],
                        text_classifier
                    )

                pool_cls_logits = get_classification_logits(pooled_img_feat ,text_classifier, self.out_vocab_logit_scale, num_templates)

            if self.use_softmax:
                is_void_prob = F.softmax(query_cls_results_final, dim=-1)[..., -1:]
                in_vocab_cls_results = query_cls_results_final[..., :-1]
                in_vocab_cls_probs = in_vocab_cls_results.softmax(-1)
                out_vocab_cls_probs = pool_cls_logits.softmax(-1)
            else:
                in_vocab_cls_probs = torch.sigmoid(query_cls_results_final)
                out_vocab_cls_probs = F.softmax(pool_cls_logits, dim=-1)
            category_overlapping_mask = self.SAM_category_overlapping_mask.to(self.device)
            alpha = self.alpha
            beta = self.beta

            eps = 1e-7


            probs_seen = (
                (in_vocab_cls_probs + eps) ** (1 - alpha) *
                (out_vocab_cls_probs + eps) ** alpha
            )


            probs_unseen = (
                (in_vocab_cls_probs + eps) ** (1 - beta) *
                (out_vocab_cls_probs + eps) ** beta
            )

            ensemble_logits = (
                probs_seen.log() * category_overlapping_mask +
                probs_unseen.log() * (1 - category_overlapping_mask)
            )

            final_probs = torch.cat([
                ensemble_logits.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob
            ], dim=-1)

            query_cls_results_final = torch.log(final_probs + 1e-8)

            mask_cls_logits = query_cls_results_final
            mask_pred_logits = outputs["pred_masks"]

            box_pred_logits = outputs["pred_boxes_xyxy"]


            results = []

            for i in range(bs):

                mask_cls_i = mask_cls_logits[i]
                mask_pred_i = mask_pred_logits[i]


                img_h_orig = batched_inputs[i]["height"]
                img_w_orig = batched_inputs[i]["width"]

                mask_pred_i = F.interpolate(
                    mask_pred_i.unsqueeze(0),
                    size=(img_h_orig, img_w_orig),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                res = {}


                if self.semantic_on:

                    mask_cls_prob = F.softmax(mask_cls_i, dim=-1)[..., :-1]
                    mask_pred_prob = mask_pred_i.sigmoid()
                    semseg = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_prob)
                    res["sem_seg"] = semseg


                    visualize_semantic = False


                    if visualize_semantic:

                        tensor_h, tensor_w = batched_inputs[i]["image"].shape[-2:]


                        mask_pred_i_square = F.interpolate(
                            mask_pred_logits[i].unsqueeze(0),
                            size=(tensor_h, tensor_w),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0).sigmoid()


                        semseg_square = torch.einsum("qc,qhw->chw", mask_cls_prob, mask_pred_i_square)
                        pred_result_square = semseg_square.argmax(0).cpu()


                        current_dataname = batched_inputs[i]["meta"]["dataname"]
                        if current_dataname in self.test_metadata:
                            meta = self.test_metadata[current_dataname]
                        else:
                            meta = MetadataCatalog.get(current_dataname)

                        try:
                            current_class_names = meta.stuff_classes
                        except:
                            current_class_names = meta.thing_classes

                if self.panoptic_on:
                    excluded_datasets = ["lvis_v1_val", "lvis_v1_train"]

                    if dataname not in excluded_datasets:
                        panoptic_seg, segments_info = self.panoptic_inference(
                            mask_cls_i, mask_pred_i, dataname
                        )
                        res["panoptic_seg"] = (panoptic_seg, segments_info)

                if self.instance_on:
                    box_pred_i = box_pred_logits[i]
                    instances = self.instance_inference(
                        mask_cls_i, mask_pred_i, box_pred_i, dataname
                    )
                    res["instances"] = instances

                results.append(res)

            return results


    def panoptic_inference(self, mask_cls, mask_pred, dataname):

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)

        num_classes = mask_cls.shape[-1]
        bg_idx = num_classes - 1 if self.use_softmax else -1

        keep = scores > self.object_mask_threshold

        if self.use_softmax:
            keep = keep & (labels != bg_idx)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_masks.sigmoid_()

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info

        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}


        meta = self.test_metadata[dataname] if dataname in self.test_metadata else MetadataCatalog.get(dataname)
        thing_ids = set(meta.thing_dataset_id_to_contiguous_id.values())

        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in thing_ids


            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue


                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred, dataname):


        image_size = mask_pred.shape[-2:]
        img_h, img_w = image_size


        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        num_classes = scores.shape[-1]


        num_queries = scores.shape[0]
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]


        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[topk_indices]
        box_pred = box_pred[topk_indices]


        if self.panoptic_on:
            meta = self.test_metadata[dataname] if dataname in self.test_metadata else MetadataCatalog.get(dataname)
            if hasattr(meta, 'thing_dataset_id_to_contiguous_id'):
                thing_ids = set(meta.thing_dataset_id_to_contiguous_id.values())
            else:

                thing_ids = set(range(len(meta.thing_classes)))

            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab.item() in thing_ids

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
            box_pred = box_pred[keep]


        result = Instances(image_size)


        mask_pred_sigmoid = mask_pred.sigmoid()
        pred_masks_binary = (mask_pred_sigmoid > 0.5).float()
        result.pred_masks = pred_masks_binary


        mask_scores_per_image = (mask_pred_sigmoid.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image


        if pred_masks_binary.numel() > 0:

            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=self.device)

            abs_boxes = box_pred * scale_fct
            result.pred_boxes = Boxes(abs_boxes)
        else:
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=self.device))

        return result


def get_dataname(batched_input):


    if "meta" in batched_input and "dataname" in batched_input["meta"]:
        return batched_input["meta"]["dataname"]
    if "dataname" in batched_input:
        return batched_input["dataname"]


    file_name = batched_input.get("file_name", "")


    file_name_lower = file_name.lower()

    if "lvis" in file_name_lower:
        return "lvis"
    elif "ade" in file_name_lower:
        return "ade20k"


    print(f"Warning: Could not infer dataname from {file_name}, using default 'lvis_v1_val'")
    return "lvis_v1_val"


def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):


    text_classifier = F.normalize(text_classifier, dim=-1)
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    if len(text_classifier.shape) == 2:
        pred_logits = logit_scale * x @ text_classifier.T
    else:
        pred_logits = logit_scale * x @ text_classifier.permute(0,2,1)


    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates:
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t

    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits
