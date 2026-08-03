# sam3/modeling_d2.py
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import numpy as np
import torchvision

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

# from maft.utils.text_templetes import VILD_PROMPT
VILD_PROMPT = ["{}"]

@META_ARCH_REGISTRY.register()
class SAM3Wrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device_type = cfg.MODEL.DEVICE
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False) 
        
        self.predict_mode = cfg.MODEL.SAM3.PREDICT_MODE
        self.streaming_semseg = cfg.MODEL.SAM3.STREAMING_SEMSEG
        self.cycle_infer_gt_classes_only = cfg.MODEL.SAM3.CYCLE_INFER_GT_CLASSES_ONLY
        self._printed_mask_shape = False

        # -------------------------------------------------------
        
        # -------------------------------------------------------       
        compile_mode = "default" if cfg.MODEL.SAM3.COMPILE else None
        
        vision_encoder = _create_vision_backbone(
            compile_mode=compile_mode, 
            enable_inst_interactivity=cfg.MODEL.SAM3.ENABLE_INST_INTERACTIVITY
        )
        text_encoder = _create_text_encoder(cfg.MODEL.SAM3.BPE_PATH)
        backbone = _create_vl_backbone(vision_encoder, text_encoder)
        transformer = _create_sam3_transformer()
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
        # -------------------------------------------------------
        
        # -------------------------------------------------------
        
        self.loss_func = None
        self.train_dataname = None
        self.test_dataname = None
        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.semantic_on = cfg.TEST.SEMANTIC_ON
        self.instance_on = cfg.TEST.INSTANCE_ON
        self.panoptic_on = cfg.TEST.PANOPTIC_ON
        self.object_mask_threshold = cfg.TEST.PANOPTIC_OBJECT_MASK_THRESHOLD
        self.overlap_threshold = cfg.TEST.PANOPTIC_OVERLAP_THRESHOLD
        self.panoptic_chunk_size = cfg.TEST.PANOPTIC_CHUNK_SIZE

        # --------------------------------------------------
        
        # --------------------------------------------------
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device = self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device = self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names)) 
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self, dataname):
        if self.training:
            if self.train_dataname != dataname:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.detector.backbone.forward_text(self.train_class_names[idx:idx+bs], device=self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
                self.train_dataname = dataname
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata)
                text_classifier = []
                language_mask = []
                language_features = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.test_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # B, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> B, L, D 
                    language_features.append(batch_text_feat.clone())
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask.unsqueeze(1)) # B, 1, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0)
                language_features = torch.cat(language_features, dim=0) # (num_classes, VILD_PROMPT, L, D)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1])
                # language_mask = language_mask.reshape(language_mask.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), language_mask.shape[-1])
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier[language_mask] = 0.0 
                text_classifier = text_classifier.mean(-2)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                self.language_features = language_features
                # self.language_features = language_features.reshape(language_features.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), language_features.shape[1], language_features.shape[2])
                self.language_mask = language_mask
                self.test_text_classifier = text_classifier
                self.test_dataname = dataname
            return self.test_text_classifier, self.test_num_templates

    @property
    def device(self):
        return self.pixel_mean.device

    def _new_panoptic_states(self, batched_inputs):
        states = []
        for input_per_image in batched_inputs:
            height, width = input_per_image["height"], input_per_image["width"]
            states.append(
                {
                    "best_weight": torch.full(
                        (height, width), -float("inf"), device=self.device
                    ),
                    "winner": torch.full(
                        (height, width), -1, dtype=torch.int32, device=self.device
                    ),
                    "winner_foreground": torch.zeros(
                        (height, width), dtype=torch.bool, device=self.device
                    ),
                    "candidates": [],
                }
            )
        return states

    def _update_panoptic_states(
        self, states, mask_logits, mask_scores, pred_class, batched_inputs
    ):
        """Online equivalent of FC-CLIP's global score-weighted mask argmax."""
        for batch_idx, state in enumerate(states):
            keep = mask_scores[batch_idx] > self.object_mask_threshold
            selected_logits = mask_logits[batch_idx][keep]
            selected_scores = mask_scores[batch_idx][keep]
            if selected_logits.shape[0] == 0:
                continue

            output_size = (
                batched_inputs[batch_idx]["height"],
                batched_inputs[batch_idx]["width"],
            )
            for start in range(0, selected_logits.shape[0], self.panoptic_chunk_size):
                end = min(start + self.panoptic_chunk_size, selected_logits.shape[0])
                chunk_masks = F.interpolate(
                    selected_logits[start:end, None],
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].sigmoid()
                chunk_scores = selected_scores[start:end]
                chunk_weights = chunk_masks * chunk_scores[:, None, None]
                chunk_best, chunk_winner = chunk_weights.max(dim=0)

                first_candidate_id = len(state["candidates"])
                candidate_ids = torch.arange(
                    first_candidate_id,
                    first_candidate_id + chunk_masks.shape[0],
                    device=self.device,
                    dtype=torch.int32,
                )
                winning_ids = candidate_ids[chunk_winner]
                winning_foreground = torch.gather(
                    chunk_masks, 0, chunk_winner.unsqueeze(0)
                )[0] >= 0.5
                update = chunk_best > state["best_weight"]
                state["best_weight"] = torch.where(
                    update, chunk_best, state["best_weight"]
                )
                state["winner"] = torch.where(
                    update, winning_ids, state["winner"]
                )
                state["winner_foreground"] = torch.where(
                    update, winning_foreground, state["winner_foreground"]
                )

                original_areas = (chunk_masks >= 0.5).flatten(1).sum(1).tolist()
                state["candidates"].extend(
                    {
                        "category_id": int(pred_class),
                        "original_area": int(original_area),
                    }
                    for original_area in original_areas
                )

    def _panoptic_inference(self, state, metadata):
        height, width = state["winner"].shape
        panoptic_seg = torch.zeros(
            (height, width), dtype=torch.int32, device=self.device
        )
        segments_info = []
        current_segment_id = 0
        stuff_memory = {}
        thing_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())

        for candidate_id, candidate in enumerate(state["candidates"]):
            owner = state["winner"] == candidate_id
            mask_area = owner.sum().item()
            original_area = candidate["original_area"]
            mask = owner & state["winner_foreground"]
            if mask_area == 0 or original_area == 0 or not mask.any().item():
                continue
            if mask_area / original_area < self.overlap_threshold:
                continue

            pred_class = candidate["category_id"]
            isthing = pred_class in thing_ids
            if not isthing and pred_class in stuff_memory:
                panoptic_seg[mask] = stuff_memory[pred_class]
                continue

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            if not isthing:
                stuff_memory[pred_class] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": bool(isthing),
                    "category_id": pred_class,
                }
            )

        return panoptic_seg, segments_info

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        # print("shape of first image:", images[0].shape)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 14)
        # print("shape of images.tensor:", images.tensor.shape)
        img_h, img_w = images.tensor.shape[-2:]

        batch_size = images.tensor.shape[0]
        

        file_names = [x["file_name"] for x in batched_inputs]
        file_names = [x.split('/')[-1].split('.')[0] for x in file_names]

        meta = batched_inputs[0]["meta"]
        
        
        backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
        img_feat = backbone_out_vision["vision_features"] # B, C, H', W'
        backbone_fpn = backbone_out_vision["backbone_fpn"]
    

        
        # text_classifier:[num_names, dim] 
        # language_features:[num_names, num_templates, L, dim] language_mask:[num_names, num_templates, L]
        text_classifier, num_templates = self.get_text_classifier(meta['dataname'])

        # others
        geometric_prompt = self.detector._get_dummy_prompt()
        
        if self.predict_mode == "Cycle": 

            infer_names_idx = None
            if self.cycle_infer_gt_classes_only:
                gt_classes = get_gt_labels_from_sem_seg(
                    batched_inputs[0]["sem_seg"].to(self.device)
                )
                gt_names_idx = []
                cur_idx = 0
                for i, num_t in enumerate(num_templates):
                    if i in gt_classes:
                        gt_names_idx += list(range(cur_idx, cur_idx + num_t))
                    cur_idx += num_t
                infer_names_idx = set(gt_names_idx)

            names_masks = []
            inferred_mask_batch_size = None
            final_seg_logits = None
            streaming_results = None
            stream_semantic = self.streaming_semseg and self.semantic_on
            if stream_semantic:
                streaming_results = [
                    {
                        "best_score": torch.full(
                            (batched_inputs[i]["height"], batched_inputs[i]["width"]),
                            -float("inf"),
                            device=self.device,
                        ),
                        "best_class": torch.zeros(
                            (batched_inputs[i]["height"], batched_inputs[i]["width"]),
                            dtype=torch.long,
                            device=self.device,
                        ),
                    }
                    for i in range(len(batched_inputs))
                ]
            name_to_class = []
            for class_idx, num_t in enumerate(num_templates):
                name_to_class.extend([class_idx] * num_t)
            panoptic_states = (
                self._new_panoptic_states(batched_inputs) if self.panoptic_on else None
            )
            for i in range(len(self.language_features)):

                if infer_names_idx is not None and i not in infer_names_idx:
                    if self.semantic_on and not stream_semantic:
                        names_masks.append(None)
                    continue

                backbone_out={
                    "img_batch_all_stages": img_feat,
                    "vision_pos_enc": backbone_out_vision["vision_pos_enc"],
                    "backbone_fpn": backbone_fpn,
                    "language_features": self.language_features[i,:,:].unsqueeze(1),
                    "language_mask": self.language_mask[i,:],
                }
                outputs = self.detector.forward_grounding(
                    backbone_out = backbone_out,
                    find_input=self.find_stage,
                    geometric_prompt= geometric_prompt,
                    find_target=None,
                )

                out_bbox = outputs["pred_boxes"]

                out_masks = outputs["pred_masks"]
                if not self._printed_mask_shape:
                    print(
                        "[SAM3Wrapper] before upsample:",
                        "pred_masks", tuple(out_masks.shape),
                        "semantic_seg", tuple(outputs["semantic_seg"].shape),
                        "input_image", (img_h, img_w),
                        flush=True,
                    )
                    self._printed_mask_shape = True
                if out_masks.dim() == 5 and out_masks.size(1) == 1:
                    out_masks = out_masks.squeeze(1)
                if out_masks.dim() != 4:
                    raise ValueError(f"Expected pred_masks with shape [B, N, H, W], got {tuple(out_masks.shape)}")
                presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)

                out_logits = outputs["pred_logits"] 
                out_probs = out_logits.sigmoid() # B, N, 1
                out_probs = (out_probs * presence_score).squeeze(-1) 

                if panoptic_states is not None:
                    self._update_panoptic_states(
                        panoptic_states,
                        out_masks,
                        out_probs,
                        name_to_class[i],
                        batched_inputs,
                    )

                if not self.semantic_on:
                    continue
                if stream_semantic:
                    out_masks = out_masks.sigmoid()
                else:
                    out_masks = interpolate(
                        out_masks,
                        (img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()

                out_semseg = outputs["semantic_seg"]
                if stream_semantic:
                    out_semseg = out_semseg.sigmoid()
                else:
                    out_semseg = F.interpolate(
                        out_semseg,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()

                # print("out_masks shape:", out_masks.shape, "out_probs shape:", out_probs.shape, "out_semseg shape:", out_semseg.shape, "presence_score shape:", presence_score.shape)
                # out_masks shape: torch.Size([1, 200, 1008, 1008]) out_probs shape: torch.Size([1, 200]) out_semseg shape: torch.Size([1, 1, 1008, 1008]) presence_score shape: torch.Size([1, 1, 1])
                instance_masks = torch.mul(out_masks,out_probs.unsqueeze(-1).unsqueeze(-1)) 
                semantic_masks = torch.mul(out_semseg , presence_score.unsqueeze(-1))
                cls_mask = torch.max(
                    torch.max(instance_masks, dim=1).values,
                    torch.max(semantic_masks, dim=1).values,

                ) # B, H, W
                if cls_mask.shape[0] != batch_size:
                    cls_mask = cls_mask.max(dim=0, keepdim=True).values
                
                if inferred_mask_batch_size is None:
                    inferred_mask_batch_size = cls_mask.shape[0]
                if stream_semantic:
                    class_idx = name_to_class[i]
                    for batch_idx, per_image_mask in enumerate(cls_mask):
                        orig_h = batched_inputs[batch_idx]["height"]
                        orig_w = batched_inputs[batch_idx]["width"]
                        resized_mask = F.interpolate(
                            per_image_mask[None, None],
                            size=(orig_h, orig_w),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        update = resized_mask > streaming_results[batch_idx]["best_score"]
                        streaming_results[batch_idx]["best_score"] = torch.where(
                            update,
                            resized_mask,
                            streaming_results[batch_idx]["best_score"],
                        )
                        streaming_results[batch_idx]["best_class"] = torch.where(
                            update,
                            torch.full_like(
                                streaming_results[batch_idx]["best_class"],
                                class_idx,
                            ),
                            streaming_results[batch_idx]["best_class"],
                        )
                    continue
                names_masks.append(cls_mask)
            
            if self.semantic_on and not stream_semantic:
                if inferred_mask_batch_size is None:
                    inferred_mask_batch_size = batch_size
                names_masks = [
                    mask if mask is not None else torch.zeros(
                        (inferred_mask_batch_size, img_h, img_w),
                        device=self.device,
                    )
                    for mask in names_masks
                ]
                names_masks = torch.stack(names_masks, dim=1)

                final_seg_logits = []
                cur_idx = 0
                for num_t in num_templates: 
                    final_seg_logits.append(names_masks[:, cur_idx: cur_idx + num_t,:,:].max(1).values)
                    cur_idx += num_t
                final_seg_logits = torch.stack(final_seg_logits, dim=1)

        panoptic_results = None
        if panoptic_states is not None:
            metadata = self.test_metadata[meta["dataname"]]
            panoptic_results = [
                self._panoptic_inference(state, metadata)
                for state in panoptic_states
            ]

        results = []
        for i in range(len(batched_inputs)):
            result = {}
            if streaming_results is not None:
                result["sem_seg_label"] = streaming_results[i]["best_class"]
            elif self.semantic_on:
                orig_size = (batched_inputs[i]["height"], batched_inputs[i]["width"])
                result["sem_seg"] = sem_seg_postprocess(
                    final_seg_logits[i],
                    (img_h, img_w),
                    orig_size[0],
                    orig_size[1],
                )
            if panoptic_results is not None:
                result["panoptic_seg"] = panoptic_results[i]
            results.append(result)
        return results

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

def get_gt_labels_from_sem_seg(sem_seg):
    ""
    
    if sem_seg.dim() == 3: 
        sem_seg = sem_seg.squeeze(0)
    
    
    classes = torch.unique(sem_seg, sorted=False, return_inverse=False, return_counts=False)
    
    
    
    gt_labels = classes[classes != 255]
    
    return gt_labels.cpu().numpy().tolist()
