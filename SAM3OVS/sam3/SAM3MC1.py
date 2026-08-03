# sam3/modeling_d2.py
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

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

from maft.utils.text_templetes import VILD_PROMPT
# VILD_PROMPT = ["{}"]

from maft.modeling.matcher import HungarianMatcher
from maft.modeling.criterion import SetCriterion
from maft.modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits

class PixelProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        ""
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


@META_ARCH_REGISTRY.register()
class SAM3MC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device_type = cfg.MODEL.DEVICE
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False) 
        
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
        self.mask_pooling = MaskPooling()
        
        self.no_object_embed = nn.Embedding(1, 256)
        
        nn.init.normal_(self.no_object_embed.weight, mean=0, std=0.02)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10))

        self.sem_seg_projector = PixelProjector(
            input_dim= 256,
            output_dim= 256,
        )

        # -------------------------------------------------------
        
        # -------------------------------------------------------
        

        self.train_dataname = None
        self.test_dataname = None
        self.test_metadata = {i: MetadataCatalog.get(i) for i in cfg.DATASETS.TEST}
        self.train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(self.train_metadata, self.train_metadata)

        self.use_aux = cfg.SOLVER.USE_AUX

        # -------------------------------------------------------
        
        # -------------------------------------------------------
        no_object_weight = cfg.SOLVER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.SOLVER.CLASS_WEIGHT
        dice_weight = cfg.SOLVER.DICE_WEIGHT
        mask_weight = cfg.SOLVER.MASK_WEIGHT

        weight_dict = {}
        criterion_weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        weight_dict.update(criterion_weight_dict)

        if self.use_aux:
            for i in range (5):
                for k in criterion_weight_dict.keys():
                    weight_dict[f"{k}_{i}"] = criterion_weight_dict[k]
        
        

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
        )
        
        losses = ["labels", "masks"]

        self.criterion = SetCriterion(
            171, 
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.SOLVER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.SOLVER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO,
        )

        # --------------------------------------------------
        
        # --------------------------------------------------

        self._freeze()

    def _freeze(self, ):
        for name, param in self.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            # elif 'dot_prod_scoring' in name:
            #     param.requires_grad = False
            elif 'geometry_encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        print('='*10,'Parameters to be trained', '='*10)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)
        # exit()

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

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
        self.vis_class_names = class_names
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
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                for idx in range(0, len(self.train_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.train_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * VILD_PROMPT,  L)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, VILD_PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.train_text_classifier = text_classifier.detach()
                self.train_dataname = dataname
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_dataname != dataname:
                self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(self.test_metadata[dataname], self.train_metadata)
                text_classifier = []
                language_mask = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.test_class_names), "classes.")
                for idx in range(0, len(self.test_class_names), bs):
                    state_text = self.detector.backbone.forward_text(self.test_class_names[idx:idx+bs], device=self.device)

                    batch_text_feat = state_text["language_features"].detach()
                    mask = state_text["language_mask"] # bs, L
                    batch_text_feat = batch_text_feat.permute(1,0,2) # -> bs, L, D 
                    text_classifier.append(batch_text_feat)
                    language_mask.append(mask) # bs, L
                text_classifier = torch.cat(text_classifier, dim=0)
                language_mask = torch.cat(language_mask, dim=0) # (num_names * VILD_PROMPT,  L)
                # average across templates and normalization.
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-2], text_classifier.shape[-1]) # num_names, VILD_PROMPT, L, D
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)

                text_classifier[language_mask.view(text_classifier.shape[0],text_classifier.shape[1],text_classifier.shape[2])] = 0.0 
                language_features = text_classifier.mean(1) # num_names, L, D
                text_classifier = text_classifier.mean(-2) 
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                text_classifier = text_classifier.mean(1)
                text_classifier /= (text_classifier.norm(dim=-1, keepdim=True) + 1e-6)
                
                self.language_features = language_features.detach() # num_names , L, D
                self.language_mask = torch.min(language_mask.view(language_features.shape[0],len(VILD_PROMPT),language_features.shape[1]), dim=1).values# [num_names, L]
                self.test_text_classifier = text_classifier.detach()
                self.test_dataname = dataname
            return self.test_text_classifier, self.test_num_templates

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # print('='*10,'Parameters to be trained', '='*10)
        # for name, param in self.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)
        # exit()


        images = [x["image"].to(self.device) for x in batched_inputs]
        # print("shape of first image:", images[0].shape)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 14)
        # print("shape of images.tensor:", images.tensor.shape)
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

            meta = batched_inputs[0]["meta"]
            
            
            backbone_out_vision = self.detector.backbone.forward_image(images.tensor)
            img_feat = backbone_out_vision["vision_features"].detach() # bs, C, H', W'
            backbone_fpn = backbone_out_vision["backbone_fpn"]
            for k in range(len(backbone_fpn)):
                backbone_fpn[k] = backbone_fpn[k].detach()


            
            # text_classifier:[num_names, dim] 
            # language_features:[num_names, num_templates, L, dim] language_mask:[num_names, num_templates, L]
            text_classifier, num_templates = self.get_text_classifier(meta['dataname'])
            no_obj_normalized = torch.nn.functional.normalize(self.no_object_embed.weight, p=2, dim=-1)
            text_classifier = torch.cat([text_classifier, no_obj_normalized], dim=0)

            # others
            geometric_prompt = self.detector._get_dummy_prompt(bs)
        
        batch_gt_names_idx = []
        for i in range(bs):
            gt_classes = get_gt_labels_from_sem_seg(batched_inputs[i]["sem_seg"].to(self.device))
            gt_names_idx = []
            cur_idx = 0
            for i,num_t in enumerate(num_templates): 
                if i in gt_classes:
                    gt_names_idx += list(range(cur_idx, cur_idx + num_t))
                cur_idx += num_t
            batch_gt_names_idx.append(gt_names_idx)

        # =======================================================
        
        language_features_input = []

        # USE_GT_NAMES_ONLY = True
        USE_GT_NAMES_ONLY = False
        
        if USE_GT_NAMES_ONLY:
            language_features_input = [self.language_features[batch_gt_names_idx[i],:,:] for i in range(bs)]
            language_features_input = torch.cat(language_features_input, dim=0) # (bs, num_names * L, dim)
            language_mask_input = [self.language_mask[batch_gt_names_idx[i],:] for i in range(bs)]
            language_mask_input = torch.cat(language_mask_input, dim=0) # (bs, num_names * L)
            if bs == 1:
                language_features_input = language_features_input.unsqueeze(0)
                language_mask_input = language_mask_input.unsqueeze(0)

        else:
            language_features_input = self.language_features.expand(bs, -1, -1, -1) # (bs, num_names, L, dim)
            language_mask_input = self.language_mask.expand(bs, -1, -1) # (bs, num_names, L)

        # print("shape of input:",language_features_input.shape, language_mask_input.shape)
        language_features_input = language_features_input.reshape(bs, -1, language_features_input.shape[-1]) # (bs, num_names * L, dim)
        language_mask_input = language_mask_input.reshape(bs, -1) # (bs, num_names * L)
        # print("shape of input after reshape:",language_features_input.shape, language_mask_input.shape)
        

        backbone_out={
            "img_batch_all_stages": img_feat,
            "vision_pos_enc": backbone_out_vision["vision_pos_enc"],
            "backbone_fpn": backbone_fpn,
            "language_features": language_features_input.permute(1, 0, 2), # (num_names * L, bs, dim)
            "language_mask": language_mask_input, # bs, (num_names * L)
        }

        #=================================
        find_input = self.find_stage

        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self.detector._encode_prompt(
                backbone_out, find_input, geometric_prompt
            )
        # Run the encoder
        with torch.profiler.record_function("SAM3Image._run_encoder"):
            backbone_out, encoder_out, _ = self.detector._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        # print("keys of out before decoder:", out.keys()) # s(['encoder_hidden_states', 'prev_encoder_out'])
        # Run the decoder
        with torch.profiler.record_function("SAM3Image._run_decoder"):
            out, hs = self.detector._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        # print("keys of out after decoder:", out.keys()) # (['encoder_hidden_states', 'prev_encoder_out', 'presence_feats', 'queries', 'presence_logit_dec', 'pred_logits', 'pred_boxes', 'pred_boxes_xyxy'])
        # Run segmentation heads
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
                aux_masks=True,
            )
        
        # if self.detector.training or self.detector.num_interactive_steps_val > 0:
        #     self.detector._compute_matching(out, self.detector.back_convert(find_target))

        #========================================
        outputs = out
        # print("outputs keys:", outputs.keys())
        # print('aux:',outputs['aux_outputs'][0].keys())

        out_masks = outputs["pred_masks"].clone()

        out_masks = out_masks.sigmoid()

        

        
        # out_semseg = F.interpolate(
        #     out_semseg,
        #     size=(img_h, img_w),
        #     mode="bilinear",
        #     align_corners=False,
        # ).sigmoid()


        # print("out_masks shape:", out_masks.shape, "out_probs shape:", out_probs.shape, "out_semseg shape:", out_semseg.shape, "presence_score shape:", presence_score.shape)
        # out_masks shape: torch.Size([1, 200, 1008, 1008]) out_probs shape: torch.Size([1, 200]) out_semseg shape: torch.Size([1, 1, 1008, 1008]) presence_score shape: torch.Size([1, 1, 1])
        

        bs, N, H, W = out_masks.shape
        C_ = text_classifier.shape[0] # num_names + 1 

        queries_masks = out_masks 

        queries = outputs["obj_queries"] # 6, bs, N, D

        text_classifier_mask = torch.zeros((C_, bs), dtype=torch.bool, device=text_classifier.device)

        
        use_aux = self.use_aux and self.training
        aux_outputs = []

        for i in range(6):
            if use_aux or i == 5 :
                tp_queries = queries[i,:,:N,:].clone() 
                tp_queries = F.normalize(tp_queries, dim=-1, p=2)

                # hs=tp_queries.unsqueeze(1)
                # hs = hs.expand(-1, C_, -1, -1).contiguous().view(bs*(C_), N, -1).unsqueeze(0) # 1, bs*(C+1), N, D
                # print("hs shape:", hs.shape)
                # prompt= text_classifier.unsqueeze(0).expand( bs, -1, -1).contiguous().view(1, bs * C_, -1) # 1, bs*(C+1), D
                # print("prompt shape", prompt.shape)

                
                #     hs= tp_queries.unsqueeze(1).expand(-1, C_, -1, -1).contiguous().view(bs*(C_), N, -1).unsqueeze(0), # 1, bs*(C+1), N, D
                #     prompt = text_classifier.unsqueeze(0).expand( bs, -1, -1).contiguous().view(1, bs * C_, -1), # 1, bs*(C+1), D
                #     prompt_mask=text_classifier_mask.expand(-1, bs).view(-1, 1), # (C+1) * bs 
                # ) 
                # out_vocab_names_results = out_vocab_names_results.view(bs, C_, N).permute(0,2,1) # bs, N, C+1

                out_vocab_names_results = torch.einsum("bnd,cd->bnc", tp_queries, text_classifier) # bs, N, C+1
                
                out_vocab_cls_results= []
                cur_idx = 0
                for num_t in num_templates: 
                    out_vocab_cls_results.append(out_vocab_names_results[:,:, cur_idx: cur_idx + num_t].max(-1).values)
                    cur_idx += num_t
                out_vocab_cls_results = torch.stack(out_vocab_cls_results, dim=-1) # bs, N, num_classes
                out_vocab_cls_results = torch.cat([out_vocab_cls_results, out_vocab_names_results[:,:,-1:]], dim=-1) 
                # print(f"aux out_vocab_cls_results[{i}] shape:", out_vocab_cls_results.shape)
                if i<5:
                    aux_outputs.append({'pred_logits': out_vocab_cls_results, 'pred_masks': outputs['aux_outputs'][i]["pred_masks"]})
                else:
                    out_vocab_cls_results_final = out_vocab_cls_results



        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            
            criterion_pred = {
                'pred_logits': out_vocab_cls_results_final,
                'pred_masks': outputs["pred_masks"],
                'aux_outputs': aux_outputs if use_aux is True else None,
            }

            losses = self.criterion(criterion_pred, targets)

            for k in list(losses.keys()):
                # print("loss:", k, losses[k].item())
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
            
        else:
            out_vocab_cls_results = out_vocab_cls_results.sigmoid() 
            queries_seg_result = torch.einsum("bnc,bnhw->bchw", out_vocab_cls_results, queries_masks) # [bs, num_classes+1, H, W]


        seg_logits = queries_seg_result 

        if USE_GT_NAMES_ONLY:
            pass
            # for b in range(bs):
            #     gtcls_mask = torch.ones_like(seg_logits[b], dtype=torch.bool)
            #     gtcls_mask[batch_gt_names_idx[b],:,:] = False
            #     seg_logits[b][gtcls_mask] = 0.0
        
        final_seg_logits = seg_logits[:, :-1, :, :]
        # print("final_seg_logits shape:", final_seg_logits.shape) # bs,num_classes+1, H, W

        # pred_result = final_seg_logits[0].argmax(0)
        # visualize_segmentation(pred_result, self.vis_class_names+['void'],batched_inputs[0]["image"],f"./show_queries/{file_names[0]}_")
        # visualize_segmentation(pred_result, self.vis_class_names+['void'],batched_inputs[0]["image"],f"./show_seg/{file_names[0]}_")

        # =======================================================
        


        results = []
        for i in range(bs):
            orig_size = (batched_inputs[i]["height"], batched_inputs[i]["width"]) 
            res = sem_seg_postprocess(
                final_seg_logits[i], 
                (img_h, img_w),
                orig_size[0],
                orig_size[1],
            )
            results.append({"sem_seg": res})
        
        return results

    def semantic_segmentation_loss(self, pred_logits, batched_inputs, images_tensor_shape):
        ""
        
        
        _, _, pad_h, pad_w = images_tensor_shape
        
        
        
        pred_logits_upsampled = F.interpolate(
            pred_logits, 
            size=(pad_h, pad_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        total_loss = 0.0
        valid_samples = 0
        
        
        for i, input_per_image in enumerate(batched_inputs):
            
            
            gt_sem_seg = input_per_image["sem_seg"].to(pred_logits.device)
            
            
            gt_h, gt_w = gt_sem_seg.shape
            
            
            
            valid_pred = pred_logits_upsampled[i, :, :gt_h, :gt_w]
            
            
            
            
            valid_pred = valid_pred.unsqueeze(0) # [1, C, H, W]
            gt_sem_seg = gt_sem_seg.unsqueeze(0) # [1, H, W]
            
            
            
            loss = F.cross_entropy(valid_pred, gt_sem_seg.long(), ignore_index=255)
            
            total_loss += loss
            valid_samples += 1

        
        return total_loss / max(valid_samples, 1)    


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


def visualize_segmentation(
    pred_result, 
    class_names, 
    original_image_tensor, 
    save_path="./show/result.png", 
    fig_size=(15, 10)
):
    ""
    print(f"Number of classes: {len(class_names)}")

    
    if isinstance(pred_result, torch.Tensor):
        pred_result = pred_result.cpu().numpy()

    
    if len(pred_result.shape) == 3 and pred_result.shape[0] == 1:
        pred_result = pred_result[0]
    
    height, width = pred_result.shape
    num_classes = len(class_names)

    
    np.random.seed(0)  
    palette = np.random.randint(0, 255, size=(num_classes, 3))

    
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_index in range(num_classes):
        color_image[pred_result == class_index] = palette[class_index]

    
    unique_classes, pixel_counts = np.unique(pred_result, return_counts=True)
    
    
    class_statistics = [
        {
            "index": class_index,
            "name": class_names[class_index],
            "count": pixel_counts[i]
        }
        for i, class_index in enumerate(unique_classes) if class_index < num_classes
    ]
    sorted_class_statistics = sorted(class_statistics, key=lambda x: x['count'], reverse=True)

    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    
    original_image = original_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    
    ax[1].imshow(color_image)
    ax[1].set_title("Segmentation Result")
    ax[1].axis('off')

    
    legend_elements = [
        Rectangle((0, 0), 1, 1, color=palette[stats["index"]] / 255.0, 
                  label=f"{stats['name']} ({stats['count']:,} px)")
        for stats in sorted_class_statistics
    ]

    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01), 
        ncol=min(4, len(legend_elements)),  
        frameon=False,
        fontsize='small'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    
    
    try:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization result saved to: {save_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close(fig)

