"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py

FC-CLIP criterion.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)


from maft.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from sam3.model.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    text_classifier = F.normalize(text_classifier, dim=-1)
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    if len(text_classifier.shape) == 2:
        pred_logits = logit_scale * x @ text_classifier.T
    else:
        pred_logits = logit_scale * x @ text_classifier.permute(0, 2, 1)

    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates:
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    return torch.stack(final_pred_logits, dim=-1)


class FcclipSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()
        current_num_classes = src_logits.shape[-1] - 1

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            # self.num_classes,
            current_num_classes,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        empty_weight = torch.ones(current_num_classes + 1, device=src_logits.device)
        empty_weight[-1] = self.eos_coef

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes,
                                #   self.empty_weight
                                  empty_weight,
        )
        losses = {"loss_cls": loss_ce}

        if "attn_cls_logits" in outputs.keys():
            attn_cls_logits = outputs["attn_cls_logits"].float()
            empty_weight = torch.ones(current_num_classes, device=src_logits.device)
            matched_logits = attn_cls_logits[idx]

            loss_attn_cls = F.cross_entropy(matched_logits, target_classes_o,
                                        empty_weight,
            )
            losses["loss_attn_cls"] = loss_attn_cls

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    # def __repr__(self):
    #     head = "Criterion " + self.__class__.__name__
    #     body = [
    #         "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
    #         "losses: {}".format(self.losses),
    #         "weight_dict: {}".format(self.weight_dict),
    #         "num_classes: {}".format(self.num_classes),
    #         "eos_coef: {}".format(self.eos_coef),
    #         "num_points: {}".format(self.num_points),
    #         "oversample_ratio: {}".format(self.oversample_ratio),
    #         "importance_sample_ratio: {}".format(self.importance_sample_ratio),
    #     ]
    #     _repr_indent = 4
    #     lines = [head] + [" " * _repr_indent + line for line in body]
    #     return "\n".join(lines)

    def loss_boxes(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)


        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 1. L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_masks

        # 2. GIoU Loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_masks

        return losses


class MapAdapterFcclipSetCriterion(FcclipSetCriterion):
    def _build_query_map_inputs(self, outputs, query_indices_per_image, extra_context):
        attn_weights = outputs.get("map_attn_weights")
        if attn_weights is None:
            raise KeyError("map_attn_weights is required for map-adapter supervision")
        if attn_weights.dim() != 4:
            raise ValueError(
                f"Expected per-layer multi-head attn weights with shape [B, H, Q, L], got {tuple(attn_weights.shape)}"
            )

        pred_masks = outputs["pred_masks"]
        map_src_feature = extra_context["map_src_feature"]
        feat_h, feat_w = map_src_feature.shape[-2:]
        if attn_weights.shape[-1] != feat_h * feat_w:
            raise ValueError(
                "Attention map token count does not match map-adapter feature resolution: "
                f"{attn_weights.shape[-1]} vs {feat_h}x{feat_w}"
            )

        bs, num_heads, num_queries, _ = attn_weights.shape
        mask_h, mask_w = pred_masks.shape[-2:]
        attn_maps = attn_weights.view(bs, num_heads, num_queries, feat_h, feat_w)
        attn_maps = attn_maps.permute(0, 2, 1, 3, 4).reshape(bs * num_queries, num_heads, feat_h, feat_w)
        attn_maps = F.interpolate(attn_maps.float(), size=(mask_h, mask_w), mode="bilinear", align_corners=False)
        attn_maps = attn_maps.view(bs, num_queries, num_heads, mask_h, mask_w)

        binary_masks = (pred_masks.sigmoid() > extra_context["mask_threshold"]).unsqueeze(2).float()
        query_map_inputs = torch.cat([attn_maps, binary_masks], dim=2)

        return [query_map_inputs[b, query_indices] for b, query_indices in enumerate(query_indices_per_image)]

    def _pool_query_features(self, map_src_feature, pool_feature, map_adapter_outputs, num_queries_per_image, num_output_maps):
        pooled_query_features = []
        for b, maps_for_pooling in enumerate(map_adapter_outputs):
            num_queries = num_queries_per_image[b]
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
            pooled = pooled.view(1, num_queries, num_output_maps, -1).mean(dim=2).squeeze(0).contiguous()
            pooled_query_features.append(pooled)
        return pooled_query_features

    def _compute_map_adapter_logits(self, outputs, indices, extra_context):
        query_indices_per_image = [src for src, _ in indices]
        total_matches = sum(len(src) for src in query_indices_per_image)
        num_classes = len(extra_context["num_templates"])
        if total_matches == 0:
            return outputs["pred_logits"].new_empty((0, num_classes))

        query_map_inputs = self._build_query_map_inputs(outputs, query_indices_per_image, extra_context)
        map_adapter_outputs = extra_context["map_adapter"](extra_context["map_src_feature"], query_map_inputs)
        pooled_query_features = self._pool_query_features(
            extra_context["map_src_feature"],
            extra_context["pool_feature"],
            map_adapter_outputs,
            [len(src) for src in query_indices_per_image],
            extra_context["num_output_maps"],
        )

        matched_logits = []
        text_classifier = extra_context["text_classifier"]
        num_templates = extra_context["num_templates"]
        out_vocab_logit_scale = extra_context["out_vocab_logit_scale"]
        for b, pooled_features in enumerate(pooled_query_features):
            if pooled_features.numel() == 0:
                continue
            cur_text_classifier = text_classifier[b:b + 1] if text_classifier.dim() == 3 else text_classifier
            cur_logits = get_classification_logits(
                pooled_features.unsqueeze(0),
                cur_text_classifier,
                out_vocab_logit_scale,
                num_templates,
            ).squeeze(0)
            matched_logits.append(cur_logits)

        if not matched_logits:
            return outputs["pred_logits"].new_empty((0, num_classes))
        return torch.cat(matched_logits, dim=0)

    def loss_labels(self, outputs, targets, indices, num_masks, extra_context=None):
        losses = super().loss_labels(outputs, targets, indices, num_masks)
        if "loss_attn_cls" in losses:
            return losses

        if not extra_context or not extra_context.get("use_map_adapter", False):
            return losses

        src_logits = outputs["pred_logits"].float()
        current_num_classes = src_logits.shape[-1] - 1
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if target_classes_o.numel() == 0:
            losses["loss_attn_cls"] = src_logits.sum() * 0.0
            return losses

        matched_logits = self._compute_map_adapter_logits(outputs, indices, extra_context)
        empty_weight = torch.ones(current_num_classes, device=src_logits.device)
        losses["loss_attn_cls"] = F.cross_entropy(matched_logits.float(), target_classes_o, empty_weight)
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_masks, extra_context=None):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss == "labels":
            return loss_map[loss](outputs, targets, indices, num_masks, extra_context=extra_context)
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, extra_context=None):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, extra_context=extra_context))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, extra_context=extra_context)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
