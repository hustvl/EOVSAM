import warnings
import os
import gc

warnings.filterwarnings("ignore")

import sys
import types

import copy
import contextlib
import itertools
import logging
import os
import time
import weakref


from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,

    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger


from maft import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    COCOSemanticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    SemSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,


)

from sam3.data.dataset_mappers.coco_combine_new_baseline_dataset_mapper_2 import COCOCombineNewBaselineDatasetMapper
from sam3.data.custom_dataset_dataloader import build_custom_train_loader
from sam3.config import add_sam3_config
from sam3.modeling_d2 import SAM3Wrapper

from sam3.EOVSAM import EOVSAM

from sam3.mask_adapter_head import MASKAdapterHead


import numpy as np
try:
    np.float = float
except AttributeError:
    pass


def infer_gradient_accumulation(cfg, world_size):
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    total_batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
    if total_batch_size <= 0:
        raise ValueError(f"SOLVER.IMS_PER_BATCH must be positive, got {total_batch_size}")
    if total_batch_size % world_size != 0:
        raise ValueError(
            f"SOLVER.IMS_PER_BATCH={total_batch_size} must be divisible by world_size={world_size}"
        )

    target_batch_per_gpu = total_batch_size // world_size
    use_diff_bs_size = bool(getattr(cfg.DATALOADER, "USE_DIFF_BS_SIZE", False))
    dataset_bs = [int(bs) for bs in getattr(cfg.DATALOADER, "DATASET_BS", [])]

    if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler" and use_diff_bs_size:
        if not dataset_bs:
            raise ValueError(
                "DATALOADER.DATASET_BS must be non-empty when USE_DIFF_BS_SIZE=True"
            )
        unique_bs = sorted(set(dataset_bs))
        if any(bs <= 0 for bs in unique_bs):
            raise ValueError(f"DATALOADER.DATASET_BS must all be positive, got {dataset_bs}")
        if len(unique_bs) != 1:
            raise ValueError(
                "Gradient accumulation with USE_DIFF_BS_SIZE=True currently requires all "
                f"DATASET_BS values to be identical so that IMS_PER_BATCH maps to an exact "
                f"effective batch size. Got {dataset_bs}"
            )
        micro_batch_size = unique_bs[0]
        micro_batch_source = "DATALOADER.DATASET_BS"
    else:
        micro_batch_size = target_batch_per_gpu
        micro_batch_source = "SOLVER.IMS_PER_BATCH/world_size"

    if micro_batch_size > target_batch_per_gpu:
        raise ValueError(
            f"Per-GPU micro-batch {micro_batch_size} exceeds target per-GPU effective batch "
            f"{target_batch_per_gpu}. Increase SOLVER.IMS_PER_BATCH or reduce DATASET_BS."
        )
    if target_batch_per_gpu % micro_batch_size != 0:
        raise ValueError(
            f"Target per-GPU effective batch {target_batch_per_gpu} must be divisible by the "
            f"per-GPU micro-batch {micro_batch_size} inferred from {micro_batch_source}."
        )

    accumulation_steps = target_batch_per_gpu // micro_batch_size
    return {
        "world_size": world_size,
        "total_batch_size": total_batch_size,
        "target_batch_per_gpu": target_batch_per_gpu,
        "micro_batch_size": micro_batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": micro_batch_size * world_size * accumulation_steps,
        "micro_batch_source": micro_batch_source,
    }


class _GradientAccumulationMixin:
    def __init__(self, *args, accumulation_steps=1, **kwargs):
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {self.accumulation_steps}"
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_micro_batch_size(data):
        if isinstance(data, (list, tuple)):
            return len(data)
        try:
            return len(data)
        except TypeError:
            return 1

    def _no_sync_context(self, is_last_micro_step):
        if is_last_micro_step or not hasattr(self.model, "no_sync"):
            return contextlib.nullcontext()
        return self.model.no_sync()

    def _write_aggregated_metrics(self, loss_sums, total_samples, data_time):
        normalizer = max(total_samples, 1)
        reduced_loss_dict = {
            key: value / normalizer for key, value in loss_sums.items()
        }

        if getattr(self, "async_write_metrics", False):
            self.concurrent_executor.submit(
                self._write_metrics, reduced_loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(reduced_loss_dict, data_time)


class GradientAccumulationSimpleTrainer(_GradientAccumulationMixin, SimpleTrainer):
    def run_step(self):
        assert self.model.training, "[GradientAccumulationSimpleTrainer] model was changed to eval mode!"

        self.optimizer.zero_grad()
        total_data_time = 0.0
        total_samples = 0
        loss_sums = {}

        for micro_step in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            total_data_time += time.perf_counter() - start

            batch_size = self._get_micro_batch_size(data)
            total_samples += batch_size
            is_last_micro_step = micro_step == self.accumulation_steps - 1

            with self._no_sync_context(is_last_micro_step):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

                scaled_losses = losses / self.accumulation_steps
                scaled_losses.backward()

            for key, value in loss_dict.items():
                detached_value = value.detach()
                loss_sums[key] = loss_sums.get(key, detached_value.new_zeros(())) + detached_value * batch_size

        if hasattr(self, "after_backward"):
            self.after_backward()
        self._write_aggregated_metrics(loss_sums, total_samples, total_data_time)
        self.optimizer.step()


class GradientAccumulationAMPTrainer(_GradientAccumulationMixin, AMPTrainer):
    def run_step(self):
        assert self.model.training, "[GradientAccumulationAMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[GradientAccumulationAMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        self.optimizer.zero_grad()
        total_data_time = 0.0
        total_samples = 0
        loss_sums = {}

        for micro_step in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            total_data_time += time.perf_counter() - start

            batch_size = self._get_micro_batch_size(data)
            total_samples += batch_size
            is_last_micro_step = micro_step == self.accumulation_steps - 1

            with self._no_sync_context(is_last_micro_step):
                with autocast():
                    loss_dict = self.model(data)
                    if isinstance(loss_dict, torch.Tensor):
                        losses = loss_dict
                        loss_dict = {"total_loss": loss_dict}
                    else:
                        losses = sum(loss_dict.values())

                scaled_losses = losses / self.accumulation_steps
                self.grad_scaler.scale(scaled_losses).backward()

            for key, value in loss_dict.items():
                detached_value = value.detach()
                loss_sums[key] = loss_sums.get(key, detached_value.new_zeros(())) + detached_value * batch_size

        if getattr(self, "log_grad_scaler", False):
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        if hasattr(self, "after_backward"):
            self.after_backward()
        self._write_aggregated_metrics(loss_sums, total_samples, total_data_time)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to FCCLIP.
    """

    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()

        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        accumulation_info = infer_gradient_accumulation(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        trainer_cls = (
            GradientAccumulationAMPTrainer if cfg.SOLVER.AMP.ENABLED else GradientAccumulationSimpleTrainer
        )
        self._trainer = trainer_cls(
            model,
            data_loader,
            optimizer,
            accumulation_steps=accumulation_info["accumulation_steps"],
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.accumulation_info = accumulation_info

        logging.getLogger("detectron2").info(
            "Gradient accumulation enabled: per_gpu_micro_batch=%d, world_size=%d, "
            "accumulation_steps=%d, effective_batch_size=%d (SOLVER.IMS_PER_BATCH=%d)",
            accumulation_info["micro_batch_size"],
            accumulation_info["world_size"],
            accumulation_info["accumulation_steps"],
            accumulation_info["effective_batch_size"],
            accumulation_info["total_batch_size"],
        )

        self.register_hooks(self.build_hooks())


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []


        if "lvis" in dataset_name:
            print("=" * 20, "Using the LVIS evaluator", "=" * 20)
            evaluator_list.append( LVISEvaluator(dataset_name, cfg, True, output_folder))
            return evaluator_list

        if "cityscape" in dataset_name:
            print("=" * 20, "Using the Cityscapes evaluator", "=" * 20)
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
            return evaluator_list

        if cfg.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))


        if cfg.TEST.PANOPTIC_ON:
            evaluator_list.append(
                COCOPanopticEvaluator(dataset_name, output_dir=output_folder)
            )


        if cfg.TEST.SEMANTIC_ON:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )


        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "No Evaluator generated. Please check your cfg.TEST.*_ON settings "
                "or the dataset evaluator_type."
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)


    @classmethod
    def build_train_loader(cls, cfg):

        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler":
            mapper = COCOCombineNewBaselineDatasetMapper(cfg, True)
            data_loader = build_custom_train_loader(cfg, mapper=mapper)
            return data_loader


        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)

        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_semantic_lsj":
            mapper = COCOSemanticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_combine_lsj":
            mapper = COCOCombineNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            print(f"Mapper {cfg.INPUT.DATASET_MAPPER_NAME} does not exist!")
            exit()
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,

            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue

                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] *  cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):

            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sam3_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    cfg.eval_only = args.eval_only

    cfg.freeze()
    default_setup(cfg, args)

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sam3")
    return cfg


def main(args):

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )


        res = Trainer.test(cfg, model)


        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
