# sam3/config.py
from detectron2.config import CfgNode as CN

def add_sam3_config(cfg):
    ""
    cfg = cfg
    
    cfg.MODEL.SAM3 = CN()
    
    cfg.MODEL.SAM3.MODEL_TYPE = "sam3_image" 
    cfg.MODEL.SAM3.BPE_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz" 
    

    
    cfg.MODEL.SAM3.ENABLE_SEGMENTATION = True
    cfg.MODEL.SAM3.ENABLE_INST_INTERACTIVITY = False
    cfg.MODEL.SAM3.COMPILE = False

    cfg.MODEL.SAM3.PREDICT_MODE = "Cycle"
    cfg.MODEL.SAM3.STREAMING_SEMSEG = True
    cfg.MODEL.SAM3.CYCLE_INFER_GT_CLASSES_ONLY = False
    

    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.IMAGE_SIZE = 1008
    cfg.INPUT.SAM3_SQUARE_SIZE = 1008
    cfg.INPUT.COLOR_AUG_SSD = False    
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0


    cfg.MODEL.PIXEL_MEAN = [127.5, 127.5, 127.5]
    cfg.MODEL.PIXEL_STD = [127.5, 127.5, 127.5]

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.0
    cfg.SOLVER.NO_OBJECT_WEIGHT = 0.1
    cfg.SOLVER.CLASS_WEIGHT = 2.0
    cfg.SOLVER.MASK_WEIGHT = 5.0
    cfg.SOLVER.DICE_WEIGHT = 5.0
    cfg.SOLVER.TRAIN_NUM_POINTS = 12544
    cfg.SOLVER.OVERSAMPLE_RATIO = 3.0
    cfg.SOLVER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.SOLVER.USE_AUX = True

    cfg.TEST.SEMANTIC_ON = True
    cfg.TEST.INSTANCE_ON = False
    cfg.TEST.PANOPTIC_ON = False
    cfg.TEST.DETECTIONS_PER_IMAGE = 100 
    cfg.TEST.PANOPTIC_OBJECT_MASK_THRESHOLD = 0.01
    cfg.TEST.PANOPTIC_OVERLAP_THRESHOLD = 0.8
    cfg.TEST.PANOPTIC_CHUNK_SIZE = 16
