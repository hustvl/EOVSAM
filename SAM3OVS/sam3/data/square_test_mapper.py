import copy
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class SAM3SquareTestDatasetMapper:
    @configurable
    def __init__(self, *, image_format, square_size):
        self.image_format = image_format
        self.square_size = square_size

    @classmethod
    def from_config(cls, cfg):
        size = cfg.INPUT.SAM3_SQUARE_SIZE
        if size <= 0:
            raise ValueError("INPUT.SAM3_SQUARE_SIZE must be positive.")
        return {"image_format": cfg.INPUT.FORMAT, "square_size": size}

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        image = T.ResizeTransform(image.shape[0], image.shape[1], self.square_size, self.square_size).apply_image(image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict