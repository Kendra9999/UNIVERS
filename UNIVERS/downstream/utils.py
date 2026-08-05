import os
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from univers import *


def init(config, checkpoint):
    cfg = Config.fromfile(config)
    cfg.gpu_ids = range(1)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()
    return model


def get_embedding(batched_data, model):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **batched_data)
    return result


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())