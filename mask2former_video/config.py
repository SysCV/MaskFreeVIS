# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 3 
    cfg.INPUT.SAMPLING_FRAME_RANGE = 5
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = True 
    cfg.INPUT.AUGMENTATIONS = [] 

    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.SAMPLING_FRAME_NUM = 4 
    cfg.INPUT.PSEUDO.SAMPLING_FRAME_RANGE = 20 
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

