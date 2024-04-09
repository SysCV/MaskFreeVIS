from detectron2.config import CfgNode as CN


def add_data_fusion_block_config(cfg):

    cfg.MODEL.DATAFUSION = CN()
    cfg.MODEL.DATAFUSION.STATUS = False
    cfg.MODEL.DATAFUSION.NAME = "build_optical_flow_fusion_block"
    cfg.MODEL.DATAFUSION.NORM = "BN"

    # Channel size of datafusion block output for feed the backbone
    cfg.MODEL.DATAFUSION.OUT_FEATURES = 3
    
    # 1x1 conv in features and output features which is takes as input raw image
    cfg.MODEL.DATAFUSION.RAW_IMAGE = CN()
    cfg.MODEL.DATAFUSION.RAW_IMAGE.IN_FEATURES = 3
    cfg.MODEL.DATAFUSION.RAW_IMAGE.OUT_FEATURES = 8

    # 1x1 conv in features and output features which is takes as input optical flow feature map
    cfg.MODEL.DATAFUSION.OPTICAL_FLOW = CN()
    cfg.MODEL.DATAFUSION.OPTICAL_FLOW.IN_FEATURES = 3
    cfg.MODEL.DATAFUSION.OPTICAL_FLOW.OUT_FEATURES = 8
