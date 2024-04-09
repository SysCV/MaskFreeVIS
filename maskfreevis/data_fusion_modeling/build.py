from .base import DataFusionBlock

from detectron2.utils.registry import Registry

DATAFUSION_REGISTRY = Registry("DATAFUSION")
DATAFUSION_REGISTRY.__doc__ = """
Registry for data fusions, which fusion raw images and additional features

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_optical_flow_fusion_block(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    If `cfg.MODEL.DATAFUSION.STATUS` parameter is true, 
    it returns the module of OpticalFlowFusionBlock class.
    """
    data_fusion_status = cfg.MODEL.DATAFUSION.STATUS
    
    if data_fusion_status:
        data_fusion_block_name = cfg.MODEL.DATAFUSION.NAME
        data_fusion_block = DATAFUSION_REGISTRY.get(data_fusion_block_name)(cfg)
        assert isinstance(data_fusion_block, DataFusionBlock)
        return data_fusion_block

    return None