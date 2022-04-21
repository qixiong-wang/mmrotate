from mmdet.models.builder import DETECTORS
from .single_stage import RotatedSingleStageDetector


@DETECTORS.register_module()
class FCOSOBB(RotatedSingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(FCOSOBB, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained)