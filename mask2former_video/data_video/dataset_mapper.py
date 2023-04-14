# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

from .augmentation import build_augmentation, build_pseudo_augmentation #build_coco_augmentation
from .datasets.ytvis import COCO_TO_YTVIS_2019, COCO_TO_YTVIS_2021
import os

from pycocotools import mask as coco_mask

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper"]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances



def _get_dummy_anno():
    return {
        "iscrowd": 0,
        "category_id": -1,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }

def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        is_tgt: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        src_dataset_name: str = "",
        tgt_dataset_name: str = "",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.is_tgt                 = is_tgt
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes

        if not is_tgt:
            self.src_metadata = MetadataCatalog.get(src_dataset_name)
            self.tgt_metadata = MetadataCatalog.get(tgt_dataset_name)
            print('tgt_dataset_name:', tgt_dataset_name)
            if tgt_dataset_name.startswith("ytvis_2019"):
                src2tgt = OVIS_TO_YTVIS_2019
            elif tgt_dataset_name.startswith("ytvis_2021"):
                src2tgt = OVIS_TO_YTVIS_2021
            elif tgt_dataset_name.startswith("ovis"):
                if src_dataset_name.startswith("ytvis_2019"):
                    src2tgt = YTVIS_2019_TO_OVIS
                elif src_dataset_name.startswith("ytvis_2021"):
                    src2tgt = YTVIS_2021_TO_OVIS
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            self.src2tgt = {}
            for k, v in src2tgt.items():
                self.src2tgt[
                    self.src_metadata.thing_dataset_id_to_contiguous_id[k]
                ] = self.tgt_metadata.thing_dataset_id_to_contiguous_id[v]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, is_tgt: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "is_tgt": is_tgt,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "tgt_dataset_name": cfg.DATASETS.TRAIN[-1],
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["video_len"] = len(video_annos)
        dataset_dict["frame_idx"] = list(selected_idx)
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno() for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            if not self.is_tgt:
                instances.gt_classes = torch.tensor(
                    [self.src2tgt[c] if c in self.src2tgt else -1 for c in instances.gt_classes.tolist()]
                )
            instances.gt_ids = torch.tensor(_gt_ids)
            instances = filter_empty_instances(instances)
            # if instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            #     instances = filter_empty_instances(instances)
            if not instances.has("gt_masks"):
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        is_tgt: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        src_dataset_name: str = "",
        tgt_dataset_name: str = "",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train               = is_train
        self.is_tgt                 = is_tgt
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range

        if not is_tgt:
            self.src_metadata = MetadataCatalog.get(src_dataset_name)
            self.tgt_metadata = MetadataCatalog.get(tgt_dataset_name)
            if tgt_dataset_name.startswith("ytvis_2019"):
                src2tgt = COCO_TO_YTVIS_2019
            elif tgt_dataset_name.startswith("ytvis_2021"):
                src2tgt = COCO_TO_YTVIS_2021
            elif tgt_dataset_name.startswith("ovis"):
                src2tgt = COCO_TO_OVIS
            else:
                raise NotImplementedError

            self.src2tgt = {}
            for k, v in src2tgt.items():
                self.src2tgt[
                    self.src_metadata.thing_dataset_id_to_contiguous_id[k]
                ] = self.tgt_metadata.thing_dataset_id_to_contiguous_id[v]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, is_tgt: bool = True):
        if is_tgt:
            augs = build_augmentation(cfg, is_train)
        else:
            # print('come here')
            augs = build_pseudo_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.PSEUDO.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.PSEUDO.SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "is_tgt": is_tgt,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "tgt_dataset_name": cfg.DATASETS.TRAIN[-1],
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        if self.is_train:
            video_length = random.randrange(16, 49)
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
        else:
            video_length = self.sampling_frame_num
            selected_idx = list(range(self.sampling_frame_num))

        dataset_dict["video_len"] = video_length
        dataset_dict["frame_idx"] = selected_idx
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(annos, image_shape)
            if not self.is_tgt:
                instances.gt_classes = torch.tensor(
                    [self.src2tgt[c] if c in self.src2tgt else -1 for c in instances.gt_classes.tolist()]
                )
            instances.gt_ids = torch.tensor(_gt_ids)
            # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # NOTE we don't need boxes
            instances = filter_empty_instances(instances)
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            else:
                instances.gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)
            dataset_dict["instances"].append(instances)

        return dataset_dict
