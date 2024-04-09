import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from mask2former_video.data_video.datasets.ytvis import register_ytvis_instances

from maskfreevis.config import get_cfg
from maskfreevis.data_fusion_modeling import add_data_fusion_block_config

from train_net_video import Trainer
from utils import ValidationLoss, CUDAMemoryOptimizer


# Define datasets
trainDatasetName = "ytvis_train_unity"
trainAnnPath = "datasets/ytvis_2019/train.json"
trainImagesDir = "datasets/ytvis_2019/train/JPEGImages"

validDatasetName = "ytvis_valid_unity"
validAnnPath = "datasets/ytvis_2019/valid.json"
validImagesDir = "datasets/ytvis_2019/valid/JPEGImages"

testDatasetName = "ytvis_test_unity"
testAnnPath = "datasets/ytvis_2019/test.json"
testImagesDir = "datasets/ytvis_2019/test/JPEGImages"

ret = {
    "thing_classes": ["arac", "insan"],
    "thing_colors": [(0, 0, 142), (220, 20, 60)],
}   

register_ytvis_instances(trainDatasetName, ret, trainAnnPath, trainImagesDir)
register_ytvis_instances(validDatasetName, ret, validAnnPath, validImagesDir)
register_ytvis_instances(testDatasetName, ret, testAnnPath, testImagesDir)

train_metadata = MetadataCatalog.get(trainDatasetName)

# Set model configs
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_maskformer2_video_config(cfg)
add_data_fusion_block_config(cfg)
cfg.merge_from_file("configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml")

cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(train_metadata.thing_classes)
cfg.MODEL.WEIGHTS = "mfvis_models/model_final_r50_0466.pth"
cfg.DATALOADER.NUM_WORKERS = 4
cfg.DATASETS.TRAIN = (trainDatasetName,)
cfg.DATASETS.TEST = (validDatasetName,)
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.MAX_ITER = 50000
cfg.MODEL.DEVICE = "cuda"
cfg.OUTPUT_DIR = "models/MaskFreeVIS_R50_50k_iter_1x_pretrained"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

with open("models/MaskFreeVIS_R50_50k_iter_1x_pretrained/config.yaml", "w") as f:
  f.write(cfg.dump()) 

# Model train
trainer = Trainer(cfg)
trainer.register_hooks([CUDAMemoryOptimizer(),
                        ValidationLoss(cfg)])
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
trainer.resume_or_load(resume=True)
trainer.train()

# Model eval
cfg.DATASETS.TEST = (testDatasetName,)
Trainer.test(cfg, trainer.model)