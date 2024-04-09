import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import copy
import random
import imageio

from torch.cuda.amp import autocast

from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from mask2former_video.data_video.datasets.ytvis import register_ytvis_instances

from maskfreevis.config import get_cfg
from maskfreevis.data_fusion_modeling import add_data_fusion_block_config

from demo_video.predictor import VisualizationDemo
from demo_video.visualizer import TrackVisualizer


def setup_config(metadata):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_data_fusion_block_config(cfg)
    cfg.merge_from_file("configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml")
    cfg.MODEL.WEIGHTS = "mfvis_models/model_final_r50_0466.pth"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(metadata.thing_classes)
    cfg.freeze()
    return cfg

def register_datasets():
    global trainDatasetName
    global testDatasetName

    trainAnnPath = "datasets/ytvis_2019/train.json"
    trainImagesDir = "datasets/ytvis_2019/train/JPEGImages"

    testAnnPath = "datasets/ytvis_2019/test.json"
    testImagesDir = "datasets/ytvis_2019/test/JPEGImages"

    ret = {"thing_colors": [[0, 0, 142], [220, 20, 60]]}

    try:
        register_ytvis_instances(trainDatasetName, ret, trainAnnPath, trainImagesDir)
        register_ytvis_instances(testDatasetName, ret, testAnnPath, testImagesDir)
    except:
        pass

def get_dataset_dict():
    global trainDatasetName
    global testDatasetName
    
    train_metadata = MetadataCatalog.get(testDatasetName)
    dataset_dicts = DatasetCatalog.get(testDatasetName)
    return train_metadata, dataset_dicts

def extract_frame_dic(dic, frame_idx):
    frame_dic = copy.deepcopy(dic)
    annos = frame_dic.get("annotations", None)
    if annos:
        frame_dic["annotations"] = annos[frame_idx]

    return frame_dic

def visualize_dataset_and_predict():
    global trainDatasetName
    global testDatasetName

    trainDatasetName = "ytvis_2019_train"
    testDatasetName = "ytvis_2019_test_unity"

    seedParam = 42
    numSamples = 5
    threshValue = 0.2
    dirname = "ytvis_2019_train_gt_visualize"
    predictOutputDir = f"ytvis_2019_train_predict_visualize_thresh_{threshValue}"

    random.seed(seedParam)
    register_datasets()
    train_metadata, dataset_dicts = get_dataset_dict()
    cfg = setup_config(train_metadata)
    demo = VisualizationDemo(cfg, train_metadata)
    
    for d in random.sample(dataset_dicts, numSamples):
        isFileExist = True
        vid_name = d["file_names"][0].split('/')[-2]
        if not os.path.isdir(os.path.join(dirname, vid_name)):
            isFileExist = False
            os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
    
        video_frames = []
        gt_images = []
        
        for idx, file_name in enumerate(d["file_names"]):
            img = read_image(file_name, format="BGR")
            video_frames.append(img)
        
            if not isFileExist:
                visualizer = TrackVisualizer(img[:, :, ::-1], metadata=train_metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
                fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
                vis.save(fpath)
                gt_images.append(vis.get_image())
        
        if gt_images:
            imageio.mimsave(os.path.join(dirname, vid_name) + ".gif", gt_images, fps=5)

        if not os.path.isdir(os.path.join(predictOutputDir, vid_name)):
            with autocast():
                predictions, visualized_output = demo.run_on_video(video_frames, threshValue)

            predicted_images = []
            os.makedirs(os.path.join(predictOutputDir, vid_name), exist_ok=True)

            for path, vis_output in zip(d["file_names"], visualized_output):
                out_filename = os.path.join(predictOutputDir, vid_name, os.path.basename(path))
                vis_output.save(out_filename)
                predicted_images.append(vis_output.get_image())
            imageio.mimsave(os.path.join(predictOutputDir, vid_name) + ".gif", predicted_images, fps=5)


if __name__ == "__main__":
    visualize_dataset_and_predict()