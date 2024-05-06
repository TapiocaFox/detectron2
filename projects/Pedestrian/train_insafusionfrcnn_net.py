import os
import sys
import time
import torch
from torchvision import transforms
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import detectron2.utils as utils
# from detectron2.data.detection_utils import read_image
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.structures import Instances, Boxes

from dataloader import register_datasets
from detectron2.modeling import BACKBONE_REGISTRY
from backbones import INSAFusion


from utils import VisableLwirDatasetMapper, FusionTrainer, PIXEL_MEAN, PIXEL_STD

# Initialize logger
setup_logger()

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("../../configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")

    # Training Dataset
    cfg.DATASETS.TRAIN = ("kaist_set00_visible_lwir", "kaist_set01_visible_lwir", "kaist_set02_visible_lwir", "kaist_set03_visible_lwir", "kaist_set04_visible_lwir", "kaist_set05_visible_lwir", "kaist_set06_visible_lwir", "kaist_set07_visible_lwir", "kaist_set08_visible_lwir", "kaist_set09_visible_lwir", "kaist_set10_visible_lwir", )  # Example, add other dataset names if required
    cfg.DATASETS.TEST = ("kaist_set11_visible_lwir", )  # Add test dataset names here if validation is required
    # cfg.DATALOADER.MAPPER = VisableLwirDatasetMapper
    # Number of data loading workers
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.BACKBONE.NAME = "INSAFusion"  # Use the custom backbone
    cfg.MODEL.BACKBONE.PRETRAINED = True  # Set to False if you do not want to use pre-trained weights

    # Initial weights path: load weights from a model pretrained on COCO

    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl"

    # Learning rate and batch size adjustments
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 150000  # Adjust based on your dataset size
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # No learning rate decay scheduling
    cfg.SOLVER.STEPS = []  # No learning rate decay scheduling
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Typically 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (people)

    # 4 Channels config
    # cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675, 120]
    # cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
    cfg.MODEL.PIXEL_MEAN = PIXEL_MEAN
    cfg.MODEL.PIXEL_STD = PIXEL_STD
    cfg.IMAGE_SHAPE = 4

    cfg.TEST.EVAL_PERIOD = 1000
    # Output directory for training artifacts
    cfg.OUTPUT_DIR = "./outputs/insafusionfrcnn"

    return cfg

def main():
    # Check if a command line argument is provided for resuming training
    resume_training = False
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        resume_training = True
        print("Resume training...")

    data_dir = 'datasets'  # Change as per your dataset location 
    register_datasets(data_dir)  # Registering the dataset

    cfg = setup_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = FusionTrainer(cfg)
    if resume_training:
        # Automatically resume from the last checkpoint if one exists
        trainer.resume_or_load(resume=True)
    else:
        # Start training from scratch (or from pre-trained weights as specified)
        trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == "__main__":
    main()
