import os
import sys
import time
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.utils.logger import setup_logger
import detectron2.utils as utils
# from detectron2.data.detection_utils import read_image
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator

from dataloader import register_datasets
from detectron2.modeling import BACKBONE_REGISTRY
from backbones import DebugVisibleResNet50FPNBackbone, ToyBackbone

from utils import VisableLwirDatasetMapper, FusionTrainer, PIXEL_MEAN, PIXEL_STD
# Initialize logger
setup_logger()


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    def after_step(self):
        if (self.trainer.iter + 1) % self._cfg.TEST.EVAL_PERIOD == 0 or self.trainer.iter + 1 == self._cfg.SOLVER.MAX_ITER:
            self.trainer.model.eval()
            evaluator = COCOEvaluator(self._cfg.DATASETS.TEST[0], self._cfg, False, output_dir=self._cfg.OUTPUT_DIR)
            results = evaluator.evaluate(self.trainer.model, self._loader)
            self.trainer.model.train()
            storage = get_event_storage()
            if "bbox/AP" in results:
                storage.put_scalar("val_bbox/AP", results["bbox/AP"], smoothing_hint=False)

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("../../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Training Dataset
    cfg.DATASETS.TRAIN = ("kaist_set00_visible_lwir",) # "kaist_set01_visible_lwir", "kaist_set02_visible_lwir", "kaist_set03_visible_lwir", "kaist_set04_visible_lwir", "kaist_set05_visible_lwir", "kaist_set06_visible_lwir", "kaist_set07_visible_lwir", "kaist_set08_visible_lwir", "kaist_set09_visible_lwir", "kaist_set10_visible_lwir", )  # Example, add other dataset names if required
    cfg.DATASETS.TEST = ("kaist_set11_visible_lwir", )  # Add test dataset names here if validation is required
    # cfg.DATALOADER.MAPPER = VisableLwirDatasetMapper
    # Number of data loading workers
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.BACKBONE.NAME = "DebugVisibleResNet50FPNBackbone"  # Use the custom backbone
    cfg.MODEL.BACKBONE.PRETRAINED = True  # Set to False if you do not want to use pre-trained weights

    # Initial weights path: load weights from a model pretrained on COCO
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

    # Learning rate and batch size adjustments
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 150000  # Adjust based on your dataset size
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # No learning rate decay scheduling
    cfg.SOLVER.STEPS = []  # No learning rate decay scheduling
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Typically 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (people)

    # 4 Channels config
    cfg.MODEL.PIXEL_MEAN = PIXEL_MEAN
    cfg.MODEL.PIXEL_STD = PIXEL_STD
    cfg.IMAGE_SHAPE = 4

    cfg.TEST.EVAL_PERIOD = 1000
    # Output directory for training artifacts
    cfg.OUTPUT_DIR = "./outputs/debugfusionfrcnn"

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
