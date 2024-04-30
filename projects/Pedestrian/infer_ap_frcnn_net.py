import os
import cv2
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from dataloader import register_datasets

setup_logger()

view_type = "visible"  # visible, lwir
set_name = "set01"
# set_name = "set11"
# Register datasets if not already registered
data_dir = 'datasets'  # Update this if your data directory is different
register_datasets(data_dir)
model_path = "./outputs/frcnn"
with open(os.path.join(model_path, "last_checkpoint"), 'r') as file:
    last_checkpoint_path = os.path.join(model_path, file.read().strip())

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("../../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = last_checkpoint_path  # load the last model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (people)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = (f"kaist_{set_name}_{view_type}", )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Run on GPU if available
    return cfg

cfg = setup_cfg()
predictor = DefaultPredictor(cfg)

# Specify the output directory
output_dir = "./outputs/frcnn/inference_images"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Create evaluator
evaluator = COCOEvaluator(f"kaist_{set_name}_{view_type}", cfg, False, output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, f"kaist_{set_name}_{view_type}")

# Perform evaluation
inference_results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(inference_results)
