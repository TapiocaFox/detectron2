import cv2
import os
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt

from dataloader import register_datasets

setup_logger()

view_type = "visible" # visible, lwir
set_name = "set11"
# Register datasets if not already registered
data_dir = 'datasets'  # Update this if your data directory is different
register_datasets(data_dir)
model_path = "./outputs/frcnn"
with open(os.path.join("./outputs/frcnn", "last_checkpoint"), 'r') as file:
    last_checkpoint_path = os.path.join("./outputs/frcnn", file.read())

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("../../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = last_checkpoint_path  # load the last model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (people)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("kaist_set11_visible", )  # set the test dataset
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Run on GPU if available
    return cfg

cfg = setup_cfg()
predictor = DefaultPredictor(cfg)

# Define your dataset and metadata
dataset_dicts = DatasetCatalog.get(f"kaist_{set_name}_{view_type}")
kaist_metadata = MetadataCatalog.get(f"kaist_{set_name}_{view_type}")

# Specify the output directory
output_dir = "./outputs/frcnn/inference_images"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
for d in dataset_dicts:
    image_id = d["image_id"]
    img = cv2.imread(d["file_name"])
    # print(img)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], metadata=kaist_metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print(outputs)
    # print(d["instances"])
    img_with_boxes = out.get_image()[:, :, ::-1]

    plt.figure(figsize=(10, 5))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.show()

    output_path = os.path.join(output_dir, f"annotated_{image_id}.jpg")
    cv2.imwrite(output_path, img_with_boxes)  # corrected to use the correct image variable
    plt.close()
    print(f"{image_id}: {output_path}")
