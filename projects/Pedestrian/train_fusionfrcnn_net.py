import os
import sys
import time
import torch
from torchvision import transforms
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.utils.logger import setup_logger
import detectron2.utils as utils
# from detectron2.data.detection_utils import read_image
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.structures import Instances, Boxes

from dataloader import register_datasets
from detectron2.modeling import BACKBONE_REGISTRY
from backbones import DualStreamFusionBackbone

from PIL import Image

# Initialize logger
setup_logger()

convert_tensor = transforms.ToTensor()

class VisableLwirDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.augmentations = [ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )] if is_train else []

        self.transform = transforms.Compose([
            transforms.Normalize(mean=cfg.MODEL.PIXEL_MEAN, std=cfg.MODEL.PIXEL_STD)
        ])

    def __call__(self, dataset_dict):
        # dataset_dict = super().__call__(dataset_dict)
        visible_tensor = torch.as_tensor(convert_tensor(Image.open(dataset_dict["file_name"]["visible"])))
        lwir_tensor = torch.as_tensor(convert_tensor(Image.open(dataset_dict["file_name"]["lwir"]).convert('L')))
        # print(visible_tensor.shape, lwir_tensor.shape)
        visible_lwir_tensor = torch.cat([visible_tensor, lwir_tensor], dim=0)
        dataset_dict["image"] = self.transform(visible_lwir_tensor)
        
        if "annotations" in dataset_dict and self.is_train:
            boxes_list = []
            classes_list = []
            for obj in dataset_dict["annotations"]:
                # Assuming "bbox" and "category_id" are keys in obj
                # print(obj)
                boxes_list.append(obj["bbox"])
                classes_list.append(obj["category_id"])

            # Convert lists to tensors
            gt_boxes = torch.tensor(boxes_list, dtype=torch.float32)
            gt_classes = torch.tensor(classes_list, dtype=torch.int64)
            
            # Create Instances object
            instances = Instances(dataset_dict["image"].shape[1:])
            instances.gt_boxes = Boxes(gt_boxes)
            instances.gt_classes = gt_classes
            # print(instances)
            
            dataset_dict["instances"] = instances

        return dataset_dict
        # return visible_tensor

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

class Trainer(DefaultTrainer):
    """
    We subclass DefaultTrainer to add a custom evaluator (if necessary).
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=VisableLwirDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=VisableLwirDatasetMapper(cfg, is_train=False))
    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1, ValidationLoss(self.cfg))
    #     return hooks


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("../../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Training Dataset
    cfg.DATASETS.TRAIN = ("kaist_set00_visible_lwir", "kaist_set01_visible_lwir", "kaist_set02_visible_lwir", "kaist_set03_visible_lwir", "kaist_set04_visible_lwir", "kaist_set05_visible_lwir", "kaist_set06_visible_lwir", "kaist_set07_visible_lwir", "kaist_set08_visible_lwir", "kaist_set09_visible_lwir", "kaist_set10_visible_lwir", )  # Example, add other dataset names if required
    cfg.DATASETS.TEST = ("kaist_set11_visible_lwir", )  # Add test dataset names here if validation is required
    # cfg.DATALOADER.MAPPER = VisableLwirDatasetMapper
    # Number of data loading workers
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.BACKBONE.NAME = "DualStreamFusionBackbone"  # Use the custom backbone
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
    cfg.MODEL.PIXEL_MEAN = [103.53, 116.28, 123.675, 120]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
    cfg.IMAGE_SHAPE = 4

    cfg.TEST.EVAL_PERIOD = 1000
    # Output directory for training artifacts
    cfg.OUTPUT_DIR = "./outputs/fusionfrcnn"

    return cfg

def main():
    # Check if a command line argument is provided for resuming training
    resume_training = False
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':
        resume_training = True
        print("Resume training...")

    data_dir = 'dataset'  # Change as per your dataset location 
    register_datasets(data_dir)  # Registering the dataset

    cfg = setup_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    if resume_training:
        # Automatically resume from the last checkpoint if one exists
        trainer.resume_or_load(resume=True)
    else:
        # Start training from scratch (or from pre-trained weights as specified)
        trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == "__main__":
    main()
