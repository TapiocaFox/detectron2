import os
import sys
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator

from dataloader import register_datasets

# Initialize logger
setup_logger()
import time

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

    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1, ValidationLoss(self.cfg))
    #     return hooks

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("../../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Training Dataset
    cfg.DATASETS.TRAIN = ("kaist_set00_visible", "kaist_set01_visible", "kaist_set02_visible", "kaist_set03_visible", "kaist_set04_visible", "kaist_set05_visible", "kaist_set06_visible", "kaist_set07_visible", "kaist_set08_visible", "kaist_set09_visible", "kaist_set10_visible", )  # Example, add other dataset names if required
    cfg.DATASETS.TEST = ("kaist_set11_visible", )  # Add test dataset names here if validation is required

    # Number of data loading workers
    cfg.DATALOADER.NUM_WORKERS = 4

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

    cfg.TEST.EVAL_PERIOD = 1000
    # Output directory for training artifacts
    cfg.OUTPUT_DIR = "./outputs/frcnn"

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
