import torch
import numpy as np
from detectron2.data import DatasetMapper
from detectron2.data.transforms import ResizeShortestEdge
from torchvision import transforms
from detectron2.structures import Instances, Boxes
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from PIL import Image

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
        # print("shape")
        # print(visible_tensor.shape)
        # print(lwir_tensor)
        # print(visible_tensor)
        
        visible_lwir_tensor = torch.cat([visible_tensor, lwir_tensor], dim=0)
        dataset_dict["image"] =  self.transform(visible_lwir_tensor)
        # dataset_dict["image"] =  visible_lwir_tensor
        
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
        # print(dataset_dict)
        return dataset_dict
        # return visible_tensor

PIXEL_MEAN = [0.3465, 0.3219, 0.2842, 0.1598]
PIXEL_STD = [1.0, 1.0, 1.0, 1.0]

def normalize(image, mean, std):
    # Convert mean and std to numpy arrays if they aren't already
    mean = np.array(mean)
    std = np.array(std)
    # Normalize the image
    normalized_image = (image - mean) / std
    return normalized_image



class FusionTrainer(DefaultTrainer):
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
