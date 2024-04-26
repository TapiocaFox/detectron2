import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def get_kaist_dicts(data_dir, set_name, view_type):
    dataset_dicts = []
    annotations_base = os.path.join(data_dir, "kaist-rgbt", "annotations_paired", set_name)
    
    for sequence in os.listdir(annotations_base):
        seq_path = os.path.join(annotations_base, sequence)
        for view in ["visible", "lwir"]:
            if view != view_type:
                continue
            img_dir = os.path.join(seq_path, view)
            for file in os.listdir(img_dir):
                if not file.endswith(".txt"):
                    continue
                record = {}
                filename = os.path.join(img_dir, file)
                height, width = 512, 640  # Assuming all images are the same size
                record["file_name"] = filename.replace(".txt", ".jpg").replace("annotations_paired", "images")
                record["image_id"] = file.replace(".txt", "")
                record["height"] = height
                record["width"] = width

                objs = []
                with open(filename) as f:
                    for line in f:
                        if line.startswith('%'):
                            continue
                        parts = line.strip().split()
                        category, x, y, w, h = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        obj = {
                            "bbox": [x, y, x+w, y+h],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 0,  # Assuming 'people' is the only category
                        }
                        objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

def register_datasets(data_dir, set_names=["set00", "set01", "set02", "set03", "set04", "set05", "set06", "set07", "set08", "set09", "set10", "set11"]):
    for set_name in set_names:  # Example sets
        for view_type in ["visible", "lwir"]:
            dataset_name = f"kaist_{set_name}_{view_type}"
            DatasetCatalog.register(dataset_name, lambda d=data_dir, s=set_name, v=view_type: get_kaist_dicts(d, s, v))
            MetadataCatalog.get(dataset_name).set(thing_classes=["people"])
            print("Dataset registered: "+dataset_name)