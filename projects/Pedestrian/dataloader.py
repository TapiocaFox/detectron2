import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def get_kaist_dicts(data_dir, set_name, view_type):
    dataset_dicts = []
    annotations_base = os.path.join(data_dir, "kaist-rgbt", "annotations_paired", set_name)
    
    for sequence in sorted(os.listdir(annotations_base)):
        seq_path = os.path.join(annotations_base, sequence)
        # print(sequence)
        for view in ["visible", "lwir"]:
            if view != view_type:
                continue
            img_dir = os.path.join(seq_path, view)
            for file in sorted(os.listdir(img_dir)):
                if not file.endswith(".txt"):
                    continue
                record = {}
                filename = os.path.join(img_dir, file)
                
                image_id = f"{set_name}_{view_type}_{sequence}_{file.replace('.txt', '')}"
                # print(file)
                height, width = 512, 640  # Assuming all images are the same size
                record["file_name"] = filename.replace(".txt", ".jpg").replace("annotations_paired", "images")
                record["image_id"] = image_id
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

def get_kaist_paired_dicts(data_dir, set_name):
    dataset_dicts = []
    annotations_base = os.path.join(data_dir, "kaist-rgbt", "annotations_paired", set_name)
    
    # Get directories for both image types
    for sequence in sorted(os.listdir(annotations_base)):
        visible_annotation_dir = os.path.join(annotations_base, sequence, "visible")
        lwir_annotation_dir = os.path.join(annotations_base, sequence, "lwir")
        visible_image_dir = visible_annotation_dir.replace("annotations_paired", "images")
        lwir_image_dir = lwir_annotation_dir.replace("annotations_paired", "images")
        for file in sorted(os.listdir(visible_annotation_dir)):
            if file.endswith(".txt"):
                record = {}
                base_filename = file.replace(".txt", "")
                visible_file = os.path.join(visible_image_dir, f"{base_filename}.jpg")
                lwir_file = os.path.join(visible_image_dir, f"{base_filename}.jpg")
                annotation_file = os.path.join(visible_annotation_dir, file)

                image_id = f"{set_name}_visible_lwir_{sequence}_{base_filename}"

                height, width = 512, 640  # Assuming all images are the same size

                record["file_name"] = {"visible": visible_file, "lwir": lwir_file}
                record["image_id"] = image_id
                record["height"] = height
                record["width"] = width

                # Load annotations from the visible image annotations
                objs = []
                with open(annotation_file) as f:
                    for line in f:
                        if line.startswith('%'):
                            continue
                        parts = line.strip().split()
                        category, x, y, w, h = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        obj = {
                            "bbox": [x, y, x + w, y + h],
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
            print("Dataset registered [Seperated]: "+dataset_name)
            
        # Register paired dataset
        paired_dataset_name = f"kaist_{set_name}_visible_lwir"
        DatasetCatalog.register(paired_dataset_name, lambda d=data_dir, s=set_name: get_kaist_paired_dicts(d, s))
        MetadataCatalog.get(paired_dataset_name).set(thing_classes=["people"])
        print("Dataset registered [ *Paired ]: " + paired_dataset_name)