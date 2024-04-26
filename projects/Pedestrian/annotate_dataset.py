from dataloader import register_datasets
import os
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

view_type = "visible" # visible/lwir
set_name = "set00"
# Use the function to register datasets
data_dir = 'dataset'  # Update this path
register_datasets(data_dir, set_names=[set_name])

# Define your dataset and metadata
dataset_dicts = DatasetCatalog.get(f"kaist_{set_name}_{view_type}")
kaist_metadata = MetadataCatalog.get(f"kaist_{set_name}_{view_type}")

# Specify the output directory
output_dir = "./outputs/ground_truth_annotations"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

for idx, d in enumerate(dataset_dicts):  # Save all images or just a subset
    
    # print(d)
    image_id = d["image_id"]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=kaist_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    
    # Define output path
    output_path = os.path.join(output_dir, f"annotated_{set_name}_{view_type}_{image_id}.jpg")
    cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
    print(f"{image_id}: {output_path}")
    # print(d["file_name"])
