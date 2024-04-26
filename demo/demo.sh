#!/bin/bash
# Path to the input directory
input_dir="./inputs"

# Base command setup without the --opts part
base_cmd="python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --output ../outputs"

# Initialize input_files string
input_files=""

# Loop through all files in the input directory
for file in $input_dir/*; do
    input_files+=" $file"
done

# Options to be added last
opts="--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

# Complete command with input files, then --opts
full_cmd="$base_cmd --input $input_files $opts"

# Display the command to be run for verification
echo "Running command: $full_cmd"

# Execute the command
eval $full_cmd