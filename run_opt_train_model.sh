#!/bin/bash

# This script removes all jpg files in opt_img/ directory and runs a Python script

echo "Removing all JPG files in opt_img/ directory..."
rm opt_img/*.jpg

echo "Running Python script..."
python opt_train.py --iters 5000 --lr 1e-3 --device cuda:0 --style_weight 1e5 --content_weight 1 --tv_weight 1 --content_img_path VanGogh.jpg --style_name rst/content/city_center
echo "Script completed."
