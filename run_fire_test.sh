#!/bin/bash

# Activate the conda environment
source /home/siddu/miniconda3/bin/activate fall

# Run the fire detection test
python test_fire_detection.py

echo "Fire detection test completed."