#!/bin/bash

# Activate the conda environment
source /home/siddu/miniconda3/bin/activate fall

# Run the complete multi-scenario detection system
python new_file.py

echo "The system is now running. Open http://localhost:5000 in your browser to access the dashboard."
