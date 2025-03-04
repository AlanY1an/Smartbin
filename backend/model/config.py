# config.py

import os

# Automatically count the number of classes from your dataset
DATASET_PATH = "../data/raw-data"  # Change this if needed

# If you wanna auto detect
# NUM_CLASSES = len([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
NUM_CLASSES = 12


print(f"Detected {NUM_CLASSES} classes in dataset.")

# You can add other configurations here if needed
