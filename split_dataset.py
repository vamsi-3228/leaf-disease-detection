import os
import shutil
import random

# CHANGE THIS PATH to where your original dataset is stored
SOURCE_DIR = "PlantVillage"

# Target directories
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Create target directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Split ratio
SPLIT_RATIO = 0.8

# Loop through each disease folder
for folder_name in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder_name)
    
    if not os.path.isdir(folder_path):
        continue

    images = os.listdir(folder_path)
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_point]
    val_images = images[split_point:]

    # Create class folders
    os.makedirs(os.path.join(TRAIN_DIR, folder_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, folder_name), exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(folder_path, img)
        dst = os.path.join(TRAIN_DIR, folder_name, img)
        shutil.copy(src, dst)

    # Copy validation images
    for img in val_images:
        src = os.path.join(folder_path, img)
        dst = os.path.join(VAL_DIR, folder_name, img)
        shutil.copy(src, dst)

print("Dataset split completed successfully!")