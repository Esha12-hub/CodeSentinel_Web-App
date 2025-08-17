import os
import shutil
import random

# Set your paths
SOURCE_DIR = "C:\\Users\\Public\\Documents\\VS_Projects\\Web_App\\flowers"
DEST_DIR = "C:\\Users\\Public\\Documents\\VS_Projects\\Web_App\\flowers_split"

SPLIT_RATIO = 0.8                   # 80% training, 20% validation

# Make sure destination directories exist
for split in ['train', 'val']:
    for class_folder in os.listdir(SOURCE_DIR):
        os.makedirs(os.path.join(DEST_DIR, split, class_folder), exist_ok=True)

# Process each flower class
for class_folder in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_folder)
    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_point]
    val_images = images[split_point:]

    # Copy train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(DEST_DIR, 'train', class_folder, img)
        shutil.copy(src, dst)

    # Copy validation images
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(DEST_DIR, 'val', class_folder, img)
        shutil.copy(src, dst)

print("✅ Dataset split complete: 'flowers_split/train/' and 'flowers_split/val/'")
