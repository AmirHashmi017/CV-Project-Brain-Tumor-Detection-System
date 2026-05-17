import os
import shutil
import random

source_dir = "dataset/Kindey Stone Dataset/Original"
train_dir = "dataset/split/train"
test_dir = "dataset/split/test"

classes = ["Non-Stone", "Stone"]
split_ratio = 0.8

print("Starting to split the dataset...")

for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    src_cls_dir = os.path.join(source_dir, cls)
    if not os.path.exists(src_cls_dir):
        print(f"Directory {src_cls_dir} not found.")
        continue
        
    images = os.listdir(src_cls_dir)
    random.shuffle(images)
    
    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    for img in train_images:
        shutil.copy(os.path.join(src_cls_dir, img), os.path.join(train_dir, cls, img))
        
    for img in test_images:
        shutil.copy(os.path.join(src_cls_dir, img), os.path.join(test_dir, cls, img))
        
    print(f"Class '{cls}' - Train: {len(train_images)}, Test: {len(test_images)}")

print("Dataset successfully split into train and test folders!")
