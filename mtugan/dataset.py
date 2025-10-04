import os
import numpy as np
import cv2
from PIL import Image
import albumentations as A
import torch
from torch.utils.data import Dataset

# Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_dirs, img_size=(256, 256), split='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_dirs = class_dirs
        self.img_size = img_size
        self.split = split
        self.class_to_idx = {'DG': 0, 'CA1': 1, 'CA3': 2}
        self.image_paths, self.mask_paths, self.labels = self.get_file_paths()
        # self.mean, self.std = self.compute_statistics()

    def get_file_paths(self):
        class_masks = {}
        for class_name, class_dir in self.class_dirs.items():
            for mask_name in os.listdir(class_dir):
                class_masks[mask_name] = class_name

        image_paths, mask_paths, labels = [], [], []
        mismatch_count = 0

        for img_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, img_name)
            # mask_name1 = img_name.replace('.tif', '_ch02_mask.png')
            mask_name1 = img_name.replace('_ch00.tif', '_ch02_mask.png')
            mask_name_for_loading_neun = img_name.replace('.tif','_mask.png')
            mask_name2 = img_name.replace('.tif', 'Mask.tif')

            mask_path1 = os.path.join(self.mask_dir, mask_name_for_loading_neun)
            mask_path2 = os.path.join(self.mask_dir, mask_name2)
            # print(len(os.listdir(self.image_dir)),img_name, mask_name1)
            if os.path.exists(mask_path1):
                mask_path = mask_path1
                mask_name = mask_name1
            elif os.path.exists(mask_path2):
                mask_path = mask_path2
                mask_name = mask_name2
            else:
                mismatch_count += 1
                continue

            if mask_name in class_masks:
                image_paths.append(img_path)
                mask_paths.append(mask_path)
                labels.append(self.class_to_idx[class_masks[mask_name]])
            else:
                print(mask_name)
                mismatch_count += 1
                # print(f"Class mismatch for mask: {mask_name}")

        print(f"Matched samples: {len(image_paths)} (DG: {labels.count(0)}, CA1: {labels.count(1)}, CA3: {labels.count(2)})")
        print(f"Mismatched samples: {mismatch_count}")
        if not image_paths:
            raise ValueError("No matched samples found. Check filename consistency.")
        return image_paths, mask_paths, labels

    def compute_statistics(self):
        if not self.image_paths:
            print("Warning: No images to compute statistics. Using defaults.")
            return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
        images = [cv2.resize(np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0, self.img_size) 
                  for p in self.image_paths[:50]]
        return np.stack(images).mean(axis=(0, 1, 2)), np.stack(images).std(axis=(0, 1, 2))

    def preprocess(self, image, mask):
        image = image/255.0 #(image / 255.0 - self.mean) / self.std
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        return image.astype(np.float32), mask.astype(np.float32)

    def augment_image(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.ElasticTransform(p=0.3),
            A.Resize(height=self.img_size[0], width=self.img_size[1])
        ])
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        image, mask = self.preprocess(np.array(img), np.array(mask))
        # Apply augmentation only to training split
        if self.split == 'train':
            image, mask = self.augment_image(image, mask)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, mask, label

# Compute class weights
def compute_class_weights(labels):
    class_counts = [labels.count(i) for i in range(3)]
    total = sum(class_counts)
    weights = [total / (3 * count) if count > 0 else 0 for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)
