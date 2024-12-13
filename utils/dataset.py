import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dir, annotations_file, transform=None):
        """
        Args:
            dir (str): Directory containing the images.
            annotations_file (str): Path to the annotations file in COCO format (JSON).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dir = dir
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        self.annotations_data = self.annotations['annotations']
        self.category_mapping = {cat['id']: cat['name'] for cat in self.annotations['categories']}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Filter annotations for this image
        annotations = [ann for ann in self.annotations_data if ann['image_id'] == img_info['id']]
        boxes = []
        labels = []
        for ann in annotations:
            # Convert COCO format bbox (x, y, width, height) to the proper format
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to (x_min, y_min, x_max, y_max)
            labels.append(ann['category_id'])

        # If no annotations are found, create an empty target
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Empty tensor for boxes
            labels = torch.zeros((0,), dtype=torch.int64)  # Empty tensor for labels

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            img, target = self.transform(img, target)

        return img, target
