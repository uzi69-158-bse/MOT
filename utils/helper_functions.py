import yaml
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
import torch


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # If targets are nested lists of dictionaries, flatten them
    if isinstance(targets[0], list) and isinstance(targets[0][0], dict):
        targets = [t[0] for t in targets]
    elif not all(isinstance(t, dict) for t in targets):
        raise ValueError("Targets are not in the expected format (list of dictionaries).")

    # Maximum number of boxes in the batch for padding
    max_num_boxes = max(len(target['boxes']) for target in targets)
    print(f"Max number of boxes in the batch: {max_num_boxes}")  # Debugging

    # Transform images to proper size and tensor format
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fixed size (224x224 in this example)
    ])

    transformed_images = []
    for image in images:
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        transformed_images.append(transform(image))
    
    transformed_images = [transforms.ToTensor()(image) for image in transformed_images]

    # Process targets (bounding boxes and labels)
    padded_targets = []
    for target in targets:
        # Get bounding boxes and labels
        boxes = target['boxes']
        labels = target['labels']

        # Debugging: Print the boxes
        # print(f"Original boxes: {boxes}")  # Debugging
        
        # Filter out boxes with zero width or height (invalid boxes)
        valid_boxes_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        valid_boxes = boxes[valid_boxes_mask]
        valid_labels = labels[valid_boxes_mask]

        # # Debugging: Print after filtering
        # print(f"Valid boxes after filtering: {valid_boxes}")  # Debugging
        # print(f"Valid labels after filtering: {valid_labels}")  # Debugging

        # Handle empty targets (no valid boxes)
        if len(valid_boxes) == 0:
            print(f"Empty target detected, padding with default box.")  # Debugging
            valid_boxes = torch.zeros(1, 4)  # Use [0.0, 0.0, 1.0, 1.0] for a valid empty box
            valid_labels = torch.full((1,), 0, dtype=torch.int64)  # Use 0 as the default label
        else:
            # Pad remaining boxes to match the max_num_boxes
            valid_boxes = torch.cat([valid_boxes, torch.zeros(max_num_boxes - len(valid_boxes), 4)], dim=0)
            valid_labels = torch.cat([valid_labels, torch.full((max_num_boxes - len(valid_labels),), 0, dtype=torch.int64)], dim=0)

        padded_targets.append({'boxes': valid_boxes, 'labels': valid_labels})

    # Stack images into a batch tensor
    images = torch.stack(transformed_images, dim=0)

    # # Debugging: Check padded targets
    # for i, target in enumerate(padded_targets):
    #     print(f"Target {i} boxes: {target['boxes']}")  # Debugging
    #     print(f"Target {i} labels: {target['labels']}")  # Debugging

    return images, padded_targets


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.show()
