import os
import time
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.dataset import ImageDataset
from torch.optim.lr_scheduler import StepLR
from models.object_detection import ObjectDetectionModel
from utils.helper_functions import load_config, collate_fn, plot_losses
from utils.transform import CustomTransform
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

def train_model(config):
    # Initialize the dataset
    image_transforms = transforms.Compose([
    transforms.Resize((config['train']['image_size'], config['train']['image_size'])),  # Resize image 
    transforms.ToTensor()           # Convert image to a tensor
    ])
    # Wrap the image transformations in the custom transform class
    transform = CustomTransform(transform= image_transforms)

    dataset = ImageDataset(
        dir =config['train']['directory'],
        annotations_file=config['detection']['file'],
        transform=transform
    )

    # Dataset splitting (90% training, 10% validation)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize the model
    model = ObjectDetectionModel(config['model'])
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config['train']['lr'], weight_decay=1e-4)

    # Scheduler for learning rate reduction
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Model saving directory
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Training variables
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    print("Training started...")
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = [img.to(model.device) for img in images]
            targets = [{k: v.to(model.device) for k, v in t.items()} for t in targets]


            # Compute loss
            loss = model.compute_loss(images, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # Log the losses to TensorBoard
            writer.add_scalar('Training Loss', avg_loss, epoch * len(train_dataloader) + batch_idx)

            print(f"Epoch [{epoch + 1}/{config['train']['epochs']}], "
                  f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Avg Loss: {avg_loss:.4f}, "
                  f"Time Elapsed: {time.time() - start_time:.2f}s")

        train_losses.append(avg_loss)

        # Validation
        avg_val_loss = validate_model(model, val_dataloader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

        val_losses.append(avg_val_loss)

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

        # Adjust learning rate
        scheduler.step()

    writer.close()  # Close the TensorBoard writer
    print("Training completed!")
    plot_losses(train_losses, val_losses)

def validate_model(model, val_dataloader):
    model.model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for images, targets in val_dataloader:
            images = [image.to(model.device) for image in images]  # Move to device
            targets = [{k: v.to(model.device) for k, v in t.items()} for t in targets]

            # Perform forward pass to get the loss values
            loss_dict = model.compute_loss(images, targets)

            # If loss_dict is a tensor, simply sum it
            if isinstance(loss_dict, torch.Tensor):
                val_loss = loss_dict.sum()
            else:
                # Ensure all tensors are dense (convert sparse tensors to dense format)
                loss_dict = {k: v.to_dense() if v.is_sparse else v for k, v in loss_dict.items()}
                # Sum all losses (classification, bbox regression, etc.)
                val_loss = sum(loss for loss in loss_dict.values())

            total_loss += val_loss

    avg_val_loss = total_loss / len(val_dataloader)  # Compute average loss
    return avg_val_loss


if __name__ == "__main__":
    config = load_config('config/config.yaml')  # Ensure config.yaml has correct paths and settings
    train_model(config)
