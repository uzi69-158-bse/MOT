import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import PIL, logging
class ObjectDetectionModel(nn.Module):
    def __init__(self, config):
        """
        Initialize the object detection model based on the provided configuration.
        Args:
            config (dict): Configuration dictionary containing model type, device, and class labels.
        """
        super(ObjectDetectionModel, self).__init__()

        # Set up the device based on availability
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load the correct model based on config
        model_config = config['type']  # Access the model config section
        if model_config == 'faster_rcnn':
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        elif model_config == 'ssd':
            self.model = ssd300_vgg16(pretrained=True)  # Use the correct SSD model
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

        # Predefined class labels from the config (extendable for custom datasets)
        self.class_labels = config.get('class_labels', []) 

        # Move the model to the specified device (GPU or CPU)
        self.model.to(self.device)
        
    def forward(self, images):
        """
        Forward pass for object detection model.
        
        Args:
            images (tensor): Input images to the model, should be a batch of images in tensor format.
        
        Returns:
            predictions (list): List of predictions from the model.
        """
        return self.model(images)
    


    def detect(self, frame, frame_idx):
        """
        Run object detection on a single frame in evaluation mode.

        Args:
            frame (PIL.Image.Image or torch.Tensor): A single frame (image) to process.
            frame_idx (int): Index of the frame being processed.

        Returns:
            list: A list of dictionaries, each containing detections for the frame.
        """
        self.model.eval()  # Set the model to evaluation mode

        # Define the transformation pipeline
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # Apply transformation to the frame
        if isinstance(frame, PIL.Image.Image):
            frame_tensor = transform(frame).unsqueeze(0)  # Add batch dimension (shape: [1, 3, H, W])
        elif isinstance(frame, torch.Tensor):
            frame_tensor = frame.unsqueeze(0)  # Add batch dimension
        else:
            raise TypeError(f"Expected a PIL Image or Tensor, got {type(frame)}")

        frame_tensor = frame_tensor.to(self.device)  # Move tensor to the appropriate device

        with torch.no_grad():
            try:
                # Make predictions with the model
                prediction = self.model(frame_tensor)[0]  # Predictions for one frame
            except Exception as e:
                logging.error(f"Error during model prediction for frame {frame_idx}: {e}")
                return []

        frame_detections = []

        # Check if required keys exist in the prediction
        required_keys = {'boxes', 'scores', 'labels'}
        if not required_keys.issubset(prediction.keys()):
            logging.warning(f"Missing keys in prediction for frame {frame_idx}. Keys found: {prediction.keys()}")
            return []  # Return empty if prediction is malformed

        for i in range(len(prediction['boxes'])):
            score = prediction['scores'][i].item()
            if score < 0.5:  # Confidence threshold
                continue

            # Extract and format the bounding box
            bbox = prediction['boxes'][i].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox

            # Get label and handle out-of-bound indices
            label_idx = int(prediction['labels'][i].item())
            label_name = self.class_labels[label_idx] if label_idx < len(self.class_labels) else 'Unknown'
            if label_name == 'Unknown':
                logging.warning(f"Unknown label index {label_idx} for frame {frame_idx}.")

            # Store detection information in dictionary format
            detection = {
                'frame_idx': frame_idx,
                'bbox': {
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                },
                'score': score,
                'label': label_name
            }
            frame_detections.append(detection)

        return frame_detections




    def compute_loss(self, images, targets):
        """
        Compute the loss based on predictions and targets during training.
        
        Args:
            images (list): List of input images.
            targets (list): List of target dictionaries, containing bounding boxes and labels.

        Returns:
            total_loss (Tensor): The total loss computed from the model.
        """
        self.model.train()  # Set the model to training mode
        # Perform forward pass to get the loss values
        loss_dict = self.model(images, targets)  # Targets are passed here during training

        # Calculate total loss by summing all individual losses (classification, bbox regression, etc.)
        total_loss = sum(loss for loss in loss_dict.values())

        return total_loss

    def save_model(self, path, epoch=None):
        """
        Save the trained model to a file.
        
        Args:
            path (str): Path to save the model.
            epoch (int, optional): The epoch at which the model is saved (useful for resuming training).
        """
        checkpoint = {'model_state_dict': self.model.state_dict()}
        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the trained model from a file.
        
        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            print(f"Model loaded from {path}, epoch {epoch}")
        else:
            print(f"Model loaded from {path}")
        self.model.to(self.device)
