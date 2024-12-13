from models.object_detection import ObjectDetectionModel
from utils.dataset import ImageDataset
from torch.utils.data import DataLoader
from utils.helper_functions import load_config
import torch
from utils.visualization import visualize_predictions
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from torchvision import transforms


def load_saved_model(config, model_path):
    """
    Load the saved model from the given file path.

    Args:
        config (dict): Configuration dictionary.
        model_path (str): Path to the saved model weights.

    Returns:
        ObjectDetectionModel: Loaded model with weights.
    """
    try:
        # Initialize the model architecture
        model = ObjectDetectionModel(config['model'])

        # Inspect the checkpoint file
        checkpoint = torch.load(model_path, map_location=config['device'])
        if 'model' in checkpoint:
            model.model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.model.load_state_dict(checkpoint['state_dict'])
        else:
            model.model.load_state_dict(checkpoint)  # Assuming raw state_dict

        # Move model to the appropriate device
        model.model.to(config['device'])
        print(f"Model loaded successfully from {model_path}")
    
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise




def test_model(config, model):
    """
    Function to evaluate the model on the test dataset.

    Args:
        config (dict): Configuration dictionary.
        model (ObjectDetectionModel): Trained object detection model.
    """
    try:
        # Initialize the test dataset and DataLoader
        print(f"Initializing test dataset from directory: {config['test']['directory']}")
        test_dataset = ImageDataset(
            dir=config['test']['directory'], 
            annotations_file=config['detection']['file'],  # Ensure this is a valid path
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=lambda x: tuple(zip(*x))
        )

        # Set the model to evaluation mode
        model.model.eval()

        # Define the transform to convert images to tensors
        transform = transforms.Compose([transforms.ToTensor()])  # Convert PIL image to tensor

        # Initialize metrics
        metric = MeanAveragePrecision()

        # Initialize a variable to accumulate loss (if needed)
        total_loss = 0.0

        # Testing loop
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_dataloader):
                print(f"Processing batch {batch_idx + 1}")
                try:
                    images = [transform(img).to(model.device) for img in images]
                    targets = [{k: v.to(model.device) for k, v in t.items()} for t in targets]
                    print("Images and targets prepared, making predictions...")

                    # Get predictions from the model
                    predictions = model.model(images)
                    print(f"Predictions made for batch {batch_idx + 1}")

                    # Update metrics
                    metric.update(predictions, targets)

                    # Visualize predictions (in a try-except block to catch visualization errors)
                    try:
                        print(f"Visualizing predictions for batch {batch_idx + 1}")
                        visualize_predictions(images, predictions, targets)
                    except Exception as e:
                        print(f"Error during visualization for batch {batch_idx + 1}: {e}")
                    
                    print(f"Test Batch [{batch_idx + 1}/{len(test_dataloader)}]")

                except Exception as e:
                    print(f"Error in processing batch {batch_idx + 1}: {e}")

        # Compute final evaluation metrics
        result = metric.compute()

        # Output the results in a structured format
        print("Evaluation Metrics:")
        for metric_name, value in result.items():
            print(f"{metric_name}: {value:.4f}")

        return result

    except Exception as e:
        print(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    try:
        # Load configuration from YAML
        config = load_config('config/config.yaml')
        
        # Path to the saved model weights
        saved_model_path = "saved_models/best_model.pth"  # Replace with your actual path

        # Load the saved model
        model = load_saved_model(config, saved_model_path)

        # Evaluate the model on the test dataset
        print("Starting model evaluation...")
        test_metrics = test_model(config, model)

        # Log final test metrics
        print("Testing completed. Metrics:")
        print(test_metrics)

    except Exception as e:
        print(f"Error in main script: {e}")
