import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' if you don't need to display plots

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import PIL
import numpy as np
import cv2
from PIL import Image
import torch


def visualize_predictions(images, predictions, targets, save_path=None):
    for idx, (image, prediction, target) in enumerate(zip(images, predictions, targets)):
        print(f"Visualizing image {idx}")
        try:
            # Convert image tensor to numpy
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype('uint8')  # Ensure proper scaling
            
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")

            # Create figure
            fig, ax = plt.subplots(1, figsize=(12, 9))
            ax.imshow(image)

            # Plot predicted boxes
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(box[0], box[1], f'{label.item()} {score.item():.2f}', fontsize=12, color='red')

            # Plot target boxes
            for box, label in zip(target['boxes'], target['labels']):
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='g', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(box[0], box[1], f'{label.item()}', fontsize=12, color='green')

            # Save or display
            if save_path:
                plt.savefig(f'{save_path}_{idx}.png')
                print(f"Saved visualization to {save_path}_{idx}.png")
            else:
                plt.show()

            plt.close()
        except Exception as e:
            print(f"Error during visualization of image {idx}: {e}")



def display_frame(image, detections, output_image_dir, frame_idx):
    """
    Draw detections on a frame and save it.

    Args:
        image (PIL.Image.Image or np.ndarray or torch.Tensor): The frame image.
        detections (list): List of detection dictionaries.
        output_image_dir (str): Directory to save the output image.
        frame_idx (int): Index of the frame.

    Returns:
        np.ndarray: Processed frame with visualized detections.
    """
    # Convert torch.Tensor to NumPy array
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Move tensor to CPU and convert to NumPy
        if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        print(f"[DEBUG] Converted tensor to array. Shape: {image.shape}, dtype: {image.dtype}")

    # Convert PIL Image to NumPy array
    if isinstance(image, PIL.Image.Image):
        processed_frame = np.array(image)
        print(f"[DEBUG] Converted PIL image to array. Shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")
    elif isinstance(image, np.ndarray):
        processed_frame = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure frame has 3 channels (for color images)
    if processed_frame.ndim == 2:  # Grayscale image
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    elif processed_frame.shape[-1] == 4:  # RGBA
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)
    elif processed_frame.shape[-1] == 3:  # RGB
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

    # Debug: Check pixel value range before processing
    print(f"[DEBUG] Processed frame min/max values: {processed_frame.min()}, {processed_frame.max()}")

    # Scale the image to 0-255 and convert to uint8 only if needed
    if processed_frame.dtype == np.float32 or processed_frame.dtype == np.float64:
        processed_frame = np.clip(processed_frame * 255, 0, 255).astype(np.uint8)
    elif processed_frame.dtype == np.uint8:
        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

    # Debug: Check pixel value range after scaling and conversion
    print(f"[DEBUG] After scaling and conversion, frame min/max values: {processed_frame.min()}, {processed_frame.max()}")

    # Draw detections
    for detection in detections:
        bbox = detection['bbox']
        try:
            x_min = int(bbox['x_min'])
            y_min = int(bbox['y_min'])
            x_max = int(bbox['x_max'])
            y_max = int(bbox['y_max'])
        except KeyError as e:
            print(f"Missing key in bbox: {e}")
            continue

        # Debug: Ensure that detection bounding box is valid
        print(f"[DEBUG] Drawing bbox: ({x_min}, {y_min}), ({x_max}, {y_max})")

        # Draw the bounding box
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), color, thickness=2)

        # Add label text
        label = detection.get('label', 'Unknown')
        score = detection.get('score', 0.0)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(processed_frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

    # Convert back to RGB before saving (OpenCV uses BGR by default)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Debug: Check final image shape and dtype
    print(f"[DEBUG] Final frame shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")

    # Save image to the specified directory
    output_path = f"{output_image_dir}/frame_{frame_idx}.jpg"
    try:
        Image.fromarray(processed_frame).save(output_path)
        print(f"Saved frame to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

    return processed_frame




def draw_detections_matplotlib(frame, tracked_objects):
    """
    Draw the detections and tracking information on the frame.

    Args:
        frame (Tensor or ndarray): The input frame (image) to be drawn on.
        tracked_objects (list of dict): The tracked objects containing 'id', 'bbox', etc.
    """
    # Ensure frame is a numpy array
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy().transpose(1, 2, 0)  # Convert from CHW to HWC format
    
    # Draw the bounding boxes for the tracked objects
    for obj in tracked_objects:
        bbox = obj['bbox']  # Format: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox

        # Draw the bounding box
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                          linewidth=2, edgecolor='red', facecolor='none'))

        # Label the object with its ID
        obj_id = obj['id']
        plt.text(x_min, y_min, f'ID: {obj_id}', color='red', fontsize=12, verticalalignment='top')

    # Show the frame with detections
    plt.imshow(frame)
    plt.axis('off')
    plt.show()


def convert_frame(frame):
    """
    Convert a frame from tensor format to numpy format for visualization.

    Args:
        frame (torch.Tensor): The input frame as a tensor in CHW format.

    Returns:
        numpy.ndarray: The frame converted to HWC format.
    """
    # Convert from torch.Tensor to numpy array
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC

    # Normalize the frame to [0, 1] range if needed
    if frame.max() > 1.0:  # If the frame has values greater than 1 (possibly float32 type)
        frame = frame / 255.0  # Scale pixel values to [0, 1] range

    return frame
