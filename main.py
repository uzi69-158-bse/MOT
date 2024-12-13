import os
import numpy as np
from torchvision import transforms
import traceback
from models.object_detection import ObjectDetectionModel
from utils.dataset import ImageDataset
from utils.visualization import display_frame
from utils.helper_functions import load_config
from utils.numa_allocator import allocate_numa_memory, deallocate_numa_memory
from utils.numa_profiler import monitor_memory_usage
from utils.numa_scheduler import display_numa_info
from trackers.deep_sort import DeepSort

# Transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    try:
        # Load configuration
        config = load_config('config/config.yaml')
        print("Configuration: Using default settings for testing.")

        # Initialize dataset, model, and tracker
        dataset = ImageDataset(config['test']['directory'], config['detection']['file'])
        model = ObjectDetectionModel(config['model'])
        model.eval()  # Set the model to evaluation mode once
        print("Object detection model initialized.")

        tracker = DeepSort(config['tracking'])
        print("Tracker initialized.")

        # Prepare output directory
        output_image_dir = 'output/detections/'
        os.makedirs(output_image_dir, exist_ok=True)
        print(f"Output images will be saved to: {output_image_dir}")
        
        # Handle NUMA-related logic based on the config
        if config.get('numa', {}).get('enable', False):
            print("[DEBUG] NUMA is enabled.")

            # Allocate NUMA memory if the node is specified
            if config.get('numa', {}).get('node', -1) >= 0:
                print(f"[DEBUG] Allocating NUMA memory to node {config['numa']['node']}.")
                allocate_numa_memory(config['numa']['node'])

            # Optionally use NUMA profiler
            if config.get('numa', {}).get('profiler', False):
                print("[DEBUG] NUMA profiler enabled.")
                monitor_memory_usage()
        else:
            print("[DEBUG] NUMA is disabled. Proceeding with normal processing.")

        # Frame processing loop
        for frame_idx, allocated_frame in enumerate(dataset):
            try:
                print(f"Processing frame {frame_idx}...")

                # If it's a tuple, extract the image
                if isinstance(allocated_frame, tuple):
                    allocated_frame = allocated_frame[0]

                print(f"Frame type: {type(allocated_frame)}")

                # Transform frame
                allocated_frame = transform(allocated_frame)

                # Object detection
                detections = model.detect(allocated_frame, frame_idx)
                if not detections:
                    print(f"No detections in frame {frame_idx}.")
                    continue
               
                # Visualize and save the frame
                processed_frame = display_frame(allocated_frame, detections, output_image_dir, frame_idx)
                save_image(processed_frame, os.path.join(output_image_dir, f"frame_{frame_idx}.jpg"))

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                traceback.print_exc()
            # After processing all frames, deallocate NUMA resources if NUMA is enabled
        if config.get('numa', {}).get('enable', False):
            # Deallocate NUMA resources
            print("[DEBUG] Deallocating NUMA resources.")
            deallocate_numa_memory()

            # Optionally, run NUMA scheduler if enabled
            if config.get('numa', {}).get('scheduler', False):
                print("[DEBUG] NUMA scheduler enabled.")
                display_numa_info()

    except Exception as e:
        print(f"Critical error occurred: {e}")
        traceback.print_exc()

# Function to save the processed frame to the output directory
def save_image(image, path):
    try:
        import cv2
        processed_frame_np = np.array(image)  # Ensure image is in numpy format
        cv2.imwrite(path, cv2.cvtColor(processed_frame_np, cv2.COLOR_RGB2BGR))
        print(f"Saved frame to {path}")
    except Exception as e:
        print(f"Error saving image {path}: {e}")

if __name__ == "__main__":
    main()
