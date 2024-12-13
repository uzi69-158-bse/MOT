# Multiple Object Tracking (MOT) System

## Overview
This project implements a **Multiple Object Tracking (MOT) system** using PyTorch. The system performs the following tasks:
- Takes video/images input and detects objects (e.g., cars).
- Tracks objects across frames, assigning unique IDs to maintain continuity.
- Calculates the speed of tracked objects in real-time.
- Displays the output video with annotations and saves it to a specified directory.

### Key Features
- **Object Detection**: Using Faster R-CNN for accurate detection.
- **Tracking Algorithm**: Integration with DeepSORT for robust tracking.
- **Speed Calculation**: Computes object speed dynamically.
- **Optimization**: Incorporates advanced NUMA-aware techniques for efficient memory management (optional).
- **Platform Compatibility**: Configured for multi-core CPUs without NUMA support (e.g., Intel Core i7 processors).

---

## Project Structure
MOT_System/ ├── data/ │ ├── train/ # Training data  │ ├── test/ # Testing data (video files) │ └── output/ # Processed output videos ├── models/ │ ├── fasterrcnn_weights.pth # Pretrained Faster R-CNN weights │ └── tracker/ # Tracking model files ├── trackers/ │ ├── deep_sort.py # DeepSORT implementation ├── utils/ │ ├── dataset.py # Dataset processing utilities │ ├── speed_calculation.py # Speed computation logic │ ├── visualization.py # Visualization utilities │ ├── numa_allocator.py # NUMA-aware memory management (optional) ├── main.py # Main entry point for running the MOT system ├── train.py # Script for model training ├── test.py # Script for model testing ├── README.md # Project documentation └── requirements.txt # Python dependencies

yaml
Copy code

---

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/MOT_System.git
   cd MOT_System
Set up a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Run the MOT System
To process a video and save the output:

bash
Copy code
python main.py --video_path <path_to_video> --output_dir <path_to_output>
Example:

bash
Copy code
python main.py --video_path data/test/video.mp4 --output_dir data/output/
Train the Detection Model
To train the object detection model on your data:

bash
Copy code
python train.py --epochs 10 --lr 0.001
Test the System
Evaluate the MOT system on the test dataset:

bash
Copy code
python test.py --test_dir data/test/
Configuration
Detection Threshold: Set in main.py (default: 0.5).
Model Configuration: Modify in models/ for detection type and tracking behavior.
NUMA Support: Optional. Modify numa_allocator.py for NUMA-aware memory allocation.
Output
The system generates a processed video in the specified output directory with the following:

Bounding boxes around detected objects.
Unique IDs for each object.
Speed annotations in km/h (or your preferred unit).
Requirements
Python 3.8 or higher
PyTorch 1.10 or higher
Additional libraries (listed in requirements.txt)
Future Enhancements
Extend support for real-time processing with GPU optimization.
Incorporate more advanced tracking algorithms.
Optimize speed calculation for better accuracy.
Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or support, please contact:

Name: Uzair