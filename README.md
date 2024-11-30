# FoodVision: Automated Food Detection Using YOLOv8

## Project Overview
FoodVision is a deep learning-based food detection system that utilizes YOLOv8 to identify and classify various food items in images. The system is capable of detecting 55 different food classes with a focus on fruits and vegetables, making it useful for dietary monitoring and nutritional analysis.

## Features
- Real-time food detection using YOLOv8
- Support for 55 different food classes
- Calorie estimation per 100g of detected food items
- Web interface using Streamlit
- Support for both image upload and camera capture
- Bounding box visualization with confidence scores

## Model Architecture
- Base model: YOLOv8n (nano version)
- Input size: 640x640 pixels
- Batch size: 32
- Learning rate: 3e-4
- Training epochs: 45

## Performance Metrics
- mAP50: ~0.8 (80% accuracy at 50% IoU)
- Precision: ~0.8
- Recall: ~0.75

## Installation

1. Clone the repository:
```bash
git clone git@github.com:2302660/aai3001_final_project.git
cd aai3001_final_project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run Sapp.py
```

2. Use the web interface to:
   - Upload images or capture them using your camera
   - View detected food items with bounding boxes
   - See confidence scores and calorie information

## Project Structure
```
.
├── Model.py            # Notebook for model training and evaluation
├── cal.py              # Core calorie calculation and detection functions
├── Sapp.py             # Streamlit web application
├── best.pt             # Trained model weights (not included in repo)
└── README.md           # Project documentation
```

## Supported Food Classes
The model can detect 55 different food items including:
- Green foods: asparagus, avocados, broccoli, cabbage, etc.
- White/Beige foods: banana, cauliflower, garlic, mushroom, etc.
- Purple/Red foods: beetroot, blackberry, cherry, eggplant, etc.
- Orange/Yellow foods: apricot, carrot, corn, mango, etc.

## Live Demo
You can try out the live demo at:
- [Hugging Face Space](https://nightey3s-aai3001-final-project.hf.space/)
- [Project Files](https://huggingface.co/spaces/nightey3s/aai3001_final_project/tree/main)

## Team Members
- Brian Tham
- Hong Ziyang
- Javier Si Zhao Hong
- Timothy Zoe Delaya

## Course Information
AAI3001 Deep Learning and Computer Vision, Trimester 1, 2024
Singapore Institute of Technology

## Future Work
- Expand the dataset to include more food categories.
- Implement portion size estimation.
- Compare uploaded food images with dietary recommendations.