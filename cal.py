# cal.py

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# Configuration class
class Config:
    
    CLASSES = ['asparagus', 'avocados', 'broccoli', 'cabbage',
               'celery', 'cucumber', 'green_apples',
               'green_beans', 'green_capsicum', 'green_grapes', 'kiwifruit',
               'lettuce', 'limes', 'peas', 'spinach']
    
    CALORIES_DICT = {
        'asparagus': 20,
        'avocados': 160,
        'broccoli': 55,
        'cabbage': 25,
        'celery': 16,
        'cucumber': 16,
        'green_apples': 52,
        'green_beans': 31,
        'green_capsicum': 20,
        'green_grapes': 69,
        'kiwifruit': 61,
        'lettuce': 15,
        'limes': 30,
        'peas': 81,
        'spinach': 23
    }

# Load the model
@st.cache_resource
def load_model():
    model = YOLO('./best.pt')
    return model

# Function to make predictions on a single image
def predict_image(image_path, model, conf_threshold=0.03):
    # Perform inference on the image
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=conf_threshold
    )
    
    # Load the image for visualization
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To store detailed information about detections
    detection_details = []
    
    # Iterate over detections
    for result in results[0].boxes.data:
        # Extract bounding box coordinates, confidence score, and class ID
        x1, y1, x2, y2, confidence, class_id = result.cpu().numpy()
        
        # Draw the bounding box with top confidence score
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
        label = f"{Config.CLASSES[int(class_id)]}: {confidence:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        
        # Save details for printing below
        detection_details.append({
            "class": Config.CLASSES[int(class_id)],
            "top_confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        })
    
    return image, detection_details

# Function to calculate detected items and their calories
def calculate_calories(detection_details):
    detected_items = []
    
    for det in detection_details:
        item = det["class"]
        calories = Config.CALORIES_DICT[item]
        confidence = det["top_confidence"]
        detected_items.append((item, calories, confidence))
    
    return detected_items
