# cal.py

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# Configuration class
class Config:
    
    CLASSES = ['asparagus', 'avocados', 'broccoli', 'cabbage',        #4
               'celery', 'cucumber', 'green_apples', 'green_beans', #4
               'green_capsicum', 'green_grapes', 'kiwifruit', #3
               'lettuce', 'limes', 'peas', 'spinach',  #4
               'Banana', 'Cauliflower', 'Date', 'Garlic', #4
               'Ginger', 'Mushroom', 'Onion', 'Parsnip', #4
               'Peach', 'Pear', 'Potato', 'Turnip', #4
               'Beetroot', 'Blackberry', 'Blueberry', 'Cherry', #4
               'Eggplant', 'Plum', 'Purple asparagus', 'Purple grapes',  #4
               'Radish', 'Raspberry', 'Red Apple', 'Red Grape', #4
               'Red cabbage', 'Red capsicum', 'Strawberry', 'Tomato', #4
               'Watermelon', 'apricot', 'carrot', 'corn', #4
               'grapefruit', 'lemon', 'mango', 'nectarine', #4
               'orange', 'pineapple', 'pumpkin', 'sweet_potato'] #4
    
    CALORIES_DICT = {
        # Green foods (existing)
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
        'spinach': 23,
        
        # White/Beige foods
        'Banana': 89,
        'Cauliflower': 25,
        'Date': 282,
        'Garlic': 149,
        'Ginger': 80,
        'Mushroom': 22,
        'Onion': 40,
        'Parsnip': 75,
        'Peach': 39,
        'Pear': 57,
        'Potato': 77,
        'Turnip': 28,
        
        # Purple/Red foods
        'Beetroot': 43,
        'Blackberry': 43,
        'Blueberry': 57,
        'Cherry': 50,
        'Eggplant': 25,
        'Plum': 46,
        'Purple asparagus': 20,
        'Purple grapes': 69,
        'Radish': 16,
        'Raspberry': 52,
        'Red Apple': 52,
        'Red Grape': 69,
        'Red cabbage': 31,
        'Red capsicum': 31,
        'Strawberry': 32,
        'Tomato': 18,
        'Watermelon': 30,
        
        # Orange/Yellow foods
        'apricot': 48,
        'carrot': 41,
        'corn': 86,
        'grapefruit': 42,
        'lemon': 29,
        'mango': 60,
        'nectarine': 44,
        'orange': 47,
        'pineapple': 50,
        'pumpkin': 26,
        'sweet_potato': 86
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
    """
    Calculate calories for detected items, keeping only the highest confidence detection for each unique food item.
    
    Args:
        detection_details: List of dictionaries containing detection information
            Each dict has keys: "class" (food name), "top_confidence" (detection confidence), "bbox"
            
    Returns:
        List of tuples: (food_item, calories, confidence) for unique items with highest confidence
    """
    # Dictionary to keep track of highest confidence detection for each food item
    unique_items = {}
    
    # Process each detection
    for det in detection_details:
        item = det["class"]
        confidence = det["top_confidence"]
        
        # Only update if this is the first instance or has higher confidence
        if item not in unique_items or confidence > unique_items[item]["confidence"]:
            unique_items[item] = {
                "calories": Config.CALORIES_DICT[item],
                "confidence": confidence
            }
    
    # Convert to list of tuples format
    detected_items = [
        (item, data["calories"], data["confidence"])
        for item, data in unique_items.items()
    ]
    
    # Sort by confidence (optional)
    detected_items.sort(key=lambda x: x[2], reverse=True)
    
    return detected_items