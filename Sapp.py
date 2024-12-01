import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cal import load_model, predict_image, calculate_calories

# Load the model
model = load_model()

# Set up the sidebar
st.sidebar.title("FoodVision")
st.sidebar.write("Upload an image or use your camera to take a picture.")

option = st.sidebar.selectbox(
    'How would you like to provide the image?',
    ('Upload an image', 'Use camera')
)

image_path = None
if option == 'Upload an image':
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_path = "uploaded_image.jpg"
        image.save(image_path)
elif option == 'Use camera':
    camera_image = st.sidebar.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image_path = "camera_image.jpg"
        image.save(image_path)

if image_path:
    # Display the image and classification results in columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Captured Image.', use_container_width=True)
        st.write("")
        st.write("Classifying...")

    # Predict the image
    image_with_boxes, detection_details = predict_image(image_path, model)

    with col2:
        # Display the image with bounding boxes and labels
        st.image(image_with_boxes, caption='Processed Image.', use_container_width=True)

    # Calculate and display detected items and their calories
    detected_items = calculate_calories(detection_details)
    st.markdown("<h3>Detection Results:</h3>", unsafe_allow_html=True)
    for item, calories, confidence in detected_items:
        st.markdown(f"<p style='font-size:18px;'>✓ Detected {item} ({calories} cal/100g) - Confidence: {confidence:.2%}</p>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            color: black;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
        <p>Food Calorie Detector © 2023</p>
    </div>
""", unsafe_allow_html=True)