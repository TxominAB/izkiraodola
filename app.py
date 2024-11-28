import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8m custom model
model = YOLO('customobject_Yolov8m_modelv2.pt')  # Replace with the path to your trained YOLOv8m model

# Function to run inference on the uploaded image
def run_inference(image):
    results = model(image)
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    annotated_image = image.copy()
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy().astype(int)
        # Draw the rectangle (bounding box) without label and confidence
        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return annotated_image

# Streamlit app layout
st.title("THC counter")
st.write("Upload an image to predict THC.")

# Image uploader
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the uploaded image
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, channels="RGB", use_column_width=True)

    # Run inference
    st.write("Running inference...")
    results = run_inference(image)

    # Draw bounding boxes on the image
    annotated_image = draw_boxes(image, results)

    # Display the image with detected objects
    st.subheader("Image with Detected Objects")
    st.image(annotated_image, channels="RGB", use_column_width=True)

    # Calculate the total number of objects detected
    total_objects = len(results[0].boxes)

    # Display the total count
    st.subheader(f"Total Objects Detected: {total_objects}")