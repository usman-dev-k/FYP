from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the model
model = YOLO("/home/usman/FYP/best.pt") 

# Predict function
def detect_objects(image: Image.Image):
    image_array = np.array(image)
    results = model(image_array)[0]
    annotated_frame = results.plot()  # draw boxes
    return Image.fromarray(annotated_frame)

