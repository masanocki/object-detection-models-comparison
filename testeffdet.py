import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from pathlib import Path

# Path to the video file
video_path = "./dataset/rugby/video/1.avi"

# Load the EfficientDet-D0 model from TensorFlow Hub
model_path = Path("./models/efficientdet-d0")
detector = tf.saved_model.load(model_path)


# Function to preprocess the video frame
def preprocess_image(image):
    # Convert the image to uint8 and add a batch dimension
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)[tf.newaxis, ...]
    return input_tensor


# Function to draw bounding boxes on the frame
def draw_detections(frame, boxes, class_names, scores, threshold=0.5):
    h, w, _ = frame.shape
    for box, class_name, score in zip(boxes, class_names, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        label = f"Class {class_name}: {int(score * 100)}%"  # Use class_name directly
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (TensorFlow expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    input_tensor = preprocess_image(rgb_frame)

    # Run inference
    detections = detector(input_tensor)

    # Extract detection results
    boxes = detections["detection_boxes"].numpy()[0]
    class_names = detections["detection_classes"].numpy()[0].astype(int)
    scores = detections["detection_scores"].numpy()[0]

    # Draw detections on the frame
    draw_detections(frame, boxes, class_names, scores, threshold=0.5)

    # Display the frame
    cv2.imshow("EfficientDet-D0 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
