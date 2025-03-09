import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Load COCO annotations
coco = COCO("./annotations/instances_val2017.json")
categories = coco.loadCats(coco.getCatIds())
category_names = [cat["name"] for cat in categories]

# Initialize detector
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="efficientdet_lite2.tflite"),
    max_results=5,
    score_threshold=0.3,
    running_mode=VisionRunningMode.IMAGE,
)

# Load and process image
image_path = "./assets/test.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Perform detection
with ObjectDetector.create_from_options(options) as detector:
    detection_result = detector.detect(mp_image)

# Visualize results
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)

for detection in detection_result.detections:
    bbox = detection.bounding_box
    x1 = bbox.origin_x
    y1 = bbox.origin_y
    x2 = x1 + bbox.width
    y2 = y1 + bbox.height

    # Create rectangle patch
    rect = patches.Rectangle(
        (x1, y1), bbox.width, bbox.height, linewidth=2, edgecolor="r", facecolor="none"
    )
    plt.gca().add_patch(rect)

    # Get category and score
    category = detection.categories[0]
    class_name = category.category_name
    score = category.score

    plt.text(
        x1,
        y1,
        f"{class_name}: {score:.2f}",
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=8,
    )

plt.axis("off")
plt.show()
