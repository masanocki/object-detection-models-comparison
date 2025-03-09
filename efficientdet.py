import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow_hub as hub
from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO("./annotations/instances_val2017.json")
categories = coco.loadCats(coco.getCatIds())
category_names = [cat["name"] for cat in categories]

# Load the model from TF Hub
model = hub.load(
    "https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d7/1"
)

# Load and preprocess image
image_path = "./assets/test.jpg"
image = Image.open(image_path)
image_np = np.array(image)

# Convert to tensor, add batch dimension
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

# Run detection
detections = model(input_tensor)

# Process detections
boxes = detections["detection_boxes"][0].numpy()
scores = detections["detection_scores"][0].numpy()
classes = detections["detection_classes"][0].numpy().astype(np.int32)

# Filter detections
threshold = 0.3
mask = scores > threshold
boxes = boxes[mask]
scores = scores[mask]
classes = classes[mask]

# Get image dimensions
height, width = image_np.shape[:2]

# Visualize results
plt.figure(figsize=(12, 8))
plt.imshow(image)

for box, score, label in zip(boxes, scores, classes):
    # Convert normalized coordinates to pixel values
    y1, x1, y2, x2 = box
    x1 = int(x1 * width)
    x2 = int(x2 * width)
    y1 = int(y1 * height)
    y2 = int(y2 * height)

    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
    )
    plt.gca().add_patch(rect)

    # Get class name from COCO annotations (subtract 1 as COCO classes are 1-indexed)
    class_name = (
        category_names[label - 1]
        if label - 1 < len(category_names)
        else f"Unknown ({label})"
    )
    plt.text(
        x1,
        y1,
        f"{class_name}: {score:.2f}",
        bbox=dict(facecolor="white", alpha=0.7),
        fontsize=8,
    )

plt.axis("off")
plt.show()
