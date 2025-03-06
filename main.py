from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    w,
)
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# Load COCO class names
with open("coco_classnames.json") as f:
    coco_classnames = json.load(f)

# Invert the dictionary to map indices to class names
COCO_INSTANCE_CATEGORY_NAMES = {v: k for k, v in coco_classnames.items()}

image_path = "./assets/test.jpg"
image = Image.open(image_path).convert("RGB")

# Preprocess the image
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform object detection
# model = fasterrcnn_resnet50_fpn_v2(
#     pretrained=True, weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# )
model = retinanet_resnet50_fpn_v2(
    pretrained=True, weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
)
model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    predictions = model(image_tensor)

# Post-process the results
pred_boxes = predictions[0]["boxes"].cpu().numpy()
pred_scores = predictions[0]["scores"].cpu().numpy()
pred_labels = predictions[0]["labels"].cpu().numpy()

# Visualize the detections
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# Draw bounding boxes
for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
    if score > 0.5:  # Only display confident detections
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        ax.text(
            x_min,
            y_min,
            f"{class_name}: {score:.2f}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

plt.show()
