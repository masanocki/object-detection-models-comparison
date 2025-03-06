from ultralytics import YOLO
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# model = YOLO("./models/yolo12n.pt")

# results = model("./assets/test.jpg")

# for result in results:
#     result.show()

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

image_path = "./assets/test.jpg"
image = Image.open(image_path).convert("RGB")

# Preprocess the image
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform object detection
model = fasterrcnn_resnet50_fpn_v2(
    pretrained=True, weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
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
