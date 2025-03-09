import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO

# Load COCO annotations
coco = COCO("./annotations/instances_val2017.json")
categories = coco.loadCats(coco.getCatIds())
category_names = [cat["name"] for cat in categories]

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

# Load and process image
image_path = "./assets/test.jpg"
image = Image.open(image_path)

# Prepare image for the model
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process predictions
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.3
)[0]

# Visualize results
plt.figure(figsize=(12, 8))
plt.imshow(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = box.tolist()
    x1, y1, x2, y2 = box

    # Create rectangle patch
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
    )
    plt.gca().add_patch(rect)

    # Get class name from COCO annotations
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
