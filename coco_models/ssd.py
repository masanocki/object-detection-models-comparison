import json
import time
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.transforms import functional as F

from metrics import *


def run_ssd_for_videos(media_path, device):

    with open("./coco_classnames.json", "r") as f:
        class_names = {int(v): k for k, v in json.load(f).items()}

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    results_data = []
    for video in media_path.glob("*.avi"):
        if device == "cuda":
            torch.cuda.empty_cache()
        cap = cv2.VideoCapture(video)
        frame_count = 0
        frame_times = []
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = F.to_tensor(rgb_frame).to(device)

                frame_start_time = time.time()
                with torch.no_grad():
                    predictions = model([image])

                if device == "cuda":
                    torch.cuda.synchronize()

                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                frame_count += 1

                boxes = predictions[0]["boxes"].cpu()
                scores = predictions[0]["scores"].cpu()
                labels = predictions[0]["labels"].cpu()

                confidence_threshold = 0.5
                mask = scores > confidence_threshold
                boxes = boxes[mask].numpy()
                scores = scores[mask].numpy()
                labels = labels[mask].numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names.get(int(label), "Unknown")
                    label_text = f"{class_name}: {score:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("SSD300 Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
            del predictions
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        metrics = calculate_fps_and_time(frame_times, total_time, frame_count)

        results_data.append(
            {
                "video_name": video.name,
                "min_fps": metrics["min_fps"],
                "avg_fps": metrics["avg_fps"],
                "max_fps": metrics["max_fps"],
                "total_detection_time": total_time,
                "avg_frame_time": metrics["avg_frame_time"],
                "device": device,
                "frames_processed": frame_count,
            }
        )
    print(results_data)


def run_ssd_for_images(media_path, device):

    with open("./coco_classnames.json", "r") as f:
        class_names = {int(v): k for k, v in json.load(f).items()}

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    results_data = []
    frame_times = []
    start_time = time.time()
    processed_count = 0

    for image_path in media_path.glob("*.jpg"):

        image = cv2.imread(image_path)

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = F.to_tensor(rgb_frame).to(device)

        frame_start_time = time.time()
        with torch.no_grad():
            predictions = model([image_tensor])

        if device == "cuda":
            torch.cuda.synchronize()

        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        processed_count += 1

        boxes = predictions[0]["boxes"].cpu()
        scores = predictions[0]["scores"].cpu()
        labels = predictions[0]["labels"].cpu()

        confidence_threshold = 0.5
        mask = scores > confidence_threshold
        boxes = boxes[mask].numpy()
        scores = scores[mask].numpy()
        labels = labels[mask].numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names.get(int(label), "Unknown")
            label_text = f"{class_name}: {score:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("SSD300 Detection", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    if device == "cuda":
        del predictions
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    metrics = calculate_fps_and_time(frame_times, total_time, processed_count)

    results_data.append(
        {
            "folder_path": media_path,
            "total_detection_time": total_time,
            "avg_image_time": metrics["avg_frame_time"],
            "device": device,
            "images_processed": processed_count,
        }
    )

    print(results_data)
