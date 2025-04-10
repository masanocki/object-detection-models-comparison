import json
import time

import cv2
import torch
from effdet import DetBenchPredict, create_model
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

from models.utils.metrics import *


def run_efficientdet_coco_videos(media_path, device):

    with open("./coco_classnames.json", "r") as f:
        class_names = {int(v): k for k, v in json.load(f).items()}

    model = create_model("tf_efficientdet_d0", pretrained=True, num_classes=90)
    model = DetBenchPredict(model).to(device)
    model.eval()

    torch.backends.cudnn.benchmark = True
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
                original_height, original_width = frame.shape[:2]
                resized_frame = cv2.resize(frame, (512, 512))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                input_tensor = (
                    to_tensor(rgb_frame).unsqueeze(0).to(device, non_blocking=True)
                )
                frame_start_time = time.time()
                with torch.no_grad():
                    detections = model(input_tensor)
                    detections = detections[0].cpu()

                if device == "cuda":
                    torch.cuda.synchronize()

                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                frame_count += 1

                boxes = detections[:, :4].cpu()
                scores = detections[:, 4].cpu()
                labels = detections[:, 5].cpu()

                confidence_threshold = 0.25
                keep = scores > confidence_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                keep = nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                scale_x = original_width / 512
                scale_y = original_height / 512
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_name = class_names.get(int(label.item()), "Unknown")
                    label_text = f"{class_name}: {score.item():.2f}"

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

                cv2.imshow("EfficientDet-D0", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
            del detections
            del input_tensor
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


def run_efficientdet_coco_images(media_path, device):

    with open("./coco_classnames.json", "r") as f:
        class_names = {int(v): k for k, v in json.load(f).items()}

    model = create_model("tf_efficientdet_d0", pretrained=True, num_classes=90)
    model = DetBenchPredict(model).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    results_data = []
    frame_times = []
    start_time = time.time()
    processed_count = 0

    for image_path in media_path.glob("*.jpg"):
        image = cv2.imread(image_path)

        original_height, original_width = image.shape[:2]
        resized_frame = cv2.resize(image, (512, 512))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_tensor = to_tensor(rgb_frame).unsqueeze(0).to(device, non_blocking=True)

        frame_start_time = time.time()
        with torch.no_grad():
            detections = model(input_tensor)
            detections = detections[0].cpu()

        if device == "cuda":
            torch.cuda.synchronize()

        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        processed_count += 1

        boxes = detections[:, :4].cpu()
        scores = detections[:, 4].cpu()
        labels = detections[:, 5].cpu()

        confidence_threshold = 0.25
        keep = scores > confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        keep = nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        scale_x = original_width / 512
        scale_y = original_height / 512
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = class_names.get(int(label.item()), "Unknown")
            label_text = f"{class_name}: {score.item():.2f}"
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
        cv2.imshow("EfficientDet-D0", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    if device == "cuda":
        del detections
        del input_tensor
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
